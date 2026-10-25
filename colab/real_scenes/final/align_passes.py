"""Re-place whole walk passes relative to each other, judged by real pixels.
The global SfM placed passes wrongly relative to each other (look-alike places: symmetric corridor, repeated grilled windows), while
each pass is right inside itself (neighbour rebuild ~35 dB, cross-pass rebuild 11-17 dB). So keep every pass as it is and only
search a similarity transform per pass:
  1. hypotheses: SIFT matches between frames of two passes, lifted to 3D with the aligned depth, RANSAC similarity; plus the
     current SfM placement (identity);
  2. judge: rebuild frames of pass A ONLY from pass B's photos under each hypothesis (ibr.synth) and keep the best;
     a look-alike match explains the look-alike object but not the rest of the view, so it scores low;
  3. group passes along the best-scoring links (maximum spanning forest, links >= MIN_DB); each group shares one frame.
Usage: python align_passes.py <cache> <keep.json or -> <out_dir> [--workers 4] [--bridge <prep cache>]. CPU only. Writes <out>/da3_sfm.npz + align_report.json."""
import argparse, json, os, time, numpy as np, cv2
from multiprocessing import Pool
import ibr

VERIFY_DB = 21.0  # a join must agree with the whole group (wrong placements measured 13-17 dB, right ones 28-37)
MIN_DB = 23.0   # 20 dB let a wrong placement through (white walls and floor score ~20 when misplaced)
G = {}

def pieces(fid): return np.cumsum(np.r_[0, np.diff(fid) > 3])

def lift(j, kp, scale):
    """Keypoints (full image px) of frame j -> world points through its aligned depth; NaN where no depth.
    scale = (sx, sy): image px per depth px. NOT one number: frames are 567x1008, depth 280x504 (different aspect)."""
    d = G['d']; z = d['depth'][j]; Hd, Wd = z.shape; x, y = kp[:, 0] / scale[0], kp[:, 1] / scale[1]
    u = np.clip(x.astype(int), 0, Wd - 1); v = np.clip(y.astype(int), 0, Hd - 1)
    zz = z[v, u]; K = d['K'][j]; P = np.stack([(x - K[0, 2]) / K[0, 0] * zz, (y - K[1, 2]) / K[1, 1] * zz, zz], 1)
    W = P @ d['c2w'][j][:3, :3].T + d['c2w'][j][:3, 3]; W[zz <= 0] = np.nan; return W

def umeyama(X, Y):
    """s, R, t with Y ~ s R X + t."""
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt; s = (S * np.diag(D)).sum() / Xc.var(0).sum()
    return s, R, my - s * R @ mx

def ransac(X, Y, it=300, tol=0.06, rng=np.random.default_rng(0)):
    best = (0, None)
    for _ in range(it):
        i = rng.choice(len(X), 3, replace=False); s, R, t = umeyama(X[i], Y[i])
        if not 0.7 < s < 1.4: continue
        inl = np.linalg.norm((s * X @ R.T + t) - Y, axis=1) < tol
        if inl.sum() > best[0]: best = (inl.sum(), inl)
    if best[0] < 25: return None
    s, R, t = umeyama(X[best[1]], Y[best[1]]); return s, R, t, int(best[0])

def feat(g):
    """SIFT of frame id g (full resolution), cached."""
    if g not in G['fe']:
        im = cv2.imread(G['files'][g], cv2.IMREAD_GRAYSCALE); kp, de = G['sift'].detectAndCompute(im, None)
        G['fe'][g] = (np.array([k.pt for k in kp], np.float32), de, (im.shape[1], im.shape[0]))
    return G['fe'][g]

def rel_pose(gi, zi, Ki, gj):
    """Camera of frame gj relative to gi (X_j = R X_i + t): gi's keypoints lifted with its depth zi (DA3 grid, intrinsics Ki),
    found again in gj, PnP + RANSAC. Frames seconds apart in the video: no look-alike confusion possible."""
    (ki, di, wi), (kj, dj, wj) = feat(gi), feat(gj)
    if di is None or dj is None or len(di) < 10 or len(dj) < 10: G['why'] = 'no features'; return None
    mm = [m for m, n in cv2.BFMatcher().knnMatch(di, dj, k=2) if m.distance < 0.8 * n.distance]
    if len(mm) < 20: G['why'] = f'{len(mm)} matches'; return None
    sx, sy = wi[0] / zi.shape[1], wi[1] / zi.shape[0]                                 # per axis: image and depth aspect differ
    pi = ki[[m.queryIdx for m in mm]]; pj = kj[[m.trainIdx for m in mm]].astype(np.float64); x, y = pi[:, 0] / sx, pi[:, 1] / sy
    u = np.clip(x.astype(int), 0, zi.shape[1] - 1); v = np.clip(y.astype(int), 0, zi.shape[0] - 1); z = zi[v, u]; ok = z > 0
    if ok.sum() < 20: G['why'] = f'{len(mm)} matches, {ok.sum()} with depth'; return None
    P = np.stack([(x - Ki[0, 2]) / Ki[0, 0] * z, (y - Ki[1, 2]) / Ki[1, 1] * z, z], 1)[ok].astype(np.float64)
    Kf = Ki.astype(np.float64).copy(); Kf[0] *= sx; Kf[1] *= sy
    r, rv, tv, inl = cv2.solvePnPRansac(P, pj[ok], Kf, None, iterationsCount=500, reprojectionError=4.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not r or inl is None or len(inl) < 15: G['why'] = f'{len(mm)} matches, {ok.sum()} with depth, {0 if inl is None else len(inl)} PnP inliers'; return None
    rv, tv = cv2.solvePnPRefineLM(P[inl[:, 0]], pj[ok][inl[:, 0]], Kf, None, rv, tv)
    return cv2.Rodrigues(rv)[0], tv[:, 0], len(inl)

def bridge(ia, ib):
    """Carry the camera from frame index ia (end of one pass) through the dropped frames of the video to ib (start of the next
    pass) by chaining PnP steps over frames that have depth. Returns T (next pass world -> this pass world) or None."""
    d, F = G['d'], G['full']; ga, gb = int(d['idx'][ia]), int(d['idx'][ib])
    chain = [(ga, d['depth'][ia], d['K'][ia])] + [(int(g), F['depth'][k], F['K'][k]) for k, g in enumerate(F['idx']) if ga < g < gb and F['has'][k]]
    chain.append((gb, None, None)); w2c = np.linalg.inv(d['c2w'][ia]); i = 0; steps = 0
    while i < len(chain) - 1:                                                        # step to the nearest of the next 3 frames that works
        for k in range(i + 1, min(i + 6, len(chain))):
            rp = rel_pose(*chain[i], chain[k][0])
            if rp is None: continue
            if not np.all(np.isfinite(rp[1])) or np.linalg.norm(rp[1]) > 0.6: G['why'] = f'implausible step {np.linalg.norm(rp[1]):.2g} m'; continue
            M = np.eye(4); M[:3, :3], M[:3, 3] = rp[0], rp[1]; w2c = M @ w2c; i = k; steps += 1; break
        else: print(f'   bridge stuck at frame {chain[i][0]} (tried {[c[0] for c in chain[i + 1:i + 6]]}; last: {G.get("why")})', flush=True); return None, steps
    Tm = np.linalg.inv(w2c) @ np.linalg.inv(d['c2w'][ib])                            # predicted pose o (next pass pose)^-1
    if not np.all(np.isfinite(Tm)) or np.linalg.norm(Tm[:3, 3]) > 5: return None, steps   # a 4e13 m 'bridge' was seen: never trust that
    return (1.0, Tm[:3, :3], Tm[:3, 3]), steps

def judge(args):
    """Median PSNR of A's test frames rebuilt only from B's frames moved by T = (s, R, t) (B world -> A world). Works on a small
    copy (B's frames + the test frames) so each worker never copies the whole depth stack. Test frames B cannot see are skipped."""
    a, b, T, tests = args; d = G['d']; idx = np.where(G['blk'] == b)[0]; sub = np.r_[idx, tests]; nB = len(idx)
    d2 = {k: v[sub].copy() for k, v in d.items()}; K2 = G['K'][sub]; has2 = G['has'][sub].copy(); has2[nB:] = False; s, R, t = T
    for j in range(nB):
        c = d2['c2w'][j].copy(); d2['c2w'][j][:3, :3] = R @ c[:3, :3]; d2['c2w'][j][:3, 3] = s * R @ c[:3, 3] + t; d2['depth'][j] *= s
    out = []
    for q in range(nB, len(sub)):
        img, m, _, _ = ibr.synth(d2, G['files'], K2, has2, G['H'], G['W'], K2[q], d2['c2w'][q], exclude={q}, k=3, cache_imgs={})
        if m.mean() > 0.3: out.append(ibr.psnr_gain(img, ibr.image(G['files'], int(d2['idx'][q]), G['W'], G['H']), m))
    return a, b, (float(np.median(out)) if out else 0.0), len(out)

def cloud(passes, step=8, cap=80000, seed=0):
    """World points of the given passes {pass: pose}: every frame's aligned depth, every `step` px, carried by the pass pose."""
    d, blk = G['d'], G['blk']; P = []
    for b, (s, R, t) in passes.items():
        for j in np.where(blk == b)[0]:
            if not G['has'][j]: continue
            z = d['depth'][j][::step, ::step]; v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]] * step; K = d['K'][j]; ok = z > 0
            X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)[ok]
            X = X @ d['c2w'][j][:3, :3].T + d['c2w'][j][:3, 3]; P.append(s * X @ R.T + t)
    P = np.concatenate(P) if P else np.zeros((0, 3)); r = np.random.default_rng(seed)
    return P[r.choice(len(P), cap, replace=False)] if len(P) > cap else P

def icp(X, tree, Y, T, iters=30, radius=0.3, keep=0.6):
    """Trimmed similarity ICP: X (pass points in its own world) -> Y (group points), start T. Returns refined T and the overlap
    fraction (points with a group point within radius). Fixes the scale/offset a rigid bridge cannot."""
    from scipy.spatial import cKDTree
    s, R, t = T; frac = 0.0
    for _ in range(iters):
        Pq = s * X @ R.T + t; dist, idx = tree.query(Pq, distance_upper_bound=radius); ok = np.isfinite(dist); frac = ok.mean()
        if ok.sum() < 500: return T, frac
        thr = np.quantile(dist[ok], keep); m = ok & (dist <= thr)
        s2, R2, t2 = umeyama(X[m], Y[idx[m]])
        if not 0.8 < s2 < 1.25: return T, frac
        done = np.linalg.norm(t2 - t) < 1e-3 and abs(s2 - s) < 1e-4; s, R, t = s2, R2, t2
        if done: break
    return (s, R, t), frac

def judge_group(members, X, TX, n_test=8):
    """Pass X placed by TX, rebuilt only from ALL passes already in the group (each at its pose): a placement must agree with
    the whole group, not just with the one pass it was matched to (a look-alike agrees with one pass and contradicts another)."""
    d, blk = G['d'], G['blk']; src = [(j, members[b]) for b in members for j in np.where(blk == b)[0] if G['has'][j]]
    xs = [j for j in np.where(blk == X)[0] if G['has'][j]]; xs = [xs[i] for i in np.linspace(0, len(xs) - 1, min(n_test, len(xs))).astype(int)]
    sub = np.array([j for j, _ in src] + xs); n = len(src); d2 = {k: v[sub].copy() for k, v in d.items()}; K2 = G['K'][sub]
    has2 = G['has'][sub].copy(); has2[n:] = False
    for q, (_, T) in enumerate(src + [(j, TX) for j in xs]):
        s, R, t = T; c = d2['c2w'][q].copy(); d2['c2w'][q][:3, :3] = R @ c[:3, :3]; d2['c2w'][q][:3, 3] = s * R @ c[:3, 3] + t; d2['depth'][q] *= s
    out = []
    for q in range(n, len(sub)):
        img, m, _, _ = ibr.synth(d2, G['files'], K2, has2, G['H'], G['W'], K2[q], d2['c2w'][q], exclude=set(range(n, len(sub))), k=3, cache_imgs={})
        if m.mean() > 0.3: out.append(ibr.psnr_gain(img, ibr.image(G['files'], int(d2['idx'][q]), G['W'], G['H']), m))
    return (float(np.median(out)) if out else None), len(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cache'); ap.add_argument('keep'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--bridge', help='prep cache (all frames, depth every 3rd): follow the video across the gaps between passes')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    d, files, K, has, H, W = ibr.load(a.cache, scale=1, keep=None if a.keep == '-' else a.keep); G.update(d=d, files=files, K=K, has=has, H=H, W=W)
    fid = d['idx']; blk = pieces(fid); G['blk'] = blk; nb = blk.max() + 1
    G['sift'] = cv2.SIFT_create(2000); G['fe'] = {}; bridges = {}
    if a.bridge:
        z = np.load(f'{a.bridge}/da3_sfm.npz'); F = {k: z[k] for k in z.files}; F['has'] = F['depth'].reshape(len(F['idx']), -1).max(1) > 0; G['full'] = F
        for A in range(nb - 1):                                                       # consecutive passes in the video
            ia, ib = np.where(blk == A)[0][-1], np.where(blk == A + 1)[0][0]
            while ia > 0 and blk[ia] == A and not has[ia]: ia -= 1
            T, n = bridge(ia, ib)
            print(f'bridge {int(fid[ia])} -> {int(fid[ib])}: ' + (f'{n} steps, moves the next pass {np.linalg.norm(T[2]):.2f} m' if T else f'failed after {n} steps'), flush=True)
            if T: bridges[(A, A + 1)] = (T, int(ia), int(ib))
    # SIFT on up to 10 depth frames per pass (full-resolution photos)
    sift = G['sift']; feats = {}; sample = {}
    for b in range(nb):
        js = [j for j in np.where(blk == b)[0] if has[j]]; sample[b] = [js[i] for i in np.linspace(0, len(js) - 1, min(12, len(js))).astype(int)] if js else []
        for j in sample[b]:
            im = cv2.imread(files[int(fid[j])], cv2.IMREAD_GRAYSCALE); kp, de = sift.detectAndCompute(im, None)
            feats[j] = (np.array([k.pt for k in kp], np.float32), de, (im.shape[1] / d['depth'].shape[2], im.shape[0] / d['depth'].shape[1]))
    print(f'{nb} passes, SIFT on {len(feats)} frames [{time.time()-t0:.0f}s]', flush=True)
    bf = cv2.BFMatcher(); jobs = []; raw = {}
    for A in range(nb):                                                              # match each pair of passes once; (B, A) = inverse
        for B in range(A + 1, nb):
            for ja in sample[A]:
                for jb in sample[B]:
                    (ka, da, sa), (kb, db, sb) = feats[ja], feats[jb]
                    if da is None or db is None or len(da) < 10 or len(db) < 10: continue
                    mm = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
                    if len(mm) < 30: continue
                    Xa = lift(ja, ka[[m.queryIdx for m in mm]], sa); Xb = lift(jb, kb[[m.trainIdx for m in mm]], sb)
                    ok = ~(np.isnan(Xa).any(1) | np.isnan(Xb).any(1))
                    if ok.sum() < 30: continue
                    r = ransac(Xb[ok], Xa[ok])
                    if r:
                        s_, R_, t_, n_ = r; Ri = R_.T
                        raw.setdefault((A, B), []).append((n_, (s_, R_, t_), ja)); raw.setdefault((B, A), []).append((n_, (1 / s_, Ri, -Ri @ t_ / s_), jb))
    for (A, B), (T, ia, ib) in bridges.items():                                     # bridges compete with the matches, judged the same way
        Ri = T[1].T; raw.setdefault((A, B), []).append((10 ** 6, T, ia)); raw.setdefault((B, A), []).append((10 ** 6, (1.0, Ri, -Ri @ T[2]), ib))
    print(f'matching done [{time.time()-t0:.0f}s]', flush=True)
    for A in range(nb):
        for B in range(nb):
            if A == B or not sample[A] or not sample[B]: continue
            hyps = sorted(raw.get((A, B), []), key=lambda h: -h[0]); kept = []      # merge hypotheses that agree (10 cm, 5 deg)
            for n, T, ja in hyps:
                same = [k for k in kept if np.linalg.norm(k[1][2] - T[2]) < 0.1 and np.degrees(np.arccos(np.clip((np.trace(k[1][1].T @ T[1]) - 1) / 2, -1, 1))) < 5]
                if same: same[0][0] += n; same[0][2].append(ja)
                else: kept.append([n, T, [ja]])
            kept = sorted(kept, key=lambda k: -k[0])[:3]                              # 3 best supported + the current placement
            Aidx = np.where(blk == A)[0]
            for n, T, jas in kept + [[0, (1.0, np.eye(3), np.zeros(3)), list(sample[A])]]:
                near = sorted({int(np.clip(np.searchsorted(Aidx, j) + o, 0, len(Aidx) - 1)) for j in jas[:2] for o in (-6, 6)})
                spread = np.linspace(0, len(Aidx) - 1, 4).astype(int).tolist()           # + 4 frames over the whole pass: agreement must hold everywhere
                jobs.append((A, B, T, sorted({Aidx[i] for i in near[:4] + spread}), n))
    print(f'{len(jobs)} hypotheses to judge [{time.time()-t0:.0f}s]', flush=True)
    with Pool(a.workers) as p: scores = p.map(judge, [j[:4] for j in jobs], chunksize=2)
    best = {}
    for (A, B, T, _, n), (_, _, s, cnt) in zip(jobs, scores):
        if cnt and ((A, B) not in best or s > best[(A, B)][0]): best[(A, B)] = (s, T, n, cnt)
        if n == 10 ** 6 and not cnt and (A, B) not in best: best[(A, B)] = (MIN_DB, T, n, 0)   # bridge, no overlap to judge: trust the video
    # Group passes. Links: (a) video bridges between consecutive passes (cannot be look-alike confusions) first, then (b) the best
    # appearance links >= MIN_DB. Every join is verified against ALL passes already in the group (judge_group >= VERIFY_DB; a bridge
    # may also join when nothing in the group overlaps it). Each group has its own frame; passes that fit nowhere stay alone.
    inv = lambda T: (1 / T[0], T[1].T, -T[1].T @ T[2] / T[0])
    comp = lambda P, Q: (P[0] * Q[0], P[1] @ Q[1], P[0] * P[1] @ Q[2] + P[2])         # P o Q
    name = lambda b: f'{int(fid[blk==b][0])}-{int(fid[blk==b][-1])}'
    bjudge = {}
    for (A, B, T, _, n), (_, _, sc, cnt) in zip(jobs, scores):
        if n == 10 ** 6: bjudge.setdefault(tuple(sorted((A, B))), []).append(sc if cnt else None)
    pose, group, links, left, tried = {}, {}, [], set(range(nb)), set(); g = 0
    while left:
        anchor = max(left, key=lambda b: (blk == b).sum()); pose[anchor] = (1.0, np.eye(3), np.zeros(3)); group[anchor] = g; left.discard(anchor)
        while True:
            mem = {b: pose[b] for b in pose if group[b] == g}; cand = []
            for (A, B), (T, _, _) in bridges.items():                                 # A < B consecutive; T: B world -> A world
                own = [x for x in bjudge.get((A, B), []) if x is not None]
                if own and max(own) < 20: continue                                    # the bridge itself contradicts the pixels
                if A in mem and B in left: cand.append((2, 0, A, B, comp(pose[A], T), 'video bridge'))
                if B in mem and A in left: cand.append((2, 0, B, A, comp(pose[B], inv(T)), 'video bridge'))
            for (P, X), v in best.items():
                if v[2] == 10 ** 6 or v[0] < MIN_DB: continue
                kind = 'matches' if v[2] else 'already placed'
                if P in mem and X in left: cand.append((1, v[0], P, X, comp(pose[P], v[1]), kind))
                if X in mem and P in left: cand.append((1, v[0], X, P, comp(pose[X], inv(v[1])), kind))
            cand.sort(key=lambda c: (-c[0], -c[1])); joined = False
            for pri, sc, P, X, TX, kind in cand:
                key = (g, P, X, kind, len(mem))
                if key in tried: continue
                tried.add(key); vs, vn = judge_group(mem, X, TX)
                Y = cloud(mem); Xp = cloud({X: (1.0, np.eye(3), np.zeros(3))}, cap=30000)
                if len(Y) > 1000 and len(Xp) > 1000:
                    from scipy.spatial import cKDTree
                    Ti, fr = icp(Xp, cKDTree(Y), Y, TX)
                    if fr > 0.2 and Ti is not TX:
                        vi, ni = judge_group(mem, X, Ti)
                        print(f'   ICP refine {name(X)}: overlap {fr:.0%}, scale x{Ti[0]/TX[0]:.3f}, moved {np.linalg.norm(Ti[2]-TX[2]):.2f} m -> '
                              f'{vi if vi is None else round(vi, 1)} dB (was {vs if vs is None else round(vs, 1)})', flush=True)
                        if vi is not None and (vs is None or vi > vs + 0.5): TX, vs, vn, kind = Ti, vi, ni, kind + ' + ICP'
                ok = (vs is not None and vs >= VERIFY_DB) or (vs is None and kind.startswith('video bridge'))
                print(f'   try {name(X)} via {name(P)} ({kind}): whole-group check ' + (f'{vs:.1f} dB on {vn} frames' if vs is not None else 'no overlap') + (' -> join' if ok else ' -> reject'), flush=True)
                if ok:
                    pose[X] = TX; group[X] = g; left.discard(X); joined = True
                    links.append(dict(group=g, pass_=name(X), joined_to=name(P), by=kind, link_psnr=round(sc, 1) if sc else None,
                                      group_check=round(vs, 1) if vs is not None else None, moved_m=round(float(np.linalg.norm(TX[2] - pose[P][2])), 2)))
                    print('LINK', json.dumps(links[-1]), flush=True); break
            if not joined: break
        g += 1
    try:                                                                             # untested on real data: never lose the groups above
        from scipy.spatial import cKDTree
        merged = True
        while merged and len(set(group.values())) > 1:
            merged = False; gs = sorted(set(group.values()), key=lambda k: -sum((blk == b).sum() for b in group if group[b] == k))
            for g1 in gs:
                for g2 in gs:
                    if g1 == g2 or merged: continue
                    m1 = {b: pose[b] for b in group if group[b] == g1}; m2 = {b: pose[b] for b in group if group[b] == g2}
                    if sum((blk == b).sum() for b in m2) > sum((blk == b).sum() for b in m1): continue   # move the smaller group
                    cands = [comp(m1[P], comp(v[1], inv(m2[X]))) for (P, X), v in best.items() if P in m1 and X in m2 and v[0] >= MIN_DB]
                    cands += [comp(m1[P], comp(inv(v[1]), inv(m2[X]))) for (X, P), v in best.items() if P in m1 and X in m2 and v[0] >= MIN_DB]
                    Y = cloud(m1); Xp = cloud(m2, cap=30000); tree = cKDTree(Y); bestm = (None, 0)
                    for Tg in cands[:6]:
                        Ti, fr = icp(Xp, tree, Y, Tg)
                        for lab, Tc in (('as linked', Tg), ('ICP', Ti)):                  # keep whichever agrees better
                            sc = [judge_group(m1, X, comp(Tc, m2[X]), n_test=4)[0] for X in m2]; sc = [x for x in sc if x is not None]
                            v = float(np.median(sc)) if sc else 0.0
                            print(f'   merge group {g2} into {g1} ({lab}, overlap {fr:.0%}): {v:.1f} dB', flush=True)
                            if v > bestm[1]: bestm = (Tc, v)
                    if bestm[0] is not None and bestm[1] >= VERIFY_DB:
                        for X in m2: pose[X] = comp(bestm[0], m2[X]); group[X] = g1
                        links.append(dict(group=g1, pass_=f'group {g2}: ' + ', '.join(name(X) for X in m2), joined_to=f'group {g1}', by='group merge + ICP',
                                          group_check=round(bestm[1], 1))); print('LINK', json.dumps(links[-1]), flush=True); merged = True
    except Exception as e:
        import traceback; traceback.print_exc(); print('group merge skipped:', e, flush=True)
    ren = {k: i for i, k in enumerate(sorted(set(group.values()), key=lambda k: -sum((blk == b).sum() for b in group if group[b] == k)))}
    group = {b: ren[k] for b, k in group.items()}; g = len(ren)
    out = {k: v.copy() for k, v in d.items()}; out['group'] = np.array([group[b] for b in blk]); out['piece'] = blk
    for b, T in pose.items():
        s, R, t = T
        for j in np.where(blk == b)[0]:
            c = d['c2w'][j]; out['c2w'][j][:3, :3] = R @ c[:3, :3]; out['c2w'][j][:3, 3] = s * R @ c[:3, 3] + t; out['depth'][j] = d['depth'][j] * s
    np.savez(f'{a.out}/da3_sfm.npz', **out)
    groups = [[name(b) for b in range(nb) if group[b] == k] for k in range(g)]
    table = {f'{name(A)} <- {name(B)}': round(v[0], 1) for (A, B), v in sorted(best.items())}
    json.dump(dict(links=links, groups=groups, best_pair_psnr=table), open(f'{a.out}/align_report.json', 'w'), indent=1)
    print(f'{nb} passes -> {g} groups that agree: {groups} [{time.time()-t0:.0f}s]', flush=True)
    if not os.path.exists(f'{a.out}/frames'): os.symlink(os.path.abspath(f'{a.cache}/frames'), f'{a.out}/frames')

if __name__ == '__main__':
    main()
