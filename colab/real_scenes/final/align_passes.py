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

MIN_DB = 23.0   # 20 dB let a wrong placement through (white walls and floor score ~20 when misplaced)
G = {}

def pieces(fid): return np.cumsum(np.r_[0, np.diff(fid) > 3])

def lift(j, kp, scale):
    """Keypoints (full image px) of frame j -> world points through its aligned depth; NaN where no depth."""
    d = G['d']; z = d['depth'][j]; Hd, Wd = z.shape; u = np.clip((kp[:, 0] / scale).astype(int), 0, Wd - 1); v = np.clip((kp[:, 1] / scale).astype(int), 0, Hd - 1)
    zz = z[v, u]; K = d['K'][j]; P = np.stack([(u - K[0, 2]) / K[0, 0] * zz, (v - K[1, 2]) / K[1, 1] * zz, zz], 1)
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
        G['fe'][g] = (np.array([k.pt for k in kp], np.float32), de, im.shape[1])
    return G['fe'][g]

def rel_pose(gi, zi, Ki, gj):
    """Camera of frame gj relative to gi (X_j = R X_i + t): gi's keypoints lifted with its depth zi (DA3 grid, intrinsics Ki),
    found again in gj, PnP + RANSAC. Frames seconds apart in the video: no look-alike confusion possible."""
    (ki, di, wi), (kj, dj, wj) = feat(gi), feat(gj)
    if di is None or dj is None or len(di) < 10 or len(dj) < 10: return None
    mm = [m for m, n in cv2.BFMatcher().knnMatch(di, dj, k=2) if m.distance < 0.75 * n.distance]
    if len(mm) < 40: return None
    sc = wi / zi.shape[1]; pi = ki[[m.queryIdx for m in mm]]; pj = kj[[m.trainIdx for m in mm]].astype(np.float64)
    u = np.clip((pi[:, 0] / sc).astype(int), 0, zi.shape[1] - 1); v = np.clip((pi[:, 1] / sc).astype(int), 0, zi.shape[0] - 1); z = zi[v, u]; ok = z > 0
    if ok.sum() < 30: return None
    P = np.stack([(pi[:, 0] / sc - Ki[0, 2]) / Ki[0, 0] * z, (pi[:, 1] / sc - Ki[1, 2]) / Ki[1, 1] * z, z], 1)[ok].astype(np.float64)
    Kf = Ki.astype(np.float64).copy(); Kf[:2] *= sc
    r, rv, tv, inl = cv2.solvePnPRansac(P, pj[ok], Kf, None, iterationsCount=500, reprojectionError=4.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not r or inl is None or len(inl) < 30: return None
    rv, tv = cv2.solvePnPRefineLM(P[inl[:, 0]], pj[ok][inl[:, 0]], Kf, None, rv, tv)
    return cv2.Rodrigues(rv)[0], tv[:, 0], len(inl)

def bridge(ia, ib):
    """Carry the camera from frame index ia (end of one pass) through the dropped frames of the video to ib (start of the next
    pass) by chaining PnP steps over frames that have depth. Returns T (next pass world -> this pass world) or None."""
    d, F = G['d'], G['full']; ga, gb = int(d['idx'][ia]), int(d['idx'][ib])
    chain = [(ga, d['depth'][ia], d['K'][ia])] + [(int(g), F['depth'][k], F['K'][k]) for k, g in enumerate(F['idx']) if ga < g < gb and F['has'][k]]
    chain.append((gb, None, None)); w2c = np.linalg.inv(d['c2w'][ia]); i = 0; steps = 0
    while i < len(chain) - 1:                                                        # step to the nearest of the next 3 frames that works
        for k in range(i + 1, min(i + 4, len(chain))):
            rp = rel_pose(*chain[i], chain[k][0])
            if rp is None: continue
            M = np.eye(4); M[:3, :3], M[:3, 3] = rp[0], rp[1]; w2c = M @ w2c; i = k; steps += 1; break
        else: return None, steps
    Tm = np.linalg.inv(w2c) @ np.linalg.inv(d['c2w'][ib])                            # predicted pose o (next pass pose)^-1
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
        if m.mean() > 0.3: out.append(ibr.psnr(img, ibr.image(G['files'], int(d2['idx'][q]), G['W'], G['H']), m))
    return a, b, (float(np.median(out)) if out else 0.0), len(out)

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
            feats[j] = (np.array([k.pt for k in kp], np.float32), de, im.shape[1] / d['depth'].shape[2])
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
                jobs.append((A, B, T, [Aidx[i] for i in near][:4], n))
    print(f'{len(jobs)} hypotheses to judge [{time.time()-t0:.0f}s]', flush=True)
    with Pool(a.workers) as p: scores = p.map(judge, [j[:4] for j in jobs], chunksize=2)
    best = {}
    for (A, B, T, _, n), (_, _, s, cnt) in zip(jobs, scores):
        if cnt and ((A, B) not in best or s > best[(A, B)][0]): best[(A, B)] = (s, T, n, cnt)
        if n == 10 ** 6 and not cnt and (A, B) not in best: best[(A, B)] = (MIN_DB, T, n, 0)   # bridge, no overlap to judge: trust the video
    # Group passes along the best links (maximum spanning forest), either direction: (P, X) carries X into P's frame,
    # (X, P) carries P into X's, so its inverse carries X into P's. Each group gets its own frame (largest pass = anchor);
    # passes that link to nothing stay alone. Nothing is dropped; the viewer only mixes photos within one group.
    inv = lambda T: (1 / T[0], T[1].T, -T[1].T @ T[2] / T[0])
    comp = lambda P, Q: (P[0] * Q[0], P[1] @ Q[1], P[0] * P[1] @ Q[2] + P[2])         # P o Q
    name = lambda b: f'{int(fid[blk==b][0])}-{int(fid[blk==b][-1])}'
    pose, group, links, left = {}, {}, [], set(range(nb)); g = 0
    while left:
        anchor = max(left, key=lambda b: (blk == b).sum()); pose[anchor] = (1.0, np.eye(3), np.zeros(3)); group[anchor] = g; left.discard(anchor)
        while True:
            cand = [(v[0], P, X, v[1]) for (P, X), v in best.items() if P in group and group[P] == g and X in left and v[0] >= MIN_DB]
            cand += [(v[0], P, X, inv(v[1])) for (X, P), v in best.items() if P in group and group[P] == g and X in left and v[0] >= MIN_DB]
            if not cand: break
            s, P, X, T = max(cand, key=lambda c: c[0]); pose[X] = comp(pose[P], T); group[X] = g; left.discard(X)
            how = best.get((P, X), best.get((X, P)))
            links.append(dict(group=g, pass_=name(X), joined_to=name(P), psnr=round(s, 1), moved_m=round(float(np.linalg.norm(T[2])), 2),
                              by='video bridge' if how and how[2] == 10 ** 6 else 'matches' if how and how[2] else 'already placed', judged=bool(how and how[3])))
            print('LINK', json.dumps(links[-1]), flush=True)
        g += 1
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
