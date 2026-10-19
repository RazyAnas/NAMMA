# Live monocular tracker v2: Depth Anything V2 depth as a pseudo depth sensor, with a sliding-window bundle adjustment
# that solves each keyframe's pose AND the scale/shift (+ optional coarse 3x3 deformation grid) of its monocular depth.
# Points are never optimised individually: each keyframe's depth is "DA disparity x few parameters", so the BA is tiny.
# Frames are processed strictly in order. Usage: live_track2.py S grid(0|1) first last step [window]
import sys, time, json, numpy as np, cv2, onnxruntime as ort
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
S = sys.argv[1]; GRID = int(sys.argv[2]); first, last, step = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
WIN = int(sys.argv[6]) if len(sys.argv) > 6 else 8
COV = int(sys.argv[7]) if len(sys.argv) > 7 else 1  # also match/anchor to older covisible keyframes (held fixed)
GBA = int(sys.argv[8]) if len(sys.argv) > 8 else 1  # final global BA over all keyframes
CAP = 150
import os
REAL = os.environ.get('REAL')  # lobby video: undistorted frames + COLMAP reference trajectory instead of ICL-NUIM
if REAL:
    sm = json.load(open(f'{REAL}/sfm_summary.json')); fx, cx, cy, k1 = sm['camera']; IW, IH = 848, 478
    GT = np.load(f'{REAL}/ref_c2w.npy', allow_pickle=True).item(); FR = sorted(int(n[:5]) for n in os.listdir(f'{REAL}/frames_all'))
else:
    fx, cx, cy, IW, IH = 525., 319.5, 239.5, 640, 480
    L = open(f'{S}/data/livingroom1-traj.txt').read().split('\n'); GT = {}
    for i in range(0, len(L) - 4, 5):
        GT[int(L[i].split()[0])] = np.array([[float(x) for x in L[i + j].split()] for j in range(1, 5)])
K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]]); Ki = np.linalg.inv(K)
sess = ort.InferenceSession(f'{S}/models/depth_anything_v2_vits.onnx', providers=['CPUExecutionProvider'])
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def disparity(rgb):
    x = cv2.resize(rgb / 255., (518, 518), interpolation=cv2.INTER_CUBIC); x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    return cv2.resize(sess.run(None, {'l_x_': x})[0][0], (IW, IH)).astype(np.float64)
sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.004); clahe = cv2.createCLAHE(3.0, (8, 8)); bf = cv2.BFMatcher(cv2.NORM_L2)
def feats(gray):
    k, d = sift.detectAndCompute(clahe.apply(gray), None); return np.array([p.pt for p in k]), d
def match(d1, d2, ratio=0.8):
    m = bf.knnMatch(d1, d2, k=2); out = [(a.queryIdx, a.trainIdx) for a, b in (x for x in m if len(x) == 2) if a.distance < ratio * b.distance]
    return np.array(out, dtype=int).reshape(-1, 2)
def sample(img, uv):
    return img[np.clip(uv[:, 1].round().astype(int), 0, IH - 1), np.clip(uv[:, 0].round().astype(int), 0, IW - 1)]
def gridw(uv):  # bilinear weights of a 3x3 grid over the image
    gx = np.clip(uv[:, 0] / (IW - 1) * 2, 0, 2 - 1e-9); gy = np.clip(uv[:, 1] / (IH - 1) * 2, 0, 2 - 1e-9); ix, iy = gx.astype(int), gy.astype(int); fx_, fy_ = gx - ix, gy - iy
    W = np.zeros((len(uv), 9))
    for dx, dy, w in ((0, 0, (1 - fx_) * (1 - fy_)), (1, 0, fx_ * (1 - fy_)), (0, 1, (1 - fx_) * fy_), (1, 1, fx_ * fy_)):
        W[np.arange(len(uv)), (iy + dy) * 3 + ix + dx] += w
    return W
NP = 17  # per keyframe: rotvec(3) t(3) log-scale(1) shift(1) grid(9)
def inv_depth(par, p, W):  # inverse depth = exp(la) * p * (1 + grid) + b
    return np.exp(par[..., 6]) * p * (1 + (W * par[..., 8:17]).sum(-1)) + par[..., 7]
def lift_w(par, uv, p, W):
    inv = np.maximum(inv_depth(par, p, W), 1e-3); Xc = (Ki @ np.c_[uv, np.ones(len(uv))].T).T / inv[:, None]
    R = Rotation.from_rotvec(par[:3]).as_matrix(); return (Xc - par[3:6]) @ R, 1 / inv  # world = R^T (Xc - t)
def robust_affine(p, z):
    y = 1 / z; w = np.ones_like(y)
    for _ in range(10):
        A = np.stack([p, np.ones_like(p)], 1) * np.sqrt(w)[:, None]; a, b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        r = np.abs(a * p + b - y); s = 1.4826 * np.median(r) + 1e-12; w = np.where(r < 1.345 * s, 1., 1.345 * s / r)
    return a, b

kfs = []; pairs = {}  # (i,j) -> match index array, F-RANSAC filtered
def add_pairs(j):
    cand = list(range(max(0, j - WIN + 1), j))
    if COV: cand = list(range(0, j))
    for i in cand:
        m = match(kfs[i]['des'], kfs[j]['des'])
        if len(m) < 20: continue
        _, msk = cv2.findFundamentalMat(kfs[i]['uv'][m[:, 0]], kfs[j]['uv'][m[:, 1]], cv2.FM_RANSAC, 1.5, 0.999)
        if msk is None: continue
        m = m[msk.ravel() > 0]
        if len(m) > CAP: m = m[np.random.default_rng(j * 1000 + i).choice(len(m), CAP, replace=False)]
        if len(m) >= (20 if i >= j - WIN + 1 else 40): pairs[(i, j)] = m

def bundle(n_fixed_pose, ids=None):
    win = list(range(max(0, len(kfs) - WIN), len(kfs))) if ids is None else ids; nf = 0
    if ids is None and COV:  # older keyframes that share matches with the window join as fixed anchors (ORB-SLAM-style local BA)
        old = {}
        for (i, j), m in pairs.items():
            if j in win and i < win[0]: old[i] = old.get(i, 0) + len(m)
        olds = sorted(sorted(old, key=lambda k: -old[k])[:6]); nf = len(olds); win = olds + win
    ids = win; loc = {k: n for n, k in enumerate(ids)}
    full = np.array([kfs[k]['par'] for k in ids]); free = np.ones_like(full, bool)
    free[:nf] = False; free[:nf + n_fixed_pose, :6] = False
    if len(kfs) <= 2: free[0, 6] = False  # gauge: first keyframe's depth scale fixes the map scale until poses do
    if not GRID: free[:, 8:] = False
    H, T, UVh, Ph, Wh, UVt = [], [], [], [], [], []
    for (i, j), m in pairs.items():
        if i in loc and j in loc:
            for a, b, ma, mb in ((i, j, m[:, 0], m[:, 1]), (j, i, m[:, 1], m[:, 0])):
                H.append(np.full(len(m), loc[a])); T.append(np.full(len(m), loc[b])); uv = kfs[a]['uv'][ma]
                UVh.append(uv); Ph.append(kfs[a]['pk'][ma]); Wh.append(gridw(uv)); UVt.append(kfs[b]['uv'][mb])
    if not H: return 0
    H, T, UVh, Ph, Wh, UVt = map(np.concatenate, (H, T, UVh, Ph, Wh, UVt)); rays = (Ki @ np.c_[UVh, np.ones(len(UVh))].T).T
    nreg = free[:, 8:].sum()
    def resid(x):
        P = full.copy(); P[free] = x; Rm = Rotation.from_rotvec(P[:, :3]).as_matrix()
        inv = np.maximum(inv_depth(P[H], Ph, Wh), 1e-3); Xc = rays / inv[:, None]
        Xw = np.einsum('nji,nj->ni', Rm[H], Xc - P[H, 3:6]); Xt = np.einsum('nij,nj->ni', Rm[T], Xw) + P[T, 3:6]
        z = np.maximum(Xt[:, 2], 1e-3); r = np.c_[fx * Xt[:, 0] / z + cx - UVt[:, 0], fx * Xt[:, 1] / z + cy - UVt[:, 1]].ravel()
        return np.r_[r, 20.0 * P[:, 8:][free[:, 8:]]] if nreg else r
    col = -np.ones_like(full, int); col[free] = np.arange(free.sum())
    Jsp = lil_matrix((2 * len(H) + nreg, free.sum()), dtype=int)
    for n in range(len(ids)):
        rows = np.where((H == n) | (T == n))[0]; cs = col[n][col[n] >= 0]
        if len(cs) and len(rows):
            rr = np.r_[2 * rows, 2 * rows + 1]; Jsp[np.repeat(rr, len(cs)), np.tile(cs, len(rr))] = 1
    if nreg: Jsp[2 * len(H) + np.arange(nreg), col[:, 8:][free[:, 8:]]] = 1
    sol = least_squares(resid, full[free], jac_sparsity=Jsp, loss='huber', f_scale=2.0, max_nfev=30, x_scale='jac', method='trf')
    full[free] = sol.x
    for n, k in enumerate(ids): kfs[k]['par'] = full[n]
    return len(H)

def T_c2w(par):
    R = Rotation.from_rotvec(par[:3]).as_matrix(); T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ par[3:6]; return T

frames = [f for f in FR if first <= f <= last][::step] if REAL else list(range(first, last + 1, step)); poses = {}; t_track, t_kf, t_ba = [], [], []; lost = 0; nres = []; relocs = 0
if REAL:
    Kd = K.copy(); dist = np.array([k1, 0, 0, 0.]); img = lambda f: cv2.undistort(cv2.imread(f'{REAL}/frames_all/{f:05d}.jpg'), Kd, dist)
else:
    img = lambda f: cv2.imread(f'{S}/data/color_raw/{f:05d}.jpg')
def new_kf(f, bgr, uv, des, par):
    p = disparity(bgr[..., ::-1]); kfs.append(dict(f=f, uv=uv, des=des, p=p, pk=sample(p, uv), par=par))
for f in frames:
    bgr = img(f); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); t0 = time.time(); uv, des = feats(gray)
    if not kfs:
        par = np.zeros(NP); new_kf(f, bgr, uv, des, par); kfs[0]['par'][6] = -np.log(np.median(kfs[0]['p'])); poses[f] = None; t_kf.append(time.time() - t0); continue
    def try_pnp(cands):
        obj, img_pts = [], []
        for k in cands:
            m = match(k['des'], des)
            if len(m) < 15: continue
            X, z = lift_w(k['par'], k['uv'][m[:, 0]], k['pk'][m[:, 0]], gridw(k['uv'][m[:, 0]])); ok = np.isfinite(z) & (z > 0.05) & (z < 50)
            obj.append(X[ok]); img_pts.append(uv[m[ok, 1]])
        if not obj or sum(len(o) for o in obj) < 20: return None
        obj, img_pts = np.vstack(obj), np.vstack(img_pts)
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, img_pts, K, None, reprojectionError=3.0, iterationsCount=200, flags=cv2.SOLVEPNP_EPNP)
        return None if not ok or inl is None or len(inl) < 15 else (obj, img_pts, rv, tv, inl)
    out = try_pnp(kfs[-3:])
    if out is None and len(kfs) > 3:  # relocalise against the 3 keyframes in the map with most matches
        cnt = [len(match(k['des'], des)) for k in kfs[:-3]]; out = try_pnp([kfs[i] for i in np.argsort(cnt)[-3:]]); relocs += out is not None
    if out is None: lost += 1; continue
    obj, img_pts, rv, tv, inl = out
    rv, tv = cv2.solvePnPRefineLM(obj[inl[:, 0]], img_pts[inl[:, 0]], K, None, rv, tv)
    par = np.zeros(NP); par[:3] = rv[:, 0]; par[3:6] = tv[:, 0]; T = T_c2w(par); poses[f] = T; t_track.append(time.time() - t0)
    lk = kfs[-1]; rel = np.linalg.inv(T_c2w(lk['par'])) @ T; med = np.median(1 / np.maximum(inv_depth(lk['par'], lk['pk'], gridw(lk['uv'])), 1e-3))
    ang = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)))
    if len(inl) < 150 or np.linalg.norm(rel[:3, 3]) > 0.12 * med or ang > 10:
        t1 = time.time(); new_kf(f, bgr, uv, des, par); k = kfs[-1]
        Xi = obj[inl[:, 0]]; R = Rotation.from_rotvec(par[:3]).as_matrix(); zc = (Xi @ R.T + par[3:6])[:, 2]
        a, b = robust_affine(sample(k['p'], img_pts[inl[:, 0]]), zc)
        if a <= 0: a, b = np.median(1 / zc) / np.median(sample(k['p'], img_pts[inl[:, 0]])), 0.
        k['par'][6], k['par'][7] = np.log(a), b; t_kf.append(time.time() - t1)
        t2 = time.time(); add_pairs(len(kfs) - 1); nres.append(bundle(1 if len(kfs) <= 2 else 2)); t_ba.append(time.time() - t2)
for k in kfs: poses[k['f']] = T_c2w(k['par'])  # keyframes carry their bundle-adjusted pose
live_poses = dict(poses); t_g = 0
if GBA:  # one global BA over every keyframe (what a background thread would run); first two poses fixed
    t3 = time.time(); bundle(2, ids=list(range(len(kfs)))); t_g = time.time() - t3
    for k in kfs: poses[k['f']] = T_c2w(k['par'])
def ate(fs):
    src = np.array([poses[f][:3, 3] for f in fs]); dst = np.array([GT[f][:3, 3] for f in fs])
    mu_s, mu_d = src.mean(0), dst.mean(0); A = (dst - mu_d).T @ (src - mu_s) / len(src); U, Sg, Vt = np.linalg.svd(A)
    Dg = np.eye(3); Dg[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ Dg @ Vt; s = (Sg * np.diag(Dg)).sum() / ((src - mu_s) ** 2).sum(1).mean(); t = mu_d - s * R @ mu_s
    e = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1); return round(100 * float(np.sqrt((e ** 2).mean())), 2), round(100 * float(np.median(e)), 2)
fs = sorted(f for f in poses if poses[f] is not None and f in GT)
tag = ('real_' if REAL else '') + f'v3_grid{GRID}_w{WIN}_cov{COV}_gba{GBA}'
res = dict(mode=tag, frames=len(frames), tracked=len(fs), lost=lost, keyframes=len(kfs), ate_rmse_cm=ate(fs)[0], ate_median_cm=ate(fs)[1],
           kf_ate_rmse_cm=ate([k['f'] for k in kfs])[0], track_ms=round(1000 * float(np.mean(t_track)), 1), keyframe_ms=round(1000 * float(np.mean(t_kf)), 1),
           ba_ms=round(1000 * float(np.mean(t_ba)), 1), ba_matches=int(np.mean(nres)), global_ba_s=round(t_g, 1), relocalised=int(relocs))
if REAL:
    ref = np.array([GT[f][:3, 3] for f in fs]); res['ref_frames'] = len(GT); res['ref_path_len'] = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    res['ate_pct_of_path'] = round(res['ate_rmse_cm'] / 100 / res['ref_path_len'] * 100, 2)
if GBA: res['kf_ate_after_global_ba_cm'] = res.pop('kf_ate_rmse_cm')
P2 = poses; poses = live_poses; res['live_ate_rmse_cm'] = ate(fs)[0]; poses = P2
print(json.dumps(res)); json.dump(res, open(f'{REAL or S + "/work"}/live_{tag}_{first}_{last}_{step}.json', 'w'))
np.save(f'{REAL or S + "/work"}/live_{tag}_{first}_{last}_{step}_poses.npy', {f: poses[f] for f in fs}, allow_pickle=True)
