# Build poses + point sources for one track (clean or degraded) of the handout.
# Train frames 1200..2216 step 16 (64), held-out test frames 1208..2216 step 16 (64).
# 1) COLMAP global SfM on train+test images (test images only help pose them, like standard NeRF benchmarks),
#    Sim3-aligned to ground truth so every track lives in the same metric frame.
# 2) Depth Anything V2 Small on train frames, anchored to the SfM points (global / local / boot, multi-view check),
#    following /mnt/project-files/monocular-points/scripts/eval_points.py.
# Usage: python prep_track.py <work> <track> <image_dir> [--clahe]
import sys, os, json, shutil, time, numpy as np, cv2, pycolmap, onnxruntime as ort
from scipy.spatial import cKDTree
W, track, IMG = sys.argv[1], sys.argv[2], sys.argv[3]; clahe = '--clahe' in sys.argv
O = f'{W}/{track}'; os.makedirs(O, exist_ok=True)
K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1]]); H, Wd = 480, 640
train = list(range(1200, 2217, 16)); test = list(range(1208, 2217, 16)); allf = sorted(train + test)
L = open(f'{W}/data/livingroom1-traj.txt').read().split('\n'); gtc2w = {}
for i in range(0, len(L) - 4, 5):
    gtc2w[int(L[i].split()[0])] = np.array([[float(x) for x in L[i + j].split()] for j in range(1, 5)])
stats = {'track': track}; t0 = time.time()

# ---- SfM
sd = f'{O}/sfm_images'; shutil.rmtree(sd, ignore_errors=True); os.makedirs(sd)
eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
for f in allf:
    im = cv2.imread(f'{IMG}/{f:05d}.jpg')
    if clahe:  # features on contrast-equalised copies (robustness thread's best setting for degraded footage)
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB); lab[..., 0] = eq.apply(lab[..., 0]); im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(f'{sd}/{f:05d}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 98])
db = f'{O}/db.db'
if os.path.exists(db): os.remove(db)
opts = pycolmap.FeatureExtractionOptions()
if clahe:
    opts.sift.max_num_features = 8192; opts.sift.peak_threshold = 0.004; opts.sift.domain_size_pooling = True; opts.sift.estimate_affine_shape = True
    opts.use_gpu = False
else:
    opts.sift.max_num_features = 4096
reader = pycolmap.ImageReaderOptions(); reader.camera_model = 'PINHOLE'; reader.camera_params = '525,525,319.5,239.5'
pycolmap.extract_features(db, sd, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=opts)
pycolmap.match_exhaustive(db)
so = f'{O}/sfm'; shutil.rmtree(so, ignore_errors=True); os.makedirs(so)
recs = pycolmap.global_mapping(db, sd, so)
rec = max(recs.values(), key=lambda r: r.num_reg_images())
reg = {int(im.name[:5]): im for im in rec.images.values()}
src = np.array([reg[f].projection_center() for f in sorted(reg)]); dst = np.array([gtc2w[f][:3, 3] for f in sorted(reg)])
mu_s, mu_d = src.mean(0), dst.mean(0); A = (dst - mu_d).T @ (src - mu_s) / len(src)
U, Sg, Vt = np.linalg.svd(A); Dg = np.eye(3); Dg[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = U @ Dg @ Vt; s = (Sg * np.diag(Dg)).sum() / ((src - mu_s) ** 2).sum(1).mean(); t = mu_d - s * R @ mu_s
err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
stats.update(registered=len(reg), of=len(allf), missing=[f for f in allf if f not in reg],
             ate_rmse_cm=float(100 * np.sqrt((err ** 2).mean())), ate_median_cm=float(100 * np.median(err)), sfm_points=rec.num_points3D())
print(json.dumps(stats), flush=True)
est = {}
for f, im in reg.items():
    w2c = im.cam_from_world().matrix(); Rc, tc = w2c[:, :3], w2c[:, 3]
    Rn = Rc @ R.T; tn = s * tc - Rn @ t
    est[f] = np.vstack([np.hstack([Rn, tn[:, None]]), [0, 0, 0, 1]])
pid2i = {pid: i for i, pid in enumerate(rec.points3D.keys())}
X3 = np.array([s * R @ p.xyz + t for p in rec.points3D.values()]); C3 = np.array([p.color for p in rec.points3D.values()])
obs = {}
for f, im in reg.items():
    uv, ii = [], []
    for p2 in im.points2D:
        if p2.has_point3D(): uv.append(p2.xy); ii.append(pid2i[p2.point3D_id])
    obs[f] = (np.array(uv).reshape(-1, 2), np.array(ii, int))
json.dump({str(f): est[f].tolist() for f in sorted(est)}, open(f'{O}/poses_w2c.json', 'w'))
train = [f for f in train if f in reg]  # an unregistered train frame cannot be used

# ---- Depth Anything V2 Small
sess = ort.InferenceSession(f'{W}/models/depth_anything_v2_vits.onnx', providers=['CPUExecutionProvider'])
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]); da = {}
img = {}
for f in train:
    im = cv2.cvtColor(cv2.imread(f'{IMG}/{f:05d}.jpg'), cv2.COLOR_BGR2RGB); img[f] = im
    x = cv2.resize(im / 255., (518, 518), interpolation=cv2.INTER_CUBIC); x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    da[f] = cv2.resize(sess.run(None, {'l_x_': x})[0][0], (Wd, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
v, u = np.mgrid[0:H, 0:Wd]

def sparse_obs(f):
    uv, ii = obs[f]; Xc = (est[f][:3, :3] @ X3[ii].T).T + est[f][:3, 3]
    ok = Xc[:, 2] > 0.05; return uv[ok], Xc[ok, 2]

def robust_affine(p, d, w=None):
    y = 1 / d; w = np.ones_like(y) if w is None else w
    for _ in range(10):
        A = np.stack([p, np.ones_like(p)], 1) * np.sqrt(w)[:, None]
        a, b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        r = np.abs(a * p + b - y); sc = 1.4826 * np.median(r) + 1e-9
        w = np.where(r < 1.345 * sc, 1., 1.345 * sc / r)
    return a, b

def da_local(f, disp, k=12, lam=4.0, anchors=None):
    uv, d = sparse_obs(f) if anchors is None else anchors; p = disp[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    a0, b0 = robust_affine(p, d); y = 1 / d
    r = np.abs(a0 * p + b0 - y); keep = r < 3 * 1.4826 * np.median(r) + 1e-9
    uv, p, y = uv[keep], p[keep], y[keep]
    q = np.stack([u.ravel(), v.ravel()], 1)
    k = min(k, len(uv)); dist, idx = cKDTree(uv).query(q, k=k); wgt = 1 / (dist + 8.)
    P_, Y_ = p[idx], y[idx]; sp = np.std(p) + 1e-6; reg_ = lam * wgt.mean(1)
    Spp = (wgt * P_ * P_).sum(1) + reg_ * sp ** 2; Sp = (wgt * P_).sum(1); S1 = wgt.sum(1) + reg_
    Spy = (wgt * P_ * Y_).sum(1) + reg_ * sp ** 2 * a0; Sy = (wgt * Y_).sum(1) + reg_ * b0
    det = Spp * S1 - Sp * Sp; a = (Spy * S1 - Sp * Sy) / det; b = (Spp * Sy - Sp * Spy) / det
    inv = (a * disp.ravel() + b).reshape(H, Wd)
    return np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.nan)

def backproject(D, T_w2c):
    Xc = np.stack([(u - K[0, 2]) / K[0, 0] * D, (v - K[1, 2]) / K[1, 1] * D, D], -1)
    Tc2w = np.linalg.inv(T_w2c); return Xc @ Tc2w[:3, :3].T + Tc2w[:3, 3]

def mv_filter(depths, n=2, tol=0.02, min_agree=2):
    out = {}
    for i, f in enumerate(train):
        Xw = backproject(depths[f], est[f]); agree = np.zeros((H, Wd)); acc = np.nan_to_num(depths[f]).copy(); cnt = np.ones((H, Wd))
        for j in range(max(0, i - n), min(len(train), i + n + 1)):
            if j == i: continue
            g = train[j]; Xc = Xw @ est[g][:3, :3].T + est[g][:3, 3]
            zc = Xc[..., 2]; uu = K[0, 0] * Xc[..., 0] / zc + K[0, 2]; vv = K[1, 1] * Xc[..., 1] / zc + K[1, 2]
            ok = (zc > 0.05) & (uu >= 0) & (uu < Wd - 1) & (vv >= 0) & (vv < H - 1) & np.isfinite(zc)
            dj = np.full((H, Wd), np.nan); dj[ok] = depths[g][vv[ok].astype(int), uu[ok].astype(int)]
            good = ok & (np.abs(dj - zc) < tol * zc); agree += good
            acc[good] += (depths[f] * dj / zc)[good]; cnt[good] += 1
        out[f] = np.where(agree >= min_agree, acc / cnt, np.nan)
    return out

gt = {f: cv2.imread(f'{W}/data/depth_raw/{f:05d}.png', -1) / 1000. for f in train}
loc = {f: da_local(f, da[f]) for f in train}; loc_mv = mv_filter(loc); boot = {}
for f in train:
    D = loc_mv[f]; ys, xs = np.nonzero(np.isfinite(D[::6, ::6])); ys, xs = ys * 6, xs * 6
    uv0, d0 = sparse_obs(f); uv = np.vstack([uv0, np.stack([xs, ys], 1)]); d = np.concatenate([d0, D[ys, xs]])
    boot[f] = da_local(f, da[f], k=16, lam=1.0, anchors=(uv, d)) if len(d) > 50 else loc[f]
boot_mv = mv_filter(boot)
gtd = {f: np.where(gt[f] > 0, gt[f], np.nan) for f in train}

def score(depths):
    e = []; cov = []
    for f in train:
        D = depths[f]; G = gt[f]; m = np.isfinite(D) & (G > 0); cov.append(m.mean()); e.append(np.stack([D[m], G[m]], 1))
    e = np.concatenate(e); d, g = e[:, 0], e[:, 1]; rel = np.abs(d - g) / g
    return dict(absrel=float(rel.mean()), within_5pct=float((rel < 0.05).mean()), median_err_cm=float(100 * np.median(np.abs(d - g))), coverage=float(np.mean(cov)))
stats['depth_boot'] = score(boot); stats['depth_boot_mv'] = score(boot_mv)

def cloud(depths, pose, stride=2):
    xyz, rgb = [], []
    for f in train:
        D = depths[f]; m = (np.isfinite(D) & (D > 0))[::stride, ::stride]
        xyz.append(backproject(D, pose(f))[::stride, ::stride][m]); rgb.append(img[f][::stride, ::stride][m])
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    key = np.floor(xyz / 0.01).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)  # 1 cm voxels
    return xyz[first], rgb[first]

def write_ply(path, xyz, rgb, n_max=150000, seed=0):
    if len(xyz) > n_max:
        sel = np.random.default_rng(seed).choice(len(xyz), n_max, replace=False); xyz, rgb = xyz[sel], rgb[sel]
    a = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    a['x'], a['y'], a['z'] = xyz.T; a['red'], a['green'], a['blue'] = np.asarray(rgb, np.uint8).T
    with open(path, 'wb') as fh:
        fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode()); fh.write(a.tobytes())
    return len(a)

pd = f'{O}/points'; os.makedirs(pd, exist_ok=True); n = {}
n['sparse'] = write_ply(f'{pd}/sparse.ply', X3, C3)
bx, bc = cloud(boot, lambda f: est[f]); n['boot'] = write_ply(f'{pd}/boot.ply', bx, bc)
gx, gc = cloud(gtd, lambda f: np.linalg.inv(gtc2w[f])); n['gtdepth'] = write_ply(f'{pd}/gtdepth.ply', gx, gc)
lo, hi = np.percentile(gx, 1, 0), np.percentile(gx, 99, 0); rng = np.random.default_rng(0)
n['random'] = write_ply(f'{pd}/random.ply', rng.uniform(lo, hi, (100000, 3)), rng.integers(0, 255, (100000, 3)))
stats['n_points'] = n

# per-frame points for MODP (what a tracker would hand it for that frame), float32 npy
for name, getter in [('sparse', lambda f: X3[obs[f][1]]),
                     ('boot_mv', lambda f: backproject(boot_mv[f], est[f])[np.isfinite(boot_mv[f])])]:
    md = f'{O}/modp_points/{name}'; os.makedirs(md, exist_ok=True); r2 = np.random.default_rng(1)
    for f in train:
        P = getter(f)
        if len(P) > 4000: P = P[r2.choice(len(P), 4000, replace=False)]
        np.save(f'{md}/point_cloud_{f}.npy', P.astype(np.float32))
stats['train_frames'] = train; stats['test_frames'] = [f for f in test if f in reg]; stats['seconds'] = time.time() - t0
json.dump(stats, open(f'{O}/stats.json', 'w'), indent=1); print(json.dumps({k: v for k, v in stats.items() if 'frames' not in k}, indent=1))
