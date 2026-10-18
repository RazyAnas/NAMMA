# Rebuild today's "boot" starting depth (prep_whole.py, functions copied unchanged) with a swappable depth source:
#   da   = Depth Anything V2 Small disparity (today's source), rebuilt here with identical code as the fair baseline
#   vggt = 1 / VGGT-1B depth (all 359 frames, Colab T4), resized 518x392 -> 640x480
# Anchors: the saved SfM points (sparse.ply) projected with our poses, nearest point per 4 px cell (the SfM run's
# per-image observation lists were not saved). Same anchors for both sources.
# ANCHORS=<npz from anchors.py> swaps in visibility-filtered anchors.
# Usage: python build_boot.py <da|vggt> <out dir>
import sys, os, json, numpy as np, cv2, onnxruntime as ort
from scipy.spatial import cKDTree
from plyfile import PlyData
src, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
Wk = '/tmp/claude-0/work'; D0 = f'{Wk}/data/namma_wholeroom_gpu/data'
K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1]]); H, Wd = 480, 640
cams = json.load(open(f'{D0}/clean/cameras.json'))
est = {c['frame']: np.array(c['w2c']) for c in cams['train']}; train = sorted(est)
pv = PlyData.read(f'{D0}/clean/points/sparse.ply')['vertex']; X3 = np.stack([pv['x'], pv['y'], pv['z']], 1).astype(np.float64)
img = {f: cv2.cvtColor(cv2.imread(f'{D0}/images/clean/{f:05d}.jpg'), cv2.COLOR_BGR2RGB) for f in train}
v, u = np.mgrid[0:H, 0:Wd]
if src == 'da':
    sess = ort.InferenceSession(f'{Wk}/models/da_vits.onnx', providers=['CPUExecutionProvider']); iname = sess.get_inputs()[0].name
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]); disp = {}
    for f in train:
        x = cv2.resize(img[f] / 255., (518, 518), interpolation=cv2.INTER_CUBIC); x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        disp[f] = cv2.resize(np.squeeze(sess.run(None, {iname: x})[0]), (Wd, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
else:
    z = np.load(f'{Wk}/vggt/vggt_depth_all.npz'); fi = {int(f): i for i, f in enumerate(z['frames'])}
    disp = {f: 1 / np.maximum(cv2.resize(z['depth'][fi[f]].astype(np.float32), (Wd, H), interpolation=cv2.INTER_LINEAR), 1e-3) for f in train}

ANC = np.load(os.environ['ANCHORS']) if os.environ.get('ANCHORS') else None
def sparse_obs(f, cell=4):
    if ANC is not None: a = ANC[str(f)]; return a[:, :2], a[:, 2]
    Xc = X3 @ est[f][:3, :3].T + est[f][:3, 3]; zc = Xc[:, 2]; ok = zc > 0.05
    uu = K[0, 0] * Xc[:, 0] / zc + K[0, 2]; vv = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
    ok &= (uu >= 0) & (uu < Wd - 1) & (vv >= 0) & (vv < H - 1)
    uu, vv, zc = uu[ok], vv[ok], zc[ok]; key = (vv // cell).astype(int) * 1000 + (uu // cell).astype(int)
    o = np.lexsort((zc, key)); _, first = np.unique(key[o], return_index=True); s = o[first]  # nearest per cell
    return np.stack([uu[s], vv[s]], 1), zc[s]

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

gt = {f: cv2.imread(f'{Wk}/floor/gtd/{f:05d}.png', -1) / 1000. for f in train}
loc = {f: da_local(f, disp[f]) for f in train}; loc_mv = mv_filter(loc); boot = {}
for f in train:
    D = loc_mv[f]; ys, xs = np.nonzero(np.isfinite(D[::6, ::6])); ys, xs = ys * 6, xs * 6
    uv0, d0 = sparse_obs(f); uv = np.vstack([uv0, np.stack([xs, ys], 1)]); d = np.concatenate([d0, D[ys, xs]])
    boot[f] = da_local(f, disp[f], k=16, lam=1.0, anchors=(uv, d)) if len(d) > 50 else loc[f]
boot_mv = mv_filter(boot)
bd = f'{O}/boot_depth'; os.makedirs(bd, exist_ok=True)
for f in train: cv2.imwrite(f'{bd}/{f:05d}.png', np.nan_to_num(np.clip(boot[f] * 1000, 0, 65535)).astype(np.uint16))

def score(depths):
    e = []; cov = []; per = {}
    for f in train:
        D = depths[f]; G = gt[f]; m = np.isfinite(D) & (G > 0); cov.append(m.mean()); e.append(np.stack([D[m], G[m]], 1))
        per[f] = float((np.abs(D[m] - G[m]) / G[m]).mean())
    e = np.concatenate(e); d, g = e[:, 0], e[:, 1]; rel = np.abs(d - g) / g
    return dict(absrel=float(rel.mean()), within_5pct=float((rel < 0.05).mean()), median_err_cm=float(100 * np.median(np.abs(d - g))), coverage=float(np.mean(cov))), per
stats = {'source': src, 'n_train': len(train), 'anchors_per_frame_median': float(np.median([len(sparse_obs(f)[1]) for f in train]))}
stats['depth_loc'], _ = score(loc); stats['depth_boot'], stats['per_frame_boot_absrel'] = score(boot); stats['depth_boot_mv'], _ = score(boot_mv)

def cloud(depths, stride=2):
    xyz, rgb = [], []
    for f in train:
        D = depths[f]; m = (np.isfinite(D) & (D > 0))[::stride, ::stride]
        xyz.append(backproject(D, est[f])[::stride, ::stride][m]); rgb.append(img[f][::stride, ::stride][m])
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    key = np.floor(xyz / 0.01).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)
    return xyz[first], rgb[first]

def write_ply(path, xyz, rgb, n_max=300000, seed=0):
    if len(xyz) > n_max:
        sel = np.random.default_rng(seed).choice(len(xyz), n_max, replace=False); xyz, rgb = xyz[sel], rgb[sel]
    a = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    a['x'], a['y'], a['z'] = xyz.T; a['red'], a['green'], a['blue'] = np.asarray(rgb, np.uint8).T
    with open(path, 'wb') as fh:
        fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode()); fh.write(a.tobytes())
    return len(a)
bx, bc = cloud(boot); stats['boot_points'] = write_ply(f'{O}/boot.ply', bx, bc); stats['boot_points_before_cap'] = len(bx)
json.dump(stats, open(f'{O}/stats.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in stats.items() if not k.startswith('per_')}, indent=1))
