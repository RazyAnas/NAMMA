"""Turn a posed COLMAP model (from our SfM, or the phone's AR poses) into MODP-ready inputs, CPU only.

Writes, for one "track" (one pose source):
  <out>/images/<frame>.jpg                 undistorted pinhole images (shared by every track of the scene)
  <out>/<track>/cameras.json               K, size, train/test frame lists, world-to-camera 4x4 per frame
  <out>/<track>/points/sparse/<frame>.npy  world xyz of the SfM points observed in that frame (what a tracker hands MODP)
  <out>/<track>/points/dense/<frame>.npy   Depth Anything V2 depth anchored to those points, multi-view checked (ours)
  <out>/<track>/prep_stats.json

python prep_modp_inputs.py --frames F --models m0[,m1,...] --out OUT --track sfm [--resize 0.5] [--metric | --cam-height 1.45]
  --models    COLMAP models already in one world frame (first one wins for frames in several)
  --metric    poses are already in metres (AR poses); otherwise the scene is scaled so the median camera height
              above the floor is --cam-height (gravity = mean camera "down" axis; a guess, roughly +-15%)
  --ba-align  bundle-adjust poses with intrinsics fixed, then Sim3 back onto the input centres (keeps AR metric scale)
"""
import argparse, json, os, shutil
import cv2, numpy as np, pycolmap, onnxruntime as ort
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('--frames', required=True); ap.add_argument('--models', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--track', required=True); ap.add_argument('--resize', type=float, default=1.0)
ap.add_argument('--metric', action='store_true'); ap.add_argument('--cam-height', type=float, default=1.45)
ap.add_argument('--ba-align', action='store_true'); ap.add_argument('--depth-model', required=True)
ap.add_argument('--test-every', type=int, default=8); ap.add_argument('--max-pts', type=int, default=4000)
a = ap.parse_args()
os.makedirs(f'{a.out}/images', exist_ok=True); T = f'{a.out}/{a.track}'
shutil.rmtree(T, ignore_errors=True); os.makedirs(f'{T}/points/sparse'); os.makedirs(f'{T}/points/dense')


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); A = (dst - mu_d).T @ (src - mu_s) / len(src); U, Sg, Vt = np.linalg.svd(A)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = (Sg * np.diag(D)).sum() / ((src - mu_s) ** 2).sum(1).mean(); return s, R, mu_d - s * R @ mu_s


recs = [pycolmap.Reconstruction(m) for m in a.models.split(',')]
if a.ba_align:
    rec = recs[0]; before = {im.name: im.projection_center() for im in rec.images.values()}
    o = pycolmap.BundleAdjustmentOptions(); o.refine_focal_length = False; o.refine_extra_params = False; o.refine_principal_point = False
    pycolmap.bundle_adjustment(rec, o)
    names = sorted(before); s, R, t = umeyama(np.array([rec.images[[i for i, im in rec.images.items() if im.name == n][0]].projection_center() for n in names]), np.array([before[n] for n in names]))
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
    moved = [np.linalg.norm(im.projection_center() - before[im.name]) for im in rec.images.values()]
    print('BA moved camera centres: median %.3f, max %.3f (model units)' % (np.median(moved), np.max(moved)), flush=True)

cam0 = list(recs[0].cameras.values())[0]; W0, H0 = cam0.width, cam0.height
fx, fy, cx, cy = (cam0.params[0], cam0.params[0], cam0.params[1], cam0.params[2]) if cam0.model.name in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL') else tuple(cam0.params[:4])
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]); W, H = int(round(W0 * a.resize)), int(round(H0 * a.resize))
Kout = K.copy(); Kout[:2] *= a.resize
# MODP builds a 4-level image pyramid by 2x2 subsampling, so both sides must be multiples of 8: centre-crop the rest
Wc, Hc = W // 8 * 8, H // 8 * 8; ox, oy = (W - Wc) // 2, (H - Hc) // 2

frames = {}   # name -> (w2c, observed world points, Kdist, dist)
for rec in recs:
    cam = list(rec.cameras.values())[0]; p = cam.params
    m = cam.model.name
    if m in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'): Kk = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
    elif m in ('PINHOLE', 'OPENCV'): Kk = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    else: raise SystemExit(f'camera model {m} not handled')
    dist = {'SIMPLE_RADIAL': lambda: np.array([p[3], 0, 0, 0.]), 'RADIAL': lambda: np.array([p[3], p[4], 0, 0.]),
            'OPENCV': lambda: np.array(p[4:8], dtype=float)}.get(m, lambda: None)()
    for im in rec.images.values():
        if im.name in frames: continue
        M = np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]])
        X = np.array([rec.points3D[p2.point3D_id].xyz for p2 in im.points2D if p2.has_point3D()]).reshape(-1, 3)
        frames[im.name] = (M, X, Kk, dist)
names = sorted(frames)
print('frames', len(names), flush=True)

# metric scale
if a.metric: scale = 1.0
else:
    C = np.array([np.linalg.inv(frames[n][0])[:3, 3] for n in names])
    down = np.mean([np.linalg.inv(frames[n][0])[:3, :3] @ [0, 1, 0] for n in names], 0); down /= np.linalg.norm(down)
    P = np.concatenate([frames[n][1] for n in names]); hp = P @ down; hc = C @ down
    floor = np.percentile(hp, 98)   # "down" grows towards the floor
    scale = a.cam_height / max(np.median(floor - hc), 1e-9)
    print('scale guess: camera height %.3f model units -> %.2f m, scale %.3f' % (np.median(floor - hc), a.cam_height, scale), flush=True)
for n in names:
    M, X, Kk, dist = frames[n]; M = M.copy(); M[:3, 3] *= scale; frames[n] = (M, X * scale, Kk, dist)

test = [n for i, n in enumerate(names) if i % a.test_every == a.test_every // 2]; train = [n for n in names if n not in test]
key = lambda n: os.path.splitext(n)[0]

# images
rgb = {}
for n in names:
    M, X, Kk, dist = frames[n]; img = cv2.imread(f'{a.frames}/{n}')
    if dist is not None: img = cv2.undistort(img, Kk, dist, None, K)
    if a.resize != 1: img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img[oy:oy + Hc, ox:ox + Wc]
    cv2.imwrite(f'{a.out}/images/{key(n)}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95]); rgb[n] = img
W, H = Wc, Hc; Kout[0, 2] -= ox; Kout[1, 2] -= oy

# sparse per-frame points (what a feature tracker would give MODP)
for n in train: np.save(f'{T}/points/sparse/{key(n)}.npy', frames[n][1].astype(np.float32))

# dense: Depth Anything V2 (relative inverse depth) anchored to this frame's sparse points
sess = ort.InferenceSession(a.depth_model, providers=['CPUExecutionProvider']); inp = sess.get_inputs()[0].name
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]); v, u = np.mgrid[0:H, 0:W]


def robust_affine(p, d):
    y = 1 / d; w = np.ones_like(y)
    for _ in range(10):
        A = np.stack([p, np.ones_like(p)], 1) * np.sqrt(w)[:, None]; a_, b_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        r = np.abs(a_ * p + b_ - y); s = 1.4826 * np.median(r) + 1e-12; w = np.where(r < 1.345 * s, 1., 1.345 * s / r)
    return a_, b_, r < 3 * s


dep, errs = {}, []
for i, n in enumerate(train):
    M, X, _, _ = frames[n]; Xc = X @ M[:3, :3].T + M[:3, 3]; ok = Xc[:, 2] > 1e-6
    uv = Xc[ok, :2] / Xc[ok, 2:] @ np.diag(Kout[[0, 1], [0, 1]]) + Kout[:2, 2]; d = Xc[ok, 2]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1); uv, d = uv[inb], d[inb]
    if len(uv) < 15: continue
    x = cv2.resize(rgb[n][..., ::-1] / 255., (518, 518), interpolation=cv2.INTER_CUBIC); x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    p = cv2.resize(sess.run(None, {inp: x})[0][0], (W, H)).astype(np.float64); pp = p[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    a0, b0, good = robust_affine(pp, d); uvg, ppg, y = uv[good], pp[good], 1 / d[good]
    # local correction: weighted affine of the k nearest anchors, pulled towards the global fit
    k = min(12, len(uvg)); q = np.stack([u[::4, ::4].ravel(), v[::4, ::4].ravel()], 1)
    dd, idx = cKDTree(uvg).query(q, k=k); wgt = 1 / (dd + 8.); P_, Y_ = ppg[idx], y[idx]; sp = np.std(ppg) + 1e-9; reg = 4.0 * wgt.mean(1)
    Spp = (wgt * P_ * P_).sum(1) + reg * sp ** 2; Sp = (wgt * P_).sum(1); S1 = wgt.sum(1) + reg
    Spy = (wgt * P_ * Y_).sum(1) + reg * sp ** 2 * a0; Sy = (wgt * Y_).sum(1) + reg * b0; det = Spp * S1 - Sp * Sp
    A_ = ((Spy * S1 - Sp * Sy) / det).reshape(u[::4, ::4].shape); B_ = ((Spp * Sy - Sp * Spy) / det).reshape(u[::4, ::4].shape)
    inv = cv2.resize(A_, (W, H)) * p + cv2.resize(B_, (W, H)); D = np.where(inv > 1e-6, 1 / np.maximum(inv, 1e-6), np.nan)
    errs.append(float(np.median(np.abs(D[uvg[:, 1].astype(int), uvg[:, 0].astype(int)] - 1 / y) * y))); dep[n] = D
    if i % 40 == 0: print('depth', i, len(train), flush=True)

counts = []
fxo, fyo, cxo, cyo = Kout[0, 0], Kout[1, 1], Kout[0, 2], Kout[1, 2]
dn = [n for n in train if n in dep]; rng = np.random.default_rng(0)
for i, n in enumerate(dn):   # keep pixels that >= 2 of the 4 nearest frames agree with (2% depth)
    D = dep[n]; Tc = np.linalg.inv(frames[n][0]); Xc = np.stack([(u - cxo) / fxo * D, (v - cyo) / fyo * D, D], -1); Xw = Xc @ Tc[:3, :3].T + Tc[:3, 3]
    agree = np.zeros((H, W))
    for j in range(max(0, i - 2), min(len(dn), i + 3)):
        if j == i: continue
        Mj = frames[dn[j]][0]; Xj = Xw @ Mj[:3, :3].T + Mj[:3, 3]; z = Xj[..., 2]; zs = np.where(z > 0, z, 1)
        uu = fxo * Xj[..., 0] / zs + cxo; vv = fyo * Xj[..., 1] / zs + cyo
        ok = (z > 0) & (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1) & np.isfinite(z)
        dj = np.full((H, W), np.nan); dj[ok] = dep[dn[j]][vv[ok].astype(int), uu[ok].astype(int)]
        agree += ok & (np.abs(dj - z) < 0.02 * z)
    m = (agree >= 2)[::4, ::4]; P = Xw[::4, ::4][m]
    if len(P) > a.max_pts: P = P[rng.choice(len(P), a.max_pts, replace=False)]
    np.save(f'{T}/points/dense/{key(n)}.npy', P.astype(np.float32)); counts.append(len(P))
for n in train:   # frames without enough anchors fall back to their sparse points
    if not os.path.exists(f'{T}/points/dense/{key(n)}.npy'): shutil.copy(f'{T}/points/sparse/{key(n)}.npy', f'{T}/points/dense/{key(n)}.npy')

cams = dict(K=Kout.tolist(), width=W, height=H, scale_applied=float(scale), metric=bool(a.metric),
            train=[dict(frame=key(n), image=f'images/{key(n)}.jpg', w2c=frames[n][0].tolist()) for n in train],
            test=[dict(frame=key(n), image=f'images/{key(n)}.jpg', w2c=frames[n][0].tolist()) for n in test])
json.dump(cams, open(f'{T}/cameras.json', 'w'))
stats = dict(frames=len(names), train=len(train), test=len(test), size=[W, H], scale=float(scale),
             sparse_pts_per_frame_median=float(np.median([len(frames[n][1]) for n in train])),
             dense_pts_per_frame_median=float(np.median(counts)) if counts else 0, dense_anchor_rel_err_median=float(np.median(errs)) if errs else None)
json.dump(stats, open(f'{T}/prep_stats.json', 'w'), indent=1); print(json.dumps(stats))
