# Stage 2 for a real video (same as prep_whole.py's depth step): Depth Anything V2 Small on every training frame,
# anchored to the SfM points (robust affine in disparity, then local kNN-weighted fit), multi-view filtered, then the
# "boot" re-anchoring on the multi-view-consistent pixels. Writes clean/boot_depth/<frame>.png (uint16 mm).
# Usage: python prep_depth.py <work> <depth_anything_v2_vits.onnx>
#    or: DEPTH_SRC=vggt python prep_depth.py <work> -   (VGGT-1B depth instead, fed as inverse depth to the same anchoring;
#        writes clean/vggt_boot_depth/ so the DA-V2 maps the splat uses stay untouched; needs a CUDA GPU and the vggt package)
import sys, os, json, time, numpy as np, cv2, onnxruntime as ort
from scipy.spatial import cKDTree
W, ONNX = sys.argv[1:3]; t0 = time.time()
st = json.load(open(f'{W}/clean/stats.json')); K = np.array(st['K']); H, Wd = st['height'], st['width']
P = json.load(open(f'{W}/clean/poses_w2c.json')); est = {int(k): np.array(v) for k, v in P.items()}
train = st['train_frames']; Z = np.load(f'{W}/clean/sfm_obs.npz'); X3 = Z['X3'].astype(np.float64)
obs = {f: (Z[f'uv_{f}'].astype(np.float64), Z[f'id_{f}']) for f in train}
if os.environ.get('DEPTH_SRC') == 'vggt':
    import torch
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval(); da = {}
    sc = 518 / max(H, Wd); th, tw = [max(14, int(round(x * sc / 14)) * 14) for x in (H, Wd)]
    for c in range(0, len(train), 30):
        fr = train[c:c + 30]
        ims = [cv2.resize(cv2.cvtColor(cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg'), cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_AREA) for f in fr]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            d = model(x)['depth'][0, ..., 0].float().cpu().numpy()
        for f, di in zip(fr, d):
            da[f] = cv2.resize(1 / np.maximum(di, 1e-4), (Wd, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
else:
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if os.environ.get('DA_GPU') else ['CPUExecutionProvider']
    sess = ort.InferenceSession(ONNX, providers=prov)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]); da = {}
    for f in train:
        im = cv2.cvtColor(cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg'), cv2.COLOR_BGR2RGB)
        x = cv2.resize(im / 255., (518, 518), interpolation=cv2.INTER_CUBIC); x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        da[f] = cv2.resize(sess.run(None, {'l_x_': x})[0][0], (Wd, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
print('depth net', time.time() - t0, flush=True)
v, u = np.mgrid[0:H, 0:Wd]

def sparse_obs(f):
    uv, ii = obs[f]; Xc = (est[f][:3, :3] @ X3[ii].T).T + est[f][:3, 3]
    ok = (Xc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < H); return uv[ok], Xc[ok, 2]

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

loc = {f: da_local(f, da[f]) for f in train}; loc_mv = mv_filter(loc); boot = {}
for f in train:
    D = loc_mv[f]; ys, xs = np.nonzero(np.isfinite(D[::6, ::6])); ys, xs = ys * 6, xs * 6
    uv0, d0 = sparse_obs(f); uv = np.vstack([uv0, np.stack([xs, ys], 1)]); d = np.concatenate([d0, D[ys, xs]])
    boot[f] = da_local(f, da[f], k=16, lam=1.0, anchors=(uv, d)) if len(d) > 50 else loc[f]
tag = 'vggt_' if os.environ.get('DEPTH_SRC') == 'vggt' else ''; bd = f'{W}/clean/{tag}boot_depth'; os.makedirs(bd, exist_ok=True)
rel = []
for f in train:
    cv2.imwrite(f'{bd}/{f:05d}.png', np.nan_to_num(np.clip(boot[f] * 1000, 0, 65535)).astype(np.uint16))
    uv, d = sparse_obs(f); b = boot[f][uv[:, 1].astype(int), uv[:, 0].astype(int)]; m = np.isfinite(b); rel.append(np.median(np.abs(b[m] - d[m]) / d[m]))
res = dict(frames=len(train), anchor_rel_err_median=float(np.median(rel)), seconds=time.time() - t0)
json.dump(res, open(f'{W}/clean/{tag}depth_stats.json', 'w')); print(json.dumps(res))
