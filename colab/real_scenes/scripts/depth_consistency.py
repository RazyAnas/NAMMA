# Multi-view depth consistency filter (the MVS "geometric consistency" check, as in COLMAP / OpenMVS fusion): a pixel of a
# training frame's depth map is kept only if, lifted to 3D and projected into its N nearest training frames, it agrees with
# their depth there within rel_tol in at least min_ok of them. Per-frame monocular depth disagrees between frames, which is
# what doubles edges in a fused mesh; this drops the pixels that disagree. Writes clean/<out>/<frame>.png (uint16 mm, 0 = dropped).
# Usage: python depth_consistency.py <work> [out=cons_depth] [neighbours=6] [min_ok=2] [rel_tol=0.02] [src=boot_depth]
import sys, os, json, numpy as np, cv2
W = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'cons_depth'
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6; need = int(sys.argv[4]) if len(sys.argv) > 4 else 2
tol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02; src = sys.argv[6] if len(sys.argv) > 6 else 'boot_depth'
st = json.load(open(f'{W}/clean/stats.json')); P = json.load(open(f'{W}/clean/poses_w2c.json')); K = np.array(st['K'])
Wd, H = st['width'], st['height']; os.makedirs(f'{W}/clean/{out}', exist_ok=True)
fr = [f for f in st['train_frames'] if os.path.exists(f'{W}/clean/{src}/{f:05d}.png')]
D = {f: cv2.imread(f'{W}/clean/{src}/{f:05d}.png', -1).astype(np.float64) / 1000. for f in fr}
E = {f: np.array(P[str(f)]) for f in fr}; C = {f: np.linalg.inv(E[f])[:3, 3] for f in fr}
v, u = np.mgrid[0:H, 0:Wd]; ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
kept = []
for f in fr:
    z = D[f]; X = (ray * z[..., None]).reshape(-1, 3); Tw = np.linalg.inv(E[f]); Xw = X @ Tw[:3, :3].T + Tw[:3, 3]
    nb = sorted((q for q in fr if q != f), key=lambda q: np.linalg.norm(C[q] - C[f]))[:N]; ok = np.zeros(len(Xw), int)
    for q in nb:
        Y = Xw @ E[q][:3, :3].T + E[q][:3, 3]; zq = Y[:, 2]; vis = zq > 0.05
        px = K[0, 0] * Y[:, 0] / np.maximum(zq, 1e-6) + K[0, 2]; py = K[1, 1] * Y[:, 1] / np.maximum(zq, 1e-6) + K[1, 2]
        vis &= (px >= 0) & (px <= Wd - 1) & (py >= 0) & (py <= H - 1)
        dq = D[q][py.round().astype(int).clip(0, H - 1), px.round().astype(int).clip(0, Wd - 1)]
        ok += vis & (np.abs(dq - zq) < tol * zq)
    keep = (ok >= need).reshape(H, Wd) & (z > 0); kept.append(keep.mean())
    cv2.imwrite(f'{W}/clean/{out}/{f:05d}.png', np.where(keep, z * 1000, 0).round().astype(np.uint16))
print(json.dumps(dict(frames=len(fr), neighbours=N, min_ok=need, rel_tol=tol, kept_mean=round(float(np.mean(kept)), 3),
                      kept_min=round(float(np.min(kept)), 3))))
