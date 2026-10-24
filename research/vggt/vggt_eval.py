# Shared evaluation for the VGGT test (same code runs in the cloud for our side and on Colab for VGGT).
# Frames: ICL-NUIM living room 1, frames 0..752 every 24th (32 frames), the room corner with the window.
# Depth is compared at VGGT's input size (518x392, the size load_and_preprocess_images gives a 640x480 frame).
import numpy as np, cv2

FRAMES = list(range(0, 753, 24))
W, H = 518, 392

def read_traj(path):
    raw = open(path).read().split('\n'); gt = {}
    for i in range(0, len(raw) - 4, 5):
        gt[int(raw[i].split()[0])] = np.array([[float(x) for x in raw[i + j].split()] for j in range(1, 5)])
    return gt                                          # camera-to-world, metres

def gt_depth(path):                                    # uint16 mm png -> metres at 518x392
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)

def umeyama(src, dst):                                 # Sim3 dst ~ s R src + t
    mu_s, mu_d = src.mean(0), dst.mean(0); X, Y = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Y.T @ X / len(src)); E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: E[2, 2] = -1
    R = U @ E @ Vt; s = np.trace(np.diag(S) @ E) / X.var(0).sum(); t = mu_d - s * R @ mu_s
    return s, R, t

def ate_cm(centres_est, centres_gt):
    s, R, t = umeyama(centres_est, centres_gt); al = (s * (R @ centres_est.T)).T + t
    e = np.linalg.norm(al - centres_gt, axis=1) * 100
    return dict(rmse_cm=float(np.sqrt((e ** 2).mean())), median_cm=float(np.median(e)), max_cm=float(e.max()))

def depth_metrics(pred, gt, mode):
    """mode: 'scale' one median scale per frame; 'scale_shift' least squares per frame;
    pred/gt lists of HxW arrays. For 'global', one scale for all frames (tests multi-view consistency)."""
    out = []
    if mode == 'global':
        k = np.median(np.concatenate([g[(g > 0) & np.isfinite(p) & (p > 0)] / p[(g > 0) & np.isfinite(p) & (p > 0)] for p, g in zip(pred, gt)]))
    for p, g in zip(pred, gt):
        m = (g > 0) & np.isfinite(p) & (p > 0)
        if mode == 'scale': q = p * np.median(g[m] / p[m])
        elif mode == 'global': q = p * k
        else:
            A = np.stack([p[m], np.ones(m.sum())], 1); a, b = np.linalg.lstsq(A, g[m], rcond=None)[0]; q = a * p + b
        e = q[m] if mode != 'scale_shift' else q[m]
        rel = np.abs(e - g[m]) / g[m]; ratio = np.maximum(e / g[m], g[m] / np.maximum(e, 1e-6))
        out.append((rel.mean(), (ratio < 1.25).mean(), np.median(np.abs(e - g[m])) * 100))
    o = np.array(out)
    return dict(absrel_pct=float(o[:, 0].mean() * 100), delta1_pct=float(o[:, 1].mean() * 100), median_err_cm=float(o[:, 2].mean()))
