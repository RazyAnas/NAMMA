# Anchor a monocular depth map to our SfM sparse points (no ground truth), the way boot points are made.
# samples: (N,3) array of u, v (pixels at the depth map's size) and z (metres, from our SfM).
import numpy as np, cv2

def sample(d, s):
    return cv2.remap(d.astype(np.float32), s[:, 0:1].astype(np.float32), s[:, 1:2].astype(np.float32), cv2.INTER_LINEAR)[:, 0]

def fit_depth_scale(d, s, iters=3, tol=0.1):
    """depth-like prediction: z = k * d. Robust: median log ratio, then refit on inliers (|log residual| < tol)."""
    p = sample(d, s); m = np.isfinite(p) & (p > 0); r = np.log(s[m, 2] / p[m]); k = np.median(r)
    for _ in range(iters):
        inl = np.abs(r - k) < tol
        if inl.sum() > 10: k = np.median(r[inl])
    return d * np.exp(k)

def fit_disparity_affine(disp, s, iters=3, tol=0.1):
    """inverse-depth-like prediction (DA-V2): 1/z = a * disp + b. Robust least squares on inliers."""
    p = sample(disp, s); y = 1 / s[:, 2]; m = np.isfinite(p); A = np.stack([p[m], np.ones(m.sum())], 1); yy = y[m]
    w = np.ones(len(yy), bool)
    for _ in range(iters + 1):
        a, b = np.linalg.lstsq(A[w], yy[w], rcond=None)[0]
        z = 1 / np.clip(A @ [a, b], 1e-3, None); w = np.abs(np.log(z * yy)) < tol
        if w.sum() < 10: break
    return 1 / np.clip(a * disp + b, 1e-3, None)
