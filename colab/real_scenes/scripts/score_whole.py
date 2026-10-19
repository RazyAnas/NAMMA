# PSNR / SSIM of OpenSplat's held-out renders (test_*.png) against the original frames, full resolution.
# psnr_cc: PSNR after the best per-image affine colour fit (3x4), which separates exposure / white-balance error from
# geometry error (the colour-corrected metric used for in-the-wild captures).
# Usage: python score_whole.py <render dir> <color dir> <out json>
import sys, glob, os, json, numpy as np, cv2
from skimage.metrics import structural_similarity
R, C, out = sys.argv[1:4]; rows = []
for f in sorted(glob.glob(f'{R}/test_*.png')):
    r = cv2.imread(f).astype(np.float64) / 255; fr = int(os.path.basename(f)[5:10])
    g = cv2.imread(f'{C}/{fr:05d}.jpg').astype(np.float64) / 255
    if g.shape != r.shape: g = cv2.resize(g, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
    A = np.c_[r.reshape(-1, 3), np.ones(r.shape[0] * r.shape[1])]; cc = A @ np.linalg.lstsq(A, g.reshape(-1, 3), rcond=None)[0]
    rows.append(dict(frame=fr, psnr=10 * np.log10(1 / ((r - g) ** 2).mean()), ssim=structural_similarity(r, g, channel_axis=2, data_range=1),
                     psnr_cc=10 * np.log10(1 / ((cc - g.reshape(-1, 3)) ** 2).mean())))
p = np.array([x['psnr'] for x in rows]); s = np.array([x['ssim'] for x in rows])
res = dict(n_views=len(rows), resolution=f'{r.shape[1]}x{r.shape[0]}', psnr_mean=float(p.mean()), psnr_median=float(np.median(p)),
           psnr_min=float(p.min()), psnr_max=float(p.max()), ssim_mean=float(s.mean()), psnr_cc_mean=float(np.mean([x['psnr_cc'] for x in rows])), per_frame=rows)
json.dump(res, open(out, 'w'), indent=1); print({k: v for k, v in res.items() if k != 'per_frame'})
