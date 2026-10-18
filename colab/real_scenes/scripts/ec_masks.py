# MODP error compensation masks (Sec. 3.2.2): pixels where SSIM(render, frame) < threshold.
# Usage: python ec_masks.py <render dir with test_<frame>.png> <color dir> <out dir> <thr> [--stride 2]
import sys, glob, os, json, numpy as np, cv2
from skimage.metrics import structural_similarity
R, C, O, thr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); os.makedirs(O, exist_ok=True)
stride = int(sys.argv[sys.argv.index('--stride') + 1]) if '--stride' in sys.argv else 1
frac = []
for f in sorted(glob.glob(f'{R}/test_*.png')):
    fr = int(os.path.basename(f)[5:10]); r = cv2.imread(f); g = cv2.imread(f'{C}/{fr:05d}.jpg')
    if g.shape != r.shape: g = cv2.resize(g, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
    _, s = structural_similarity(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), full=True, data_range=255)
    m = (s < thr).astype(np.uint8) * 255
    if stride > 1: keep = np.zeros_like(m); keep[::stride, ::stride] = 1; m = m * keep
    cv2.imwrite(f'{O}/{fr:05d}.png', m); frac.append(float((s < thr).mean()))
print(json.dumps(dict(frames=len(frac), thr=thr, mean_frac=float(np.mean(frac)), max_frac=float(np.max(frac)))))
