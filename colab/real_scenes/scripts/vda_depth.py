# Temporally consistent depth for the whole walk with Video Depth Anything (Chen et al. 2025), then ONE scale for the whole
# clip fitted to our SfM-anchored boot depth (instead of one scale per frame, which is what lets per-frame depths disagree
# and double edges in the fused mesh). Metric model: depth = s * d. Relative model (disparity): 1/depth = a * d + b.
# Writes clean/vda_depth/<frame>.png (uint16 mm) for every posed frame, and the fit to clean/vda_depth.json.
# Usage: python vda_depth.py <work> <Video-Depth-Anything checkout> [encoder=vitl] [metric=1]
import sys, os, json, glob, numpy as np, cv2, torch
W, REPO = sys.argv[1:3]; ENC = sys.argv[3] if len(sys.argv) > 3 else 'vitl'; MET = (sys.argv[4] if len(sys.argv) > 4 else '1') == '1'
sys.path.insert(0, REPO); from video_depth_anything.video_depth import VideoDepthAnything
cfg = {'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
       'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])}[ENC]
m = VideoDepthAnything(**cfg, metric=MET)
m.load_state_dict(torch.load(f"{REPO}/checkpoints/{'metric_' if MET else ''}video_depth_anything_{ENC}.pth", map_location='cpu'), strict=True)
m = m.cuda().eval()
P = json.load(open(f'{W}/clean/poses_w2c.json')); fr = sorted(int(k) for k in P)
imgs = np.stack([cv2.cvtColor(cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg'), cv2.COLOR_BGR2RGB) for f in fr])
with torch.no_grad(): d, _ = m.infer_video_depth(imgs, 30, input_size=518, device='cuda', fp32=False)
d = np.asarray(d, np.float64); H, Wd = imgs.shape[1:3]
if d.shape[1:] != (H, Wd): d = np.stack([cv2.resize(x, (Wd, H), interpolation=cv2.INTER_LINEAR) for x in d])
xs, ys = [], []
for i, f in enumerate(fr):   # sample pixels where the anchored per-frame depth exists
    p = f'{W}/clean/boot_depth/{f:05d}.png'
    if not os.path.exists(p): continue
    b = cv2.imread(p, -1).astype(np.float64) / 1000; ok = (b > 0.1) & (d[i] > 1e-4)
    idx = np.flatnonzero(ok.ravel())[::97]; xs.append(d[i].ravel()[idx]); ys.append(b.ravel()[idx])
x, y = np.concatenate(xs), np.concatenate(ys)
if MET:
    s = float(np.median(y / x)); out = d * s; fit = dict(model='metric', scale=s)
else:
    A = np.c_[x, np.ones_like(x)]; a, b0 = np.linalg.lstsq(A, 1 / y, rcond=None)[0]
    for _ in range(3):   # robust: refit on the 80% best-agreeing samples
        r = np.abs(A @ [a, b0] - 1 / y); k = r < np.quantile(r, 0.8); a, b0 = np.linalg.lstsq(A[k], 1 / y[k], rcond=None)[0]
    out = 1 / np.clip(a * d + b0, 1e-3, None); fit = dict(model='relative', a=float(a), b=float(b0))
pred = (x * fit['scale']) if MET else 1 / np.clip(fit['a'] * x + fit['b'], 1e-3, None)
fit.update(frames=len(fr), absrel_vs_boot_median=float(np.median(np.abs(pred - y) / y)))
os.makedirs(f'{W}/clean/vda_depth', exist_ok=True)
for i, f in enumerate(fr): cv2.imwrite(f'{W}/clean/vda_depth/{f:05d}.png', np.clip(out[i] * 1000, 0, 65535).round().astype(np.uint16))
json.dump(fit, open(f'{W}/clean/vda_depth.json', 'w'), indent=1); print(json.dumps(fit))
