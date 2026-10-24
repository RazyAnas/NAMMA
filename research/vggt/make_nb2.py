# Builds VGGT_anchor_test.ipynb: VGGT-1B depth anchored to OUR SfM sparse points (no ground truth), the realistic
# way it would feed boot points. Prints one JSON line between RESULT markers.
import json
R = 'https://raw.githubusercontent.com/RazyAnas/NAMMA/claude/project-thread-gxgkd9/research/vggt/'
cells = []
md = lambda s: cells.append(dict(cell_type='markdown', metadata={}, source=s))
code = lambda s: cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s))
md("# NammaSpace: VGGT depth anchored to our SfM (T4 or CPU, Run all, nothing to upload)")
code(f"""!pip -q install git+https://github.com/facebookresearch/vggt.git opencv-python-headless
import os, zipfile, urllib.request
B = 'https://github.com/isl-org/open3d_downloads/releases/download/augmented-icl-nuim/'
for f in ['livingroom1-color.zip', 'livingroom1-depth-clean.zip', 'livingroom1-traj.txt']:
    if os.path.exists(f) and f.endswith('.zip') and not zipfile.is_zipfile(f): os.remove(f)  # half-downloaded
    if not os.path.exists(f): urllib.request.urlretrieve(B + f, f)
for f in ['vggt_eval.py', 'anchor.py', 'sparse_samples.json']: urllib.request.urlretrieve('{R}' + f, f)
print('data ok')""")
code("""import os, zipfile, torch, numpy as np, json, time
from vggt_eval import *; from anchor import *
os.makedirs('icl', exist_ok=True)
zc, zd = zipfile.ZipFile('livingroom1-color.zip'), zipfile.ZipFile('livingroom1-depth-clean.zip')
for f in FRAMES:
    open(f'icl/{f:05d}.jpg', 'wb').write(zc.read(f'{f:05d}.jpg')); open(f'icl/d{f:05d}.png', 'wb').write(zd.read(f'{f:05d}.png'))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # CPU fallback when Colab has no GPU left (slow, fp32)\ndtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if dev == 'cuda' else torch.float32\nt0 = time.time()
model = VGGT.from_pretrained('facebook/VGGT-1B').to(dev).eval()
imgs = load_and_preprocess_images([f'icl/{f:05d}.jpg' for f in FRAMES]).to(dev)
with torch.no_grad(), torch.autocast(device_type=dev, dtype=dtype, enabled=dev == 'cuda'):
    pred = model(imgs)
depth = pred['depth'][0, ..., 0].float().cpu().numpy(); conf = pred['depth_conf'][0].float().cpu().numpy()
secs = time.time() - t0; print('depth', depth.shape, 'imgs', tuple(imgs.shape), f'{secs:.0f} s on {dev}')""")
code("""S = json.load(open('sparse_samples.json')); G = [gt_depth(f'icl/d{f:05d}.png') for f in FRAMES]
assert depth.shape[1:] == (H, W), depth.shape
def raw(pred, gt, mask=None):
    o = []
    for i, (p, g) in enumerate(zip(pred, gt)):
        m = (g > 0) & np.isfinite(p) & (p > 0)
        if mask is not None: m &= mask[i]
        rel = np.abs(p[m] - g[m]) / g[m]; o.append((rel.mean(), (np.maximum(p[m] / g[m], g[m] / p[m]) < 1.25).mean(), np.median(np.abs(p[m] - g[m])) * 100))
    o = np.array(o); return dict(absrel_pct=float(o[:, 0].mean() * 100), delta1_pct=float(o[:, 1].mean() * 100), median_err_cm=float(o[:, 2].mean()))
anc = [fit_depth_scale(d, np.array(S[str(f)])) for d, f in zip(depth, FRAMES)]
top70 = [c >= np.percentile(c, 30) for c in conf]
res = dict(device=torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', seconds=secs,
           vggt_anchored_to_our_sfm=raw(anc, G),
           vggt_anchored_top70_conf=raw(anc, G, top70),
           vggt_true_scale_per_frame=depth_metrics(list(depth), G, 'scale'),
           vggt_top70_conf_one_scale=depth_metrics([np.where(t, d, np.nan) for d, t in zip(depth, top70)], G, 'global'),
           vggt_one_scale=depth_metrics(list(depth), G, 'global'))
print('RESULT' + json.dumps(res) + 'RESULT')""")
nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'), kernelspec=dict(name='python3', display_name='Python 3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=0)
json.dump(nb, open('VGGT_anchor_test.ipynb', 'w'), indent=1); print('ok')
