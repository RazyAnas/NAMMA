# Our side: read the zip the friend sends back and build the final comparison table + image.
# Usage: python3 ingest_results.py namma_results_YYYYMMDD_HHMM.zip [out dir]
# Writes <out>/final_comparison.md, final_comparison.png and final_comparison.json (default out: returned/<zip name>/).
import json, os, sys, zipfile, glob
import numpy as np
from PIL import Image, ImageDraw

zp = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[2] if len(sys.argv) > 2 else f'{here}/returned/{os.path.basename(zp)[:-4]}'
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(zp) as z:
    for m in z.namelist():   # extract safely: only plain relative paths under results/
        if m.startswith('results/') and '..' not in m and not m.startswith('/'):
            z.extract(m, out)
R = f'{out}/results'
LABEL = {
    'clean_sparse': 'Sparse tracker points (what MODP is fed)', 'clean_random': 'Random points',
    'clean_boot': 'Our anchored monocular points', 'clean_gtdepth': 'True depth (upper bound)',
    'degraded_sparse': 'Degraded video: sparse points', 'degraded_boot': 'Degraded video: our monocular points',
    'modp_clean_sparse': 'MODP, sparse tracker points', 'modp_clean_boot_mv': 'MODP, our checked monocular points',
}
runs = {}
for mp in sorted(glob.glob(f'{R}/*/metrics.json')):
    m = json.load(open(mp)); runs[m['run']] = m
if not runs: sys.exit('no metrics.json found in the zip: the friend may have sent the zip before run_all.sh finished')
# compare on the frames every run rendered (some degraded test frames could not be posed)
common = set.intersection(*[{r['frame'] for r in m['per_frame']} for m in runs.values()])
keys = ['psnr', 'psnr_cc', 'ssim', 'lpips_alex', 'lpips_vgg']
table = []
for name, m in runs.items():
    rows = [r for r in m['per_frame'] if r['frame'] in common]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]}
    info = m.get('info', {})
    table.append(dict(run=name, label=LABEL.get(name, name), frames=len(rows), **mean,
                      train_min=info.get('train_seconds', float('nan')) / 60, gaussians=info.get('final_gaussians'),
                      steps=info.get('steps', 'MODP default' if name.startswith('modp') else None), gpu=info.get('gpu')))
order = list(LABEL) + sorted(set(runs) - set(LABEL))
table.sort(key=lambda t: order.index(t['run']))
f = lambda v, d: '-' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'
env = json.load(open(f'{R}/environment.json')) if os.path.exists(f'{R}/environment.json') else {}
md = ['# Full-quality GPU run: which starting points give the best 3D model?', '',
      f'Measured on the friend\'s GPU ({env.get("gpu", "unknown GPU")}, driver {env.get("driver", "?")}), returned in `{os.path.basename(zp)}`.',
      f'Scored on {len(common)} held-out ICL-NUIM living-room frames the models never trained on, against the clean frames. Higher PSNR/SSIM and lower LPIPS are better.', '',
      '| Starting points | PSNR | PSNR (colour-matched) | SSIM | LPIPS (Alex) | LPIPS (VGG) | Train time (min) | Gaussians |',
      '|---|---|---|---|---|---|---|---|']
for t in table:
    md.append(f'| {t["label"]} | {f(t.get("psnr"), 2)} | {f(t.get("psnr_cc"), 2)} | {f(t.get("ssim"), 3)} | {f(t.get("lpips_alex"), 3)} | '
              f'{f(t.get("lpips_vgg"), 3)} | {f(t["train_min"], 1)} | {t["gaussians"] or "-"} |')
missing = [n for n in LABEL if n not in runs]
if missing: md += ['', 'Runs not in the zip (failed or skipped): ' + ', '.join(missing) + '. See results/logs/ for why.']
md += ['', 'Paper numbers for context (not comparable: other scenes, full resolution, RTX 4090): MODP 25.45 dB on TUM, 35.85 dB on Replica.',
       'Our earlier CPU-only check (1,000 steps, 320x240): about 24.3-24.7 dB for every pose source.', '', '![comparison](final_comparison.png)']
open(f'{out}/final_comparison.md', 'w').write('\n'.join(md) + '\n')
json.dump(dict(common_frames=sorted(common), table=table, environment=env), open(f'{out}/final_comparison.json', 'w'), indent=1)
# image: ground truth + every run, for the example frames
ex = sorted({os.path.basename(p)[:5] for p in glob.glob(f'{R}/*/examples/*.jpg')})[:3]
names = [t['run'] for t in table]; tw, th, top = 320, 240, 40
sheet = Image.new('RGB', (tw * (len(names) + 1), top + th * len(ex)), 'white'); d = ImageDraw.Draw(sheet)
gt_dir = os.path.join(here, 'package', 'data', 'images', 'clean')
for j, lab in enumerate(['Ground truth'] + [LABEL.get(n, n) for n in names]):
    words, line, y = lab.split(), '', 2
    for w in words:  # wrap labels to the tile width
        if len(line + ' ' + w) > 42: d.text((tw * j + 4, y), line, fill='black'); line, y = w, y + 12
        else: line = (line + ' ' + w).strip()
    d.text((tw * j + 4, y), line, fill='black')
for i, fr in enumerate(ex):
    gp = f'{gt_dir}/{fr}.jpg'
    if os.path.exists(gp): sheet.paste(Image.open(gp).convert('RGB').resize((tw, th)), (0, top + th * i))
    for j, n in enumerate(names):
        p = f'{R}/{n}/examples/{fr}.jpg'
        if os.path.exists(p): sheet.paste(Image.open(p).convert('RGB').resize((tw, th)), (tw * (j + 1), top + th * i))
sheet.save(f'{out}/final_comparison.png')
print('\n'.join(md)); print(f'\nwrote {out}/final_comparison.md and final_comparison.png')
