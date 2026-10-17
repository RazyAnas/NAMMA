"""Score every finished run on its held-out frames: PSNR, SSIM, LPIPS (AlexNet, as MODP's paper), plus a side-by-side image.
  python score.py <results dir>        (each run folder holds run_meta.json from run_all.py and renders/)
Writes <results>/summary.md, summary.json, side_by_side_<scene>.jpg. LPIPS is skipped (shown as n/a) if the lpips package
or its weights are missing, e.g. on a laptop without internet.
"""
import json, os, re, sys
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

R = sys.argv[1]
try:
    import torch, lpips
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; LP = lpips.LPIPS(net='alex', verbose=False).to(dev)
    def lp(a, b):
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) * 2 - 1
        with torch.no_grad(): return float(LP(t(a), t(b)))
except Exception as e:
    print('LPIPS unavailable:', e); lp = None

rows, by_scene = [], {}
for name in sorted(os.listdir(R), key=lambda n: (n.split('_')[0], 'stock' not in n, n)):
    d = f'{R}/{name}'
    if not os.path.exists(f'{d}/run_meta.json'): continue
    meta = json.load(open(f'{d}/run_meta.json')); cams = json.load(open(f'{meta["track"]}/cameras.json')); scene = os.path.dirname(meta['track'])
    info = json.load(open(f'{d}/run_info.json')) if os.path.exists(f'{d}/run_info.json') else {}
    native = json.load(open(f'{d}/native_eval.json')) if os.path.exists(f'{d}/native_eval.json') else {}
    ps, ss, ls, used = [], [], [], []
    for c in cams['test']:
        f = f'{d}/renders/{int(re.findall(r"[0-9]+", c["frame"])[-1]):06d}.png'
        if not os.path.exists(f): continue
        gt = np.asarray(Image.open(f'{scene}/{c["image"]}').convert('RGB'), dtype=np.float32) / 255
        im = np.asarray(Image.open(f).convert('RGB'), dtype=np.float32) / 255
        ps.append(peak_signal_noise_ratio(gt, im, data_range=1)); ss.append(structural_similarity(gt, im, channel_axis=2, data_range=1))
        if lp: ls.append(lp(gt, im))
        used.append((c['image'], f))
    if not ps: print(name, 'no renders yet'); continue
    row = dict(run=name, scene=meta['scene'], label=meta['label'], heldout_frames=len(ps), psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
               lpips=float(np.mean(ls)) if ls else None, native=native, **{k: info.get(k) for k in ('online_fps', 'online_seconds', 'post_refinement_seconds', 'final_gaussians', 'peak_vram_gb', 'gpu')})
    rows.append(row); by_scene.setdefault(meta['scene'], []).append((row, used, scene))

json.dump(rows, open(f'{R}/summary.json', 'w'), indent=1, default=str)
fmt = lambda x, n=2: 'n/a' if x is None else f'{x:.{n}f}'
out = ['# MODP runs on our footage (measured on this Colab GPU)', '',
       'Held-out = every 8th frame, never shown to MODP; scored against the real photo. Higher PSNR/SSIM and lower LPIPS are better.', '']
for sc, items in by_scene.items():
    out += [f'## {sc}', '', '| Run | Inputs | PSNR | SSIM | LPIPS | Online FPS | Post-refine s | Gaussians | GPU |', '|---|---|---|---|---|---|---|---|---|']
    for r, _, _ in items:
        out.append(f"| {r['run']} | {r['label']} | {fmt(r['psnr'])} | {fmt(r['ssim'], 3)} | {fmt(r['lpips'], 3)} | {fmt(r['online_fps'])} | {r['post_refinement_seconds']} | {r['final_gaussians']} | {r['gpu']} |")
    out.append('')
open(f'{R}/summary.md', 'w').write('\n'.join(out)); print('\n'.join(out))

for sc, items in by_scene.items():   # GT | each run, for 3 held-out frames
    common = set.intersection(*[set(g for g, _ in used) for _, used, _ in items]); picks = sorted(common)[:: max(1, len(common) // 3)][:3]
    if not picks: continue
    tiles = []
    for g in picks:
        row = [Image.open(f'{items[0][2]}/{g}').convert('RGB')] + [Image.open(dict(used)[g]).convert('RGB') for _, used, _ in items]
        w, h = row[0].size; s = 320 / max(w, h); row = [t.resize((int(w * s), int(h * s))) for t in row]
        strip = Image.new('RGB', (sum(t.width for t in row), row[0].height)); x = 0
        for t in row: strip.paste(t, (x, 0)); x += t.width
        tiles.append(strip)
    sheet = Image.new('RGB', (tiles[0].width, sum(t.height for t in tiles))); y = 0
    for t in tiles: sheet.paste(t, (0, y)); y += t.height
    sheet.save(f'{R}/side_by_side_{sc}.jpg', quality=90)
    print(f'side_by_side_{sc}.jpg: columns = real photo, ' + ', '.join(r['run'] for r, _, _ in items))
