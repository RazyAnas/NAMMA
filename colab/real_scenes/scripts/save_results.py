# Collect one finished run: scores, timings, Gaussian count, a held-out photo | render grid (worst, median, best frame),
# and the splat with non-finite rows removed, into <out dir>/<scene>_<run>_*.
# Usage: python save_results.py <work> <run dir name> <out dir>
import sys, os, json, numpy as np, cv2
W, run, out = sys.argv[1:4]; R = f'{W}/{run}'; s = os.path.basename(W.rstrip('/')); os.makedirs(out, exist_ok=True); tag = f'{s}_{run}'
sc = json.load(open(f'{R}/scores.json')); t = lambda n: int(open(f'{R}/{n}.txt').read())
with open(f'{R}/splat.ply', 'rb') as fh:
    hdr = []
    while True:
        l = fh.readline(); hdr.append(l)
        if l.strip() == b'end_header': break
    n = int([h for h in hdr if h.startswith(b'element vertex')][0].split()[-1]); k = sum(h.startswith(b'property') for h in hdr)
    g = np.frombuffer(fh.read(), np.float32).reshape(n, k)
ok = np.isfinite(g).all(1); hdr = [h if not h.startswith(b'element vertex') else f'element vertex {ok.sum()}\n'.encode() for h in hdr]
with open(f'{out}/{tag}_splat.ply', 'wb') as fh: fh.write(b''.join(hdr)); fh.write(g[ok].tobytes())
summ = dict(scene=s, run=run, psnr=round(sc['psnr_mean'], 2), ssim=round(sc['ssim_mean'], 3), psnr_min=round(sc['psnr_min'], 2),
            views=sc['n_views'], resolution=sc['resolution'], gaussians=int(ok.sum()), dropped_nonfinite=int((~ok).sum()),
            minutes=round((t('end') - t('start')) / 60, 1), per_frame={r['frame']: round(r['psnr'], 2) for r in sc['per_frame']})
pf = sorted(sc['per_frame'], key=lambda r: r['psnr']); rows = []
for r in [pf[0], pf[len(pf) // 2], pf[-1]]:
    a = cv2.imread(f"{R}/test_{r['frame']:05d}.png"); b = cv2.resize(cv2.imread(f"{W}/data/color_raw/{r['frame']:05d}.jpg"), (a.shape[1], a.shape[0]))
    row = np.hstack([b, a]); cv2.putText(row, f"frame {r['frame']}: photo | render {r['psnr']:.1f} dB", (10, 30), 0, 0.7, (0, 255, 255), 2); rows.append(row)
grid = np.hstack(rows) if rows[0].shape[0] > rows[0].shape[1] else np.vstack(rows)
if max(grid.shape[:2]) > 1800: f_ = 1800 / max(grid.shape[:2]); grid = cv2.resize(grid, None, fx=f_, fy=f_, interpolation=cv2.INTER_AREA)
cv2.imwrite(f'{out}/{tag}_heldout_photo_vs_render.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(summ, open(f'{out}/{tag}_summary.json', 'w'), indent=1); print(json.dumps({k: v for k, v in summ.items() if k != 'per_frame'}))
