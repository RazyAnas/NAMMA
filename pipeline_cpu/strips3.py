"""Round-3 pictures: strips photo | round-2 hybrid | round-3 hybrid, and wall error maps (|hybrid - photo| on the wall mask).
python strips3.py <round2 work> <round3 work> <out dir> frame [frame ...]"""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from stages import read_img
from walls import wall_mask, mpsnr

def setblend(W):
    a = json.load(open(f'{W}/args.json')); E.BLEND.update(mode=a.get('blend', 'hard'), p=a.get('blend_p', 2), n0=a.get('blend_n0', 4))

r2, r3, out = sys.argv[1:4]; frames = [int(x) for x in sys.argv[4:]]
os.makedirs(f'{out}/strips', exist_ok=True)
A = E.setup(r2); B = E.setup(r3); S = A[0]; K = S['K']; H, Wd = S['height'], S['width']
for f in frames:
    gt = read_img(r3, f).astype(np.float64); wm = wall_mask(gt)
    setblend(r2); ha = E.render_all(*A, S['w2c'][f], K, H, Wd)[0]['hybrid']
    setblend(r3); hb = E.render_all(*B, S['w2c'][f], K, H, Wd)[0]['hybrid']
    cols = [E.label(gt, f'photo {f}'), E.label(ha, f'R2 hybrid, wall {mpsnr(ha, gt, wm):.1f}dB'), E.label(hb, f'R3 hybrid, wall {mpsnr(hb, gt, wm):.1f}dB')]
    cv2.imwrite(f'{out}/strips/strip_{f:05d}.jpg', np.concatenate(cols, 1))
    def emap(h):
        e = np.clip(np.abs(h - gt).mean(-1) / 0.2, 0, 1); c = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1] / 255.
        c[~wm] = 0.35 * c[~wm] + 0.65 * gt[~wm] * 0.4; return c        # non-wall pixels dimmed
    em = [E.label(gt * np.where(wm[..., None], 1, 0.4), 'wall mask'), E.label(emap(ha), 'R2 wall |error|'), E.label(emap(hb), 'R3 wall |error|')]
    cv2.imwrite(f'{out}/wall_error_{f:05d}.jpg', np.concatenate(em, 1))
    print(f, 'saved', flush=True)
