"""Camera check with real pixels: rebuild EVERY frame from its neighbours (never itself) at low resolution and score it against
the real photo. A frame whose rebuild is far worse than its neighbours' has a wrong camera (or its sources do).
Proof 1 found 4/16 held-out frames at 12-15 dB showing a window where the photo shows a door: misplaced cameras.
Usage: python ibr_check.py <cache> <out> [--workers 4]. CPU only."""
import argparse, json, os, time, numpy as np
from multiprocessing import Pool
from PIL import Image
import ibr

G = {}
def score(i):
    d, files, K, has, H, W = G['data']
    img, m, srcs, _ = ibr.synth(d, files, K, has, H, W, K[i], d['c2w'][i], exclude={i}, k=3)
    gt = ibr.image(files, int(d['idx'][i]), W, H)
    return i, (ibr.psnr(img, gt, m) if m.mean() > 0.3 else 0.0), float(m.mean()), [int(s) for s in srcs]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cache'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    G['data'] = ibr.load(a.cache, scale=1)                                          # 280 x 504: fast, enough to see a wrong camera
    d = G['data'][0]; n = len(d['idx'])
    with Pool(a.workers) as p: res = sorted(p.map(score, range(n), chunksize=8))
    ps = np.array([r[1] for r in res]); fid = d['idx']
    loc = np.array([np.median(ps[max(0, i - 6):i + 7]) for i in range(n)])            # local median along the walk
    bad = [i for i in range(n) if ps[i] < 18 and ps[i] < loc[i] - 5]
    print(f'{n} frames in {time.time()-t0:.0f}s | median {np.median(ps):.1f} dB | below 18 dB: {(ps < 18).sum()} | flagged (also 5 dB under '
          f'their neighbours): {len(bad)} -> frames {[int(fid[i]) for i in bad][:80]}', flush=True)
    runs, cur = [], []                                                                # consecutive flagged frames = a misplaced stretch
    for i in bad:
        if cur and i != cur[-1] + 1: runs.append(cur); cur = []
        cur.append(i)
    if cur: runs.append(cur)
    print('flagged stretches (frame ids):', [(int(fid[r[0]]), int(fid[r[-1]])) for r in runs], flush=True)
    json.dump(dict(psnr={int(fid[i]): float(ps[i]) for i in range(n)}, flagged=[int(fid[i]) for i in bad],
                   stretches=[(int(fid[r[0]]), int(fid[r[-1]])) for r in runs]), open(f'{a.out}/pose_check.json', 'w'))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 3)); plt.plot(fid, ps, lw=.8); plt.scatter(fid[bad], ps[bad], c='r', s=8); plt.ylabel('rebuild PSNR (dB)')
    plt.xlabel('frame'); plt.title('Rebuilding each frame from its neighbours: red = camera probably wrong'); plt.tight_layout(); plt.savefig(f'{a.out}/pose_check.png', dpi=110)
    worst = np.argsort(ps)[:6]; d_, files, K, has, H, W = G['data']; rows = []
    for i in worst:
        img, m, _, _ = ibr.synth(d_, files, K, has, H, W, K[i], d_['c2w'][i], exclude={i}, k=3)
        rows.append(np.concatenate([ibr.image(files, int(fid[i]), W, H), img], 1))
    Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(f'{a.out}/pose_check_worst.jpg', quality=90)
