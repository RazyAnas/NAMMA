"""Camera check with real pixels: rebuild EVERY frame from its neighbours (never itself) at low resolution and score it against
the real photo. A frame whose rebuild is far worse than its neighbours' has a wrong camera (or its sources do).
Proof 1 found 4/16 held-out frames at 12-15 dB showing a window where the photo shows a door: misplaced cameras.
Usage: python ibr_check.py <cache> <out> [--workers 4] [--keep good_frames.json]. CPU only."""
import argparse, json, os, time, numpy as np
from multiprocessing import Pool
from PIL import Image
import ibr

G = {'drop': set()}
def score(i):
    d, files, K, has, H, W = G['data']
    img, m, srcs, _ = ibr.synth(d, files, K, has, H, W, K[i], d['c2w'][i], exclude={i} | G['drop'], k=3)
    gt = ibr.image(files, int(d['idx'][i]), W, H)
    return i, (ibr.psnr(img, gt, m) if m.mean() > 0.3 else 0.0), float(m.mean()), [int(s) for s in srcs]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cache'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--keep'); ap.add_argument('--prune', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    G['data'] = ibr.load(a.cache, scale=1, keep=a.keep)                                          # 280 x 504: fast, enough to see a wrong camera
    d = G['data'][0]; n = len(d['idx'])
    with Pool(a.workers) as p: res = sorted(p.map(score, range(n), chunksize=8))
    ps = np.array([r[1] for r in res]); fid = d['idx']
    # --prune R: a misplaced STRETCH of frames vouches for itself (its frames rebuild each other fine) and pollutes the correct
    # frames it lands next to. 1) Cut the walk where the camera jumps (step per frame id > 6x the median and > 10 cm): a short
    # piece between two jumps was placed somewhere else -> drop it whole. 2) Repeat: drop frames < 20 dB, re-score only frames
    # that used a dropped frame as a source; stop when nothing new is dropped.
    srcs = {r[0]: r[3] for r in res}; new = set()
    if a.prune:
        c = d['c2w'][:, :3, 3]; step = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(fid), 1)
        cut = [0] + [i + 1 for i in np.where((step > 6 * np.median(step)) & (step > 0.10))[0]] + [n]
        segs = [(x, y) for x, y in zip(cut, cut[1:]) if y > x]; minseg = max(10, n // 25)
        for x, y in segs:
            if y - x < minseg: new |= set(range(x, y))
        print(f'walk cut at {len(cut)-2} camera jumps into {len(segs)} pieces; short pieces dropped (frame ids): '
              f'{[(int(fid[x]), int(fid[y-1])) for x, y in segs if y - x < minseg]}', flush=True)
    for rnd in range(a.prune):
        if not (rnd == 0 and new): new |= {i for i in range(n) if i not in G['drop'] and ps[i] < 20}   # round 1: jumps only, then re-score
        if not new: break
        G['drop'] |= new; new = set(); todo = [i for i in range(n) if i not in G['drop'] and set(srcs[i]) & G['drop']]
        with Pool(a.workers) as p:
            for i, v, _, sr in p.map(score, todo, chunksize=4): ps[i] = v; srcs[i] = sr
        live = [i for i in range(n) if i not in G['drop']]
        print(f'prune round {rnd+1}: dropped total {len(G["drop"])}, re-scored {len(todo)} | kept {len(live)} | '
              f'median {np.median(ps[live]):.1f} dB | kept below 20 dB: {int((ps[live] < 20).sum())}', flush=True)
    if a.prune:
        keep = [int(fid[i]) for i in range(n) if i not in G['drop']]
        json.dump(dict(rule=f'short pieces between camera jumps dropped, then real-pixel rebuild >= 20 dB, iterative ({a.prune} rounds max)', keep=keep),
                  open(f'{a.out}/good_frames.json', 'w'))
        ps = np.where(np.isin(np.arange(n), sorted(G['drop'])), np.nan, ps)          # plots/stats below: kept frames only
    loc = np.array([np.nanmedian(ps[max(0, i - 6):i + 7]) for i in range(n)])            # local median along the walk
    bad = [i for i in range(n) if ps[i] < 18 and ps[i] < loc[i] - 5]
    print(f'{n} frames in {time.time()-t0:.0f}s | median {np.nanmedian(ps):.1f} dB | below 18 dB: {(ps < 18).sum()} | flagged (also 5 dB under '
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
