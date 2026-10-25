"""Put the feed-forward depth on the SfM cameras: SfM gives consistent poses (arbitrary scale), DA3 gives metric dense depth.
Global metric scale = median over frames of (DA3 depth / SfM depth) at the SfM points; each frame's DA3 depth is then rescaled to
agree with the SfM geometry. Output has the same layout as da3.npz (depth, K at DA3 resolution, c2w, rgb, idx)."""
import numpy as np

def align(sfm_npz, da3_npz, out_npz, min_pts=20):
    s = np.load(sfm_npz, allow_pickle=True); z = np.load(da3_npz); d = {k: z[k] for k in z.files}   # materialise: d['depth'][j] on a lazy npz re-reads the whole array (OOM on Kaggle)
    Hd, Wd = d['depth'].shape[1:]; pos = {int(g): j for j, g in enumerate(d['idx'])}
    rows = []
    for n, K, w2c, (w, h), obs in zip(s['names'], s['K'], s['w2c'], s['wh'], s['obs']):
        g = int(str(n).split('.')[0])
        if g not in pos or len(obs) < min_pts: continue
        j = pos[g]; u = np.clip((obs[:, 0] * Wd / w).astype(int), 0, Wd - 1); v = np.clip((obs[:, 1] * Hd / h).astype(int), 0, Hd - 1)
        zd = d['depth'][j][v, u]; ok = zd > 0
        if ok.sum() < min_pts: continue
        rows.append((g, j, K, w2c, w, h, float(np.median(obs[ok, 2] / zd[ok]))))   # SfM units per DA3 metre, this frame
    if len(rows) < 10: raise RuntimeError(f'only {len(rows)} frames usable for alignment')
    r = np.array([x[-1] for x in rows]); m = 1.0 / np.median(r)                   # SfM units -> metres
    have = {x[0]: x for x in rows}
    D, Ks, C, RGB, IDX = [], [], [], [], []
    for n, K, w2c, (w, h) in zip(s['names'], s['K'], s['w2c'], s['wh']):   # EVERY registered frame; depth only where DA3 ran
        g = int(str(n).split('.')[0])
        if g in have: _, j, _, _, _, _, ri = have[g]; D.append(d['depth'][j] * ri * m); RGB.append(d['rgb'][j])
        else: D.append(np.zeros((Hd, Wd), np.float32)); RGB.append(np.zeros((Hd, Wd, 3), np.uint8))
        Kd = K.copy(); Kd[0] *= Wd / w; Kd[1] *= Hd / h; Ks.append(Kd)
        c2w = np.linalg.inv(w2c); c2w[:3, 3] *= m; C.append(c2w); IDX.append(g)
    o = np.argsort(IDX); D, Ks, C, RGB, IDX = [np.array(x)[o] for x in (D, Ks, C, RGB, IDX)]
    np.savez(out_npz, depth=D.astype(np.float32), K=Ks, c2w=C, rgb=RGB, idx=IDX)
    print(f'{len(IDX)} SfM frames, {len(rows)} with dense depth')
    spread = float(np.std(np.log(r)))
    print(f'aligned {len(rows)} frames | metric scale {m:.4f} m per SfM unit | per-frame DA3/SfM scale spread {spread*100:.1f}% '
          f'(how much the chunked DA3 scale wandered)')
    return len(rows), m, spread
