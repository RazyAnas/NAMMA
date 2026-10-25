"""Put the feed-forward depth on the SfM cameras: SfM gives consistent poses (arbitrary scale), DA3 gives metric dense depth.
Global metric scale = median over frames of (DA3 depth / SfM depth) at the SfM points; each frame's DA3 depth is then rescaled to
agree with the SfM geometry. Output has the same layout as da3.npz (depth, K at DA3 resolution, c2w, rgb, idx)."""
import numpy as np

def align(sfm_npz, da3_npz, out_npz, min_pts=20):
    s = np.load(sfm_npz, allow_pickle=True); d = np.load(da3_npz)
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
    D, Ks, C, RGB, IDX = [], [], [], [], []
    for g, j, K, w2c, w, h, ri in rows:
        D.append(d['depth'][j] * ri * m)                                          # this frame's depth, consistent with SfM geometry
        Kd = K.copy(); Kd[0] *= Wd / w; Kd[1] *= Hd / h; Ks.append(Kd)
        c2w = np.linalg.inv(w2c); c2w[:3, 3] *= m; C.append(c2w); RGB.append(d['rgb'][j]); IDX.append(g)
    np.savez(out_npz, depth=np.array(D, np.float32), K=np.array(Ks), c2w=np.array(C), rgb=np.array(RGB), idx=np.array(IDX))
    spread = float(np.std(np.log(r)))
    print(f'aligned {len(rows)} frames | metric scale {m:.4f} m per SfM unit | per-frame DA3/SfM scale spread {spread*100:.1f}% '
          f'(how much the chunked DA3 scale wandered)')
    return len(rows), m, spread
