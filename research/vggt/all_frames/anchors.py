# Visible-anchor approximation: project sparse.ply with our poses, nearest per 4 px cell, then drop points lying more than
# TOL behind the nearest point within a WIN x WIN cell neighbourhood (they are most likely hidden behind that nearer surface).
# Fixed a priori: WIN=5 (20 px), TOL=5%. Writes anchors.npz {frame: (uv, z)}. Prints anchor-vs-GT error as a check only.
import json, numpy as np, cv2, sys
from scipy.ndimage import minimum_filter
from plyfile import PlyData
Wk = '/tmp/claude-0/work'; D0 = f'{Wk}/data/namma_wholeroom_gpu/data'; WIN, TOL, cell = 5, 0.05, 4
K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1]]); H, Wd = 480, 640
est = {c['frame']: np.array(c['w2c']) for c in json.load(open(f'{D0}/clean/cameras.json'))['train']}
pv = PlyData.read(f'{D0}/clean/points/sparse.ply')['vertex']; X3 = np.stack([pv['x'], pv['y'], pv['z']], 1).astype(np.float64)
out = {}; st = []
for f in sorted(est):
    Xc = X3 @ est[f][:3, :3].T + est[f][:3, 3]; zc = Xc[:, 2]; ok = zc > 0.05
    uu = K[0, 0] * Xc[:, 0] / zc + K[0, 2]; vv = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
    ok &= (uu >= 0) & (uu < Wd - 1) & (vv >= 0) & (vv < H - 1); uu, vv, zc = uu[ok], vv[ok], zc[ok]
    cy, cx = (vv // cell).astype(int), (uu // cell).astype(int); key = cy * 1000 + cx
    o = np.lexsort((zc, key)); _, first = np.unique(key[o], return_index=True); s = o[first]
    zb = np.full((H // cell + 1, Wd // cell + 1), np.inf); zb[cy[s], cx[s]] = zc[s]
    near = minimum_filter(zb, size=WIN, mode='constant', cval=np.inf)
    s = s[zc[s] <= (1 + TOL) * near[cy[s], cx[s]]]
    uv, z = np.stack([uu[s], vv[s]], 1), zc[s]; out[str(f)] = np.concatenate([uv, z[:, None]], 1)
    g = cv2.imread(f'{Wk}/floor/gtd/{f:05d}.png', -1) / 1000.; gd = g[uv[:, 1].astype(int), uv[:, 0].astype(int)]; m = gd > 0
    r = np.abs(z[m] - gd[m]) / gd[m]; st.append((len(z), np.median(r), (r > 0.1).mean()))
np.savez_compressed(sys.argv[1], **out); st = np.array(st)
print('anchors/frame median', np.median(st[:, 0]), 'min', st[:, 0].min(), '| median anchor err', np.median(st[:, 1]).round(4), '| frac >10% off: median', np.median(st[:, 2]).round(3), 'max', st[:, 2].max().round(3))
