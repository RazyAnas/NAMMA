# Post-hoc cleanup of an OpenSplat splat for viewing off the camera path. No retraining.
#   needle k   : a Gaussian whose longest axis is more than k times its middle axis is a needle (a flat
#                disc on a wall is fine: that is long-long-thin). Shrink the long axis to k x middle.
#   near r     : drop Gaussians within r metres of any training camera centre. The cameras never look at
#                what sits right on top of them, so these are unconstrained blobs that hang in the walkway.
#   fog s, o   : drop Gaussians larger than s metres with opacity below o (big faint haze).
import numpy as np
from scipy.spatial import cKDTree

def clean(names, a, cams, needle=None, near=None, fog=None):
    ix = {n: i for i, n in enumerate(names)}; a = a.copy(); keep = np.ones(len(a), bool); st = {}
    sc = np.exp(a[:, [ix['scale_0'], ix['scale_1'], ix['scale_2']]])
    if needle:
        o = np.argsort(-sc, 1); r = np.arange(len(sc))
        s1, s2 = sc[r, o[:, 0]], sc[r, o[:, 1]]; bad = s1 > needle * s2
        sc[r[bad], o[bad, 0]] = needle * s2[bad]; st['needles_shrunk'] = int(bad.sum())
        for k in range(3): a[:, ix[f'scale_{k}']] = np.log(sc[:, k])
    if near:
        d = cKDTree(cams).query(a[:, [ix['x'], ix['y'], ix['z']]])[0]; m = d < near; keep &= ~m; st['near_removed'] = int(m.sum())
    if fog:
        s, o = fog; op = 1 / (1 + np.exp(-a[:, ix['opacity']])); m = (sc.max(1) > s) & (op < o); keep &= ~m; st['fog_removed'] = int(m.sum())
    st['kept'] = int(keep.sum()); st['of'] = len(a)
    return a[keep], st
