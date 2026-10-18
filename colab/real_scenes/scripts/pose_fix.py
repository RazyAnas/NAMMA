# Camera refinement, held-out side (as MODP does for frames it did not map): OpenSplat refined every training camera
# (OPENSPLAT_POSE_OUT); each held-out camera gets the correction of its neighbouring training cameras, interpolated in
# time (rotation slerp, translation lerp of the world-side correction). Writes a poses JSON for all frames.
# Usage: python pose_fix.py <work> <poses.txt> <out.json>
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
W, pf, out = sys.argv[1:4]
P = {int(k): np.array(v) for k, v in json.load(open(f'{W}/clean/poses_w2c.json')).items()}
st = json.load(open(f'{W}/clean/stats.json'))
L = open(pf).read().split('\n'); s = float(L[0].split()[1]); corr = {}
for l in L[1:]:
    if not l.strip(): continue
    t = l.split(); f = int(t[0].replace('test_', '')[:5]); C = np.array([float(x) for x in t[1:]]).reshape(4, 4)
    w = P[f].copy(); w[:3, :3] = C[:3, :3] @ P[f][:3, :3]; w[:3, 3] = C[:3, :3] @ P[f][:3, 3] + C[:3, 3] / s
    corr[f] = w
D = {f: corr[f] @ np.linalg.inv(P[f]) for f in corr}   # world-side correction: w2c' = D w2c
tr = sorted(D); new = dict(corr)
for f in st['test_frames']:
    a = max([g for g in tr if g < f], default=None); b = min([g for g in tr if g > f], default=None)
    if a is None or b is None:
        Df = D[a if b is None else b]
    else:
        u = (f - a) / (b - a); r = Slerp([0, 1], Rot.from_matrix([D[a][:3, :3], D[b][:3, :3]]))(u).as_matrix()
        Df = np.eye(4); Df[:3, :3] = r; Df[:3, 3] = (1 - u) * D[a][:3, 3] + u * D[b][:3, 3]
    new[f] = Df @ P[f]
for f in P:
    new.setdefault(f, P[f])
mv = [np.linalg.norm((-np.linalg.inv(new[f])[:3, 3]) + np.linalg.inv(P[f])[:3, 3]) for f in st['test_frames']]
json.dump({str(f): new[f].tolist() for f in sorted(new)}, open(out, 'w'))
print(json.dumps(dict(train_refined=len(corr), test_moved_cm_median=float(100 * np.median(mv)), test_moved_cm_max=float(100 * np.max(mv)))))
