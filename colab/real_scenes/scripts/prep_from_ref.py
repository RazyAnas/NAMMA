# Stage 1 alternative: take poses and per-frame sparse points from an earlier prepared set (namma_modp_colab.zip:
# lobby = incremental COLMAP merged from 4 pieces, phone = ARCore poses refined by bundle adjustment) instead of our
# global SfM, writing the same clean/ files prep_sfm.py writes. Used where our global SfM is shown to be worse.
# Usage: python prep_from_ref.py <work> <cameras.json> <points/sparse dir>
import sys, os, re, json, numpy as np
W, CJ, SP = sys.argv[1:4]; os.makedirs(f'{W}/clean', exist_ok=True)
cam = json.load(open(CJ)); K = np.array(cam['K']); Wd, H = cam['width'], cam['height']; num = lambda s: int(re.sub(r'\D', '', s))
P = {num(e['frame']): np.array(e['w2c']) for e in cam['train'] + cam['test']}
train = sorted(num(e['frame']) for e in cam['train']); test = sorted(num(e['frame']) for e in cam['test'])
files = {num(os.path.splitext(n)[0]): n for n in os.listdir(SP)}
X, obs, n0 = [], {}, 0
for f in train:
    x = np.load(f'{SP}/{files[f]}').astype(np.float64) if f in files else np.zeros((0, 3))
    Xc = x @ P[f][:3, :3].T + P[f][:3, 3]; z = Xc[:, 2]; ok = z > 0.05
    uv = np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    X.append(x[ok]); obs[f'uv_{f}'] = uv[ok].astype(np.float32); obs[f'id_{f}'] = np.arange(n0, n0 + ok.sum(), dtype=np.int32); n0 += int(ok.sum())
X3 = np.concatenate(X)
np.savez_compressed(f'{W}/clean/sfm_obs.npz', X3=X3.astype(np.float32), C3=np.full((len(X3), 3), 128, np.uint8), **obs)
json.dump({str(f): P[f].tolist() for f in sorted(P)}, open(f'{W}/clean/poses_w2c.json', 'w'))
stats = dict(K=K.tolist(), width=Wd, height=H, registered=len(P), of=len(P), missing=[], sfm_points=int(len(X3)), poses_from=CJ,
             vs_reference=dict(centre_rmse_cm=0.0, centre_median_cm=0.0, rot_median_deg=0.0, rot_p90_deg=0.0),
             train_frames=train, test_frames=test)
json.dump(stats, open(f'{W}/clean/stats.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in stats.items() if 'frames' not in k}), 'obs per frame median', int(np.median([len(v) for k, v in obs.items() if k.startswith('id_')])))
