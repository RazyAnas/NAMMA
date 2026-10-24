# Off-path test split: hold out one contiguous stretch of the walk instead of every 8th frame, so the test views are
# ones the training cameras never came close to. Test = every posed frame in [lo, hi) of the walk (fractions);
# train = the original training frames outside [lo - gap, hi + gap). Rewrites clean/stats.json (keeps a backup) and
# prints how far each test camera is from the nearest training camera. Default 0.78-0.90: the 12% stretch of the phone-room
# walk farthest from the rest (22 cm and 13.5 deg median to the nearest training camera, vs 7.6 cm for the middle).
# Usage: python make_block_split.py <work> [lo=0.78] [hi=0.90] [gap_frames=4]
import sys, json, shutil, os, numpy as np
W = sys.argv[1]; lo, hi = (float(x) for x in (sys.argv[2:4] if len(sys.argv) > 3 else (0.78, 0.90))); gap = int(sys.argv[4]) if len(sys.argv) > 4 else 4
p = f'{W}/clean/stats.json'; bak = f'{W}/clean/stats_every8.json'
if not os.path.exists(bak): shutil.copy(p, bak)
st = json.load(open(bak)); P = json.load(open(f'{W}/clean/poses_w2c.json'))
allf = sorted(int(k) for k in P); n = len(allf); a, b = int(lo * n), int(hi * n)
test = allf[a:b]; ban = set(allf[max(0, a - gap):min(n, b + gap)])
train = [f for f in st['train_frames'] if f not in ban]
C = {f: np.linalg.inv(np.array(P[str(f)]))[:3, 3] for f in allf}
d = [min(np.linalg.norm(C[t] - C[f]) for f in train) for t in test]
st.update(train_frames=train, test_frames=test, split=dict(kind='block', lo=lo, hi=hi, gap_frames=gap))
json.dump(st, open(p, 'w'), indent=1)
print(json.dumps(dict(train=len(train), test=len(test), test_range=[test[0], test[-1]],
                      test_to_nearest_train_m=dict(median=round(float(np.median(d)), 3), max=round(float(np.max(d)), 3)))))
