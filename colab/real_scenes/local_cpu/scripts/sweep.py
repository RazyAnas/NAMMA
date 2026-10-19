# Score cleanup variants: on-path PSNR on the held-out test frames (must not drop), and off-path renders
# (camera moved 0.6 m back and 0.4 m sideways from a test camera) saved side by side for viewing.
import sys, json, time, numpy as np, cv2
sys.path.insert(0, '.')
from plyio import read
from render import gaussians, render
from clean_splat import clean
scene, member = sys.argv[1], sys.argv[2]
hdr, names, a = read('/Users/mariyafatima/Downloads/namma_phone_sfm_lobby_30k.zip', member)
st = json.load(open(f'{scene}_stats.json')); P = json.load(open(f'{scene}_poses.json'))
cams = np.array([np.linalg.inv(np.array(v))[:3, 3] for v in P.values()])
test = st['test_frames']; K, W, H = st['K'], st['width'], st['height']
variants = {'base': {}, 'needle6': dict(needle=6), 'needle4': dict(needle=4), 'near25': dict(near=0.25),
            'fog': dict(fog=(0.1, 0.35)), 'all6': dict(needle=6, near=0.25, fog=(0.1, 0.35)), 'all4': dict(needle=4, near=0.25, fog=(0.1, 0.35))}
off_frames = test[::9][:4]
def offpose(f):
    m = np.array(P[str(f)], dtype=np.float64); c2w = np.linalg.inv(m); c2w[:3, 3] += c2w[:3, :3] @ np.array([0.4, 0.0, -0.6]); return np.linalg.inv(c2w)
res = {}; offs = {}
for name, kw in variants.items():
    t = time.time(); b, stats = clean(names, a, cams, **kw); g = gaussians(names, b); ps = []
    for f in test:
        gt = cv2.cvtColor(cv2.imread(f'frames/{scene}/{f:05d}.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        ps.append(10 * np.log10(1 / ((render(g, P[str(f)], K, W, H) - gt) ** 2).mean()))
    offs[name] = [render(g, offpose(f), K, W, H) for f in off_frames]
    res[name] = dict(stats, psnr_mean=round(float(np.mean(ps)), 3), psnr_min=round(float(np.min(ps)), 2),
                     delta_vs_base=round(float(np.mean(ps)) - res['base']['psnr_mean'], 3) if 'base' in res else 0.0, seconds=round(time.time() - t))
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(f'sweep_{scene}.json', 'w'), indent=1)
for i, f in enumerate(off_frames):
    row = np.concatenate([offs[n][i] for n in variants], 1)
    cv2.imwrite(f'off_{scene}_{f}.jpg', cv2.cvtColor((row * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print('order:', list(variants))
