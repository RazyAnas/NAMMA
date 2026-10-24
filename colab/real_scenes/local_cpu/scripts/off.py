import sys, json, numpy as np, cv2
sys.path.insert(0, '.')
from plyio import read; from render import gaussians, render; from clean_splat import clean
scene, member = 'phone_sfm', 'phone_sfm_run30k_splat.ply'
hdr, names, a = read('/Users/mariyafatima/Downloads/namma_phone_sfm_lobby_30k.zip', member)
st = json.load(open(f'{scene}_stats.json')); P = json.load(open(f'{scene}_poses.json'))
cams = np.array([np.linalg.inv(np.array(v))[:3, 3] for v in P.values()])
K, W, H = st['K'], st['width'], st['height']; frames = st['test_frames'][::9][:4]
def offpose(f):
    c2w = np.linalg.inv(np.array(P[str(f)], dtype=np.float64)); c2w[:3, 3] += c2w[:3, :3] @ np.array([0.4, 0.0, -0.6]); return np.linalg.inv(c2w)
vs = {'base': {}, 'needle6': dict(needle=6), 'fog': dict(fog=(0.1, 0.35))}
gs = {n: gaussians(names, clean(names, a, cams, **kw)[0]) for n, kw in vs.items()}
rows = []
for f in frames:
    rows.append(np.concatenate([render(gs[n], offpose(f), K, W, H) for n in vs], 1))
img = np.concatenate(rows, 0); img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
cv2.imwrite('off_phone_grid.jpg', cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)); print('ok', frames)
