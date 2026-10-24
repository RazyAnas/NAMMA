# Difix3D / Difix3D+ (nv-tlabs/Difix3D, CVPR 2025) on top of a round-1 gsplat splat, on the same off-path test:
#   base     the loaded splat rendered at the held-out cameras (should match its round-1 score)
#   difix3d  progressive 3D update: for each held-out camera, step from its nearest training camera towards it
#            (1/R, 2/R .. 1 of the way), render, clean the render with Difix (nearest training frame as reference),
#            add the cleaned views as extra training images and fine-tune; R rounds
#   plus     difix3d renders cleaned once more by Difix at render time (the "+" of Difix3D+)
# Only held-out camera poses are used, never their photos, as in the paper's novel-trajectory protocol.
# Writes <out>/{base,difix3d,plus}/test_XXXXX.png (score each with score_whole.py) and <out>/difix3d/splat.ply.
# Usage: python gsplat_difix.py <work> <splat.ply> <out> [rounds=3] [steps_per_round=1500]
# Needs DIFIX_SRC (Difix3D/src) on sys.path and its pinned diffusers / transformers (colab_round2.sh installs them).
import sys, os, json, time, numpy as np, cv2, torch, torch.nn.functional as F
sys.path[:0] = [p for p in (os.environ.get('DIFIX_PKGS'), os.environ.get('DIFIX_SRC')) if p]
from plyfile import PlyData, PlyElement
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from gsplat import rasterization
W, PLY, OUT = sys.argv[1:4]; R = int(sys.argv[4]) if len(sys.argv) > 4 else 3; SPR = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
dev = 'cuda'; torch.manual_seed(0); np.random.seed(0); t0 = time.time()
st = json.load(open(f'{W}/clean/stats.json')); P = json.load(open(f'{W}/clean/poses_w2c.json'))
H, Wd = st['height'], st['width']; K = torch.tensor(st['K'], dtype=torch.float32, device=dev)
train, test = st['train_frames'], st['test_frames']

def w2c(f):
    m = np.array(P[str(f)], dtype=np.float64); return m if m.shape[0] == 4 else np.vstack([m, [0, 0, 0, 1.]])
def load_rgb(f): return cv2.cvtColor(cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg'), cv2.COLOR_BGR2RGB)
imgs = {f: torch.from_numpy(load_rgb(f)).float().div(255) for f in train}
C2W = {f: np.linalg.inv(w2c(f)) for f in train + test}
nearest = {t: min(train, key=lambda f: np.linalg.norm(C2W[f][:3, 3] - C2W[t][:3, 3])) for t in test}

def between(a, b, s):   # camera-to-world s of the way from a to b (slerp rotation, lerp centre) -> world-to-camera
    rot = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))([s]).as_matrix()[0]
    m = np.eye(4); m[:3, :3] = rot; m[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]; return np.linalg.inv(m)

# splat.ply as written by gsplat_train.py (3DGS layout, SH degree 3)
v = PlyData.read(PLY)['vertex']; n = len(v['x']); g = lambda k: np.asarray(v[k], dtype=np.float32)
rest = np.stack([g(f'f_rest_{i}') for i in range(45)], 1).reshape(n, 3, 15).transpose(0, 2, 1)
init = {'means': np.stack([g('x'), g('y'), g('z')], 1), 'scales': np.stack([g(f'scale_{i}') for i in range(3)], 1),
        'quats': np.stack([g(f'rot_{i}') for i in range(4)], 1), 'opacities': g('opacity'),
        'sh0': np.stack([g(f'f_dc_{i}') for i in range(3)], 1)[:, None], 'shN': rest}
cams = np.stack([C2W[f][:3, 3] for f in train]); scene_scale = float(np.linalg.norm(cams - cams.mean(0), axis=1).max()) * 1.1
lrs = {'means': 1.6e-5 * scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}  # means at the end-of-schedule lr
splats = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(x)) for k, x in init.items()}).to(dev)
opt = {k: torch.optim.Adam([{'params': splats[k], 'lr': lr}], eps=1e-15) for k, lr in lrs.items()}
print('loaded', n, 'gaussians', flush=True)

def render(vm):
    vm = torch.tensor(vm, dtype=torch.float32, device=dev) if not torch.is_tensor(vm) else vm
    c, a, _ = rasterization(splats['means'], splats['quats'], torch.exp(splats['scales']), torch.sigmoid(splats['opacities']),
                            torch.cat([splats['sh0'], splats['shN']], 1), vm[None], K[None], Wd, H, sh_degree=3, packed=False)
    return c[0].clamp(0, 1)

def ssim(x, y, w=11):
    gk = torch.exp(-((torch.arange(w, device=dev) - w // 2) ** 2) / (2 * 1.5 ** 2)); gk = gk / gk.sum(); k = (gk[:, None] * gk[None])[None, None].repeat(3, 1, 1, 1)
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]; f = lambda z: F.conv2d(z, k, padding=w // 2, groups=3)
    mx, my = f(x), f(y); sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4) * (sxx + syy + 9e-4))).mean()

to_u8 = lambda t: (t.detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
def save(d, f, rgb): os.makedirs(d, exist_ok=True); cv2.imwrite(f'{d}/test_{f:05d}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

from pipeline_difix import DifixPipeline
pipe = DifixPipeline.from_pretrained('nvidia/difix_ref', trust_remote_code=True).to(dev); pipe.set_progress_bar_config(disable=True)
def fix(rgb, ref_frame):   # uint8 HxWx3 -> uint8 HxWx3 (both sides are multiples of 8 for our frames)
    out = pipe('remove degradation', image=Image.fromarray(rgb), ref_image=Image.fromarray(load_rgb(ref_frame)),
               num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    return np.asarray(out.resize((rgb.shape[1], rgb.shape[0]), Image.BICUBIC))

with torch.no_grad():
    for f in test: save(f'{OUT}/base', f, to_u8(render(np.linalg.inv(C2W[f]))))

pseudo = {}   # key -> (world-to-camera, cleaned image tensor)
for r in range(1, R + 1):
    s = r / R
    with torch.no_grad():
        for t in test:
            vm = between(C2W[nearest[t]], C2W[t], s)
            pseudo[(t, r)] = (torch.tensor(vm, dtype=torch.float32, device=dev), torch.from_numpy(fix(to_u8(render(vm)), nearest[t])).float().div(255))
    keys = [('train', f) for f in train] + [('pseudo', k) for k in pseudo]; order = []
    for step in range(SPR):
        if not order: order = list(np.random.permutation(len(keys)))
        kind, k = keys[order.pop()]
        if kind == 'train': vm, gt, wgt = torch.tensor(np.linalg.inv(C2W[k]), dtype=torch.float32, device=dev), imgs[k], 1.0
        else: (vm, gt), wgt = pseudo[k], 0.5   # cleaned renders are guesses: half weight
        img = render(vm); gt = gt.to(dev)
        loss = wgt * (0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img, gt)))
        loss.backward()
        for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
    print(f'round {r}/{R} at {s:.2f} of the way: {len(pseudo)} cleaned views, loss {loss.item():.4f}, t {time.time() - t0:.0f}s', flush=True)

with torch.no_grad():
    for f in test:
        rgb = to_u8(render(np.linalg.inv(C2W[f]))); save(f'{OUT}/difix3d', f, rgb); save(f'{OUT}/plus', f, fix(rgb, nearest[f]))
    sp = {k: splats[k].detach().cpu().numpy() for k in splats}; q = sp['quats'] / np.linalg.norm(sp['quats'], axis=1, keepdims=True)
    cols = [('x', sp['means'][:, 0]), ('y', sp['means'][:, 1]), ('z', sp['means'][:, 2]), ('nx', np.zeros(n)), ('ny', np.zeros(n)), ('nz', np.zeros(n))]
    cols += [(f'f_dc_{i}', sp['sh0'][:, 0, i]) for i in range(3)]
    cols += [(f'f_rest_{i}', x) for i, x in enumerate(sp['shN'].transpose(0, 2, 1).reshape(n, -1).T)]
    cols += [('opacity', sp['opacities'])] + [(f'scale_{i}', sp['scales'][:, i]) for i in range(3)] + [(f'rot_{i}', q[:, i]) for i in range(4)]
    a = np.empty(n, dtype=[(k, 'f4') for k, _ in cols])
    for k, x in cols: a[k] = x
    PlyData([PlyElement.describe(a, 'vertex')]).write(f'{OUT}/difix3d/splat.ply')
json.dump(dict(base=PLY, rounds=R, steps_per_round=SPR, pseudo_views=len(pseudo), seconds=time.time() - t0), open(f'{OUT}/difix_stats.json', 'w'))
print('done difix', round(time.time() - t0), 's')
