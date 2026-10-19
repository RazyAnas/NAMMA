# Same data as run_scene.sh (our poses, our MODP init cloud, our train / test_ split, our anchored boot depth), trained
# with gsplat instead of OpenSplat, so open-source training recipes can be compared on the held-out stretch:
#   default  gsplat DefaultStrategy (3DGS densification: refine every 100, 500-15000, opacity reset every 3000)
#   mcmc     3DGS-MCMC (MCMCStrategy: relocate dead Gaussians, add up to cap, position noise; opacity + scale reg 0.01)
#   depth    mcmc + L1 in disparity against our boot depth (simple_trainer's depth loss, dense instead of sparse)
#   dn       depth + DN-Splatter-style normal loss (normals of rendered depth vs normals of boot depth) + flatten loss
#   2dgs     2D Gaussian surfels (rasterization_2dgs), normal consistency from 7k and distortion from 3k, depth loss
# Writes <out>/test_XXXXX.png for every test frame (score with score_whole.py) and <out>/splat.ply (3DGS ply).
# Usage: python gsplat_train.py <work> <init xyzrgb.ply> <out> <variant> [steps=30000]
import sys, os, json, math, time, numpy as np, cv2, torch, torch.nn.functional as F
from plyfile import PlyData, PlyElement
from gsplat import rasterization, rasterization_2dgs
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.utils import depth_to_normal
W, PLY, OUT, VAR = sys.argv[1:5]; STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 30000
os.makedirs(OUT, exist_ok=True); dev = 'cuda'; torch.manual_seed(0); np.random.seed(0); t0 = time.time()
st = json.load(open(f'{W}/clean/stats.json')); P = json.load(open(f'{W}/clean/poses_w2c.json'))
H, Wd = st['height'], st['width']; K = torch.tensor(st['K'], dtype=torch.float32, device=dev)
TWO_D = VAR == '2dgs'; USE_DEPTH = VAR in ('depth', 'dn', '2dgs'); MCMC = VAR in ('mcmc', 'depth', 'dn')

def view(f):
    w2c = torch.tensor(P[str(f)], dtype=torch.float32, device=dev); return w2c if w2c.shape[0] == 4 else torch.cat([w2c, torch.tensor([[0, 0, 0, 1.]], device=dev)])
train = st['train_frames']; test = st['test_frames']
imgs = {f: torch.from_numpy(cv2.cvtColor(cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg'), cv2.COLOR_BGR2RGB)).float().div(255) for f in train}
deps = {}
if USE_DEPTH:
    for f in train:
        d = cv2.imread(f'{W}/clean/boot_depth/{f:05d}.png', cv2.IMREAD_UNCHANGED)
        if d is not None: deps[f] = torch.from_numpy(d.astype(np.float32) / 1000)
VM = {f: view(f) for f in train + test}
cams = torch.stack([torch.linalg.inv(VM[f])[:3, 3] for f in train]); scene_scale = float((cams - cams.mean(0)).norm(dim=1).max()) * 1.1

v = PlyData.read(PLY)['vertex']; pts = torch.tensor(np.stack([v['x'], v['y'], v['z']], 1), dtype=torch.float32)
rgb = torch.tensor(np.stack([v['red'], v['green'], v['blue']], 1), dtype=torch.float32) / 255
from scipy.spatial import cKDTree
nn = cKDTree(pts.numpy()).query(pts.numpy(), k=4)[0][:, 1:]; dist = torch.tensor(np.sqrt((nn ** 2).mean(1)), dtype=torch.float32).clamp_min(1e-4)
N = len(pts); init_scale = 0.1 if MCMC else 1.0; init_opa = 0.5 if MCMC else 0.1
C0 = 0.28209479177387814; sh = torch.zeros(N, 16, 3); sh[:, 0] = (rgb - 0.5) / C0
params = {'means': (pts, 1.6e-4 * scene_scale), 'scales': (torch.log(dist * init_scale)[:, None].repeat(1, 3), 5e-3),
          'quats': (torch.rand(N, 4), 1e-3), 'opacities': (torch.logit(torch.full((N,), init_opa)), 5e-2),
          'sh0': (sh[:, :1].clone(), 2.5e-3), 'shN': (sh[:, 1:].clone(), 2.5e-3 / 20)}
splats = torch.nn.ParameterDict({k: torch.nn.Parameter(x) for k, (x, _) in params.items()}).to(dev)
opt = {k: torch.optim.Adam([{'params': splats[k], 'lr': lr, 'name': k}], eps=1e-15) for k, (_, lr) in params.items()}
sched = torch.optim.lr_scheduler.ExponentialLR(opt['means'], gamma=0.01 ** (1.0 / STEPS))
if MCMC: strat = MCMCStrategy(cap_max=int(os.environ.get('CAP_MAX', 1_000_000)), refine_stop_iter=int(STEPS * 0.83) if STEPS < 30000 else 25000)
else: strat = DefaultStrategy(refine_stop_iter=STEPS // 2, reset_every=3000, key_for_gradient='gradient_2dgs' if TWO_D else 'means2d')
strat.check_sanity(splats, opt); state = strat.initialize_state() if MCMC else strat.initialize_state(scene_scale=scene_scale)

def render(vm, sh_deg, mode='RGB+ED'):
    kw = dict(means=splats['means'], quats=splats['quats'], scales=torch.exp(splats['scales']), opacities=torch.sigmoid(splats['opacities']),
              colors=torch.cat([splats['sh0'], splats['shN']], 1), viewmats=vm[None], Ks=K[None], width=Wd, height=H, sh_degree=sh_deg, render_mode=mode)
    if TWO_D:
        c, a, n, ns, dist_, med, info = rasterization_2dgs(**kw, distloss=True)
        return c[0], a[0], info, dict(normals=n[0], surf=ns[0], dist=dist_[0])
    c, a, info = rasterization(**kw, packed=False, absgrad=False)
    return c[0], a[0], info, {}

def ssim(x, y, w=11):
    g = torch.exp(-((torch.arange(w, device=dev) - w // 2) ** 2) / (2 * 1.5 ** 2)); g = (g / g.sum()); k = (g[:, None] * g[None])[None, None].repeat(3, 1, 1, 1)
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]; f = lambda z: F.conv2d(z, k, padding=w // 2, groups=3)
    mx, my = f(x), f(y); sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4) * (sxx + syy + 9e-4))).mean()

def dnormal(d, vm):   # world-frame normals of a z-depth map [H, W]
    return depth_to_normal(d[None, :, :, None], torch.linalg.inv(vm)[None], K[None])[0]

order = []
for step in range(STEPS):
    if not order: order = list(np.random.permutation(train))
    f = int(order.pop()); vm = VM[f]; gt = imgs[f].to(dev); sh_deg = min(3, step // 1000)
    col, alpha, info, ex = render(vm, sh_deg)
    strat.step_pre_backward(splats, opt, state, step, info)
    rgb_, dep = col[..., :3], col[..., 3]
    loss = 0.8 * (rgb_ - gt).abs().mean() + 0.2 * (1 - ssim(rgb_, gt))
    if USE_DEPTH and f in deps:
        dg = deps[f].to(dev); m = dg > 0.05
        lam = 0.2 if step < STEPS // 2 else 0.05   # strong while densifying, weaker while refining
        loss = loss + lam * scene_scale * (1 / dep.clamp_min(1e-3)[m] - 1 / dg[m]).abs().mean()
        if VAR == 'dn' and step > 3000:
            ng, nr = dnormal(dg, vm), dnormal(dep, vm); mm = m & (ng.norm(dim=-1) > 0.5) & (nr.norm(dim=-1) > 0.5)
            loss = loss + 0.05 * (1 - (ng[mm].detach() * nr[mm]).sum(-1)).mean()
    if VAR == 'dn': loss = loss + 0.01 * torch.exp(splats['scales']).min(1).values.mean() / max(scene_scale, 1e-6) * 10
    if TWO_D:
        if step > 7000: loss = loss + 0.05 * (1 - (ex['normals'] * (ex['surf'] * alpha.detach())).sum(-1)).mean()
        if step > 3000: loss = loss + 1e-2 * ex['dist'].mean()
    if MCMC: loss = loss + 0.01 * torch.sigmoid(splats['opacities']).mean() + 0.01 * torch.exp(splats['scales']).mean()
    loss.backward()
    for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
    sched.step()
    if MCMC: strat.step_post_backward(splats, opt, state, step, info, lr=sched.get_last_lr()[0])
    else: strat.step_post_backward(splats, opt, state, step, info, packed=False)
    if step % 1000 == 0: print(f'step {step} loss {loss.item():.4f} n {len(splats["means"])} t {time.time() - t0:.0f}s', flush=True)

with torch.no_grad():
    for f in test:
        col = render(VM[f], 3, 'RGB')[0][..., :3].clamp(0, 1)
        cv2.imwrite(f'{OUT}/test_{f:05d}.png', cv2.cvtColor((col.cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    s = {k: splats[k].detach().cpu().numpy() for k in splats}; n = len(s['means'])
    if TWO_D: s['scales'][:, 2] = np.log(1e-6)   # flat surfel as a thin 3DGS Gaussian for the viewer
    q = s['quats'] / np.linalg.norm(s['quats'], axis=1, keepdims=True); rest = s['shN'].transpose(0, 2, 1).reshape(n, -1)
    cols = [('x', s['means'][:, 0]), ('y', s['means'][:, 1]), ('z', s['means'][:, 2]), ('nx', np.zeros(n)), ('ny', np.zeros(n)), ('nz', np.zeros(n))]
    cols += [(f'f_dc_{i}', s['sh0'][:, 0, i]) for i in range(3)] + [(f'f_rest_{i}', rest[:, i]) for i in range(rest.shape[1])]
    cols += [('opacity', s['opacities'])] + [(f'scale_{i}', s['scales'][:, i]) for i in range(3)] + [(f'rot_{i}', q[:, i]) for i in range(4)]
    a = np.empty(n, dtype=[(k, 'f4') for k, _ in cols])
    for k, x in cols: a[k] = x
    PlyData([PlyElement.describe(a, 'vertex')]).write(f'{OUT}/splat.ply')
json.dump(dict(variant=VAR, steps=STEPS, gaussians=n, seconds=time.time() - t0, scene_scale=scene_scale), open(f'{OUT}/train_stats.json', 'w'))
print('done', VAR, n, 'gaussians', round(time.time() - t0), 's')
