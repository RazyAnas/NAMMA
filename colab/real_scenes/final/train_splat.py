"""Short Gaussian-splat fit (gsplat) from the merged feed-forward cloud + solved cameras, supervised by the REAL frames.
Input: da3.npz (K, c2w for 504x280 views, idx = frame ids), cloud.npz (pts, col), frames dir. Output: splat.ply (standard 3DGS)."""
import glob, sys, time, numpy as np
from PIL import Image

def load_views(da3, frames_dir, scale=2):
    import torch
    d = np.load(da3); files = sorted(glob.glob(f'{frames_dir}/*.jpg')); imgs, Ks, V = [], [], []
    for j, g in enumerate(d['idx']):
        im = Image.open(files[g]).convert('RGB'); W, H = d['rgb'].shape[2] * scale, d['rgb'].shape[1] * scale
        imgs.append(torch.from_numpy(np.asarray(im.resize((W, H), Image.LANCZOS))).float() / 255)
        K = d['K'][j].copy(); K[:2] *= scale; Ks.append(K); V.append(np.linalg.inv(d['c2w'][j]))
    return torch.stack(imgs), torch.tensor(np.array(Ks), dtype=torch.float32), torch.tensor(np.array(V), dtype=torch.float32)

def train(da3, cloud, frames_dir, out_ply, steps=3000, max_seconds=720, dev='cuda'):
    import torch, torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    imgs, Ks, V = load_views(da3, frames_dir); imgs, Ks, V = imgs.to(dev), Ks.to(dev), V.to(dev); N, H, W, _ = imgs.shape
    c = np.load(cloud); pts, col = c['pts'], c['col'] / 255.0
    sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 300000), replace=False); pts, col = pts[sel], col[sel]
    from scipy.spatial import cKDTree
    dist = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(1); dist = np.clip(dist, 1e-3, 0.2)
    scene_scale = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
    params = torch.nn.ParameterDict({
        'means': torch.tensor(pts, dtype=torch.float32), 'scales': torch.log(torch.tensor(dist, dtype=torch.float32))[:, None].repeat(1, 3),
        'quats': torch.nn.functional.normalize(torch.randn(len(pts), 4), dim=-1), 'opacities': torch.logit(torch.full((len(pts),), 0.5)),
        'sh0': ((torch.tensor(col, dtype=torch.float32) - 0.5) / 0.28209479)[:, None, :], 'shN': torch.zeros(len(pts), 0, 3)}).to(dev)
    lrs = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 1.25e-4}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strat = DefaultStrategy(verbose=False, refine_stop_iter=int(steps * 0.7)); strat.check_sanity(params, opts); st = strat.initialize_state(scene_scale=scene_scale)
    t0 = time.time(); gen = np.random.default_rng(1)
    for step in range(steps):
        i = int(gen.integers(N))
        r, a, info = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                   torch.cat([params['sh0'], params['shN']], 1), V[i:i + 1], Ks[i:i + 1], W, H, sh_degree=0, packed=False)
        strat.step_pre_backward(params, opts, st, step, info)
        loss = (r[0] - imgs[i]).abs().mean() + 0.1 * F.mse_loss(r[0], imgs[i])
        loss.backward(); strat.step_post_backward(params, opts, st, step, info, packed=False)
        for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
        if step % 250 == 0: print(f'step {step} loss {loss.item():.4f} gaussians {len(params["means"])} {time.time()-t0:.0f}s', flush=True)
        if time.time() - t0 > max_seconds: print('time cap at step', step); break
    with torch.no_grad():                                        # held-out-free sanity: PSNR on the training views (NOT a quality claim)
        ps = []
        for i in range(N):
            r, _, _ = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                    torch.cat([params['sh0'], params['shN']], 1), V[i:i + 1], Ks[i:i + 1], W, H, sh_degree=0)
            ps.append((-10 * torch.log10(F.mse_loss(r[0].clamp(0, 1), imgs[i]))).item())
        print(f'train-view PSNR mean {np.mean(ps):.2f} dB over {N} views (training views, not held-out)')
        for i in (0, N // 2, N - 1):
            r, _, _ = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                    torch.cat([params['sh0'], params['shN']], 1), V[i:i + 1], Ks[i:i + 1], W, H, sh_degree=0)
            Image.fromarray((torch.cat([imgs[i], r[0].clamp(0, 1)], 1).cpu().numpy() * 255).astype(np.uint8)).save(out_ply.replace('.ply', f'_cmp{i}.jpg'))
    write_ply(out_ply, {k: v.detach().cpu().numpy() for k, v in params.items()})

def write_ply(path, p):
    n = len(p['means']); sh0 = p['sh0'][:, 0, :]; fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    q = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True)
    data = np.concatenate([p['means'], np.zeros((n, 3)), sh0, p['opacities'][:, None], p['scales'], q], 1).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % k for k in fields) + 'end_header\n').encode()); f.write(data.tobytes())
