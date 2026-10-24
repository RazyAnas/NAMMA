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
    return (torch.stack(imgs), torch.tensor(np.array(Ks), dtype=torch.float32), torch.tensor(np.array(V), dtype=torch.float32),
            torch.tensor(np.asarray(d['depth']), dtype=torch.float32))

def ssim(a, b):
    import torch.nn.functional as F
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]; ma, mb = F.avg_pool2d(a, 7, 1, 3), F.avg_pool2d(b, 7, 1, 3)
    saa, sbb, sab = F.avg_pool2d(a * a, 7, 1, 3) - ma ** 2, F.avg_pool2d(b * b, 7, 1, 3) - mb ** 2, F.avg_pool2d(a * b, 7, 1, 3) - ma * mb
    return (((2 * ma * mb + 1e-4) * (2 * sab + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (saa + sbb + 9e-4))).mean()

def delta_pose(c):
    """6-vector (axis-angle, translation) -> 4x4 world-space correction. sqrt(sum+eps) keeps the gradient finite at zero."""
    import torch
    w, t = c[:3], c[3:]; th = torch.sqrt((w * w).sum() + 1e-12); k = w / th; z = torch.zeros_like(k[0])
    Kx = torch.stack([z, -k[2], k[1], k[2], z, -k[0], -k[1], k[0], z]).reshape(3, 3)
    R = torch.eye(3, device=c.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)
    top = torch.cat([R, t[:, None]], 1); return torch.cat([top, torch.tensor([[0., 0., 0., 1.]], device=c.device)], 0)

def train(da3, cloud, frames_dir, out_ply, steps=15000, max_seconds=600, dev='cuda', pose_opt=False, depth_w=0.0, max_init=500000):
    """pose_opt: learn a small per-view camera correction (the cameras come from chained feed-forward chunks, so they drift).
    depth_w: weight of an L1 loss against the feed-forward depth, fading to 0 at 60% of training (anchors geometry early)."""
    import torch, torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    imgs, Ks, V, DEP = load_views(da3, frames_dir); imgs, Ks, V, DEP = imgs.to(dev), Ks.to(dev), V.to(dev), DEP.to(dev); N, H, W, _ = imgs.shape
    c = np.load(cloud); pts, col = c['pts'], c['col'] / 255.0
    sel = np.random.default_rng(0).choice(len(pts), min(len(pts), max_init), replace=False); pts, col = pts[sel], col[sel]
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
    cam = torch.zeros(N, 6, device=dev, requires_grad=True); opt_cam = torch.optim.Adam([cam], lr=3e-4) if pose_opt else None
    def view(i, refined=True): return (delta_pose(cam[i]) @ V[i])[None] if (pose_opt and refined) else V[i:i + 1]
    test = [i for i in range(3, N, 8)]; train_ids = [i for i in range(N) if i not in test]   # every 8th frame held out (neighbours are close: mild test)
    t0 = time.time(); gen = np.random.default_rng(1)
    for step in range(steps):
        i = train_ids[int(gen.integers(len(train_ids)))]
        out, a, info = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                     torch.cat([params['sh0'], params['shN']], 1), view(i), Ks[i:i + 1], W, H, sh_degree=0, packed=False, render_mode='RGB+ED')
        r, ed = out[0, ..., :3], out[0, ..., 3]
        strat.step_pre_backward(params, opts, st, step, info)
        loss = 0.8 * (r - imgs[i]).abs().mean() + 0.2 * (1 - ssim(r, imgs[i]))
        if depth_w > 0 and step < 0.6 * steps:
            gt = F.interpolate(DEP[i][None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]; m = (gt > 0) & (a[0, ..., 0] > 0.5)
            if m.any(): loss = loss + depth_w * (1 - step / (0.6 * steps)) * ((ed - gt).abs() / (gt + 0.2))[m].mean()
        loss.backward(); strat.step_post_backward(params, opts, st, step, info, packed=False)
        for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
        if opt_cam is not None: opt_cam.step(); opt_cam.zero_grad(set_to_none=True)
        if step % 500 == 0: print(f'step {step} loss {loss.item():.4f} gaussians {len(params["means"])} {time.time()-t0:.0f}s', flush=True)
        if time.time() - t0 > max_seconds: print('time cap at step', step); break
    with torch.no_grad():
        def render(i, refined=True):
            r, _, _ = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                    torch.cat([params['sh0'], params['shN']], 1), view(i, refined), Ks[i:i + 1], W, H, sh_degree=0)
            return r[0].clamp(0, 1)
        psnr = lambda i, rf=True: (-10 * torch.log10(F.mse_loss(render(i, rf), imgs[i]))).item()
        if pose_opt: print(f'camera correction: mean |t| {cam[:, 3:].norm(dim=1).mean().item()*100:.1f} cm, mean |rot| {cam[:, :3].norm(dim=1).mean().item()*57.3:.2f} deg')
        print(f'[{out_ply.split("/")[-1]}] train-view PSNR {np.mean([psnr(i) for i in train_ids]):.2f} dB | HELD-OUT PSNR {np.mean([psnr(i, False) for i in test]):.2f} dB, '
              f'SSIM {np.mean([ssim(render(i, False), imgs[i]).item() for i in test]):.3f} over {len(test)} held-out views (original cameras, neighbouring frames: optimistic)', flush=True)
        for i in test[:3]: Image.fromarray((torch.cat([imgs[i], render(i, False)], 1).cpu().numpy() * 255).astype(np.uint8)).save(out_ply.replace('.ply', f'_heldout{i}.jpg'))
    write_ply(out_ply, {k: v.detach().cpu().numpy() for k, v in params.items()})
    del params, opts, imgs, DEP; import gc; gc.collect(); torch.cuda.empty_cache()

def write_ply(path, p, y_up=True):
    p = dict(p)
    if y_up:                                                     # camera frame is y-down: rotate 180 deg about x so viewers show it upright
        p['means'] = p['means'] * np.array([1, -1, -1]); q = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True)
        p['quats'] = np.stack([-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]], 1)     # (0,1,0,0) (x) q, quaternions as (w,x,y,z)
    n = len(p['means']); sh0 = p['sh0'][:, 0, :]; fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    q = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True)
    data = np.concatenate([p['means'], np.zeros((n, 3)), sh0, p['opacities'][:, None], p['scales'], q], 1).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % k for k in fields) + 'end_header\n').encode()); f.write(data.tobytes())
