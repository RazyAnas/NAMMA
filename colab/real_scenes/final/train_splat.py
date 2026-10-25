"""Short Gaussian-splat fit (gsplat) from the merged feed-forward cloud + solved cameras, supervised by the REAL frames.
Input: da3.npz (K, c2w for 504x280 views, idx = frame ids), cloud.npz (pts, col), frames dir. Output: splat.ply (standard 3DGS)."""
import glob, sys, time, numpy as np
from PIL import Image

def load_views(da3, frames_dir, scale=2):
    import torch
    import os, ibr
    z = np.load(da3); d = ibr.keep_filter({k: z[k] for k in z.files}, os.environ.get('KEEP_FRAMES'))   # env KEEP_FRAMES=good_frames.json: train only on checked cameras
    files = sorted(glob.glob(f'{frames_dir}/*.jpg')); imgs, Ks, V = [], [], []
    for j, g in enumerate(d['idx']):
        im = Image.open(files[g]).convert('RGB'); W, H = d['rgb'].shape[2] * scale, d['rgb'].shape[1] * scale
        imgs.append(torch.from_numpy(np.asarray(im.resize((W, H), Image.LANCZOS))).float() / 255)
        K = d['K'][j].copy(); K[:2] *= scale; Ks.append(K); V.append(np.linalg.inv(d['c2w'][j]))
    return (torch.stack(imgs), torch.tensor(np.array(Ks), dtype=torch.float32), torch.tensor(np.array(V), dtype=torch.float32),
            torch.tensor(np.asarray(d['depth']), dtype=torch.float32), np.asarray(d['idx']))

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

def train(da3, cloud, frames_dir, out_ply, steps=15000, max_seconds=600, dev='cuda', pose_opt=False, depth_w=0.0, max_init=500000, mcmc_cap=0, sh=0, depth_floor=0.0, aniso=0.0, aniso_max=10.0):
    """pose_opt: learn a small per-view camera correction (the cameras come from chained feed-forward chunks, so they drift).
    depth_w: weight of an L1 loss against the feed-forward depth, fading to 0 at 60% of training (anchors geometry early)."""
    import torch, torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    from gsplat.strategy.ops import reset_opa
    imgs, Ks, V, DEP, FID = load_views(da3, frames_dir); imgs, Ks, V, DEP = imgs.to(dev), Ks.to(dev), V.to(dev), DEP.to(dev); N, H, W, _ = imgs.shape
    c = np.load(cloud); pts, col = c['pts'], c['col'] / 255.0
    sel = np.random.default_rng(0).choice(len(pts), min(len(pts), max_init), replace=False); pts, col = pts[sel], col[sel]
    from scipy.spatial import cKDTree
    dist = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(1); dist = np.clip(dist, 1e-3, 0.2)
    cc = np.load(da3)['c2w'][:, :3, 3]; scene_scale = 1.1 * float(np.linalg.norm(cc - cc.mean(0), axis=1).max())   # camera extent (gsplat convention); the point extent was ~3x larger and made position steps too big
    params = torch.nn.ParameterDict({
        'means': torch.tensor(pts, dtype=torch.float32), 'scales': torch.log(torch.tensor(dist, dtype=torch.float32))[:, None].repeat(1, 3),
        'quats': torch.nn.functional.normalize(torch.randn(len(pts), 4), dim=-1), 'opacities': torch.logit(torch.full((len(pts),), 0.1 if mcmc_cap else 0.5)),
        'sh0': ((torch.tensor(col, dtype=torch.float32) - 0.5) / 0.28209479)[:, None, :], 'shN': torch.zeros(len(pts), (sh + 1) ** 2 - 1, 3)}).to(dev)
    lrs = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    if mcmc_cap:                                                 # MCMC: an explicit splat budget (relocates dead splats, adds up to cap_max)
        strat = MCMCStrategy(cap_max=mcmc_cap, refine_stop_iter=int(steps * 0.8), verbose=False); strat.check_sanity(params, opts); st = strat.initialize_state()
    else: strat = DefaultStrategy(verbose=False, refine_stop_iter=int(steps * 0.7)); strat.check_sanity(params, opts); st = strat.initialize_state(scene_scale=scene_scale)
    cam = torch.zeros(N, 6, device=dev, requires_grad=True); opt_cam = torch.optim.Adam([cam], lr=3e-4) if pose_opt else None
    def view(i, refined=True): return (delta_pose(cam[i]) @ V[i])[None] if (pose_opt and refined) else V[i:i + 1]
    test = [i for i in range(N) if FID[i] % 8 == 3]; train_ids = [i for i in range(N) if FID[i] % 8 != 3]   # frames 3, 11, 19, ... held out (neighbours are close: mild test)
    t0 = time.time(); gen = np.random.default_rng(1)
    diag = train_ids[::max(1, len(train_ids) // 2)][:2] + test[:2]                       # same views before and after training
    def snap(tag):
        with torch.no_grad():
            row = [torch.cat([imgs[i], rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']),
                   torch.sigmoid(params['opacities']), torch.cat([params['sh0'], params['shN']], 1), V[i:i + 1], Ks[i:i + 1], W, H,
                   sh_degree=0)[0][0].clamp(0, 1)], 1) for i in diag]
            Image.fromarray((torch.cat(row, 0).cpu().numpy() * 255).astype(np.uint8)).resize((2 * W // 2, len(diag) * H // 2)).save(out_ply.replace('.ply', f'_{tag}.jpg'))
    snap('step0')                                                                         # init cloud seen from the cameras: geometry/camera check
    for step in range(steps):
        i = train_ids[int(gen.integers(len(train_ids)))]
        out, a, info = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                     torch.cat([params['sh0'], params['shN']], 1), view(i), Ks[i:i + 1], W, H, sh_degree=min(sh, step // 1000), packed=False, render_mode='RGB+ED')
        r, ed = out[0, ..., :3], out[0, ..., 3]
        if not mcmc_cap: strat.step_pre_backward(params, opts, st, step, info)
        loss = 0.8 * (r - imgs[i]).abs().mean() + 0.2 * (1 - ssim(r, imgs[i]))
        if depth_w > 0 and (step < 0.6 * steps or depth_floor > 0):
            gt = F.interpolate(DEP[i][None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]; m = (gt > 0) & (a[0, ..., 0] > 0.5)
            if m.any(): loss = loss + depth_w * max(depth_floor, 1 - step / (0.6 * steps)) * ((ed - gt).abs() / (gt + 0.2))[m].mean()
        if aniso > 0:                                            # needles: penalise splats longer than aniso_max x their thinnest axis
            s3 = torch.exp(params['scales']); loss = loss + aniso * torch.relu(s3.max(1).values / s3.min(1).values.clamp_min(1e-8) - aniso_max).mean()
        if mcmc_cap: loss = loss + 0.01 * torch.sigmoid(params['opacities']).mean() + 0.01 * torch.exp(params['scales']).mean()   # keeps splats from all staying big/opaque
        loss.backward()
        lr_m = lrs['means'] * 0.01 ** (step / steps)             # exponential decay of the position lr (gsplat default schedule)
        for g in opts['means'].param_groups: g['lr'] = lr_m
        for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)   # optimizer first, then densify/prune (gsplat's trainer order)
        if mcmc_cap: strat.step_post_backward(params, opts, st, step, info, lr=lr_m)
        else:
            strat.step_post_backward(params, opts, st, step, info, packed=False)
            # gsplat 1.5.3 never resets opacities ('step % reset_every == 0 & step > 0' is always False: & binds first), so floaters
            # in front of cameras are never cleared. Do the standard 3DGS reset ourselves: every 3000 steps while densifying.
            if 0 < step < strat.refine_stop_iter and step % 3000 == 0: reset_opa(params, opts, st, value=0.01)
        if opt_cam is not None: opt_cam.step(); opt_cam.zero_grad(set_to_none=True)
        if step == 0: t0 = time.time()                               # first step includes the one-off CUDA build
        if step % 500 == 0: print(f'step {step} loss {loss.item():.4f} gaussians {len(params["means"])} {time.time()-t0:.0f}s', flush=True)
        if time.time() - t0 > max_seconds: print('time cap at step', step); break
    for p in params.values(): p.requires_grad_(False)
    snap('final')
    with torch.no_grad():
        sz = torch.exp(params['scales']).max(1).values; op = torch.sigmoid(params['opacities'])
        q = torch.quantile(sz[torch.randperm(len(sz), device=dev)[:200000]], torch.tensor([.5, .9, .99], device=dev)).tolist()
        derr = []
        for i in test:
            gt = F.interpolate(DEP[i][None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]
            if (gt > 0).sum() < 100: continue
            o_, a_, _ = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), op,
                                      torch.cat([params['sh0'], params['shN']], 1), V[i:i + 1], Ks[i:i + 1], W, H, sh_degree=0, render_mode='ED')
            mk = gt > 0; derr.append(((o_[0, ..., 0] - gt).abs() / gt)[mk].median().item())
        s3 = torch.exp(params['scales']); ar = (s3.max(1).values / s3.min(1).values.clamp_min(1e-8))
        print(f'needles: splats stretched > 10x: {(ar > 10).float().mean().item()*100:.1f}%, > 30x: {(ar > 30).float().mean().item()*100:.1f}%', flush=True)
        print(f'splat size (largest axis) median {q[0]*100:.1f} cm, 90% {q[1]*100:.1f} cm, 99% {q[2]*100:.1f} cm | splats > 20 cm: '
              f'{(sz > 0.2).float().mean().item()*100:.1f}% | mean opacity {op.mean().item():.2f} | held-out depth error vs prior '
              f'(median rel.) {np.median(derr)*100 if derr else float("nan"):.1f}% over {len(derr)} views', flush=True)
    def render(i, refined=True):
        r, _, _ = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                torch.cat([params['sh0'], params['shN']], 1), view(i, refined), Ks[i:i + 1], W, H, sh_degree=sh)
        return r[0].clamp(0, 1)
    if pose_opt:   # standard protocol when training refines cameras: align each held-out camera to the frozen model first
        opt_t = torch.optim.Adam([cam], lr=5e-4)
        for it in range(150):                                    # one view at a time: summing 25 graphs would not fit next to 2M splats
            for i in test: (render(i) - imgs[i]).abs().mean().backward()
            opt_t.step(); opt_t.zero_grad(set_to_none=True)
    with torch.no_grad():
        psnr = lambda i, rf=True: (-10 * torch.log10(F.mse_loss(render(i, rf), imgs[i]))).item()
        if pose_opt: print(f'camera correction (train views): mean |t| {cam[train_ids, 3:].norm(dim=1).mean().item()*100:.1f} cm, mean |rot| {cam[train_ids, :3].norm(dim=1).mean().item()*57.3:.2f} deg')
        msg = f'[{out_ply.split("/")[-1]}] views {N} | train-view PSNR {np.mean([psnr(i) for i in train_ids]):.2f} dB | HELD-OUT PSNR {np.mean([psnr(i, False) for i in test]):.2f} dB raw cameras'
        if pose_opt: msg += f', {np.mean([psnr(i) for i in test]):.2f} dB after aligning held-out cameras'
        msg += f' | SSIM {np.mean([ssim(render(i, pose_opt), imgs[i]).item() for i in test]):.3f} | {len(params["means"])} splats | held-out = frames 3,11,19,... ({len(test)})'
        print(msg, flush=True)
        for i in test[:3]: Image.fromarray((torch.cat([imgs[i], render(i, pose_opt)], 1).cpu().numpy() * 255).astype(np.uint8)).save(out_ply.replace('.ply', f'_heldout{FID[i]}.jpg'))
    write_ply(out_ply, {k: v.detach().cpu().numpy() for k, v in params.items()})
    del params, opts, imgs, DEP; import gc; gc.collect(); torch.cuda.empty_cache()

def write_ply(path, p, y_up=True):
    p = dict(p)
    if y_up:                                                     # camera frame is y-down: rotate 180 deg about x so viewers show it upright
        p['means'] = p['means'] * np.array([1, -1, -1]); q = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True)
        p['quats'] = np.stack([-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]], 1)     # (0,1,0,0) (x) q, quaternions as (w,x,y,z)
        if p['shN'].shape[1]:                                       # (x,y,z)->(x,-y,-z): each real SH basis is even or odd in (y,z)
            sign = np.array([-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1], np.float32)[:p['shN'].shape[1]]
            p['shN'] = p['shN'] * sign[None, :, None]
    n = len(p['means']); sh0 = p['sh0'][:, 0, :]; rest = p['shN'].transpose(0, 2, 1).reshape(n, -1)   # 3DGS order: all R, then G, then B
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{k}' for k in range(rest.shape[1])] + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    q = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True)
    data = np.concatenate([p['means'], np.zeros((n, 3)), sh0, rest, p['opacities'][:, None], p['scales'], q], 1).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % k for k in fields) + 'end_header\n').encode()); f.write(data.tobytes())
