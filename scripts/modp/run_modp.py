"""Run MODP's online mapper unchanged on one prepared run, then render the held-out views and save the model.

Run from inside the MODP repository with its environment:
  cd Online-3DGS-Monocular && python /path/run_modp.py <run dir made by make_modp_dataset.py> <results dir> [--set Loss.lambda_depth=0.05 ...]
--set overrides a key of MODP's config (dotted path, YAML value). Loss.g_loss_type=log_ratio is ours: it penalises
log(longest/shortest axis) above log(min_scale_ratio), the same idea as MODP's own "min_constraint" but independent of the
scene's metric size, so it actually stops needle-shaped splats. Every run also renders orbit views (orbit/*.png) around a
held-out camera: viewpoints the video never saw, to judge how the model holds up from the side (there is no ground truth).
Writes <results>/renders/<frame>.png (held-out views), point_cloud.ply (MODP's 3DGS model), native_eval.json (MODP's own
score on the frames it mapped, which is what the paper reports), run_info.json (timings, FPS, memory).
Held-out poses: MODP refines keyframe poses while mapping, so a held-out view gets the same correction MODP applies to its
own non-keyframes (interpolated between the neighbouring keyframes' original -> refined poses, MODP's eval_utils formula).
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run_dir'); ap.add_argument('res'); ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args(); run_dir, res = os.path.abspath(a.run_dir), os.path.abspath(a.res)
    mp.set_start_method('spawn')
    import yaml
    from utils_new import loss_utils
    from utils_new.tool_utils import load_config, mkdir_p, interpolate_pose
    from utils_new.scene_mapper import SceneMapper
    from utils_new.eval_utils import eval_gaussians
    from utils_new.dataset import load_dataset
    config = load_config(f'{run_dir}/config.yaml')
    config['Results']['save_dir'] = f'{res}/modp_logs'; mkdir_p(config['Results']['save_dir'])
    config['Mapper'].setdefault('scene_exposure_gain', 20.0)   # slam_new.py's default
    for kv in a.set:
        path, value = kv.split('=', 1); node = config; keys = path.split('.')
        for k in keys[:-1]: node = node[k]
        if keys[-1] not in node: sys.exit(f'!! --set {path}: MODP config has no such key')
        node[keys[-1]] = yaml.safe_load(value); print('config override:', path, '=', node[keys[-1]], flush=True)
    _gloss = loss_utils.GaussianLoss.forward
    def gaussian_loss(self, gaussians, level=-1):
        if self.g_loss_type != 'log_ratio' or self.gloss_weight == 0.0: return _gloss(self, gaussians, level)
        ls = torch.log(gaussians.get_scaling(level).clamp_min(1e-8))
        excess = torch.relu(ls.max(dim=1)[0] - ls.min(dim=1)[0] - math.log(self.min_scale_ratio))
        if not getattr(self, '_announced', False): print('log_ratio anisotropy loss active', flush=True); self._announced = True
        return self.gloss_weight * (excess ** 2).mean()
    loss_utils.GaussianLoss.forward = gaussian_loss
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mapper = SceneMapper(config)
    post = {}
    _pr = mapper.post_refinement
    def timed_post_refinement():   # time MODP's offline post-refinement separately from the online part
        torch.cuda.synchronize(); t = time.time(); _pr(); torch.cuda.synchronize(); post['s'] = time.time() - t
    mapper.post_refinement = timed_post_refinement
    meta, infos = mapper.run()
    torch.cuda.synchronize(); map_s = time.time() - t0
    native = eval_gaussians(mapper.gaussians, infos, config)
    mapper.gaussians.save_as_ply(f'{res}/point_cloud.ply')

    # held-out views
    train = [c['image'] for c in json.load(open(f'{run_dir}/train/trajectory.json'))['cameras']]
    num = lambda name: int(name.split('.')[0].split('_')[-1])
    tnum = np.array([num(n) for n in train]); pairs = infos['poses_pair']   # train index -> (original, refined, is keyframe)
    kfs = sorted(i for i, p in pairs.items() if p[2])
    tcfg = dict(config['Testset']); tcfg['dataset_path'] = f'{run_dir}/test'; tcfg['scene_exposure_gain'] = config['Mapper']['scene_exposure_gain']
    test = load_dataset(tcfg, ret_original_dataset=True)
    mapper.gaussians.set_vignette_img(None)
    os.makedirs(f'{res}/renders', exist_ok=True); t1 = time.time()
    with torch.no_grad():
        for i in range(len(test)):
            cam = test[i]; f = num(cam.name); orig = cam.get_pose().cpu().numpy().astype(np.float64)
            j = int(np.searchsorted(tnum, f))      # first training frame after this one
            before = [k for k in kfs if k < j]; after = [k for k in kfs if k >= j]
            if before and after:
                k0, k1 = before[-1], after[0]; al = (f - tnum[k0]) / max(tnum[k1] - tnum[k0], 1)
                io = interpolate_pose(pairs[k0][0], pairs[k1][0], al); ia = interpolate_pose(pairs[k0][1], pairs[k1][1], al)
            else:
                k = (before or after)[-1 if before else 0]; io, ia = pairs[k][0], pairs[k][1]
            cam.update_pose_numpy((orig @ np.linalg.inv(io) @ ia).astype(np.float32)); cam.to_device('cuda:0')
            img = mapper.gaussians.render(cam)['render'].clamp(0, 1)
            if img.shape[0] == 3: img = img.permute(1, 2, 0)
            Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(f'{res}/renders/{f:06d}.png')
    torch.cuda.synchronize(); fps = len(test) / (time.time() - t1)

    # orbit views around the middle held-out camera: swing left/right around the point it looks at, and look down from above.
    # Extras only: if they fail, say so and still save the run.
    shape = {}
    try:
        os.makedirs(f'{res}/orbit', exist_ok=True)
        with torch.no_grad():
            cam = test[len(test) // 2]; W = cam.get_pose().cpu().numpy().astype(np.float64); c2w = np.linalg.inv(W)
            C, fwd, down = c2w[:3, 3], c2w[:3, 2], c2w[:3, 1]
            d = (mapper.gaussians.get_xyz().detach().cpu().numpy() - C) @ fwd   # pivot = median depth of the scene in front
            d = float(np.median(d[(d > 0.2) & (d < 8)])) if np.any((d > 0.2) & (d < 8)) else 2.0
            pivot = C + fwd * d; up = -down
            def look_at(eye):
                z = pivot - eye; z /= np.linalg.norm(z); x = np.cross(-up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
                M = np.eye(4); M[:3, :3] = np.stack([x, y, z]); M[:3, 3] = -M[:3, :3] @ eye; return M
            def rot(v, ang):   # rotate v about the up axis through the pivot (Rodrigues)
                k = up / np.linalg.norm(up); v = v - pivot
                return pivot + v * math.cos(ang) + np.cross(k, v) * math.sin(ang) + k * (k @ v) * (1 - math.cos(ang))
            views = {f'yaw{deg:+03d}': rot(C, math.radians(deg)) for deg in (-60, -30, 0, 30, 60)}
            views['above'] = C + up * 0.6 * d + fwd * 0.2 * d
            for tag, eye in views.items():
                cam.update_pose_numpy(look_at(eye).astype(np.float32)); cam.to_device('cuda:0')
                img = mapper.gaussians.render(cam)['render'].clamp(0, 1)
                if img.shape[0] == 3: img = img.permute(1, 2, 0)
                Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(f'{res}/orbit/{tag}.png')
        sc = mapper.gaussians.get_scaling().detach()   # how needle-like the final splats are
        an = (sc.max(1)[0] / sc.min(1)[0].clamp_min(1e-9)).cpu().numpy()
        shape = dict(aniso_p50=round(float(np.percentile(an, 50)), 1), aniso_p99=round(float(np.percentile(an, 99)), 1),
                     frac_aniso_over_10=round(float((an > 10).mean()), 4))
    except Exception as e:
        print('!! orbit views / splat-shape stats failed (the run itself is fine):', repr(e), flush=True)
    json.dump(native, open(f'{res}/native_eval.json', 'w'), indent=1, default=str)
    nframes = int(meta['num_processed_frames'])
    info = dict(method='MODP (facebookresearch/Online-3DGS-Monocular @7d75169)', run=os.path.basename(run_dir), map_seconds=round(map_s, 1),
                online_seconds=round(map_s - post.get('s', 0), 1), post_refinement_seconds=round(post.get('s', 0), 1),
                online_fps=round(nframes / max(map_s - post.get('s', 0), 1e-6), 2), render_fps=round(fps, 1),
                final_gaussians=int(meta['num_gaussians']), keyframes=int(meta['num_keyframes']), frames=nframes,
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                config_overrides=a.set, **shape)
    json.dump(info, open(f'{res}/run_info.json', 'w'), indent=1)
    print(json.dumps(info))
