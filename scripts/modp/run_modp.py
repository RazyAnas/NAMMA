"""Run MODP's online mapper unchanged on one prepared run, then render the held-out views and save the model.

Run from inside the MODP repository with its environment:
  cd Online-3DGS-Monocular && python /path/run_modp.py <run dir made by make_modp_dataset.py> <results dir>
Writes <results>/renders/<frame>.png (held-out views), point_cloud.ply (MODP's 3DGS model), native_eval.json (MODP's own
score on the frames it mapped, which is what the paper reports), run_info.json (timings, FPS, memory).
Held-out poses: MODP refines keyframe poses while mapping, so a held-out view gets the same correction MODP applies to its
own non-keyframes (interpolated between the neighbouring keyframes' original -> refined poses, MODP's eval_utils formula).
"""
import json, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

if __name__ == '__main__':
    run_dir, res = [os.path.abspath(x) for x in sys.argv[1:3]]
    mp.set_start_method('spawn')
    from utils_new.tool_utils import load_config, mkdir_p, interpolate_pose
    from utils_new.scene_mapper import SceneMapper
    from utils_new.eval_utils import eval_gaussians
    from utils_new.dataset import load_dataset
    config = load_config(f'{run_dir}/config.yaml')
    config['Results']['save_dir'] = f'{res}/modp_logs'; mkdir_p(config['Results']['save_dir'])
    config['Mapper'].setdefault('scene_exposure_gain', 20.0)   # slam_new.py's default
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
    json.dump(native, open(f'{res}/native_eval.json', 'w'), indent=1, default=str)
    nframes = int(meta['num_processed_frames'])
    info = dict(method='MODP (facebookresearch/Online-3DGS-Monocular @7d75169)', run=os.path.basename(run_dir), map_seconds=round(map_s, 1),
                online_seconds=round(map_s - post.get('s', 0), 1), post_refinement_seconds=round(post.get('s', 0), 1),
                online_fps=round(nframes / max(map_s - post.get('s', 0), 1e-6), 2), render_fps=round(fps, 1),
                final_gaussians=int(meta['num_gaussians']), keyframes=int(meta['num_keyframes']), frames=nframes,
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    json.dump(info, open(f'{res}/run_info.json', 'w'), indent=1)
    print(json.dumps(info))
