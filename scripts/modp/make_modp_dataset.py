"""Write one MODP (facebookresearch/Online-3DGS-Monocular) run's inputs from our prepared scene data.

MODP's "aria" loader reads: trajectory.json {"cameras": [{"image", "T_camera_world"}]}, rectified/<image>,
point_cloud/point_cloud_<n>.npy (world xyz the tracker saw in frame n). Image names must end in _<integer>.
  python make_modp_dataset.py <track dir, e.g. data/lobby/sfm> <sparse|dense> <out dir> [--post-steps 1000] [--no-cam-opt]
Writes <out>/train (frames MODP maps), <out>/test (held-out frames, never shown to MODP), <out>/config.yaml.
The config is MODP's own paper config (configs/aria/orb_tracking/aria_base.yaml) with only the dataset swapped in.
"""
import argparse, copy, json, os, re, shutil
import numpy as np, yaml

ap = argparse.ArgumentParser(); ap.add_argument('track'); ap.add_argument('points', choices=['sparse', 'dense']); ap.add_argument('out')
ap.add_argument('--post-steps', type=int, default=None, help="MODP post-refinement steps (paper config: 1000; 0 = online result only)")
ap.add_argument('--no-cam-opt', action='store_true', help='turn off MODP pose refinement (paper config: on)')
ap.add_argument('--max-frames', type=int, default=0, help='quick test: only the first N training frames')
a = ap.parse_args()
T = os.path.abspath(a.track); scene = os.path.dirname(T); out = os.path.abspath(a.out)
cams = json.load(open(f'{T}/cameras.json'))
num = lambda frame: int(re.findall(r'\d+', frame)[-1])


def write(folder, entries, points):
    shutil.rmtree(folder, ignore_errors=True); os.makedirs(f'{folder}/rectified'); os.makedirs(f'{folder}/point_cloud')
    traj = []
    for c in entries:
        name = f'frame_{num(c["frame"])}.jpg'; shutil.copy(f'{scene}/{c["image"]}', f'{folder}/rectified/{name}')
        traj.append({'image': name, 'T_camera_world': c['w2c']})
        np.save(f'{folder}/point_cloud/point_cloud_{num(c["frame"])}.npy', points(c['frame']).astype(np.float32))
    json.dump({'cameras': traj}, open(f'{folder}/trajectory.json', 'w'))


train = cams['train'][:a.max_frames] if a.max_frames else cams['train']
write(f'{out}/train', train, lambda f: np.load(f'{T}/points/{a.points}/{f}.npy'))
# held-out views: rendered after mapping, never shown to MODP while it maps (one dummy point keeps the loader happy)
write(f'{out}/test', cams['test'], lambda f: np.zeros((1, 3)))
K = cams['K']
calib = dict(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], width=cams['width'], height=cams['height'], near=0.01, far=1000)
ds = dict(name=os.path.basename(out), type='aria', data_source='aria', dataset_path=f'{out}/train', num_threads=2,
          begin_cutoff=0, end_cutoff=0, vignette=False, use_vignette_type='post-render', Calibration=calib)
config = {'inherit_from': 'configs/aria/orb_tracking/aria_base.yaml', 'Dataset': ds, 'Testset': copy.deepcopy(ds), 'Mapper': {}}
if a.no_cam_opt: config['Mapper']['CameraOptimizer'] = {'use_camera_opt': False}
if a.post_steps is not None: config['Mapper']['post_refinement'] = {'max_steps': a.post_steps}
yaml.safe_dump(config, open(f'{out}/config.yaml', 'w'), sort_keys=False)
print(f'wrote {out}: {len(train)} train frames ({a.points} points), {len(cams["test"])} held-out frames')
