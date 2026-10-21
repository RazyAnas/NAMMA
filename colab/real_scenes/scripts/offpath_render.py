# Off-path views from a trained Nerfstudio model: for 4 held-out cameras, render the camera itself and the camera moved
# sideways (right), backwards and up by a fraction of the camera path's size, and save one strip per camera:
#   real photo | render at the photo's camera | moved right | moved back | moved up
# The photo shows what the room looks like; the moved views show what someone walking off the filmed path would see.
# Usage: python offpath_render.py <config.yml> <out dir> [shift=0.06 of the path extent]
import sys, os, numpy as np, torch, imageio.v2 as imageio
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
cfg, out = Path(sys.argv[1]), Path(sys.argv[2]); frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.06
out.mkdir(parents=True, exist_ok=True)
_, pipeline, _, _ = eval_setup(cfg, test_mode='test')
dm = pipeline.datamanager; ds = dm.eval_dataset; cams = ds.cameras
train_c2w = dm.train_dataset.cameras.camera_to_worlds[:, :3, 3]
extent = float((train_c2w.max(0).values - train_c2w.min(0).values).norm()); step = frac * extent
print('path extent (scene units)', round(extent, 3), 'shift', round(step, 3))
idx = np.linspace(0, len(cams) - 1, 4).astype(int)
for i in idx:
    tiles = [np.asarray(ds.get_image_float32(int(i))[..., :3])]
    base = cams[int(i):int(i) + 1].to(pipeline.device)
    c2w = base.camera_to_worlds[0]
    for d in (None, c2w[:3, 0], c2w[:3, 2], c2w[:3, 1]):   # nerfstudio camera: x right, y up, z back
        cam = base
        if d is not None:
            cam = base.__class__(camera_to_worlds=base.camera_to_worlds.clone(), fx=base.fx, fy=base.fy, cx=base.cx, cy=base.cy,
                                 width=base.width, height=base.height, camera_type=base.camera_type)
            cam.camera_to_worlds[0, :3, 3] += step * d / d.norm()
        with torch.no_grad():
            o = pipeline.model.get_outputs_for_camera(cam)
        tiles.append(o['rgb'].clamp(0, 1).cpu().numpy())
    h = min(t.shape[0] for t in tiles)
    strip = np.concatenate([t[:h] for t in tiles], 1)
    imageio.imwrite(out / f'offpath_{int(i):03d}.jpg', (strip * 255).astype(np.uint8), quality=88)
print('wrote', len(idx), 'strips to', out)
