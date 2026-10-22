# Off-path views from a trained Nerfstudio model: for 4 held-out cameras, render the camera itself and the camera moved
# off the filmed path, and save one strip per camera:
#   real photo | render at the photo's camera | right S | right L | back L | up S
# S and L are fractions of the camera path's extent (defaults 1% and 3%: roughly 10-15 cm and 30-45 cm on a 10-15 m walk).
# Cameras come from the datamanager's own eval loader, the same ones ns-eval scores, so photo and render line up.
# Also writes sheet.jpg: all strips stacked, 1600 px wide, for a quick look.
# Usage: python offpath_render.py <config.yml> <out dir> [S=0.01] [L=0.03]
import sys, numpy as np, torch, imageio.v2 as imageio
from pathlib import Path
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
cfg, out = Path(sys.argv[1]), Path(sys.argv[2])
S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
L = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
out.mkdir(parents=True, exist_ok=True)
_, pipeline, _, _ = eval_setup(cfg, test_mode='test')
dm = pipeline.datamanager
train_t = dm.train_dataset.cameras.camera_to_worlds[:, :3, 3]
extent = float((train_t.max(0).values - train_t.min(0).values).norm())
print('path extent (scene units)', round(extent, 3), 'S', round(S * extent, 4), 'L', round(L * extent, 4))
views = list(dm.fixed_indices_eval_dataloader)
pick = np.linspace(0, len(views) - 1, 4).astype(int)
rows = []
for i in pick:
    camera, batch = views[int(i)]
    camera = camera.to(pipeline.device)
    c2w = camera.camera_to_worlds[0]
    tiles = [batch['image'][..., :3].float().cpu().numpy()]
    if tiles[0].max() > 1.5: tiles[0] = tiles[0] / 255.0
    for axis, frac in ((None, 0), (0, S), (0, L), (2, L), (1, S)):   # nerfstudio camera: x right, y up, z back
        cam = camera
        if axis is not None:
            cam = camera.__class__(camera_to_worlds=camera.camera_to_worlds.clone(), fx=camera.fx, fy=camera.fy, cx=camera.cx,
                                   cy=camera.cy, width=camera.width, height=camera.height, camera_type=camera.camera_type)
            d = c2w[:3, axis]
            cam.camera_to_worlds[0, :3, 3] += frac * extent * d / d.norm()
        with torch.no_grad():
            o = pipeline.model.get_outputs_for_camera(cam)
        tiles.append(o['rgb'].clamp(0, 1).cpu().numpy())
    h = min(t.shape[0] for t in tiles); w = min(t.shape[1] for t in tiles)
    strip = (np.concatenate([t[:h, :w] for t in tiles], 1) * 255).astype(np.uint8)
    imageio.imwrite(out / f'offpath_{int(i):03d}.jpg', strip, quality=90)
    rows.append(Image.fromarray(strip).resize((1600, int(strip.shape[0] * 1600 / strip.shape[1]))))
sheet = Image.new('RGB', (1600, sum(r.height for r in rows)))
y = 0
for r in rows: sheet.paste(r, (0, y)); y += r.height
sheet.save(out / 'sheet.jpg', quality=88)
print('wrote', len(rows), 'strips and sheet.jpg to', out, '| columns: photo | render | right S | right L | back L | up S')
