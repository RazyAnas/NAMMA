# Debug: does a render at an eval camera match its photo? Prints shapes, ranges and PSNR for 3 eval views.
import torch
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
_, p, _, _ = eval_setup(Path('/content/ns_lobby/out/lobby/splatfacto/base/config.yml'), test_mode='test')
views = list(p.datamanager.fixed_indices_eval_dataloader)
print('views', len(views), type(views[0][0]).__name__)
for camera, batch in views[:3]:
    print('c2w', tuple(camera.camera_to_worlds.shape), 'wh', camera.width.flatten().tolist(), camera.height.flatten().tolist(), 'img', tuple(batch['image'].shape), batch['image'].dtype, float(batch['image'].float().max()))
    with torch.no_grad():
        o = p.model.get_outputs_for_camera(camera.to(p.device))
        m, _ = p.model.get_image_metrics_and_images(o, batch)
    gt = batch['image'][..., :3].float().to(p.device); gt = gt / 255 if gt.max() > 1.5 else gt
    print('rgb', tuple(o['rgb'].shape), 'model psnr', round(float(m['psnr']), 2), 'mine', round(float(-10 * torch.log10(((o['rgb'][..., :3] - gt) ** 2).mean())), 2))
