# Monocular depth prior for DN-Splatter, aligned to the SfM points (replaces its ZoeDepth script, which breaks on torch 2.1).
# For every frame in transforms.json: Depth Anything V2 relative disparity, fitted per image to the COLMAP points seen in that
# image as 1/z = a*d + b (least squares, refit after dropping the worst 20%), saved as z-depth in COLMAP units to
# <data>/mono_depth/<stem>_aligned.npy, the name and order DN-Splatter's normal-nerfstudio dataparser looks for.
# Usage: python mono_depth_prior.py <ns-process-data output dir> [model=depth-anything/Depth-Anything-V2-Base-hf]
import sys, json, numpy as np, torch, pycolmap
from pathlib import Path
from PIL import Image
from transformers import pipeline
data = Path(sys.argv[1]); model = sys.argv[2] if len(sys.argv) > 2 else 'depth-anything/Depth-Anything-V2-Base-hf'
rec = pycolmap.Reconstruction(str(data / 'colmap/sparse/0'))
by_name = {im.name: im for im in rec.images.values()}
pipe = pipeline('depth-estimation', model=model, device=0 if torch.cuda.is_available() else -1)
out = data / 'mono_depth'; out.mkdir(exist_ok=True)
frames = json.load(open(data / 'transforms.json'))['frames']
fits, zmeds = [], []
for f in frames:
    path = data / f['file_path']; im = by_name.get(path.name)
    img = Image.open(path).convert('RGB'); W, H = img.size
    d = pipe(img)['predicted_depth'].squeeze().float()
    d = torch.nn.functional.interpolate(d[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    uv, z = [], []
    if im is not None:
        cam = rec.cameras[im.camera_id]; sx, sy = W / cam.width, H / cam.height
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world   # a method from pycolmap 3.12
        R, t = cfw.rotation.matrix(), cfw.translation
        for p2 in im.points2D:
            if not p2.has_point3D(): continue
            pc = R @ rec.points3D[p2.point3D_id].xyz + t
            if pc[2] <= 0: continue
            u, v = int(p2.xy[0] * sx), int(p2.xy[1] * sy)
            if 0 <= u < W and 0 <= v < H: uv.append((v, u)); z.append(pc[2])
    if len(z) >= 20:
        uv, z = np.array(uv), np.array(z); x = d[uv[:, 0], uv[:, 1]]; keep = np.ones(len(z), bool)
        for _ in range(2):
            A = np.stack([x[keep], np.ones(keep.sum())], 1); a, b = np.linalg.lstsq(A, 1 / z[keep], rcond=None)[0]
            err = np.abs(1 / np.clip(a * x + b, 1e-6, None) - z) / z; keep = err <= np.quantile(err, 0.8)
        depth = 1 / np.clip(a * d + b, 1e-6, None); fits.append(float(np.median(err[keep]))); zmeds.append(float(np.median(z)))
    else:   # too few points: keep the shape only, scaled to the median depth seen so far
        depth = (np.median(zmeds) if zmeds else 1.0) / np.clip(d / max(d.mean(), 1e-6), 1e-3, None)
    zmax = np.percentile(depth, 99); depth = np.clip(depth, 0, zmax * 1.5).astype(np.float32)
    np.save(out / f'{path.stem}_aligned.npy', depth)
print('aligned depths', len(frames), 'median relative error at SfM points', round(float(np.median(fits)), 3) if fits else 'n/a')
