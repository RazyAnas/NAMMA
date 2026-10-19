"""Train a Gaussian splat with gsplat (CUDA) on an exported COLMAP dataset; the Colab / NVIDIA trainer.

Same recipe as the GPU handout's trainer (gpu-handout: scripts/train_gs.py), which ran on a Tesla T4 on
2026-10-17 (ICL-NUIM: 34.48 dB from monocular init points at 3,000 steps). Only the data loading
changed: it reads `<dataset>/sparse/0` + `<dataset>/images` as written by tracking/sfm or the phone
render job, holds out every 8th image for a PSNR score, and writes `<output>/splat.ply`.

    python scripts/train_gsplat.py --dataset data/projects/<id>/training/dataset --output out --steps 7000

Requires torch with CUDA, gsplat==1.5.3, pytorch-msssim, pycolmap. `--appearance` (on by default)
learns a per-image colour transform, which absorbs phone auto-exposure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


def load_dataset(dataset: Path, downscale: int, holdout_every: int):
    import pycolmap
    from PIL import Image

    reconstruction = pycolmap.Reconstruction(str(dataset / "sparse" / "0"))
    images = sorted((im for im in reconstruction.images.values() if im.has_pose), key=lambda im: im.name)
    views = []
    for index, image in enumerate(images):
        camera = reconstruction.cameras[image.camera_id]
        params = list(camera.params)
        # PINHOLE: fx fy cx cy; SIMPLE_*: f cx cy (+ distortion, ignored: COLMAP's k1 on phone video is small)
        fx, fy, cx, cy = (params[0], params[1], params[2], params[3]) if camera.model.name in ("PINHOLE", "OPENCV") else (params[0], params[0], params[1], params[2])
        picture = Image.open(dataset / "images" / image.name).convert("RGB")
        width, height = picture.size
        if downscale > 1:
            width, height = width // downscale, height // downscale
            picture = picture.resize((width, height), Image.BICUBIC)
        scale_x, scale_y = width / camera.width, height / camera.height
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :4] = image.cam_from_world().matrix()
        views.append({
            "name": image.name,
            "image": np.asarray(picture, dtype=np.float32) / 255.0,
            "K": np.array([[fx * scale_x, 0, cx * scale_x], [0, fy * scale_y, cy * scale_y], [0, 0, 1]], dtype=np.float32),
            "w2c": world_to_camera,
            "test": holdout_every > 0 and index % holdout_every == holdout_every // 2,
        })
    points = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float32)
    colours = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.float32) / 255.0
    return views, points, colours


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--sh", type=int, default=3)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--holdout-every", type=int, default=8, help="0 = train on every image")
    parser.add_argument("--no-appearance", action="store_true")
    parser.add_argument("--max-points", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from gsplat import export_splats, rasterization
    from gsplat.strategy import DefaultStrategy
    from pytorch_msssim import ssim

    if not torch.cuda.is_available():
        parser.error("gsplat training needs an NVIDIA GPU (on a Mac use the OpenSplat trainer).")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    args.output.mkdir(parents=True, exist_ok=True)
    views, xyz, rgb = load_dataset(args.dataset, args.downscale, args.holdout_every)
    if len(xyz) == 0:
        parser.error("The dataset has no 3D points to start from.")
    if len(xyz) > args.max_points:
        keep = rng.choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    train = [v for v in views if not v["test"]] or views
    test = [v for v in views if v["test"]]
    print(f"{len(train)} training views, {len(test)} held out, {len(xyz)} initial points", flush=True)

    images = [torch.from_numpy(v["image"]).to(device) for v in train]
    w2c = torch.tensor(np.stack([v["w2c"] for v in train]), device=device)
    intrinsics = torch.tensor(np.stack([v["K"] for v in train]), device=device)
    count = len(xyz)
    points = torch.from_numpy(xyz).to(device)
    nearest = []
    for start in range(0, count, 4096):  # initial scale = mean distance to the 3 nearest neighbours
        nearest.append(torch.cdist(points[start:start + 4096], points).topk(4, largest=False).values[:, 1:])
    distance = torch.cat(nearest).clamp_min(1e-7).mean(1)
    centres = torch.linalg.inv(w2c)[:, :3, 3]
    scene_scale = float((centres - centres.mean(0)).norm(dim=1).max() * 1.1) or 1.0
    c0 = 0.28209479177387814
    sh = torch.zeros(count, (args.sh + 1) ** 2, 3, device=device)
    sh[:, 0] = (torch.from_numpy(rgb).to(device) - 0.5) / c0
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points.clone()),
        "scales": torch.nn.Parameter(torch.log(distance)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.rand(count, 4, device=device), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((count,), 0.1, device=device))),
        "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
    })
    rates = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": rates[k], "name": k}], eps=1e-15) for k in params}
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.steps))
    appearance = not args.no_appearance
    identity = torch.cat([torch.eye(3), torch.zeros(3, 1)], 1).to(device)
    if appearance:
        colour_fix = torch.nn.Parameter(identity[None].repeat(len(train), 1, 1).clone())
        colour_optimizer = torch.optim.Adam([colour_fix], lr=1e-3)
    long_run = args.steps >= 10000
    strategy = DefaultStrategy(
        refine_start_iter=500 if long_run else max(1, args.steps // 10),
        refine_stop_iter=min(15000, args.steps // 2),
        refine_every=100 if long_run else max(10, args.steps // 10),
        reset_every=3000 if long_run else 10**9,
        absgrad=False, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(view_w2c, K, width, height, degree):
        return rasterization(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                             torch.cat([params["sh0"], params["shN"]], 1), view_w2c[None], K[None], width, height,
                             sh_degree=degree, packed=False)

    sh_interval = max(1, args.steps // 30)
    started = time.time()
    for step in range(args.steps):
        i = int(rng.integers(len(train)))
        target = images[i]
        height, width = target.shape[:2]
        degree = min(args.sh, step // sh_interval)
        out, _, info = render(w2c[i], intrinsics[i], width, height, degree)
        out = out[0]
        if appearance:
            out = out @ colour_fix[i, :, :3].T + colour_fix[i, :, 3]
        l1 = (out - target).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(out.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1.0, size_average=True))
        if appearance:
            loss = loss + ((colour_fix.mean(0) - identity) ** 2).sum()
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if appearance:
            colour_optimizer.step()
            colour_optimizer.zero_grad(set_to_none=True)
        schedule.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} gaussians {len(params['means'])} {time.time() - started:.0f}s", flush=True)
    train_seconds = time.time() - started

    psnrs = []
    with torch.no_grad():
        for view in test:  # held-out views get no colour fix: a conservative score
            target = torch.from_numpy(view["image"]).to(device)
            height, width = target.shape[:2]
            out, _, _ = render(torch.from_numpy(view["w2c"]).to(device), torch.from_numpy(view["K"]).to(device), width, height, args.sh)
            mse = ((out[0].clamp(0, 1) - target) ** 2).mean().item()
            psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
    export_splats(params["means"].detach(), params["scales"].detach(), F.normalize(params["quats"].detach(), dim=-1),
                  params["opacities"].detach(), params["sh0"].detach(), params["shN"].detach(), format="ply",
                  save_to=str(args.output / "splat.ply"))
    info = {
        "trainer": "gsplat", "steps": args.steps, "downscale": args.downscale, "appearance": appearance,
        "train_views": len(train), "heldout_views": len(test), "init_points": int(count),
        "final_gaussians": int(len(params["means"])), "train_seconds": round(train_seconds, 1),
        "heldout_psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else None,
        "gpu": torch.cuda.get_device_name(0), "measured": True,
    }
    (args.output / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
