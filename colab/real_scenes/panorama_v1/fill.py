"""Hole-only generative fill for panorama splat worlds (H4).

Rule from the owner (video NOTES 2026-10-24): generation may only fill gaps,
never replace what was filmed, and generated content stays flagged.

  1. Walkable novel cameras: between consecutive panoramas, on the route and
     a sideways step either side, looking forward/left/back/right, level and
     tilted down 20 deg (to see the road).
  2. Render the trained model (+ sky dome). A pixel is a HOLE when the
     splats are thin there (alpha < 0.6) and the ray is not sky: below the
     horizon nothing can be sky; above it, the ray must cross a voxel no real
     panorama ever saw (carve grid unknown) before leaving the scene box.
  3. Views with 0.5-35 % holes are painted with SDXL inpainting (RealVisXL V4
     inpainting + LCM-LoRA, as in World v1), holes only, prompt without objects.
     Bigger holes are skipped: big gaps invite invented objects (video NOTES).
  4. MoGe-2 depth on the painted view, scaled to the rendered depth on solid
     pixels, gives 3D points for the hole pixels (written as generated points).
  5. The view is saved as a training camera whose loss is masked to the hole
     pixels, with low weight (trainer: 0.3), marked kind="generated".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import offpath
import scene_io

PROMPT = "photo of the same street, buildings, pavement, road surface, trees, natural daylight, realistic"
NEG = ("people, person, car, vehicle, text, watermark, logo, frame, blurry, low quality, cartoon, painting, "
       "distorted, fisheye, duplicate")


def fill_cameras(cameras, centre, scale, up, max_views: int = 160, seed: int = 0):
    pc = scene_io.pano_centres(cameras)
    names = sorted(pc)
    C = np.stack([(pc[n] - centre) * scale for n in names])
    spacing = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    cams = []
    for i in range(len(C) - 1):
        a, b = C[i], C[i + 1]
        fwd = b - a
        fwd = fwd - up * np.dot(fwd, up)
        if np.linalg.norm(fwd) < 1e-9:
            continue
        fwd /= np.linalg.norm(fwd)
        side = np.cross(up, fwd)
        for lat in (-0.5, 0.0, 0.5):
            eye = (a + b) / 2 + side * lat * spacing
            for yaw in (0, 90, 180, 270):
                d = offpath.rotate_about(fwd, up, yaw)
                for pitch in (0, -20):
                    dd = offpath.rotate_about(d, np.cross(up, d), -pitch) if pitch else d
                    cams.append((f"seg{i:03d}_lat{lat:+.1f}_yaw{yaw}_p{pitch}", offpath.look_at(eye, dd, up)))
    rng = np.random.default_rng(seed)
    if len(cams) > max_views:
        cams = [cams[k] for k in sorted(rng.choice(len(cams), max_views, replace=False))]
    return cams, spacing


def unseen_ray_mask(grid, V, K, w, h, up, device, max_t: float = 3.0, step: float | None = None):
    """True where the ray meets a never-observed voxel before leaving the box."""
    import torch

    step = step or float(np.min(grid.size))
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    Vt = torch.tensor(V, dtype=torch.float32, device=device)
    d = torch.stack([(xs + 0.5 - Kt[0, 2]) / Kt[0, 0], (ys + 0.5 - Kt[1, 2]) / Kt[1, 1],
                     torch.ones_like(xs, dtype=torch.float32)], -1).reshape(-1, 3)
    d = torch.nn.functional.normalize(d @ Vt[:3, :3], dim=-1)
    o = -(Vt[:3, :3].T @ Vt[:3, 3])
    hit = torch.zeros(len(d), dtype=torch.bool, device=device)
    ts = torch.arange(1, int(max_t / step), device=device, dtype=torch.float32) * step
    for c0 in range(0, len(d), 65536):
        dd = d[c0:c0 + 65536]
        pts = o + dd[:, None, :] * ts[None, :, None]
        ijk, inside = grid._index(pts.reshape(-1, 3))
        ijk = torch.minimum(ijk.clamp_min(0), torch.tensor(grid.dims - 1, device=device))
        e = grid.empty[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        s = grid.surface[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        unknown = (e == 0) & (s == 0) & inside
        hit[c0:c0 + 65536] = unknown.reshape(len(dd), -1).any(-1)
    return hit.reshape(h, w)


class Painter:
    def __init__(self, device: str = "cuda"):
        import torch
        from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLInpaintPipeline

        self.device = device
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = None
        for mid in ("OzzyGT/RealVisXL_V4.0_inpainting", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
            try:
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16,
                                                                        variant="fp16").to(device)
                self.model_id = mid
                break
            except Exception as exc:  # noqa: BLE001 - try the fallback painter
                print("painter", mid, "failed:", repr(exc)[:200], flush=True)
        if pipe is None:
            raise RuntimeError("no SDXL inpainting model could be loaded")
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def __call__(self, rgb: np.ndarray, hole: np.ndarray, seed: int) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image

        H, W = hole.shape
        k = (1024 * 1024 / (H * W)) ** 0.5
        w8, h8 = max(64, round(W * k / 64) * 64), max(64, round(H * k / 64) * 64)
        sm = cv2.resize(rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hm = cv2.resize(hole.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
        pre = cv2.resize(cv2.inpaint(sm, hm, 5, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_LINEAR)
        src = np.where(hole[..., None], pre, rgb).astype(np.uint8)
        with torch.no_grad():
            out = self.pipe(prompt=PROMPT, negative_prompt=NEG,
                            image=Image.fromarray(src).resize((w8, h8), Image.LANCZOS),
                            mask_image=Image.fromarray(hole.astype(np.uint8) * 255).resize((w8, h8), Image.NEAREST),
                            width=w8, height=h8, num_inference_steps=8, guidance_scale=1.5, strength=1.0,
                            generator=torch.Generator(self.device).manual_seed(int(seed))).images[0]
        out = np.asarray(out.convert("RGB").resize((W, H), Image.LANCZOS))
        if out.std() < 2:
            raise RuntimeError("painter returned a blank image")
        # keep real (rendered) pixels exactly; colour-match the painted ones to the ring around the hole
        ring = cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~hole
        if ring.sum() > 50:
            for ch in range(3):
                a, b = out[..., ch][ring].astype(np.float32), rgb[..., ch][ring].astype(np.float32)
                out[..., ch] = np.clip((out[..., ch] - a.mean()) * (b.std() + 1) / (a.std() + 1) + b.mean(), 0, 255)
        return np.where(hole[..., None], out, rgb).astype(np.uint8)


def run_fill(scene_dir: Path, model_dir: Path, out_dir: Path, device: str = "cuda", size: int = 768,
             max_views: int = 160, min_hole: float = 0.005, max_hole: float = 0.35, log=print) -> dict:
    import cv2
    import torch
    from PIL import Image

    import depth_memory

    scene_dir, model_dir, out_dir = Path(scene_dir), Path(model_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "sheet").mkdir(parents=True, exist_ok=True)
    cameras, points, meta = scene_io.read_scene(scene_dir)
    ck = torch.load(model_dir / "splats.pt", map_location=device, weights_only=False)
    centre, scale, up = np.asarray(ck["centre"]), float(ck["scale"]), np.asarray(ck["up"])
    grid = depth_memory.CarveGrid.load(model_dir / "carve.pt", device) if (model_dir / "carve.pt").is_file() else None
    render = offpath.make_renderer(ck, device)
    cams, spacing = fill_cameras(cameras, centre, scale, up, max_views)
    f = size / 2 / math.tan(math.radians(45))
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    painter = Painter(device)
    moge = depth_memory.load_moge(device)
    new_cams, gen_points, stats = [], [], {"candidates": len(cams), "painted": 0, "skipped_small": 0,
                                            "skipped_big": 0, "hole_fraction": []}
    for j, (label, V) in enumerate(cams):
        rgb, alpha, depth = render(V, K, size, size, with_depth=True)
        thin = alpha < 0.6
        # elevation of each pixel ray
        ys, xs = np.mgrid[0:size, 0:size]
        d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones((size, size))], -1)
        d = d @ V[:3, :3]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        below = (d @ up) < math.sin(math.radians(3))
        unknown = unseen_ray_mask(grid, V, K, size, size, up, device).cpu().numpy() if grid is not None else below
        hole = thin & (below | unknown)
        hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        hole = cv2.dilate(hole, np.ones((7, 7), np.uint8)).astype(bool)
        frac = float(hole.mean())
        stats["hole_fraction"].append(frac)
        if frac < min_hole:
            stats["skipped_small"] += 1
            continue
        if frac > max_hole:
            stats["skipped_big"] += 1
            continue
        painted = painter(rgb, hole, seed=j)
        # depth for the painted pixels: MoGe on the painted view, scaled to the rendered depth where solid
        md, valid = depth_memory.moge_view_depth(moge, painted, 90.0, device)
        solid = (alpha > 0.9) & (depth > 0) & valid & ~hole
        if solid.sum() > 200:
            s_, inl = depth_memory.robust_scale(depth[solid], md[solid])
        else:
            s_, inl = None, 0.0
        name = f"gen/{label}.jpg"
        Image.fromarray(painted).save(out_dir / "images" / f"{label}.jpg", quality=94)
        Image.fromarray(hole.astype(np.uint8) * 255).save(out_dir / "masks" / f"{label}.png")
        # world-space camera (undo normalisation): x_cam = R x_n + t_n, x_n = (x - centre) * scale
        w2c = np.eye(4)
        w2c[:3, :3] = V[:3, :3]
        w2c[:3, 3] = V[:3, 3] / scale - V[:3, :3] @ centre
        new_cams.append({"name": name, "path": str(out_dir / "images" / f"{label}.jpg"), "pano": f"gen-{label}",
                         "width": size, "height": size, "K": K, "w2c": w2c, "split": "train",
                         "kind": "generated", "mask": str(out_dir / "masks" / f"{label}.png"),
                         "weight": str(out_dir / "masks" / "zero.png")})
        if s_ is not None and inl > 0.3:
            dz = md * s_ / scale  # normalised -> world units
            keep = hole & valid
            gen_points.append(depth_memory.backproject(dz, painted, K, w2c, stride=4, keep=keep))
        stats["painted"] += 1
        if stats["painted"] <= 8:
            sheet = np.concatenate([rgb, np.where(hole[..., None], [255, 0, 255], rgb).astype(np.uint8), painted], 1)
            Image.fromarray(sheet).resize((size * 3 // 2, size // 2)).save(out_dir / "sheet" / f"{label}.jpg", quality=88)
        if j % 20 == 0:
            log(f"[fill] {j}/{len(cams)} painted {stats['painted']} hole {frac:.3f}")
    Image.fromarray(np.zeros((size, size), np.uint8)).save(out_dir / "masks" / "zero.png")
    gp = np.concatenate(gen_points) if gen_points else np.zeros((0, 6), np.float32)
    np.save(out_dir / "generated_points.npy", gp)
    hf = np.asarray(stats.pop("hole_fraction"))
    stats.update({"hole_fraction_median": float(np.median(hf)) if len(hf) else None,
                  "generated_points": int(len(gp)), "painter": painter.model_id, "spacing_norm": spacing})
    (out_dir / "fill_report.json").write_text(json.dumps(stats, indent=2))
    # new scene = real cameras + generated cameras, points = old + generated
    scene_io.write_scene(out_dir / "scene", cameras + new_cams,
                         np.concatenate([points, gp]) if len(gp) else points,
                         {**meta, "fill": stats})
    return stats


def run_fixer_pass(scene_dir: Path, model_dir: Path, out_dir: Path, work: Path, device: str = "cuda",
                   size: int = 768, max_views: int = 120, thr: float = 0.08, log=print) -> dict:
    """NVIDIA Fixer (Apache-2.0 code, NVIDIA Open Model License weights) on off-path renders.

    Only pixels Fixer changed by more than thr (mean abs, 0..1) become supervision, masked and
    low-weight, so real panorama content is never overwritten (video NOTES 30 Oct: unmasked Fixer
    retraining made walls blotchy).
    """
    import cv2
    import torch
    from PIL import Image

    import fixer_lib

    scene_dir, model_dir, out_dir = Path(scene_dir), Path(model_dir), Path(out_dir)
    for sub in ("images", "masks", "sheet"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    cameras, points, meta = scene_io.read_scene(scene_dir)
    ck = torch.load(model_dir / "splats.pt", map_location=device, weights_only=False)
    centre, scale, up = np.asarray(ck["centre"]), float(ck["scale"]), np.asarray(ck["up"])
    render = offpath.make_renderer(ck, device)
    cams, _ = fill_cameras([c for c in cameras if c.get("kind") != "generated"], centre, scale, up,
                           max_views, seed=1)
    fixer_lib.setup(str(work), say=log)
    fix = fixer_lib.load(str(work), str(Path(__file__).resolve().parent / "fixer_shim"), device, say=log)
    f = size / 2 / math.tan(math.radians(45))
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    new_cams, changed = [], []
    for j, (label, V) in enumerate(cams):
        rgb, alpha, _ = render(V, K, size, size)
        fixed = fix(rgb)
        diff = np.abs(fixed.astype(np.float32) - rgb.astype(np.float32)).mean(-1) / 255.0
        m = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        frac = float(m.mean())
        changed.append(frac)
        if frac < 0.002:
            continue
        Image.fromarray(fixed).save(out_dir / "images" / f"fx_{label}.jpg", quality=94)
        Image.fromarray(m * 255).save(out_dir / "masks" / f"fx_{label}.png")
        w2c = np.eye(4)
        w2c[:3, :3] = V[:3, :3]
        w2c[:3, 3] = V[:3, 3] / scale - V[:3, :3] @ centre
        new_cams.append({"name": f"fix/{label}.jpg", "path": str(out_dir / "images" / f"fx_{label}.jpg"),
                         "pano": f"fix-{label}", "width": size, "height": size, "K": K, "w2c": w2c,
                         "split": "train", "kind": "generated", "mask": str(out_dir / "masks" / f"fx_{label}.png"),
                         "weight": str(out_dir / "masks" / "zero.png")})
        if len(new_cams) <= 8:
            sheet = np.concatenate([rgb, fixed, np.repeat(m[..., None] * 255, 3, -1)], 1)
            Image.fromarray(sheet).resize((size * 3 // 2, size // 2)).save(out_dir / "sheet" / f"fx_{label}.jpg", quality=88)
        if j % 20 == 0:
            log(f"[fixer] {j}/{len(cams)} changed {frac:.3f}")
    Image.fromarray(np.zeros((size, size), np.uint8)).save(out_dir / "masks" / "zero.png")
    rep = {"views": len(cams), "used": len(new_cams), "changed_median": float(np.median(changed)) if changed else None}
    (out_dir / "fixer_report.json").write_text(json.dumps(rep, indent=2))
    scene_io.write_scene(out_dir / "scene", cameras + new_cams, points, {**meta, "fixer": rep})
    return rep
