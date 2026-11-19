"""H9: dense crisp views along walkable paths (Marble's densification step, sized for a T4).

Key rule (the "hi-res fix" from Matrix-3D-based workflows): wherever any REAL panorama saw a point,
the dense frame takes that panorama's full-resolution pixel (depth-checked image-based rendering);
splat renders are used only where the model is confident, and a video model only paints true holes.

Splats look soft off the capture path because each surface was seen from few places. Here we:
  1. lay dense camera paths along the route (every ~1 m, walking eye height, lanes left/centre/right,
     looking forward and back);
  2. render the trained model along each path as a 49-frame clip;
  3. refine each clip with Wan2.1-T2V-1.3B video-to-video at low strength (SDEdit): the structure of
     our render is kept, the video model (trained on many worlds) restores consistent detail;
  4. add the refined frames as low-weight training views and continue training.
Real panorama views keep full weight, so generated frames can sharpen but never override real data.
T4 recipe from the video pipeline (final/fill_holes.py): text encoder on the CPU, fp16 transformer,
fp32 VAE with tiling.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import offpath
import scene_io

W, H = 832, 480
PROMPT = ("a sharp, high-detail photo of a city street, crisp building facades, windows, road markings, "
          "natural daylight, photorealistic, no blur")
NEG = "blurry, smeared, low detail, distorted, cartoon, painting, text, watermark, people, cars, noise"


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image (higher = crisper)."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def dense_paths(cameras, centre, scale, up, eye_drop_units: float, step_units: float,
                lanes_units=(-0.6, 0.0, 0.6), frames: int = 49) -> list[tuple[str, list[np.ndarray]]]:
    """Clips of camera matrices (trainer's normalised frame) along the route."""
    pc = scene_io.pano_centres([c for c in cameras if c.get("kind") != "generated"])
    C = np.stack([(pc[n] - centre) * scale for n in sorted(pc)])
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    clips = []
    starts = np.arange(0, max(total - step_units * frames, 0) + 1e-9, step_units * frames * 0.8)
    for k, s0 in enumerate(starts):
        for lane in lanes_units:
            for heading in (0, 180):
                mats = []
                for f in range(frames):
                    t = min(s0 + f * step_units, total - 1e-6)
                    i = int(np.searchsorted(s, t, side="right") - 1)
                    i = min(i, len(C) - 2)
                    a = (t - s[i]) / max(seg[i], 1e-9)
                    p = C[i] + a * (C[i + 1] - C[i])
                    fwd = C[i + 1] - C[i]
                    fwd = fwd - up * np.dot(fwd, up)
                    fwd /= np.linalg.norm(fwd)
                    side = np.cross(up, fwd)
                    if heading:
                        fwd = -fwd
                    eye = p + side * lane - up * eye_drop_units
                    look = offpath.rotate_about(fwd, np.cross(up, fwd), 8)  # slightly down: road and facades
                    mats.append(offpath.look_at(eye, look, up))
                clips.append((f"clip{k:03d}_lane{lane:+.1f}_h{heading}", mats))
    return clips


def metres_per_norm_unit(cameras, centre, scale, manifest: dict | None) -> float:
    """Metres per trainer unit from GPS spacing between consecutive panoramas (Street View / Panoramax)."""
    from export_game import enu

    pc = scene_io.pano_centres([c for c in cameras if c.get("kind") != "generated"])
    names = sorted(pc)
    C = np.stack([(pc[n] - centre) * scale for n in names])
    d_norm = np.linalg.norm(np.diff(C, axis=0), axis=1)
    if manifest and manifest.get("panoramas"):
        by = {f"{i:04d}": p for i, p in enumerate(manifest["panoramas"])}
        ll = [by.get(n.split("_")[0]) for n in names]
        if all(p and p.get("lat") is not None for p in ll):
            E = np.array([enu(ll[0]["lat"], ll[0]["lon"], p["lat"], p["lon"]) for p in ll])
            d_m = np.linalg.norm(np.diff(E, axis=0), axis=1)
            ok = (d_m > 1) & (d_norm > 1e-6)
            if ok.sum() >= 3:
                return float(np.median(d_m[ok] / d_norm[ok]))
    return 10.0 / float(np.median(d_norm))  # Street View spacing is ~10 m


class Refiner:
    def __init__(self, device: str = "cuda"):
        import gc

        import torch
        from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import UMT5EncoderModel

        mid = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        te = UMT5EncoderModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVideoToVideoPipeline.from_pretrained(mid, vae=vae, text_encoder=te, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        with torch.no_grad():
            self.pe, self.npe = pipe.encode_prompt(prompt=PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True,
                                                   device=torch.device("cpu"), dtype=torch.float32)
        pipe.text_encoder = None
        del te
        gc.collect()
        pipe.vae.to(device)
        try:
            pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001
            pass
        pipe.transformer.to(device)
        self.pipe, self.device = pipe, device

    def __call__(self, frames: list[np.ndarray], strength: float = 0.35, steps: int = 20, seed: int = 0) -> list[np.ndarray]:
        import torch
        from PIL import Image

        vid = [Image.fromarray(f) for f in frames]
        with torch.no_grad():
            out = self.pipe(video=vid, prompt_embeds=self.pe.to(self.device, torch.float16),
                            negative_prompt_embeds=self.npe.to(self.device, torch.float16), height=H, width=W,
                            strength=strength, num_inference_steps=steps, guidance_scale=5.0,
                            generator=torch.Generator(self.device).manual_seed(seed), output_type="np").frames[0]
        out = np.nan_to_num(np.asarray(out, np.float32))
        if out.std() < 1e-3:
            raise RuntimeError("Wan returned a blank clip (fp16 overflow)")
        return [(np.clip(o, 0, 1) * 255).astype(np.uint8) for o in out]


def run_densify(scene_dir: Path, model_dir: Path, out_dir: Path, device: str = "cuda", max_clips: int = 40,
                step_m: float = 1.0, lane_m: float = 1.5, walk_eye_m: float = 1.6,
                camera_height_m: float | None = None, manifest: dict | None = None, frame_w: int = 1280,
                frame_h: int = 720, fill_holes: bool = True, max_minutes: float = 0.0, log=print) -> dict:
    import time

    import torch
    from PIL import Image

    scene_dir, model_dir, out_dir = Path(scene_dir), Path(model_dir), Path(out_dir)
    for sub in ("images", "sheet"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    cameras, points, meta = scene_io.read_scene(scene_dir)
    ck = torch.load(model_dir / "splats.pt", map_location=device, weights_only=False)
    centre, scale, up = np.asarray(ck["centre"]), float(ck["scale"]), np.asarray(ck["up"])
    m_per_norm = metres_per_norm_unit(cameras, centre, scale, manifest)
    eye_drop = max(0.0, ((camera_height_m or 2.0) - walk_eye_m) / m_per_norm)
    clips = dense_paths(cameras, centre, scale, up, eye_drop, step_m / m_per_norm,
                        lanes_units=(-lane_m / m_per_norm, 0.0, lane_m / m_per_norm))
    if len(clips) > max_clips:
        clips = [clips[i] for i in np.linspace(0, len(clips) - 1, max_clips).round().astype(int)]
    render = offpath.make_renderer(ck, device)
    f = frame_w / 2 / math.tan(math.radians(40))
    K = np.array([[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1.0]])
    real = RealViews(cameras, device, render=render, centre=centre, scale=scale)
    log(f"[densify] {len(real.cams)} real views in RAM, {len(clips)} clips, {m_per_norm:.2f} m per unit")
    filler = None
    new_cams, crisp_splat, crisp_ibr, real_frac, gen_frac = [], [], [], [], []
    t0 = time.time()
    for ci, (label, mats) in enumerate(clips):
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            log(f"[densify] time budget {max_minutes:.0f} min reached after {ci} clips")
            break
        frames, holes, splat_frames, rmasks = [], [], [], []
        for V in mats:
            rgb, alpha, depth = render(V, K, frame_w, frame_h, with_depth=True)
            depth = np.where(alpha > 0.5, depth, 0.0)
            ibr, rmask = ibr_composite(real, V, K, frame_w, frame_h, depth, centre, scale)
            confident = alpha > 0.95
            comp = np.where(rmask[..., None], ibr, rgb)          # real pixels first, splat render second
            hole = ~rmask & ~confident
            frames.append(comp.astype(np.uint8))
            holes.append(hole)
            rmasks.append(rmask)
            splat_frames.append(rgb)
            real_frac.append(float(rmask.mean()))
        hole_frac = float(np.mean([h.mean() for h in holes]))
        if fill_holes and hole_frac > 0.01:
            if filler is None:
                filler = HoleFiller(device)
            frames = filler(frames, holes, seed=ci)
            gen_frac.append(hole_frac)
        crisp_splat += [sharpness(x) for x in splat_frames[::8]]
        crisp_ibr += [sharpness(x) for x in frames[::8]]
        for fi, (V, img, hole, rm) in enumerate(zip(mats, frames, holes, rmasks)):
            if fi % 2:
                continue
            name = f"{label}_f{fi:02d}"
            Image.fromarray(img).save(out_dir / "images" / f"{name}.jpg", quality=95)
            w2c = np.eye(4)
            w2c[:3, :3] = V[:3, :3]
            w2c[:3, 3] = V[:3, 3] / scale - V[:3, :3] @ centre
            keep = np.ones_like(hole) if fill_holes else ~hole
            mask_path = out_dir / "images" / f"{name}_mask.png"
            Image.fromarray((keep * 255).astype(np.uint8)).save(mask_path)
            weight_path = out_dir / "images" / f"{name}_weight.png"
            Image.fromarray((rm * 255).astype(np.uint8)).save(weight_path)  # real pixels: full weight
            new_cams.append({"name": f"dense/{name}.jpg", "path": str(out_dir / "images" / f"{name}.jpg"),
                             "pano": f"dense-{label}", "width": frame_w, "height": frame_h, "K": K, "w2c": w2c,
                             "split": "train", "kind": "generated", "mask": str(mask_path),
                             "weight": str(weight_path)})
        if ci < 6:
            pick = (0, 24, 48)
            sheet = np.concatenate([np.concatenate([splat_frames[i], frames[i],
                                                    np.where(holes[i][..., None], [255, 0, 255], frames[i]).astype(np.uint8)], 1)
                                    for i in pick], 0)
            Image.fromarray(sheet).resize((frame_w * 3 // 2, frame_h * len(pick) // 2)).save(
                out_dir / "sheet" / f"{label}.jpg", quality=88)
        log(f"[densify] {ci + 1}/{len(clips)} clips, real px {np.mean(real_frac[-49:]):.2f}, "
            f"holes {hole_frac:.3f}, {(time.time() - t0) / 60:.1f} min")
    Image.fromarray(np.zeros((frame_h, frame_w), np.uint8)).save(out_dir / "images" / "zero.png")
    rep = {"clips": len(clips), "frames_added": len(new_cams), "frame_size": [frame_w, frame_h],
           "real_pixel_fraction_median": float(np.median(real_frac)) if real_frac else None,
           "generated_pixel_fraction_mean": float(np.mean(gen_frac)) if gen_frac else 0.0,
           "sharpness_splat_render_median": float(np.median(crisp_splat)) if crisp_splat else None,
           "sharpness_dense_frame_median": float(np.median(crisp_ibr)) if crisp_ibr else None,
           "metres_per_norm_unit": m_per_norm, "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "densify_report.json").write_text(json.dumps(rep, indent=2))
    scene_io.write_scene(out_dir / "scene", cameras + new_cams, points, {**meta, "densify": rep})
    return rep


# ------------------------------------------------------- image-based composite from real panoramas
class RealViews:
    """All real rig views: full-resolution colour in RAM, aligned depth for visibility checks."""

    def __init__(self, cameras, device: str = "cuda", render=None, centre=None, scale: float = 1.0,
                 depth_w: int = 640):
        """render: splat renderer (offpath.make_renderer). When given, the visibility depth of every real
        view is rendered from the splats, so it agrees with the novel view's splat depth; per-view MoGe
        depth (log-std ~0.2 between views) rejected nearly every pixel at a 6% tolerance on H8."""
        from PIL import Image

        self.cams = [c for c in cameras if c.get("kind") != "generated" and (render is not None or c.get("depth"))]
        self.device = device
        self.imgs, self.depths = [], []
        for c in self.cams:
            self.imgs.append(np.asarray(Image.open(c["path"]).convert("RGB")))
            if render is not None:
                s_ = depth_w / c["width"]
                dh = max(1, round(c["height"] * s_))
                K = np.array(c["K"], dtype=np.float64).copy()
                K[:2] *= s_
                w2c = np.asarray(c["w2c"], dtype=np.float64)
                V = np.eye(4)
                V[:3, :3] = w2c[:3, :3]
                V[:3, 3] = (w2c[:3, :3] @ np.asarray(centre) + w2c[:3, 3]) * scale
                _, alpha, dn = render(V, K, depth_w, dh, with_depth=True)
                self.depths.append(np.where(alpha > 0.5, dn / scale, 0.0).astype(np.float32))  # world units
            else:
                self.depths.append(np.load(c["depth"]).astype(np.float32))
        self.centres = np.stack([scene_io.camera_centre(np.asarray(c["w2c"])) for c in self.cams])

    def nearest(self, eye_world: np.ndarray, k: int = 36) -> list[int]:
        d = np.linalg.norm(self.centres - eye_world, axis=1)
        return list(np.argsort(d)[:k])


def ibr_composite(real: RealViews, V_n: np.ndarray, K: np.ndarray, w: int, h: int, depth_n: np.ndarray,
                  centre: np.ndarray, scale: float, k: int = 36, rel_tol: float = 0.06):
    """Return (rgb uint8, real_mask) for a novel camera from the best real view per pixel.

    depth_n: z-depth of the novel view in trainer units (rendered from the splats). Each pixel is
    lifted to 3D, projected into the k nearest real views, kept only where that view's own aligned
    depth agrees (visible, not occluded), and coloured from the view whose ray is closest in direction.
    """
    import torch
    import torch.nn.functional as F

    dev = real.device
    Rn, tn = V_n[:3, :3], V_n[:3, 3]
    eye_n = -Rn.T @ tn
    eye_w = eye_n / scale + centre
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    z = torch.tensor(depth_n, dtype=torch.float32, device=dev)
    cam = torch.stack([(xs + 0.5 - Kt[0, 2]) / Kt[0, 0] * z, (ys + 0.5 - Kt[1, 2]) / Kt[1, 1] * z, z], -1)
    Xn = (cam - torch.tensor(tn, dtype=torch.float32, device=dev)) @ torch.tensor(Rn, dtype=torch.float32, device=dev)
    Xw = Xn / scale + torch.tensor(centre, dtype=torch.float32, device=dev)
    ray_n = F.normalize(Xw - torch.tensor(eye_w, dtype=torch.float32, device=dev), dim=-1)
    best = torch.full((h, w), -2.0, device=dev)
    out = torch.zeros((h, w, 3), device=dev)
    valid_px = z > 0
    for i in real.nearest(eye_w, k):
        c = real.cams[i]
        w2c = torch.tensor(np.asarray(c["w2c"]), dtype=torch.float32, device=dev)
        Xc = Xw @ w2c[:3, :3].T + w2c[:3, 3]
        zc = Xc[..., 2]
        img = real.imgs[i]
        Hc, Wc = img.shape[:2]
        Kc = np.array(c["K"], dtype=np.float64).copy()
        Kc[:2] *= Wc / c["width"]
        Kc = torch.tensor(Kc, dtype=torch.float32, device=dev)
        u = Xc[..., 0] / zc.clamp_min(1e-6) * Kc[0, 0] + Kc[0, 2]
        v = Xc[..., 1] / zc.clamp_min(1e-6) * Kc[1, 1] + Kc[1, 2]
        ok = valid_px & (zc > 1e-4) & (u >= 0) & (u < Wc - 1) & (v >= 0) & (v < Hc - 1)
        if not bool(ok.any()):
            continue
        dmap = real.depths[i]
        dh, dw = dmap.shape
        D = torch.tensor(dmap, device=dev)
        du = (u * dw / Wc).long().clamp(0, dw - 1)
        dv = (v * dh / Hc).long().clamp(0, dh - 1)
        dz = D[dv, du]
        ok &= (dz > 0) & ((dz - zc).abs() < rel_tol * zc)
        cc = torch.tensor(scene_io.camera_centre(np.asarray(c["w2c"])), dtype=torch.float32, device=dev)
        ray_c = F.normalize(Xw - cc, dim=-1)
        score = torch.where(ok, (ray_c * ray_n).sum(-1), torch.full_like(zc, -2.0))
        better = score > best
        if not bool(better.any()):
            continue
        I = torch.tensor(img, device=dev).permute(2, 0, 1)[None].float()
        # u, v are continuous coordinates with pixel centres at +0.5; align_corners=True puts centres at integers
        grid = torch.stack([(u - 0.5) / (Wc - 1) * 2 - 1, (v - 0.5) / (Hc - 1) * 2 - 1], -1)[None]
        col = F.grid_sample(I, grid, mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        out = torch.where(better[..., None], col, out)
        best = torch.where(better, score, best)
    real_mask = (best > 0.97).cpu().numpy()  # ray directions within ~14 deg: parallax small enough to trust
    return out.clamp(0, 255).byte().cpu().numpy(), real_mask


class HoleFiller:
    """Wan2.1-VACE-1.3B masked video inpainting (T4 recipe from the video pipeline's fill_holes.py)."""

    def __init__(self, device: str = "cuda"):
        import gc

        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import UMT5EncoderModel

        mid = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        te = UMT5EncoderModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVACEPipeline.from_pretrained(mid, vae=vae, text_encoder=te, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        with torch.no_grad():
            self.pe, self.npe = pipe.encode_prompt(prompt=PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True,
                                                   device=torch.device("cpu"), dtype=torch.float32)
        pipe.text_encoder = None
        del te
        gc.collect()
        pipe.vae.to(device)
        try:
            pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001
            pass
        pipe.transformer.to(device)
        self.pipe, self.device = pipe, device

    def __call__(self, frames, holes, steps: int = 20, seed: int = 0):
        import torch
        from PIL import Image

        small = [np.asarray(Image.fromarray(f).resize((W, H), Image.LANCZOS)) for f in frames]
        hs = [np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127 for m in holes]
        vid = [Image.fromarray(np.where(m[..., None], 128, f).astype(np.uint8)) for f, m in zip(small, hs)]
        msk = [Image.fromarray(m.astype(np.uint8) * 255) for m in hs]
        with torch.no_grad():
            out = self.pipe(video=vid, mask=msk, prompt_embeds=self.pe.to(self.device, torch.float16),
                            negative_prompt_embeds=self.npe.to(self.device, torch.float16), height=H, width=W,
                            num_frames=len(vid), num_inference_steps=steps, guidance_scale=5.0,
                            generator=torch.Generator(self.device).manual_seed(seed), output_type="np").frames[0]
        out = np.nan_to_num(np.asarray(out, np.float32))
        res = []
        for f, m, o in zip(frames, holes, out):
            big = np.asarray(Image.fromarray((np.clip(o, 0, 1) * 255).astype(np.uint8)).resize(
                (f.shape[1], f.shape[0]), Image.LANCZOS))
            res.append(np.where(m[..., None], big, f).astype(np.uint8))   # only hole pixels are generated
        return res
