"""H9: dense generated views along walkable paths (Marble's densification step, sized for a T4).

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
                strength: float = 0.35, step_m: float = 1.0, lane_m: float = 1.5, walk_eye_m: float = 1.6,
                camera_height_m: float | None = None, manifest: dict | None = None, log=print) -> dict:
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
    f = W / 2 / math.tan(math.radians(40))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    refiner = Refiner(device)
    new_cams, before, after = [], [], []
    t0 = time.time()
    for ci, (label, mats) in enumerate(clips):
        frames, alphas = [], []
        for V in mats:
            rgb, alpha, _ = render(V, K, W, H)
            frames.append(rgb)
            alphas.append(alpha)
        refined = refiner(frames, strength=strength, seed=ci)
        before += [sharpness(x) for x in frames[::8]]
        after += [sharpness(x) for x in refined[::8]]
        for fi, (V, img, a) in enumerate(zip(mats, refined, alphas)):
            if fi % 2:  # every other frame is plenty for training (~2 m baseline between kept frames x2 lanes)
                continue
            name = f"{label}_f{fi:02d}"
            Image.fromarray(img).save(out_dir / "images" / f"{name}.jpg", quality=94)
            w2c = np.eye(4)
            w2c[:3, :3] = V[:3, :3]
            w2c[:3, 3] = V[:3, 3] / scale - V[:3, :3] @ centre
            mask_path = out_dir / "images" / f"{name}_mask.png"
            Image.fromarray(((a > 0.5) * 255).astype(np.uint8)).save(mask_path)  # never train empty space from fill
            new_cams.append({"name": f"dense/{name}.jpg", "path": str(out_dir / "images" / f"{name}.jpg"),
                             "pano": f"dense-{label}", "width": W, "height": H, "K": K, "w2c": w2c,
                             "split": "train", "kind": "generated", "mask": str(mask_path),
                             "weight": str(out_dir / "images" / "zero.png")})
        if ci < 6:
            sheet = np.concatenate([np.concatenate([frames[i], refined[i]], 1) for i in (0, 24, 48)], 0)
            Image.fromarray(sheet).resize((W, int(H * 1.5))).save(out_dir / "sheet" / f"{label}.jpg", quality=88)
        log(f"[densify] {ci + 1}/{len(clips)} clips, {(time.time() - t0) / 60:.1f} min")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(out_dir / "images" / "zero.png")
    rep = {"clips": len(clips), "frames_added": len(new_cams), "strength": strength,
           "sharpness_render_median": float(np.median(before)) if before else None,
           "sharpness_refined_median": float(np.median(after)) if after else None,
           "metres_per_norm_unit": m_per_norm, "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "densify_report.json").write_text(json.dumps(rep, indent=2))
    scene_io.write_scene(out_dir / "scene", cameras + new_cams, points, {**meta, "densify": rep})
    return rep
