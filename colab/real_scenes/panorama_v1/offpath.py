"""Scorecard + off-path contact sheet for a trained panorama splat model.

The sheet is what a person checks: held-out panorama views (real photo next
to render) and novel cameras that no panorama occupied (midpoints between
panoramas, sideways steps, a raised view looking down the street).
Scorecard numbers are computed from the splats themselves.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import scene_io


def look_at(eye: np.ndarray, fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV world->camera matrix (x right, y down, z forward)."""
    f = fwd / np.linalg.norm(fwd)
    down = -(up - f * np.dot(up, f))
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    R = np.stack([right, down, f])
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def rotate_about(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(a) + np.cross(axis, v) * math.sin(a)
            + axis * np.dot(axis, v) * (1 - math.cos(a)))


def offpath_cameras(cameras, centre, scale, up, count: int = 10):
    """Novel cameras in the trainer's normalised frame."""
    pc = scene_io.pano_centres(cameras)
    names = sorted(pc)
    C = np.stack([(pc[n] - centre) * scale for n in names])
    if len(C) < 2:
        return []
    spacing = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    picks = np.linspace(0, len(C) - 2, num=min(count, len(C) - 1)).round().astype(int)
    cams = []
    for j, i in enumerate(picks):
        a, b = C[i], C[i + 1]
        mid = (a + b) / 2
        fwd = b - a
        fwd = fwd - up * np.dot(fwd, up)
        if np.linalg.norm(fwd) < 1e-9:
            continue
        fwd /= np.linalg.norm(fwd)
        side = np.cross(up, fwd)
        kind = j % 5
        if kind == 0:
            cams.append(("mid-forward", look_at(mid, fwd, up)))
        elif kind == 1:
            cams.append(("mid-left", look_at(mid, rotate_about(fwd, up, 90), up)))
        elif kind == 2:
            eye = mid + side * 0.6 * spacing
            cams.append(("side-step", look_at(eye, fwd, up)))
        elif kind == 3:
            eye = mid + up * 1.5 * spacing
            cams.append(("raised", look_at(eye, rotate_about(fwd, side, -25), up)))
        else:
            cams.append(("mid-back", look_at(mid, -fwd, up)))
        if j % 2 == 0:
            # where the capture vehicle stood: steeply down at the panorama position
            cams.append(("car-spot-down", look_at(a + up * 0.05 * spacing, rotate_about(fwd, side, 65), up)))
    return cams


def scorecard(params, radius_units: float = 10.0, grid=None) -> dict:
    import torch

    with torch.no_grad():
        means = params["means"].float()
        sc = torch.exp(params["scales"].float())
        op = torch.sigmoid(params["opacities"].float())
        r = torch.linalg.norm(means, dim=-1)
        ratio = sc.max(-1).values / sc.min(-1).values.clamp_min(1e-12)
        mid = sc.sort(-1).values[:, 1]
        needle = (ratio > 8) & (sc.max(-1).values > 4 * mid)
        return {
            "splats": int(len(means)),
            "anisotropy_p50": float(ratio.median()),
            "needle_fraction": float(needle.float().mean()),
            "opacity_below_0_05": float((op < 0.05).float().mean()),
            "floater_fraction_far": float((r > radius_units).float().mean()),
            "floater_fraction_far_visible": float(((r > radius_units) & (op > 0.2)).float().mean()),
            "floater_fraction_in_seen_empty_space": (float(grid.is_empty(means).float().mean())
                                                     if grid is not None else None),
        }


def make_renderer(ck: dict, device: str = "cuda"):
    """render(V, K, w, h, with_depth=False) -> (rgb uint8, alpha [H,W], depth [H,W] normalised or None)."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    params = {k: v.to(device) for k, v in ck["params"].items()}
    up = np.asarray(ck["up"])
    sh_deg = int(ck.get("sh_degree", 0))
    env = None if ck.get("env") is None else ck["env"].to(device)

    def render(V, K, w, h, with_depth=False):
        with torch.no_grad():
            colors = torch.cat([params["sh0"], params["shN"]], 1)
            img, alpha, _ = rasterization(
                params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), colors,
                torch.tensor(V, dtype=torch.float32, device=device)[None],
                torch.tensor(K, dtype=torch.float32, device=device)[None], w, h, sh_degree=sh_deg,
                render_mode="RGB+ED" if with_depth else "RGB")
            rgb = img[0, ..., :3]
            if env is not None:
                from train_splats import sky_colour
                ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
                Kt = torch.tensor(K, dtype=torch.float32, device=device)
                d = torch.stack([(xs + 0.5 - Kt[0, 2]) / Kt[0, 0], (ys + 0.5 - Kt[1, 2]) / Kt[1, 1],
                                 torch.ones_like(xs, dtype=torch.float32)], -1)
                d = F.normalize(d @ torch.tensor(V[:3, :3], dtype=torch.float32, device=device), dim=-1)
                upv = torch.tensor(up, dtype=torch.float32, device=device)
                helper = (torch.tensor([1.0, 0, 0], device=device) if abs(float(upv[0])) < 0.9
                          else torch.tensor([0, 0, 1.0], device=device))
                fwd = F.normalize(torch.cross(upv, helper, dim=0), dim=0)
                right = torch.cross(fwd, upv, dim=0)
                dirs = torch.stack([d @ right, d @ upv, d @ fwd], -1)
                rgb = rgb + (1 - alpha[0]) * sky_colour(env, dirs)
            out_rgb = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            a = alpha[0, ..., 0].cpu().numpy()
            depth = img[0, ..., 3].cpu().numpy() if with_depth else None
        return out_rgb, a, depth

    return render


def build_sheet(model_dir: Path, scene_dir: Path, out_png: Path, device: str = "cuda",
                width: int = 512, height: int = 384) -> dict:
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from PIL import Image, ImageDraw

    ck = torch.load(Path(model_dir) / "splats.pt", map_location=device, weights_only=False)
    params = {k: v.to(device) for k, v in ck["params"].items()}
    centre, scale, up = np.asarray(ck["centre"]), float(ck["scale"]), np.asarray(ck["up"])
    sh_deg = int(ck.get("sh_degree", 0))
    cameras, _, _ = scene_io.read_scene(scene_dir)

    render_full = make_renderer(ck, device)

    def render(V, K, w, h):
        return render_full(V, K, w, h)[0]

    tiles = []
    # held-out panorama views: photo | render
    val = [c for c in cameras if c["split"] == "val"]
    for cam in val[:: max(1, len(val) // 3)][:3]:
        photo = Image.open(cam["path"]).convert("RGB")
        s = width / photo.width
        photo = photo.resize((width, round(photo.height * s)))
        K = np.array(cam["K"], dtype=np.float64)
        K[:2] *= s
        w2c = np.asarray(cam["w2c"])
        V = np.eye(4)
        V[:3, :3] = w2c[:3, :3]
        V[:3, 3] = (w2c[:3, :3] @ centre + w2c[:3, 3]) * scale
        tiles.append(("held-out photo", np.asarray(photo)))
        tiles.append(("held-out render", render(V, K, photo.width, photo.height)))
    f = width / 2 / math.tan(math.radians(45))
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    crisp = []
    for label, V in offpath_cameras(cameras, centre, scale, up):
        img = render(V, K, width, height)
        tiles.append((label, img))
        import cv2

        crisp.append(float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var()))

    cols = 4
    rows = math.ceil(len(tiles) / cols)
    tile_h = max(t[1].shape[0] for t in tiles)
    sheet = Image.new("RGB", (cols * width, rows * (tile_h + 18)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, (label, arr) in enumerate(tiles):
        x, y = (k % cols) * width, (k // cols) * (tile_h + 18)
        sheet.paste(Image.fromarray(arr), (x, y + 18))
        draw.text((x + 4, y + 2), label, fill=(240, 240, 240))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, quality=90)
    grid = None
    if (Path(model_dir) / "carve.pt").is_file():
        import depth_memory
        grid = depth_memory.CarveGrid.load(Path(model_dir) / "carve.pt", device)
    card = scorecard(params, grid=grid)
    card["offpath_sharpness_median"] = float(np.median(crisp)) if crisp else None  # Laplacian variance
    return card


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    card = build_sheet(a.model, a.scene, a.out / "offpath_sheet.jpg")
    (a.out / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print("[offpath] scorecard", json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
