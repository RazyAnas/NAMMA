"""Convert equirectangular Street View panoramas into overlapping perspective images.

Input bundle:
  manifest.json
  panoramas/<pano-id>.jpg

Output:
  images/p00000_v00.jpg ...
  views_manifest.json

This module is intentionally independent of the video capture/frame-extraction code.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

EARTH_RADIUS_M = 6_378_137.0


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see panorama_v1/README.md for the bundle schema.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version", 1) != 1:
        raise ValueError("Only manifest schema_version=1 is supported.")
    panoramas = data.get("panoramas")
    if not isinstance(panoramas, list) or len(panoramas) < 8:
        raise ValueError("At least 8 panoramas are required for a useful multi-view reconstruction.")
    ids = set()
    for i, pano in enumerate(panoramas):
        pano_id = str(pano.get("pano_id") or f"pano-{i:05d}")
        if pano_id in ids:
            raise ValueError(f"Duplicate pano_id: {pano_id}")
        ids.add(pano_id)
        rel = Path(str(pano.get("image") or ""))
        if not str(rel) or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Panorama {pano_id} has an unsafe or missing image path.")
        full = (root / rel).resolve()
        if root.resolve() not in full.parents or not full.is_file():
            raise FileNotFoundError(f"Image for panorama {pano_id} does not exist: {rel}")
        pano["pano_id"] = pano_id
        pano["image"] = rel.as_posix()
        pano.setdefault("sequence", i)
        if (pano.get("lat") is None) != (pano.get("lon") is None):
            raise ValueError(f"Panorama {pano_id}: lat and lon must either both be present or both be absent.")
        if pano.get("lat") is not None:
            lat, lon = float(pano["lat"]), float(pano["lon"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"Panorama {pano_id}: invalid GPS coordinate.")
    data["panoramas"] = sorted(panoramas, key=lambda p: (p.get("sequence", 0), str(p["pano_id"])))\n    return data


def perspective_maps(width: int, height: int, source_width: int, source_height: int,
                     yaw_deg: float, hfov_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Create remap arrays for one pinhole view sampled from a 360x180 equirectangular image."""
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    cx, cy = width / 2.0, height / 2.0
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - cx) / focal
    y = -(v - cy) / focal
    z = np.ones_like(x)
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx = c * x + s * z
    dy = y
    dz = -s * x + c * z
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    longitude = np.arctan2(dx, dz)
    latitude = np.arcsin(np.clip(dy, -1.0, 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * source_width) % source_width
    map_y = (0.5 - latitude / math.pi) * source_height
    return map_x.astype(np.float32), np.clip(map_y, 0, source_height - 1).astype(np.float32), focal


def make_views(root: Path, output: Path, image_size: int = 1280, view_count: int = 12,
               hfov_deg: float = 60.0, max_panoramas: int = 200) -> dict[str, Any]:
    manifest = load_manifest(root)
    panos = manifest["panoramas"][:max_panoramas]
    if len(panos) < 8:
        raise ValueError("Fewer than 8 panoramas remain after applying max_panoramas.")
    width, height = int(image_size), int(round(image_size * 0.75))
    if width < 384 or width > 1536 or view_count < 6 or view_count > 24:
        raise ValueError("image_size must be 384..1536 and view_count must be 6..24.")
    out_images = output / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    for old in out_images.glob("p*_v*.jpg"):
        old.unlink()

    views: list[dict[str, Any]] = []
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    for pi, pano in enumerate(panos):
        source = root / pano["image"]
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read panorama: {source}")
        sh, sw = image.shape[:2]
        ratio = sw / max(sh, 1)
        if not 1.75 <= ratio <= 2.25:
            raise ValueError(f"{pano['pano_id']} is {sw}x{sh}; expected an equirectangular 2:1 panorama.")
        base_heading = float(pano.get("heading_deg", 0.0) or 0.0)
        for vi in range(view_count):
            yaw = base_heading + 360.0 * vi / view_count
            mx, my, _ = perspective_maps(width, height, sw, sh, yaw, hfov_deg)
            view = cv2.remap(image, mx, my, interpolation=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
            name = f"p{pi:05d}_v{vi:02d}.jpg"
            if not cv2.imwrite(str(out_images / name), view,
                               [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f"Failed to write perspective view {name}")
            views.append({
                "image": name, "pano_index": pi, "pano_id": pano["pano_id"],
                "sequence": int(pano.get("sequence", pi)), "view_index": vi,
                "yaw_deg": yaw, "lat": pano.get("lat"), "lon": pano.get("lon"),
                "altitude_m": pano.get("altitude_m"), "heading_deg": base_heading
            })
        if (pi + 1) % 10 == 0 or pi + 1 == len(panos):
            print(f"Prepared {pi + 1}/{len(panos)} panoramas ({len(views)} perspective views)", flush=True)

    report = {
        "schema_version": 1,
        "source_panoramas": len(panos),
        "perspective_views": len(views),
        "image_width": width,
        "image_height": height,
        "horizontal_fov_deg": hfov_deg,
        "view_count": view_count,
        "camera_model": "PINHOLE",
        "fx_initial": fx,
        "fy_initial": fy,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "views": views,
        "panoramas": panos,
        "scene_name": manifest.get("scene_name", "NAMMA panorama scene")
    }
    (output / "views_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=1280)
    p.add_argument("--view-count", type=int, default=12)
    p.add_argument("--hfov", type=float, default=60.0)
    p.add_argument("--max-panoramas", type=int, default=200)
    a = p.parse_args()
    rep = make_views(a.input, a.output, a.image_size, a.view_count, a.hfov, a.max_panoramas)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("views", "panoramas")}, indent=2))


if __name__ == "__main__":
    main()
