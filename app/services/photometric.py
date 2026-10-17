"""Training-free exposure, white-balance and flicker compensation from SfM tracks.

Every sparse 3D point is observed in several frames. In linear light each observation is
approximately `gain[frame, channel] * radiance[point, channel]`, so in log space
`log c = log g + log r` is a linear problem. We solve it with robust alternating least
squares and divide each frame by its gains before Gaussian training. The fixed point is
the average exposure of the capture (mean log gain = 0).

Inspired by gain compensation in panorama stitching (Brown & Lowe) and the per-image
appearance terms of NeRF-W / bilateral-grid splatting, but solved once on the CPU in
seconds instead of being learned during GPU training.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap

GAMMA = 2.2


def solve_gains(
    frame_index: np.ndarray,
    point_index: np.ndarray,
    log_colour: np.ndarray,
    frame_count: int,
    point_count: int,
    iterations: int = 30,
    huber: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log gains per frame x RGB, log radiance per point x RGB)."""
    gains = np.zeros((frame_count, 3))
    radiance = np.zeros((point_count, 3))
    weights = np.ones_like(log_colour)
    for _ in range(iterations):
        radiance = _weighted_mean(point_index, log_colour - gains[frame_index], weights, point_count)
        gains = _weighted_mean(frame_index, log_colour - radiance[point_index], weights, frame_count)
        gains -= gains.mean(axis=0)
        residual = np.abs(log_colour - gains[frame_index] - radiance[point_index])
        weights = np.minimum(1.0, huber / np.maximum(residual, 1e-9))
    return gains, radiance


def _weighted_mean(index: np.ndarray, values: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.stack(
        [
            np.bincount(index, weights[:, c] * values[:, c], size)
            / np.maximum(np.bincount(index, weights[:, c], size), 1e-9)
            for c in range(3)
        ],
        axis=1,
    )


def _linear(image_bgr: np.ndarray) -> np.ndarray:
    return (image_bgr[:, :, ::-1].astype(np.float32) / 255.0) ** GAMMA


def compensate_dataset(model_dir: Path, images_dir: Path, report_path: Path | None = None) -> dict:
    """Rewrite the images under `images_dir` with per-frame gains removed.

    `model_dir` is the COLMAP model whose image names match `images_dir`. Point colours in
    the model are replaced with the solved radiance, so splats start from consistent colours.
    """
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    image_ids = sorted(reconstruction.images)
    frame_of = {image_id: k for k, image_id in enumerate(image_ids)}
    point_ids = sorted(reconstruction.points3D)
    point_of = {point_id: k for k, point_id in enumerate(point_ids)}

    frames, points, colours = [], [], []
    for image_id in image_ids:
        image = reconstruction.images[image_id]
        pixels = cv2.imread(str(images_dir / image.name))
        if pixels is None:
            raise FileNotFoundError(f"Missing training image {image.name}")
        linear = _linear(cv2.blur(pixels, (3, 3)))
        height, width = linear.shape[:2]
        for observation in image.points2D:
            if not observation.has_point3D():
                continue
            x, y = int(round(observation.xy[0])), int(round(observation.xy[1]))
            if 0 <= x < width and 0 <= y < height:
                colour = linear[y, x]
                # Clipped or black samples carry no exposure information.
                if colour.max() < 0.97 and colour.min() > 0.003:
                    frames.append(frame_of[image_id])
                    points.append(point_of[observation.point3D_id])
                    colours.append(colour)

    if len(colours) < 100:
        return {"schema_version": 1, "applied": False, "reason": "Too few tracked colour samples.", "samples": len(colours)}

    frame_index, point_index = np.array(frames), np.array(points)
    log_colour = np.log(np.array(colours))
    gains, radiance = solve_gains(frame_index, point_index, log_colour, len(image_ids), len(point_ids))
    residual = float(np.median(np.abs(log_colour - gains[frame_index] - radiance[point_index])))

    per_frame = {}
    for image_id in image_ids:
        name = reconstruction.images[image_id].name
        gain = np.exp(gains[frame_of[image_id]])
        pixels = cv2.imread(str(images_dir / name))
        corrected = np.clip(_linear(pixels) / gain, 0.0, 1.0) ** (1.0 / GAMMA)
        cv2.imwrite(str(images_dir / name), (corrected[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        per_frame[name] = [round(float(value), 4) for value in gain]

    observed = np.bincount(point_index, minlength=len(point_ids)) > 0
    for point_id, k in point_of.items():
        if observed[k]:
            colour = np.clip(np.exp(radiance[k]), 0.0, 1.0) ** (1.0 / GAMMA)
            reconstruction.points3D[point_id].color = (colour * 255.0).astype(np.uint8)
    reconstruction.write(str(model_dir))

    stops = np.log2(np.exp(gains).mean(axis=1))
    report = {
        "schema_version": 1,
        "applied": True,
        "samples": len(colours),
        "median_residual_log": residual,
        "exposure_range_stops": float(stops.max() - stops.min()),
        "white_balance_spread": float(np.abs(gains - gains.mean(axis=1, keepdims=True)).max()),
        "gains_rgb": per_frame,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
