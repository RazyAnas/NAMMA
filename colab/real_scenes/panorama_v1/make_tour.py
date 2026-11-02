"""Build a route-oriented NAMMA tour plan from the panorama SfM camera model.

The Gaussian model and camera poses come from gsplat's COLMAP Parser, so they share
exactly the same normalization. The output plan is consumed by the existing static
SuperSplat tour packager as a read-only viewer utility; no video pipeline stage runs.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def gps_to_enu(panoramas: list[dict]) -> tuple[dict[int, np.ndarray], tuple[float, float] | None]:
    good = [(i, p) for i, p in enumerate(panoramas) if p.get("lat") is not None and p.get("lon") is not None]
    if not good:
        return {}, None
    lat0, lon0 = float(good[0][1]["lat"]), float(good[0][1]["lon"])
    lat0r = math.radians(lat0)
    out = {}
    for i, p in good:
        lat, lon = math.radians(float(p["lat"])), math.radians(float(p["lon"]))
        east = 6_378_137.0 * math.cos(lat0r) * (lon - math.radians(lon0))
        north = 6_378_137.0 * (lat - lat0r)
        out[i] = np.array([east, north], dtype=np.float64)
    return out, (lat0, lon0)


def fit_similarity_2d(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares row-vector similarity: target ~= scale * source @ R + t."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("At least 3 paired 2D camera/GPS locations are required.")
    sx, sy = source.mean(0), target.mean(0)
    x, y = source - sx, target - sy
    u, singular, vt = np.linalg.svd(x.T @ y)
    fix = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        fix[-1, -1] = -1
    q = u @ fix @ vt
    scale = float(np.sum(singular * np.diag(fix)) / max(np.sum(x * x), 1e-12))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not estimate a positive GPS scale.")
    t = sy - scale * (sx @ q)
    return scale, q, t


def robust_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    keep = np.ones(len(source), dtype=bool)
    for _ in range(5):
        scale, q, t = fit_similarity_2d(source[keep], target[keep])
        residual = np.linalg.norm(scale * source @ q + t - target, axis=1)
        values = residual[keep]
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        threshold = max(3.0, med + 3.0 * 1.4826 * mad)
        updated = residual <= threshold
        if updated.sum() < 3 or np.array_equal(updated, keep):
            break
        keep = updated
    scale, q, t = fit_similarity_2d(source[keep], target[keep])
    residual = np.linalg.norm(scale * source @ q + t - target, axis=1)
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    return scale, q, t, keep, rms


def smooth_rows(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if len(values) < 3 or radius <= 0:
        return values.copy()
    radius = min(radius, max(1, (len(values) - 1) // 4))
    weights = np.exp(-0.5 * (np.arange(-2 * radius, 2 * radius + 1) / radius) ** 2)
    weights /= weights.sum()
    padded = np.pad(values, ((2 * radius, 2 * radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, k], weights, mode="valid") for k in range(values.shape[1])], axis=1)


def build_plan(data_dir: Path, run_dir: Path, tour_dir: Path, examples_dir: Path,
               camera_height_m: float = 2.2) -> dict:
    sys.path.insert(0, str(examples_dir))
    from datasets.colmap import Parser

    data_dir, run_dir, tour_dir = map(Path, (data_dir, run_dir, tour_dir))
    views_meta = json.loads((data_dir.parent / "views_manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((data_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    panoramas = source_manifest["panoramas"]
    parser = Parser(str(data_dir), factor=1, normalize=True, test_every=8)
    names = list(parser.image_names)
    cameras = np.asarray(parser.camtoworlds, dtype=np.float64)
    points = np.asarray(parser.points, dtype=np.float64)
    if len(names) != len(cameras) or len(cameras) < 30:
        raise RuntimeError(f"Tour export needs at least 30 registered views; found {len(cameras)}.")
    by_pano: dict[int, list[int]] = {}
    for idx, name in enumerate(names):
        match = re.match(r"p(\d{5})_v\d+\.jpg$", name)
        if match:
            by_pano.setdefault(int(match.group(1)), []).append(idx)
    pano_indices = sorted(by_pano, key=lambda i: int(panoramas[i].get("sequence", i)))
    if len(pano_indices) < 6:
        raise RuntimeError(f"Only {len(pano_indices)} panorama positions survived SfM; at least 6 are needed for a route.")

    # Each set of perspective crops shares a physical camera centre. Median aggregation
    # suppresses small per-view drift introduced by bundle adjustment.
    centres, pano_order = [], []
    for pi in pano_indices:
        ix = by_pano[pi]
        centres.append(np.median(cameras[ix, :3, 3], axis=0))
        pano_order.append(pi)
    centres = np.asarray(centres, dtype=np.float64)

    # Synthetic crops share a vertical axis. Estimate it from their camera-down vectors;
    # PCA of the camera centres is the fallback for unusual or heavily rolled captures.
    down = np.median(cameras[:, :3, 1], axis=0)
    up = -down
    if np.linalg.norm(up) < 1e-5:
        centered = centres - centres.mean(0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        up = vh[-1]
    up /= max(np.linalg.norm(up), 1e-9)
    if float(up @ (-np.mean(cameras[:, :3, 1], axis=0))) < 0:
        up = -up

    horizontal = centres - np.outer(centres @ up, up)
    centred_h = horizontal - horizontal.mean(0)
    _, _, vh = np.linalg.svd(centred_h, full_matrices=False)
    axis1 = vh[0] - up * float(vh[0] @ up)
    axis1 /= max(np.linalg.norm(axis1), 1e-9)
    axis2 = np.cross(axis1, up)
    axis2 /= max(np.linalg.norm(axis2), 1e-9)
    source_xy = np.stack([centres @ axis1, centres @ axis2], axis=1)

    # The average forward vectors of all synthetic views at one panorama cancel
    # (the views surround 360 degrees). Use the ordered physical pano centres to
    # form a stable route tangent instead.
    forwards = []
    for i in range(len(centres)):
        if i == 0:
            direction = centres[min(1, len(centres) - 1)] - centres[0]
        elif i == len(centres) - 1:
            direction = centres[-1] - centres[-2]
        else:
            direction = centres[i + 1] - centres[i - 1]
        direction = direction - up * float(direction @ up)
        if np.linalg.norm(direction) < 1e-8:
            direction = axis1.copy()
        forwards.append(direction / max(np.linalg.norm(direction), 1e-9))
    forwards = np.asarray(forwards, dtype=np.float64)

    gps, origin = gps_to_enu(panoramas)
    paired = [(k, pano_order.index(pi), gps[pi]) for k, pi in enumerate(pano_order) if pi in gps]
    gps_rms = None
    gps_inliers = 0
    if len(paired) >= 4:
        source_fit = np.stack([source_xy[k] for k, _, _ in paired])
        target_fit = np.stack([target for _, _, target in paired])
        scale, q, t2, inliers, gps_rms = robust_fit(source_fit, target_fit)
        gps_inliers = int(inliers.sum())
        # A wildly implausible fit usually means the SfM track collapsed or the metadata
        # coordinates refer to a different set of panoramas. Refuse to claim metric scale.
        if not (0.03 <= scale <= 500.0) or gps_inliers < 4:
            scale, q, t2, gps_rms, gps_inliers, origin = 1.0, np.eye(2), -source_xy.mean(0), None, 0, None
    else:
        scale, q, t2 = 1.0, np.eye(2), -source_xy.mean(0)

    # Build a right-handed world frame: X=east, Y=up, Z=north. q maps the
    # source horizontal coordinates to GPS ENU. Vertical translation puts camera
    # centres at the typical Street View capture height above the local road plane.
    rotation = np.zeros((3, 3), dtype=np.float64)
    rotation[0] = q[0, 0] * axis1 + q[1, 0] * axis2
    rotation[1] = up
    rotation[2] = q[0, 1] * axis1 + q[1, 1] * axis2
    # The transform is applied as (points @ R.T + translation) * scale.
    # GPS translation is already in metres, so convert it back to SfM units
    # before the final uniform scale; otherwise the offset is scaled twice.
    translation = np.array([
        t2[0] / scale,
        camera_height_m / scale - np.median(centres @ up),
        t2[1] / scale
    ], dtype=np.float64)

    transform_points = (points @ rotation.T + translation) * scale
    if len(transform_points):
        lo = np.percentile(transform_points, 1, axis=0)
        hi = np.percentile(transform_points, 99, axis=0)
    else:
        transformed_centres = (centres @ rotation.T + translation) * scale
        lo, hi = transformed_centres.min(0), transformed_centres.max(0)
    span = np.maximum(hi - lo, 1.0)
    lo -= 0.05 * span
    hi += 0.05 * span

    # The existing tour converter accepts an XYZ Euler rotation, translation and
    # uniform scale. Its viewer convention is preserved; only the source-to-world
    # similarity differs per panorama job.
    euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    plan = {
        "rotate": euler.tolist(),
        "translate": translation.tolist(),
        "scale": float(scale),
        "box": lo.tolist() + hi.tolist()
    }
    tour_dir.mkdir(parents=True, exist_ok=True)
    (tour_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # Convert all registered camera poses into the exact same world frame as the
    # transformed Gaussian model. Use one pose per panorama for the guided route.
    world_centres = (centres @ rotation.T + translation) * scale
    world_forwards = forwards @ rotation.T
    world_forwards /= np.maximum(np.linalg.norm(world_forwards, axis=1, keepdims=True), 1e-9)
    route_dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(world_centres, axis=0), axis=1))]
    duration = max(10.0, float(route_dist[-1] / 5.0))
    fov = 60.0
    look = max(4.0, min(20.0, float(np.linalg.norm(hi - lo)) * 0.035))

    def pose(i: int) -> dict:
        return {
            "position": [round(float(x), 4) for x in world_centres[i]],
            "target": [round(float(x), 4) for x in world_centres[i] + look * world_forwards[i]],
            "fov": fov
        }

    key_ids = np.unique(np.linspace(0, len(world_centres) - 1, min(len(world_centres), 120)).astype(int))
    route_smooth = smooth_rows(world_centres, 2)
    forward_smooth = smooth_rows(world_forwards, 2)
    forward_smooth /= np.maximum(np.linalg.norm(forward_smooth, axis=1, keepdims=True), 1e-9)
    frames = [int(round((route_dist[k] / max(route_dist[-1], 1e-6)) * duration * 30)) for k in key_ids]
    track = {
        "name": "Street View route",
        "duration": round(duration, 2),
        "frameRate": 30,
        "loopMode": "repeat",
        "interpolation": "spline",
        "smoothness": 1,
        "keyframes": {
            "times": frames,
            "values": {
                "position": [float(x) for k in key_ids for x in route_smooth[k]],
                "target": [float(x) for k in key_ids for x in route_smooth[k] + look * forward_smooth[k]],
                "fov": [fov] * len(key_ids)
            }
        }
    }
    spots = np.linspace(0, len(world_centres) - 1, min(10, len(world_centres))).astype(int)
    annotations = []
    for n, i in enumerate(spots):
        annotations.append({
            "position": pose(i)["target"],
            "title": f"Street View {n + 1}",
            "text": "Reconstructed panorama viewpoint",
            "camera": {"initial": pose(i)}
        })
    settings = {
        "version": 2,
        "tonemapping": "none",
        "highPrecisionRendering": False,
        "background": {"color": [0.055, 0.06, 0.07]},
        "postEffectSettings": {
            "sharpness": {"enabled": True, "amount": 0.2},
            "bloom": {"enabled": False, "intensity": 0.1, "blurLevel": 2},
            "grading": {"enabled": False, "brightness": 1, "contrast": 1, "saturation": 1, "tint": [1, 1, 1]},
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5}
        },
        "animTracks": [track],
        "cameras": [{"initial": pose(0)}],
        "annotations": annotations,
        "startMode": "animTrack"
    }
    (tour_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    tour_info = {
        "pipeline": "panorama_v1",
        "panoramas_in_manifest": len(panoramas),
        "panoramas_registered": len(pano_order),
        "registered_perspective_views": len(names),
        "gps_aligned": origin is not None,
        "gps_origin_lat_lon": list(origin) if origin else None,
        "gps_inliers": gps_inliers,
        "gps_alignment_rms_m": round(float(gps_rms), 3) if gps_rms is not None else None,
        "metres_per_sfm_unit": round(float(scale), 6),
        "world_axes": "X=east, Y=up, Z=north" if origin else "Y=up; horizontal scale is arbitrary",
        "route_length_m": round(float(route_dist[-1]), 2) if origin else None,
        "walk_mode": False,
        "collision": False,
        "spots": len(annotations)
    }
    (tour_dir / "tour.json").write_text(json.dumps(tour_info, indent=2), encoding="utf-8")
    print("PANORAMA TOUR PLAN", json.dumps(tour_info), flush=True)
    return tour_info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tour", type=Path, required=True)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--camera-height", type=float, default=2.2)
    args = ap.parse_args()
    build_plan(args.data, args.run, args.tour, args.examples, args.camera_height)
