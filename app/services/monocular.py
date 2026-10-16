from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from scipy.optimize import least_squares

from app.config import PROJECT_ROOT, Settings
from app.services.geometry import GeometryError, build_collision_candidate


def _robust_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(source) & np.isfinite(target) & (source > 0) & (target > 0)
    source, target = source[valid], target[valid]
    if len(source) < 3:
        raise GeometryError("Too few valid sparse observations to align monocular depth.")
    matrix = np.column_stack([source, np.ones_like(source)])
    initial = np.linalg.lstsq(matrix, target, rcond=None)[0]
    scale = max(float(np.median(np.abs(target - np.median(target)))), 1e-3)
    result = least_squares(
        lambda parameters: parameters[0] * source + parameters[1] - target,
        initial,
        loss="soft_l1",
        f_scale=scale,
    )
    predicted = result.x[0] * source + result.x[1]
    relative_error = float(np.median(np.abs(predicted - target) / np.maximum(target, 1e-6)))
    return float(result.x[0]), float(result.x[1]), relative_error


def fit_depth_alignment(predicted: np.ndarray, geometric: np.ndarray) -> dict:
    """Choose direct-depth or inverse-depth affine alignment from sparse observations."""
    direct = _robust_affine(predicted, geometric)
    inverse = _robust_affine(1.0 / np.maximum(predicted, 1e-6), geometric)
    mode, result = min((("direct", direct), ("inverse", inverse)), key=lambda item: item[1][2])
    return {"mode": mode, "scale": result[0], "shift": result[1], "median_relative_error": result[2]}


def _apply_alignment(depth: np.ndarray, alignment: dict) -> np.ndarray:
    values = depth if alignment["mode"] == "direct" else 1.0 / np.maximum(depth, 1e-6)
    return alignment["scale"] * values + alignment["shift"]


def _camera_distortion(camera: pycolmap.Camera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=float)
    if camera.model_name == "SIMPLE_RADIAL":
        return np.asarray([params[3], 0.0, 0.0, 0.0, 0.0])
    if camera.model_name == "RADIAL":
        return np.asarray([params[3], params[4], 0.0, 0.0, 0.0])
    if camera.model_name == "OPENCV":
        return np.asarray([params[4], params[5], params[6], params[7], 0.0])
    return np.zeros(5)


def _image_observations(
    reconstruction: pycolmap.Reconstruction,
    image: pycolmap.Image,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = image.cam_from_world().matrix()
    predicted, geometric = [], []
    height, width = depth.shape
    for point in image.points2D:
        if not point.has_point3D():
            continue
        x, y = np.rint(point.xy).astype(int)
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        world = np.asarray(reconstruction.points3D[point.point3D_id].xyz)
        camera_point = transform[:, :3] @ world + transform[:, 3]
        value = float(depth[y, x])
        if camera_point[2] > 0 and np.isfinite(value) and value > 0:
            predicted.append(value)
            geometric.append(float(camera_point[2]))
    return np.asarray(predicted), np.asarray(geometric)


def _run_predictor(frames_dir: Path, output_dir: Path, log_path: Path, settings: Settings) -> dict:
    template = settings.monocular_depth_command
    if not template:
        raise GeometryError(
            "No monocular depth predictor is configured. Set MONOCULAR_DEPTH_COMMAND with "
            "{images} and {output} placeholders."
        )
    command_text = template.format(images=str(frames_dir), output=str(output_dir))
    command = shlex.split(command_text, posix=os.name != "nt")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.setdefault("HF_HOME", str(PROJECT_ROOT / "tmp" / "huggingface"))
    environment.setdefault("TORCH_HOME", str(PROJECT_ROOT / "tmp" / "torch-cache"))
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise GeometryError(
            f"Monocular depth predictor failed with exit code {result.returncode}. Log tail: {tail}"
        )
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GeometryError("Monocular predictor completed without a manifest.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_monocular_geometry(
    frames_dir: Path,
    sparse_model_dir: Path,
    output_dir: Path,
    log_path: Path,
    settings: Settings,
) -> dict:
    depth_dir = output_dir / "monocular-depth"
    if depth_dir.exists():
        shutil.rmtree(depth_dir)
    predictor_manifest = _run_predictor(frames_dir, depth_dir, log_path, settings)
    reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))

    aligned: dict[str, tuple[np.ndarray, dict]] = {}
    all_predicted, all_geometric = [], []
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        path = depth_dir / f"{Path(image.name).stem}.npy"
        if not path.is_file():
            continue
        depth = np.load(path, allow_pickle=False)
        predicted, geometric = _image_observations(reconstruction, image, depth)
        raw[image.name] = (depth, predicted, geometric)
        if len(predicted):
            all_predicted.append(predicted)
            all_geometric.append(geometric)
    if not all_predicted:
        raise GeometryError("No sparse COLMAP points overlap the predicted depth maps.")
    global_alignment = fit_depth_alignment(np.concatenate(all_predicted), np.concatenate(all_geometric))

    clouds = []
    frame_reports = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        if image.name not in raw:
            continue
        depth, predicted, geometric = raw[image.name]
        alignment = global_alignment
        if len(predicted) >= settings.monocular_min_alignment_points:
            candidate = fit_depth_alignment(predicted, geometric)
            if candidate["median_relative_error"] <= settings.monocular_max_alignment_error:
                alignment = candidate
        metric_depth = _apply_alignment(depth, alignment).astype(np.float32)
        positive_sparse = geometric[geometric > 0]
        if len(positive_sparse):
            near = max(float(np.percentile(positive_sparse, 2)) * 0.5, 1e-3)
            far = float(np.percentile(positive_sparse, 98)) * 1.75
        else:
            near, far = 1e-3, float(np.nanpercentile(metric_depth, 95))
        metric_depth[(metric_depth < near) | (metric_depth > far)] = 0

        camera = reconstruction.cameras[image.camera_id]
        calibration = np.asarray(camera.calibration_matrix(), dtype=float)
        color = cv2.imread(str(frames_dir / image.name), cv2.IMREAD_COLOR)
        if color is None:
            continue
        distortion = _camera_distortion(camera)
        metric_depth = cv2.undistort(metric_depth, calibration, distortion)
        color = cv2.undistort(color, calibration, distortion)
        stride = settings.monocular_pixel_stride
        ys, xs = np.mgrid[0 : metric_depth.shape[0] : stride, 0 : metric_depth.shape[1] : stride]
        z = metric_depth[::stride, ::stride]
        valid = np.isfinite(z) & (z > 0)
        x = (xs[valid] - calibration[0, 2]) * z[valid] / calibration[0, 0]
        y = (ys[valid] - calibration[1, 2]) * z[valid] / calibration[1, 1]
        camera_points = np.column_stack([x, y, z[valid]])
        transform = image.cam_from_world().matrix()
        world_points = (transform[:, :3].T @ (camera_points - transform[:, 3]).T).T
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world_points))
        colors = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)[::stride, ::stride][valid] / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)
        clouds.append(cloud)
        frame_reports.append({"image": image.name, "sparse_samples": len(predicted), **alignment})
    if not clouds:
        raise GeometryError("No aligned monocular point clouds were generated.")
    fused = clouds[0]
    for cloud in clouds[1:]:
        fused += cloud
    fused = fused.voxel_down_sample(settings.monocular_fusion_voxel_size)
    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    fused_path = dense_dir / "fused-monocular.ply"
    o3d.io.write_point_cloud(str(fused_path), fused, write_ascii=False)
    artifact = build_collision_candidate(
        fused_path,
        output_dir / "collision",
        settings,
        source_description="../dense/fused-monocular.ply",
    )
    artifact["depth_source"] = predictor_manifest.get("model", "unknown")
    artifact["depth_license"] = predictor_manifest.get("license")
    artifact["alignment"] = {
        "global": global_alignment,
        "frames": frame_reports,
    }
    artifact["warnings"].append(
        "Relative monocular depth was aligned to sparse COLMAP observations and may drift between views."
    )
    (output_dir / "collision" / "manifest.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    return artifact
