from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import trimesh

from app.config import Settings
from app.services.geometry import GeometryError, read_collision_manifest


class NavigationPreparationError(RuntimeError):
    pass


def _camera_centers_and_up(reconstruction: pycolmap.Reconstruction) -> tuple[np.ndarray, np.ndarray]:
    centers, ups = [], []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        centers.append(np.asarray(image.cam_from_world().inverse().translation, dtype=float))
        ups.append(image.cam_from_world().rotation.matrix().T @ np.asarray([0.0, -1.0, 0.0]))
    if len(centers) < 3:
        raise NavigationPreparationError("At least three registered camera poses are required.")
    up = np.mean(ups, axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    return np.asarray(centers), up


def detect_floor_plane(
    fused_path: Path,
    sparse_model_dir: Path,
    settings: Settings,
) -> dict:
    cloud = o3d.io.read_point_cloud(str(fused_path), remove_nan_points=True, remove_infinite_points=True)
    if len(cloud.points) < settings.navigation_min_floor_points:
        raise NavigationPreparationError("Fused geometry has too few points for floor detection.")
    reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
    camera_centers, scene_up = _camera_centers_and_up(reconstruction)
    working = cloud
    points = np.asarray(cloud.points)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    threshold = max(diagonal * settings.navigation_floor_distance_ratio, 1e-4)
    candidates = []
    for _ in range(settings.navigation_floor_candidate_count):
        if len(working.points) < settings.navigation_min_floor_points:
            break
        plane, indices = working.segment_plane(
            distance_threshold=threshold,
            ransac_n=3,
            num_iterations=settings.navigation_floor_ransac_iterations,
        )
        if len(indices) < settings.navigation_min_floor_points:
            break
        normal = np.asarray(plane[:3], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-9)
        offset = float(plane[3])
        if float(np.median(camera_centers @ normal + offset)) < 0:
            normal = -normal
            offset = -offset
        camera_distance = camera_centers @ normal + offset
        alignment = abs(float(np.dot(normal, scene_up)))
        candidates.append(
            {
                "normal": normal.tolist(),
                "offset": offset,
                "inlier_count": len(indices),
                "inlier_ratio": len(indices) / len(cloud.points),
                "camera_median_distance": float(np.median(camera_distance)),
                "normal_alignment": alignment,
                "score": float(len(indices) * alignment * max(np.median(camera_distance), 0.0)),
            }
        )
        working = working.select_by_index(indices, invert=True)
    if not candidates:
        raise NavigationPreparationError("No candidate floor plane was found.")
    floor = max(candidates, key=lambda candidate: candidate["score"])
    floor["confidence"] = float(
        min(1.0, floor["normal_alignment"])
        * min(1.0, floor["inlier_ratio"] / settings.navigation_floor_min_inlier_ratio)
    )
    floor["accepted"] = floor["confidence"] >= settings.navigation_min_floor_confidence
    return {
        "plane": floor,
        "candidates": candidates,
        "scene_up": scene_up.tolist(),
        "camera_count": len(camera_centers),
        "distance_threshold": threshold,
    }


def prepare_navigation_geometry(
    geometry_dir: Path,
    sparse_model_dir: Path,
    settings: Settings,
) -> dict:
    manifest = read_collision_manifest(geometry_dir)
    source = (geometry_dir / manifest["source_dense_cloud"]).resolve()
    # Accept artifacts written by the pre-navigation geometry schema, which
    # used ../dense/... while the dense directory is inside geometry/.
    if not source.is_file():
        legacy_source = geometry_dir / "dense" / Path(manifest["source_dense_cloud"]).name
        if legacy_source.is_file():
            source = legacy_source.resolve()
    if not source.is_relative_to(geometry_dir.resolve()) or not source.is_file():
        raise NavigationPreparationError("Collision manifest points to a missing fused geometry source.")
    result = detect_floor_plane(source, sparse_model_dir, settings)
    result.update(
        {
            "schema_version": 1,
            "status": "floor_review_required",
            "scale_calibrated": bool(manifest.get("scale_calibrated", False)),
            "unknown_space_policy": "block_unknown",
            "navmesh_ready": False,
            "reason": "Scale, floor orientation, unknown-space mask and human review are required before NavMesh generation.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    destination = geometry_dir / "navigation" / "preparation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def calibrate_collision_scale(
    geometry_dir: Path,
    known_distance_meters: float,
    observed_distance_units: float,
    reference: str,
) -> dict:
    if known_distance_meters <= 0 or observed_distance_units <= 0:
        raise NavigationPreparationError("Both calibration distances must be positive.")
    manifest = read_collision_manifest(geometry_dir)
    model_path = geometry_dir / "collision" / manifest["model_path"]
    scale = known_distance_meters / observed_distance_units
    loaded = trimesh.load(str(model_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    mesh.apply_scale(scale)
    mesh.export(model_path)
    preview_path = geometry_dir / "collision" / manifest.get("preview_path", "")
    if preview_path.is_file():
        preview = o3d.io.read_triangle_mesh(str(preview_path))
        preview.scale(scale, center=(0, 0, 0))
        o3d.io.write_triangle_mesh(str(preview_path), preview, write_ascii=False)
    manifest.update(
        {
            "coordinate_units": "meters",
            "scale_calibrated": True,
            "scale_factor_meters_per_unit": scale,
            "scale_reference": reference,
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256(model_path.read_bytes()).hexdigest(),
            "bounds_min": mesh.bounds[0].tolist(),
            "bounds_max": mesh.bounds[1].tolist(),
            "voxel_size": float(manifest.get("voxel_size", 0.0)) * scale,
            "safe_for_navigation": False,
            "status": "review_required",
        }
    )
    manifest.setdefault("warnings", []).append(
        "Scale was calibrated, but floor orientation and unknown-space review are still required."
    )
    (geometry_dir / "collision" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
