"""Fuse the downloaded TUM RGB-D benchmark into the collision artifact format.

This validates metric RGB-D fusion and mesh publication independently from the
COLMAP PatchMatch CUDA path. It is a benchmark utility, not a monocular fallback.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.services.geometry import build_collision_candidate
from app.services.navigation import (
    build_navigation_mesh,
    detect_floor_plane_from_observations,
    validate_navigation_mesh,
)


def read_index(path: Path) -> list[tuple[float, Path]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            timestamp, relative_path = line.split()[:2]
            rows.append((float(timestamp), Path(relative_path)))
    return rows


def read_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()[:8]])
    data = np.asarray(rows)
    return data[:, 0], data[:, 1:]


def nearest_index(timestamps: np.ndarray, target: float) -> int:
    position = int(np.searchsorted(timestamps, target))
    candidates = [max(0, position - 1), min(len(timestamps) - 1, position)]
    return min(candidates, key=lambda index: abs(timestamps[index] - target))


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    translation = pose[:3]
    qx, qy, qz, qw = pose[3:]
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = translation
    return camera_to_world


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=41)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--voxel-size", type=float, default=0.015)
    parser.add_argument("--navigation-agent-radius", type=float, default=0.15)
    parser.add_argument("--navigation-safety-margin", type=float, default=0.05)
    parser.add_argument("--navigation-min-component-cells", type=int, default=5)
    args = parser.parse_args()

    rgb_rows = read_index(args.dataset / "rgb.txt")
    depth_rows = read_index(args.dataset / "depth.txt")
    pose_timestamps, poses = read_poses(args.dataset / "groundtruth.txt")
    rgb_timestamps = np.asarray([row[0] for row in rgb_rows])
    depth_timestamps = np.asarray([row[0] for row in depth_rows])

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.voxel_size * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3)
    associations = []
    camera_centers = []
    camera_ups = []
    for output_index in range(args.sample_count):
        video_time = output_index / args.sample_fps
        source_index = min(round(video_time * args.video_fps), len(rgb_rows) - 1)
        rgb_timestamp, rgb_relative = rgb_rows[source_index]
        depth_index = nearest_index(depth_timestamps, rgb_timestamp)
        pose_index = nearest_index(pose_timestamps, rgb_timestamp)
        if abs(depth_timestamps[depth_index] - rgb_timestamp) > 0.04:
            continue
        color = o3d.io.read_image(str(args.dataset / rgb_relative))
        depth = o3d.io.read_image(str(args.dataset / depth_rows[depth_index][1]))
        rgbd = o3d.geometry.RGBDImage.create_from_tum_format(color, depth, convert_rgb_to_intensity=False)
        camera_to_world = pose_matrix(poses[pose_index])
        extrinsic = np.linalg.inv(camera_to_world)
        volume.integrate(rgbd, intrinsic, extrinsic)
        camera_centers.append(camera_to_world[:3, 3])
        camera_ups.append(camera_to_world[:3, :3] @ np.asarray([0.0, -1.0, 0.0]))
        associations.append(
            {"rgb_timestamp": rgb_timestamp, "depth_timestamp": depth_timestamps[depth_index]}
        )

    cloud = volume.extract_point_cloud()
    if len(cloud.points) == 0:
        raise RuntimeError("TUM TSDF fusion produced no points.")
    dense_dir = args.output / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    fused_path = dense_dir / "fused-rgbd.ply"
    o3d.io.write_point_cloud(str(fused_path), cloud, write_ascii=False)
    settings = get_settings()
    # The Freiburg desk sequence observes only a narrow floor region. Keep this
    # benchmark profile explicit and separate from the production adult-agent
    # defaults used by the API.
    settings.agent_radius_meters = args.navigation_agent_radius
    settings.safety_margin_meters = args.navigation_safety_margin
    settings.navigation_min_floor_points_per_cell = 1
    settings.navigation_component_min_cells = args.navigation_min_component_cells
    manifest = build_collision_candidate(
        fused_path,
        args.output / "collision",
        settings,
        coordinate_units="meters",
        scale_calibrated=True,
        source_description="dense/fused-rgbd.ply",
    )
    manifest["depth_source"] = "TUM RGB-D sensor depth"
    manifest["integrated_frames"] = len(associations)
    (args.output / "collision" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "associations.json").write_text(
        json.dumps(associations, indent=2), encoding="utf-8"
    )
    scene_up = np.mean(np.asarray(camera_ups), axis=0)
    scene_up /= np.linalg.norm(scene_up)
    preparation = detect_floor_plane_from_observations(
        fused_path, np.asarray(camera_centers), scene_up, settings
    )
    preparation.update(
        {
            "schema_version": 1,
            "status": "floor_review_required",
            "scale_calibrated": True,
            "unknown_space_policy": "block_unknown",
            "navmesh_ready": False,
            "source": "TUM RGB-D metric benchmark",
        }
    )
    navigation_dir = args.output / "navigation"
    navigation_dir.mkdir(parents=True, exist_ok=True)
    (navigation_dir / "preparation.json").write_text(
        json.dumps(preparation, indent=2), encoding="utf-8"
    )
    navmesh = build_navigation_mesh(
        args.output,
        Path("unused-for-tum-benchmark"),
        settings,
        camera_centers_source=np.asarray(camera_centers),
    )
    validation = validate_navigation_mesh(args.output)
    benchmark = {
        "collision": manifest,
        "navigation": {
            "width": navmesh["width"],
            "height": navmesh["height"],
            "walkable_count": navmesh["walkable_count"],
            "blocked_obstacle_count": navmesh["blocked_obstacle_count"],
        },
        "validation": validation,
        "benchmark_navigation_profile": {
            "agent_radius_m": settings.agent_radius_meters,
            "safety_margin_m": settings.safety_margin_meters,
            "minimum_component_cells": settings.navigation_component_min_cells,
            "note": "TUM Freiburg desk has limited visible floor and is not the production adult-agent profile.",
        },
    }
    (args.output / "benchmark-report.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
