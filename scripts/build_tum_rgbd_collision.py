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
        extrinsic = np.linalg.inv(pose_matrix(poses[pose_index]))
        volume.integrate(rgbd, intrinsic, extrinsic)
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
    manifest = build_collision_candidate(
        fused_path,
        args.output / "collision",
        settings,
        coordinate_units="meters",
        scale_calibrated=True,
        source_description="../dense/fused-rgbd.ply",
    )
    manifest["depth_source"] = "TUM RGB-D sensor depth"
    manifest["integrated_frames"] = len(associations)
    (args.output / "collision" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "associations.json").write_text(
        json.dumps(associations, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
