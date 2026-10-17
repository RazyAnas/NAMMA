"""Build a walking test scene from an RGB-D sequence with known poses.

Made for Augmented ICL-NUIM living room 1 (depth PNGs in mm, Open3D .log
camera-to-world trajectory, dense ground-truth point cloud in metres). The
depth maps are back-projected with depth-dependent noise and flying pixels,
roughly what monocular depth gives, then run through NAMMA's collision and
NavMesh code. The ground-truth cloud is meshed separately as the reference
surface the benchmark checks against.

Usage:
  python scripts/build_walk_scene_from_rgbd.py --depth DIR --trajectory traj.txt \
      --ground-truth livingroom.ply --output data/walk-icl-livingroom [--splat model.ply]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.services.geometry import build_collision_candidate  # noqa: E402
from app.services.navigation import (  # noqa: E402
    approve_navigation_mesh,
    build_navigation_mesh,
    calibrate_collision_scale,
)


def read_log_trajectory(path: Path) -> list[np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    poses = []
    for start in range(0, len(rows), 5):
        poses.append(np.array([[float(v) for v in row] for row in rows[start + 1:start + 5]]))
    return poses


def back_project(depth_dir: Path, poses, intrinsics, stride: int, rng, noise: float, flying: float):
    fx, fy, cx, cy = intrinsics
    clouds, centres = [], []
    for png in sorted(depth_dir.glob("*.png")):
        index = int(png.stem)
        depth = np.asarray(Image.open(png), dtype=np.float64)[::stride, ::stride] / 1000.0
        v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]] * stride
        valid = depth > 0.1
        z = depth[valid] * (1.0 + rng.normal(0.0, noise, valid.sum()))
        # Flying pixels: a few points pulled along the ray, as at depth edges.
        pull = rng.random(len(z)) < flying
        z[pull] *= rng.uniform(0.3, 0.95, pull.sum())
        x = (u[valid] - cx) / fx * z
        y = (v[valid] - cy) / fy * z
        pose = poses[index]
        points = np.column_stack([x, y, z]) @ pose[:3, :3].T + pose[:3, 3]
        clouds.append(points)
        centres.append(pose[:3, 3])
    return np.concatenate(clouds), np.asarray(centres)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splat", type=Path, help="Gaussian PLY in the same world frame, copied for the viewer")
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[525.0, 525.0, 319.5, 239.5])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    parser.add_argument("--depth-noise", type=float, default=0.01, help="relative depth noise (sigma)")
    parser.add_argument("--flying-pixels", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output.resolve()
    geometry = out / "geometry"
    (geometry / "dense").mkdir(parents=True, exist_ok=True)

    poses = read_log_trajectory(args.trajectory)
    points, centres = back_project(args.depth, poses, args.intrinsics, 4, rng, args.depth_noise, args.flying_pixels)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.01)
    o3d.io.write_point_cloud(str(geometry / "dense" / "fused.ply"), cloud)
    settings = Settings(_env_file=None)
    build_collision_candidate(geometry / "dense" / "fused.ply", geometry / "collision", settings,
                              source_description="dense/fused.ply")
    calibrate_collision_scale(geometry, 1.0, 1.0, "ground-truth metric poses")

    # Floor: the densest height slab low in the room along the known up axis.
    up = np.asarray(args.up, dtype=float)
    up /= np.linalg.norm(up)
    heights = np.asarray(cloud.points) @ up
    low = heights[heights < np.percentile(heights, 30)]
    histogram, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.02, 0.01))
    floor = float(edges[np.argmax(histogram)] + 0.005)
    (geometry / "navigation").mkdir(exist_ok=True)
    (geometry / "navigation" / "preparation.json").write_text(json.dumps({
        "status": "floor_review_required",
        "plane": {"normal": up.tolist(), "offset": -floor, "accepted": True, "confidence": 1.0},
    }), encoding="utf-8")
    navmesh = build_navigation_mesh(geometry, geometry, settings, camera_centers_source=centres,
                                    source_to_viewer=np.eye(4).tolist())
    approve_navigation_mesh(geometry, "rgbd-test", "Ground-truth poses and depth")

    # Reference surface for the benchmark: mesh the dense ground-truth cloud.
    gt = o3d.io.read_point_cloud(str(args.ground_truth))
    gt = gt.voxel_down_sample(0.01)
    gt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    gt.orient_normals_towards_camera_location(centres.mean(axis=0))
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(gt, depth=10)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, 0.05))
    reference = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    reference.export(out / "ground_truth.glb")
    reference.export(out / "ground_truth_obstacles.glb")
    if args.splat:
        shutil.copy(args.splat, out / "splat.ply")

    bounds = np.percentile(np.asarray(gt.points), [1, 99], axis=0)
    (out / "scene.json").write_text(json.dumps({
        "description": f"RGB-D walking scene from {args.depth}",
        "synthetic_room": False,
        "floor_height": floor,
        "room": {"bounds_min": bounds[0].tolist(), "bounds_max": bounds[1].tolist()},
        "world_to_source": np.eye(4).tolist(),
        "source_to_viewer": np.eye(4).tolist(),
        "point_count": int(len(cloud.points)),
    }, indent=2), encoding="utf-8")
    print(json.dumps({k: navmesh[k] for k in ("width", "height", "walkable_count", "component_count")}
                     | {"standable": navmesh["movement"]["standable_count"], "floor_y": floor}))


if __name__ == "__main__":
    main()
