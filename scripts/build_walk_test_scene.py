"""Build a synthetic living room with known geometry for walking/collision tests.

The room contains the obstacles that break naive roaming: walls, a sofa, two
tables with thin legs, a 3 cm lamp pole, a 15 cm raised floor, a staircase, a
ramp and a glass coffee table. Ground-truth geometry is exported alongside a
simulated monocular reconstruction (noise, missing thin parts, a textureless
wall, floaters) that is then run through NAMMA's real collision and NavMesh
code, in an arbitrary COLMAP-like frame, so the viewer transforms are exercised
end to end.

Usage: python scripts/build_walk_test_scene.py --output data/walk-test-room
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.services.geometry import build_collision_candidate  # noqa: E402
from app.services.navigation import (  # noqa: E402
    approve_navigation_mesh,
    build_navigation_mesh,
    calibrate_collision_scale,
)

ROOM_X, ROOM_Z, CEILING = 7.0, 6.0, 2.6


def box(x0, x1, y0, y1, z0, z1, tag):
    mesh = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    mesh.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    mesh.metadata["tag"] = tag
    return mesh


def legs(x0, x1, z0, z1, height, size, tag):
    inset = size
    return [
        box(x - size / 2, x + size / 2, 0.0, height, z - size / 2, z + size / 2, tag)
        for x in (x0 + inset, x1 - inset)
        for z in (z0 + inset, z1 - inset)
    ]


def ground_truth_parts() -> list[trimesh.Trimesh]:
    t = 0.1
    parts = [
        box(-t, ROOM_X + t, -t, 0.0, -t, ROOM_Z + t, "floor"),
        box(-t, ROOM_X + t, CEILING, CEILING + t, -t, ROOM_Z + t, "ceiling"),
        box(-t, 0.0, 0.0, CEILING, -t, ROOM_Z + t, "wall"),
        box(ROOM_X, ROOM_X + t, 0.0, CEILING, -t, 1.5, "wall"),
        box(ROOM_X, ROOM_X + t, 0.0, CEILING, 1.5, 4.0, "blank_wall"),
        box(ROOM_X, ROOM_X + t, 0.0, CEILING, 4.0, ROOM_Z + t, "wall"),
        box(-t, ROOM_X + t, 0.0, CEILING, -t, 0.0, "wall"),
        box(-t, ROOM_X + t, 0.0, CEILING, ROOM_Z, ROOM_Z + t, "wall"),
        box(0.3, 2.3, 0.0, 0.85, 0.2, 1.1, "sofa"),
        # Dining table: top at 0.72-0.76 m on four 5 cm legs.
        box(4.5, 6.0, 0.72, 0.76, 0.5, 1.4, "table_top"),
        *legs(4.5, 6.0, 0.5, 1.4, 0.72, 0.05, "thin_leg"),
        # Glass coffee table: the top is invisible to a camera.
        box(2.8, 3.8, 0.40, 0.45, 2.0, 2.6, "glass_top"),
        *legs(2.8, 3.8, 2.0, 2.6, 0.40, 0.04, "thin_leg"),
        # Floor lamp: 3 cm pole on a thin base.
        box(3.35, 3.65, 0.0, 0.03, 4.35, 4.65, "lamp_base"),
        box(3.485, 3.515, 0.03, 1.6, 4.485, 4.515, "thin_pole"),
        # Raised floor (15 cm) in one corner.
        box(5.2, ROOM_X, 0.0, 0.15, 3.5, ROOM_Z, "step"),
        # Three 16 cm stairs onto a landing.
        box(0.0, 1.0, 0.0, 0.16, 3.0, ROOM_Z, "stairs"),
        box(0.0, 1.0, 0.16, 0.32, 3.3, ROOM_Z, "stairs"),
        box(0.0, 1.0, 0.32, 0.48, 3.6, ROOM_Z, "stairs"),
        # 14 degree ramp up to a 30 cm plateau.
        box(2.7, 3.5, 0.0, 0.30, 5.0, ROOM_Z, "ramp"),
    ]
    ramp = trimesh.convex.convex_hull(
        np.array([
            [1.5, 0.0, 5.0], [1.5, 0.0, ROOM_Z], [2.7, 0.0, 5.0], [2.7, 0.0, ROOM_Z],
            [2.7, 0.30, 5.0], [2.7, 0.30, ROOM_Z],
        ])
    )
    ramp.metadata["tag"] = "ramp"
    parts.append(ramp)
    return parts


def simulate_reconstruction(parts, rng, floaters=150, density=2500.0, noise=0.01):
    """Sample a noisy point cloud the way monocular depth tends to fail."""
    clouds = []
    for part in parts:
        tag = part.metadata["tag"]
        if tag in {"blank_wall", "glass_top"}:
            continue  # textureless / transparent: no depth at all
        count = int(part.area * density)
        if tag in {"thin_leg", "thin_pole"}:
            count = int(count * 0.25)  # thin structures are under-reconstructed
        points, faces = trimesh.sample.sample_surface(part, max(count, 1), seed=int(rng.integers(1 << 31)))
        # Drop faces hidden inside or against another solid (box bottoms on
        # the floor, stacked stairs): a camera never sees them.
        probe = points + part.face_normals[faces] * 0.005
        hidden = np.zeros(len(points), dtype=bool)
        for other in parts:
            if other is not part:
                hidden |= other.contains(probe)
        points = points[~hidden]
        # Keep only points inside the room (outer wall faces are never seen).
        inside = (
            (points[:, 0] >= -0.01) & (points[:, 0] <= ROOM_X + 0.01)
            & (points[:, 2] >= -0.01) & (points[:, 2] <= ROOM_Z + 0.01)
            & (points[:, 1] >= -0.01) & (points[:, 1] <= CEILING + 0.01)
        )
        clouds.append(points[inside])
    points = np.concatenate(clouds)
    points += rng.normal(0.0, noise, points.shape)
    stray = rng.uniform([0.3, 0.2, 0.3], [ROOM_X - 0.3, 2.0, ROOM_Z - 0.3], size=(floaters, 3))
    return np.concatenate([points, stray])


COLOURS = {
    "floor": (0.55, 0.42, 0.30), "ceiling": (0.92, 0.92, 0.90), "wall": (0.82, 0.80, 0.74),
    "blank_wall": (0.95, 0.95, 0.95), "sofa": (0.20, 0.32, 0.55), "table_top": (0.45, 0.28, 0.16),
    "thin_leg": (0.15, 0.12, 0.10), "glass_top": (0.70, 0.85, 0.90), "lamp_base": (0.1, 0.1, 0.1),
    "thin_pole": (0.85, 0.75, 0.30), "step": (0.62, 0.50, 0.36), "stairs": (0.50, 0.36, 0.24),
    "ramp": (0.40, 0.55, 0.40),
}


def write_splat(parts, world_to_viewer: np.ndarray, path: Path, rng, density=4000.0) -> None:
    """Flat-coloured Gaussian PLY of the true room, so the viewer has a scene."""
    positions, colours = [], []
    for part in parts:
        tag = part.metadata["tag"]
        points, _ = trimesh.sample.sample_surface(part, max(int(part.area * density), 1), seed=int(rng.integers(1 << 31)))
        positions.append(points)
        shade = np.asarray(COLOURS[tag]) * rng.uniform(0.9, 1.05, (len(points), 1))
        colours.append(np.clip(shade, 0, 1))
    positions = np.concatenate(positions) @ world_to_viewer[:3, :3].T + world_to_viewer[:3, 3]
    colours = np.concatenate(colours)
    scale = np.log(0.012 * np.cbrt(abs(np.linalg.det(world_to_viewer[:3, :3]))))
    count = len(positions)
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros((count, len(fields)), dtype="<f4")
    data[:, 0:3] = positions
    data[:, 3:6] = (colours - 0.5) / 0.28209479
    data[:, 6] = 4.0
    data[:, 7:10] = scale
    data[:, 10] = 1.0
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % count
    header += "".join(f"property float {name}\n" for name in fields) + "end_header\n"
    path.write_bytes(header.encode("ascii") + data.tobytes())


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_between(a, b) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--floaters", type=int, default=150, help="stray depth points in free space")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output.resolve()
    geometry = out / "geometry"
    (geometry / "dense").mkdir(parents=True, exist_ok=True)

    parts = ground_truth_parts()
    gt = trimesh.util.concatenate(parts)
    gt.export(out / "ground_truth.glb")
    tags = {part.metadata["tag"] for part in parts}
    gt_obstacles = trimesh.util.concatenate([p for p in parts if p.metadata["tag"] not in {"floor", "ceiling"}])
    gt_obstacles.export(out / "ground_truth_obstacles.glb")

    # World (metres, Y up) -> arbitrary COLMAP-like source frame.
    units_per_meter = 0.37
    rotation = random_rotation(rng)
    translation = np.array([0.4, -1.2, 2.0])
    world_to_source = np.eye(4)
    world_to_source[:3, :3] = rotation * units_per_meter
    world_to_source[:3, 3] = translation

    points_world = simulate_reconstruction(parts, rng, floaters=args.floaters)
    points_source = points_world @ world_to_source[:3, :3].T + translation
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_source))
    o3d.io.write_point_cloud(str(geometry / "dense" / "fused.ply"), cloud)

    settings = Settings(_env_file=None)
    build_collision_candidate(
        geometry / "dense" / "fused.ply", geometry / "collision", settings,
        source_description="dense/fused.ply",
    )
    calibrate_collision_scale(geometry, 1.0, units_per_meter, "synthetic ground truth")

    normal = rotation @ np.array([0.0, 1.0, 0.0])
    offset = -float(normal @ translation)
    (geometry / "navigation").mkdir(exist_ok=True)
    (geometry / "navigation" / "preparation.json").write_text(json.dumps({
        "status": "floor_review_required",
        "plane": {"normal": normal.tolist(), "offset": offset, "accepted": True, "confidence": 1.0},
    }), encoding="utf-8")

    # A nerfstudio-like viewer frame: Z up, centred, rescaled.
    source_to_viewer = np.eye(4)
    align = rotation_between(normal, np.array([0.0, 0.0, 1.0]))
    centre = np.median(points_source, axis=0)
    viewer_scale = 0.8
    source_to_viewer[:3, :3] = align * viewer_scale
    source_to_viewer[:3, 3] = -align @ centre * viewer_scale

    loop = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    cameras_world = np.column_stack([3.5 + 1.6 * np.cos(loop), np.full_like(loop, 1.4), 3.0 + 1.3 * np.sin(loop)])
    cameras_source = cameras_world @ world_to_source[:3, :3].T + translation
    navmesh = build_navigation_mesh(
        geometry, geometry, settings,
        camera_centers_source=cameras_source, source_to_viewer=source_to_viewer.tolist(),
    )
    approve_navigation_mesh(geometry, "synthetic-test", "Ground-truth synthetic scene")
    write_splat(parts, source_to_viewer @ world_to_source, out / "splat.ply", rng)

    (out / "scene.json").write_text(json.dumps({
        "description": "Synthetic living room with known geometry (metres, Y up, floor y=0).",
        "room": {"x": ROOM_X, "z": ROOM_Z, "ceiling": CEILING},
        "world_to_source": world_to_source.tolist(),
        "source_to_viewer": source_to_viewer.tolist(),
        "tags": sorted(tags),
        "point_count": int(len(points_source)),
    }, indent=2), encoding="utf-8")
    print(json.dumps({k: navmesh[k] for k in ("width", "height", "walkable_count", "component_count")}))


if __name__ == "__main__":
    main()
