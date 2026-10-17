from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import base64
import zlib

import numpy as np
import open3d as o3d
import pycolmap
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from app.config import Settings
from app.services.geometry import GeometryError, read_collision_manifest


class NavigationPreparationError(RuntimeError):
    pass


def invalidate_navigation_artifacts(
    geometry_dir: Path, reason: str, *, include_preparation: bool = False
) -> None:
    names = ["navmesh.json"]
    if include_preparation:
        names.append("preparation.json")
    for name in names:
        path = geometry_dir / "navigation" / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload.update(
            {
                "status": "stale",
                "navmesh_ready": False,
                "invalidated_reason": reason,
                "invalidated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_fused_source(geometry_dir: Path, manifest: dict) -> Path:
    source = (geometry_dir / manifest["source_dense_cloud"]).resolve()
    # Older manifests used ../dense/... even though dense artifacts are stored
    # below geometry/. Keep those projects readable without rewriting them.
    if not source.is_file():
        legacy_source = geometry_dir / "dense" / Path(manifest["source_dense_cloud"]).name
        if legacy_source.is_file():
            source = legacy_source.resolve()
    if not source.is_relative_to(geometry_dir.resolve()) or not source.is_file():
        raise NavigationPreparationError("Collision manifest points to a missing fused geometry source.")
    return source


def encode_walkable_grid(grid: np.ndarray) -> str:
    """Encode a boolean grid compactly for the browser viewer."""
    return base64.b64encode(zlib.compress(np.asarray(grid, dtype=np.uint8).tobytes(), 6)).decode("ascii")


def decode_walkable_grid(encoded: str, width: int, height: int) -> np.ndarray:
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, zlib.error) as exc:
        raise NavigationPreparationError("Navigation grid encoding is invalid.") from exc
    expected = int(width) * int(height)
    if expected <= 0 or len(raw) != expected:
        raise NavigationPreparationError("Navigation grid dimensions do not match its payload.")
    grid = np.frombuffer(raw, dtype=np.uint8)
    if np.any(grid > 1):
        raise NavigationPreparationError("Navigation grid contains non-binary cells.")
    return grid.reshape((int(height), int(width))).astype(bool)


def validate_navigation_mesh(geometry_dir: Path) -> dict:
    path = geometry_dir / "navigation" / "navmesh.json"
    if not path.is_file():
        raise NavigationPreparationError("Navigation mesh has not been generated.")
    navmesh = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if navmesh.get("status") == "stale":
        errors.append("Navigation mesh is stale and must be regenerated.")
    if navmesh.get("unknown_space_policy") != "block_unknown":
        errors.append("Unknown-space policy must block unknown cells.")
    if navmesh.get("walkable_encoding") != "zlib_uint8_row_major_base64":
        errors.append("Unsupported walkability encoding.")
    try:
        grid = decode_walkable_grid(
            navmesh.get("walkable_grid_base64", ""), navmesh.get("width", 0), navmesh.get("height", 0)
        )
    except NavigationPreparationError as exc:
        errors.append(str(exc))
        grid = np.zeros((0, 0), dtype=bool)
    if grid.size:
        actual_walkable = int(np.count_nonzero(grid))
        if actual_walkable != int(navmesh.get("walkable_count", -1)):
            errors.append("Walkable cell count does not match the grid payload.")
        expected_hash = navmesh.get("walkable_sha256")
        actual_hash = sha256(grid.astype(np.uint8).tobytes()).hexdigest()
        if expected_hash != actual_hash:
            errors.append("Walkable grid integrity hash does not match.")
        spawn = navmesh.get("spawn", {}).get("grid", [-1, -1])
        if len(spawn) != 2:
            errors.append("Spawn cell is missing.")
        else:
            spawn_x, spawn_y = int(spawn[0]), int(spawn[1])
            if not (0 <= spawn_x < grid.shape[1] and 0 <= spawn_y < grid.shape[0]):
                errors.append("Spawn cell is outside the navigation grid.")
            elif not grid[spawn_y, spawn_x]:
                errors.append("Spawn cell is not walkable.")
        _, components = ndimage.label(grid, structure=np.ones((3, 3), dtype=np.uint8))
        if components != 1:
            errors.append("Approved walkability grid must contain exactly one connected component.")
    manifest = read_collision_manifest(geometry_dir)
    if not manifest.get("scale_calibrated", False):
        errors.append("Collision geometry scale is not calibrated.")
    if float(navmesh.get("source_units_to_meters", 0.0)) <= 0:
        errors.append("Source-to-meter scale is missing.")
    if float(navmesh.get("viewer_units_to_meters", 0.0)) <= 0:
        errors.append("Viewer-to-meter scale is missing.")
    for name in ("source_to_viewer", "viewer_to_source"):
        matrix = np.asarray(navmesh.get(name, []), dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append(f"{name} transform is invalid.")
    return {
        "passed": not errors,
        "errors": errors,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "walkable_count": int(np.count_nonzero(grid)),
    }


def _floor_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-9)
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, up))) > 0.9:
        reference = np.asarray([0.0, 0.0, 1.0])
    axis_x = np.cross(reference, up)
    axis_x /= max(np.linalg.norm(axis_x), 1e-9)
    axis_y = np.cross(up, axis_x)
    axis_y /= max(np.linalg.norm(axis_y), 1e-9)
    return axis_x, axis_y, up


def _disk_structure(radius_cells: int) -> np.ndarray:
    radius_cells = max(0, int(radius_cells))
    size = radius_cells * 2 + 1
    yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells


def _supported_by_neighbours(points: np.ndarray, mask: np.ndarray, settings: Settings) -> np.ndarray:
    """Keep obstacle points that sit on a surface, dropping flying pixels.

    Counts occupied 2 cm voxels within a small radius, which makes the test
    independent of how densely the source cloud was sampled.
    """
    keep = np.zeros(len(points), dtype=bool)
    if not np.any(mask):
        return keep
    voxel = 0.02
    keys, inverse = np.unique(np.floor(points[mask] / voxel).astype(np.int64), axis=0, return_inverse=True)
    centres = (keys + 0.5) * voxel
    counts = cKDTree(centres).query_ball_point(
        centres, r=settings.navigation_obstacle_neighbor_radius_m, return_length=True
    )
    keep[np.flatnonzero(mask)] = counts[inverse.ravel()] >= int(settings.navigation_min_obstacle_neighbor_voxels)
    return keep


def _movement_layers(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    local_xy: np.ndarray,
    height: np.ndarray,
    width: int,
    height_cells: int,
    cell_size: float,
    settings: Settings,
) -> tuple[np.ndarray, dict]:
    """Per-cell support height and a standable mask for physics-based roaming.

    The flat ``walkable`` grid treats every point between 0.15 m and head
    height as a wall, so steps, stairs and ramps are blocked and a single
    floater removes a 0.8 m disk of floor. Here each cell instead gets the
    height of its lowest well-supported surface; only points well above that
    surface (higher than a climbable step) count as obstacles, and only when
    several agree, so isolated floaters are ignored. The viewer's character
    controller handles step-up, slopes and gravity against the collision mesh
    and this height field; the mask blocks unseen space and thin obstacles that
    Poisson meshing tends to drop.
    """
    cells = grid_y.astype(np.int64) * width + grid_x
    surface = height <= settings.navigation_max_support_height_m
    support = np.full(width * height_cells, np.inf)
    observed = np.zeros(width * height_cells, dtype=bool)
    min_points = int(settings.navigation_min_floor_points_per_cell)
    if np.any(surface):
        order = np.lexsort((height[surface], cells[surface]))
        sorted_cells = cells[surface][order]
        sorted_heights = height[surface][order]
        starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
        counts = np.diff(np.r_[starts, len(sorted_cells)])
        # The k-th lowest height ignores a stray point below the real surface.
        enough = counts >= min_points
        kth = starts[enough] + min_points - 1
        support[sorted_cells[starts[enough]]] = sorted_heights[kth]
        observed[sorted_cells[starts[enough]]] = True
    support_at_point = support[cells]
    above_support = height - np.where(np.isfinite(support_at_point), support_at_point, 0.0)
    obstacle = (
        (above_support > settings.navigation_step_clearance_m)
        & (above_support <= settings.agent_height_meters)
    )
    obstacle &= _supported_by_neighbours(np.column_stack([local_xy, height]), obstacle, settings)
    obstacle_counts = np.bincount(cells[obstacle], minlength=width * height_cells)
    hard = obstacle_counts >= int(settings.navigation_min_obstacle_points_per_cell)
    hard_points = local_xy[obstacle & hard[cells]]
    # Cells seen only as obstacles (wall faces, legs) are known, not unknown.
    observed |= obstacle_counts > 0
    observed = observed.reshape(height_cells, width)
    hard = hard.reshape(height_cells, width)
    observed = ndimage.binary_closing(observed, structure=np.ones((3, 3), dtype=bool)) | observed
    # Both obstacles and unseen space are inflated by the agent radius so the
    # capsule's edge, not its centre, stops at them (a blank wall leaves no
    # points, only the end of the observed floor).
    radius_cells = int(np.ceil(settings.agent_radius_meters / cell_size))
    # Unseen space gets one extra cell: the observed edge is only known to
    # within a cell after closing, and nothing on a blank wall marks its face.
    unseen = ndimage.binary_dilation(~observed, structure=_disk_structure(radius_cells + 1), border_value=1)
    # Obstacles use the exact distance from each cell centre to the nearest
    # obstacle point rather than cell dilation, which would close every
    # body-width gap between furniture in a cluttered room.
    near_obstacle = np.zeros((height_cells, width), dtype=bool)
    if len(hard_points):
        yy, xx = np.mgrid[0:height_cells, 0:width]
        centres = np.column_stack([(xx.ravel() + 0.5) * cell_size, (yy.ravel() + 0.5) * cell_size])
        distance, _ = cKDTree(hard_points).query(centres, distance_upper_bound=settings.agent_radius_meters)
        near_obstacle = (distance < settings.agent_radius_meters).reshape(height_cells, width)
    blocked = near_obstacle | unseen
    standable = ~blocked
    support = support.reshape(height_cells, width)
    # Unknown cells inherit the nearest known support so the height field has
    # no cliffs at the edge of the scan; the standable mask still blocks them.
    known = np.isfinite(support)
    if np.any(known):
        _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
        support = support[iy, ix]
    else:
        support = np.zeros_like(support)
    support_mm = np.clip(np.round(support * 1000.0), -32768, 32767).astype("<i2")
    return standable, {
        "encoding": "zlib_row_major_base64",
        "standable_base64": encode_walkable_grid(standable),
        "standable_count": int(np.count_nonzero(standable)),
        "observed_count": int(np.count_nonzero(observed)),
        "hard_obstacle_count": int(np.count_nonzero(hard)),
        "support_height_mm_base64": base64.b64encode(zlib.compress(support_mm.tobytes(), 9)).decode("ascii"),
        "support_height_dtype": "int16_le_mm",
        "step_clearance_m": float(settings.navigation_step_clearance_m),
        "max_step_m": float(settings.navigation_max_step_m),
        "inflation_radius_m": float(settings.agent_radius_meters),
    }


def build_navigation_mesh(
    geometry_dir: Path,
    sparse_model_dir: Path,
    settings: Settings,
    camera_centers_source: np.ndarray | None = None,
    source_to_viewer: list[list[float]] | None = None,
) -> dict:
    """Build a conservative 2-D walkability grid from calibrated fused points.

    This deliberately blocks unknown cells. It is a review artifact rather
    than a claim that monocular geometry is metrically or semantically perfect.
    """
    manifest = read_collision_manifest(geometry_dir)
    if not manifest.get("scale_calibrated", False):
        raise NavigationPreparationError("Scale calibration is required before NavMesh generation.")
    preparation_path = geometry_dir / "navigation" / "preparation.json"
    if not preparation_path.is_file():
        raise NavigationPreparationError("Run navigation preparation and review the floor candidate first.")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("status") == "stale":
        raise NavigationPreparationError("Navigation preparation is stale; run floor preparation again.")
    plane = preparation.get("plane", {})
    if not plane.get("accepted", False):
        raise NavigationPreparationError("Floor candidate did not pass confidence review.")

    source = _resolve_fused_source(geometry_dir, manifest)
    cloud = o3d.io.read_point_cloud(str(source), remove_nan_points=True, remove_infinite_points=True)
    source_points = np.asarray(cloud.points, dtype=np.float64)
    scale = float(manifest.get("scale_factor_meters_per_unit", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise NavigationPreparationError("Collision scale factor is invalid.")
    points = source_points * scale
    if len(points) < settings.navigation_min_floor_points:
        raise NavigationPreparationError("Fused geometry has too few points for NavMesh generation.")
    normal = np.asarray(plane["normal"], dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-9)
    offset = float(plane["offset"]) * scale
    axis_x, axis_y, up = _floor_basis(normal)
    horizontal_x = points @ axis_x
    horizontal_y = points @ axis_y
    height = points @ up + offset
    cell_size = float(settings.navigation_cell_size_m)
    if cell_size <= 0:
        raise NavigationPreparationError("Navigation cell size must be positive.")
    padding = settings.agent_radius_meters + settings.safety_margin_meters + cell_size
    min_x = float(horizontal_x.min() - padding)
    min_y = float(horizontal_y.min() - padding)
    width = int(np.ceil((horizontal_x.max() + padding - min_x) / cell_size)) + 1
    height_cells = int(np.ceil((horizontal_y.max() + padding - min_y) / cell_size)) + 1
    max_dimension = int(settings.navigation_grid_max_dimension)
    if width > max_dimension or height_cells > max_dimension:
        raise NavigationPreparationError(
            f"Navigation grid would be {width}x{height_cells}; reduce scene extent or increase NAVIGATION_GRID_MAX_DIMENSION."
        )

    grid_x = np.floor((horizontal_x - min_x) / cell_size).astype(np.int32)
    grid_y = np.floor((horizontal_y - min_y) / cell_size).astype(np.int32)
    valid = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height_cells)
    grid_x, grid_y, height = grid_x[valid], grid_y[valid], height[valid]
    floor_mask = (
        (height >= -settings.navigation_floor_support_below_m)
        & (height <= settings.navigation_floor_support_above_m)
    )
    floor_counts = np.zeros((height_cells, width), dtype=np.uint16)
    np.add.at(floor_counts, (grid_y[floor_mask], grid_x[floor_mask]), 1)
    floor_support = floor_counts >= int(settings.navigation_min_floor_points_per_cell)

    obstacle_mask = (
        (height >= settings.navigation_obstacle_min_height_m)
        & (height <= settings.agent_height_meters + settings.safety_margin_meters)
    )
    obstacle_cells = np.zeros((height_cells, width), dtype=bool)
    obstacle_cells[grid_y[obstacle_mask], grid_x[obstacle_mask]] = True
    inflation_cells = int(np.ceil((settings.agent_radius_meters + settings.safety_margin_meters) / cell_size))
    inflated_obstacles = ndimage.binary_dilation(obstacle_cells, structure=_disk_structure(inflation_cells))
    walkable = floor_support & ~inflated_obstacles

    if camera_centers_source is None:
        reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
        camera_centers, _ = _camera_centers_and_up(reconstruction)
    else:
        camera_centers = np.asarray(camera_centers_source, dtype=float)
        if len(camera_centers) < 3:
            raise NavigationPreparationError("At least three camera centers are required for spawn selection.")
    camera_centers *= scale
    camera_mean = camera_centers.mean(axis=0)
    camera_mean -= (float(camera_mean @ up) + offset) * up
    spawn_x = int(np.clip(np.floor((camera_mean @ axis_x - min_x) / cell_size), 0, width - 1))
    spawn_y = int(np.clip(np.floor((camera_mean @ axis_y - min_y) / cell_size), 0, height_cells - 1))

    local_xy = np.column_stack([(horizontal_x - min_x)[valid], (horizontal_y - min_y)[valid]])
    standable, movement = _movement_layers(grid_x, grid_y, local_xy, height, width, height_cells, cell_size, settings)
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = ndimage.label(walkable, structure=structure)
    largest = int(np.bincount(labels.ravel())[1:].max()) if component_count else 0
    walkable_source = "floor_band"
    if largest < settings.navigation_component_min_cells:
        # Stray depth points can block the whole floor band; the movement
        # layers ignore them, so they stand in for the legacy grid.
        labels, component_count = ndimage.label(standable, structure=structure)
        walkable_source = "movement_standable"
    if component_count == 0:
        raise NavigationPreparationError("No walkable floor cells were found after obstacle inflation.")
    component_id = int(labels[spawn_y, spawn_x])
    if component_id == 0:
        counts = np.bincount(labels.ravel())[1:]
        component_id = int(np.argmax(counts) + 1)
        spawn_y, spawn_x = np.argwhere(labels == component_id)[0]
    component_size = int(np.count_nonzero(labels == component_id))
    if component_size < settings.navigation_component_min_cells:
        raise NavigationPreparationError("Largest walkable component is too small for safe roaming.")
    walkable = labels == component_id
    spawn_floor_x = min_x + (int(spawn_x) + 0.5) * cell_size
    spawn_floor_y = min_y + (int(spawn_y) + 0.5) * cell_size
    eye_height_m = min(1.6, max(0.5, settings.agent_height_meters - 0.1))
    spawn_world_m = axis_x * spawn_floor_x + axis_y * spawn_floor_y - up * offset + up * eye_height_m
    walkable_bytes = walkable.astype(np.uint8).tobytes()
    source_to_viewer_matrix = np.asarray(source_to_viewer if source_to_viewer is not None else np.eye(4), dtype=float)
    if source_to_viewer_matrix.shape != (4, 4) or not np.isfinite(source_to_viewer_matrix).all():
        raise NavigationPreparationError("Gaussian source-to-viewer transform is invalid.")
    try:
        viewer_to_source_matrix = np.linalg.inv(source_to_viewer_matrix)
    except np.linalg.LinAlgError as exc:
        raise NavigationPreparationError("Gaussian source-to-viewer transform is singular.") from exc
    viewer_units_per_source_unit = float(abs(np.linalg.det(source_to_viewer_matrix[:3, :3])) ** (1.0 / 3.0))
    if viewer_units_per_source_unit <= 1e-9:
        raise NavigationPreparationError("Gaussian source-to-viewer scale is invalid.")

    result = {
        "schema_version": 1,
        "status": "review_required",
        "navmesh_ready": False,
        "coordinate_system": "colmap_world_floor_basis",
        "units": "meters",
        "source_units_to_meters": scale,
        "viewer_units_to_meters": scale / viewer_units_per_source_unit,
        "source_to_viewer": source_to_viewer_matrix.tolist(),
        "viewer_to_source": viewer_to_source_matrix.tolist(),
        "unknown_space_policy": "block_unknown",
        "source_dense_cloud": str(source.relative_to(geometry_dir.resolve())).replace("\\", "/"),
        "cell_size_m": cell_size,
        "origin_floor_coordinates": [min_x, min_y],
        "width": width,
        "height": height_cells,
        "walkable_count": int(np.count_nonzero(walkable)),
        "walkable_source": walkable_source,
        "floor_support_count": int(np.count_nonzero(floor_support)),
        "blocked_obstacle_count": int(np.count_nonzero(inflated_obstacles)),
        "component_count": int(component_count),
        "component_size": component_size,
        "agent_radius_m": float(settings.agent_radius_meters),
        "agent_height_m": float(settings.agent_height_meters),
        "safety_margin_m": float(settings.safety_margin_meters),
        "max_slope_degrees": float(settings.max_slope_degrees),
        "basis": {
            "axis_x": axis_x.tolist(),
            "axis_y": axis_y.tolist(),
            "up": up.tolist(),
            "plane_offset": offset,
        },
        "spawn": {
            "grid": [int(spawn_x), int(spawn_y)],
            "source_world": (spawn_world_m / scale).tolist(),
            "eye_height_m": eye_height_m,
        },
        "walkable_encoding": "zlib_uint8_row_major_base64",
        "walkable_grid_base64": encode_walkable_grid(walkable),
        "walkable_sha256": sha256(walkable_bytes).hexdigest(),
        "movement": movement,
        "reason": "Generated conservatively; human review must approve floor, scale, and unknown-space boundaries before enabling movement collision.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = geometry_dir / "navigation" / "navmesh.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def approve_navigation_mesh(geometry_dir: Path, reviewer: str, notes: str = "") -> dict:
    if not reviewer.strip():
        raise NavigationPreparationError("A reviewer name is required.")
    path = geometry_dir / "navigation" / "navmesh.json"
    if not path.is_file():
        raise NavigationPreparationError("Navigation mesh has not been generated.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") not in {"review_required", "approved"}:
        raise NavigationPreparationError("Navigation mesh is not in a reviewable state.")
    validation = validate_navigation_mesh(geometry_dir)
    if not validation["passed"]:
        raise NavigationPreparationError("Navigation validation failed: " + "; ".join(validation["errors"]))
    result.update(
        {
            "status": "approved",
            "navmesh_ready": True,
            "approved_by": reviewer.strip(),
            "approval_notes": notes.strip(),
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "validation": validation,
        }
    )
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = read_collision_manifest(geometry_dir)
    manifest.update({"safe_for_navigation": True, "status": "approved"})
    (geometry_dir / "collision" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return result


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
    reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
    camera_centers, scene_up = _camera_centers_and_up(reconstruction)
    return detect_floor_plane_from_observations(fused_path, camera_centers, scene_up, settings)


def detect_floor_plane_from_observations(
    fused_path: Path,
    camera_centers: np.ndarray,
    scene_up: np.ndarray,
    settings: Settings,
) -> dict:
    cloud = o3d.io.read_point_cloud(str(fused_path), remove_nan_points=True, remove_infinite_points=True)
    if len(cloud.points) < settings.navigation_min_floor_points:
        raise NavigationPreparationError("Fused geometry has too few points for floor detection.")
    camera_centers = np.asarray(camera_centers, dtype=float)
    scene_up = np.asarray(scene_up, dtype=float)
    scene_up /= max(np.linalg.norm(scene_up), 1e-9)
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
    source = _resolve_fused_source(geometry_dir, manifest)
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
    invalidate_navigation_artifacts(
        geometry_dir,
        "Metric scale changed; regenerate and review the navigation mesh.",
    )
    model_path = geometry_dir / "collision" / manifest["model_path"]
    scale = known_distance_meters / observed_distance_units
    current_scale = float(manifest.get("scale_factor_meters_per_unit", 1.0))
    correction = scale / current_scale if manifest.get("scale_calibrated", False) else scale
    loaded = trimesh.load(str(model_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    mesh.apply_scale(correction)
    mesh.export(model_path)
    preview_path = geometry_dir / "collision" / manifest.get("preview_path", "")
    if preview_path.is_file():
        preview = o3d.io.read_triangle_mesh(str(preview_path))
        preview.scale(correction, center=(0, 0, 0))
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
            "voxel_size": float(manifest.get("voxel_size", 0.0)) * correction,
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
