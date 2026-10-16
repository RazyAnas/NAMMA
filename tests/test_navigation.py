import json
from hashlib import sha256
from pathlib import Path

import pytest
import trimesh
import numpy as np
import open3d as o3d

from app.config import Settings
from app.services.navigation import (
    NavigationPreparationError,
    approve_navigation_mesh,
    build_navigation_mesh,
    calibrate_collision_scale,
    encode_walkable_grid,
)
import app.services.navigation as navigation_service


def test_scale_calibration_rejects_non_positive_values(tmp_path: Path) -> None:
    with pytest.raises(NavigationPreparationError, match="positive"):
        calibrate_collision_scale(tmp_path, 0, 1, "test")


def test_navigation_settings_have_conservative_unknown_policy() -> None:
    settings = Settings()
    assert settings.navigation_min_floor_confidence >= 0.5


def test_scale_calibration_requires_collision_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        calibrate_collision_scale(tmp_path, 1, 1, "test")


def test_scale_calibration_updates_glb_and_manifest(tmp_path: Path) -> None:
    collision = tmp_path / "collision"
    collision.mkdir()
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    mesh.export(collision / "collision.glb")
    (collision / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": "collision.glb",
                "preview_path": "missing.ply",
                "size_bytes": (collision / "collision.glb").stat().st_size,
                "voxel_size": 0.1,
                "scale_calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    navigation = tmp_path / "navigation"
    navigation.mkdir()
    (navigation / "navmesh.json").write_text(
        json.dumps({"status": "approved", "navmesh_ready": True}), encoding="utf-8"
    )
    result = calibrate_collision_scale(tmp_path, 4.0, 2.0, "door width")
    assert result["scale_calibrated"] is True
    assert result["scale_factor_meters_per_unit"] == 2.0
    assert result["coordinate_units"] == "meters"
    stale = json.loads((navigation / "navmesh.json").read_text(encoding="utf-8"))
    assert stale["status"] == "stale"
    assert stale["navmesh_ready"] is False


def test_navmesh_requires_metric_scale(tmp_path: Path) -> None:
    collision = tmp_path / "collision"
    collision.mkdir()
    model = collision / "collision.glb"
    model.write_bytes(b"glTF")
    (collision / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": model.name,
                "size_bytes": model.stat().st_size,
                "scale_calibrated": False,
                "source_dense_cloud": "dense/fused.ply",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(NavigationPreparationError, match="Scale calibration"):
        build_navigation_mesh(tmp_path, tmp_path / "sparse", Settings())


def test_walkable_grid_encoding_is_compact_and_binary() -> None:
    import base64
    import zlib

    grid = __import__("numpy").array([[True, False], [False, True]], dtype=bool)
    encoded = encode_walkable_grid(grid)
    decoded = zlib.decompress(base64.b64decode(encoded))
    assert decoded == bytes([1, 0, 0, 1])


def test_navigation_approval_marks_artifact_ready(tmp_path: Path) -> None:
    geometry = tmp_path
    collision = geometry / "collision"
    navigation = geometry / "navigation"
    collision.mkdir()
    navigation.mkdir()
    model = collision / "collision.glb"
    model.write_bytes(b"glTF")
    (collision / "manifest.json").write_text(
        json.dumps({"model_path": model.name, "size_bytes": model.stat().st_size, "scale_calibrated": True}),
        encoding="utf-8",
    )
    (navigation / "navmesh.json").write_text(
        json.dumps(
            {
                "status": "review_required",
                "navmesh_ready": False,
                "unknown_space_policy": "block_unknown",
                "walkable_encoding": "zlib_uint8_row_major_base64",
                "walkable_grid_base64": encode_walkable_grid(np.asarray([[True]])),
                "walkable_sha256": sha256(bytes([1])).hexdigest(),
                "walkable_count": 1,
                "width": 1,
                "height": 1,
                "source_units_to_meters": 1.0,
                "spawn": {"grid": [0, 0]},
            }
        ),
        encoding="utf-8",
    )
    result = approve_navigation_mesh(geometry, "student-reviewer", "floor and walls checked")
    assert result["navmesh_ready"] is True
    assert result["approved_by"] == "student-reviewer"
    assert json.loads((collision / "manifest.json").read_text())["safe_for_navigation"] is True


def test_navmesh_builds_floor_grid_and_inflates_wall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = tmp_path / "geometry"
    dense = geometry / "dense"
    collision = geometry / "collision"
    navigation = geometry / "navigation"
    dense.mkdir(parents=True)
    collision.mkdir()
    navigation.mkdir()
    model = collision / "collision.glb"
    model.write_bytes(b"glTF")
    (collision / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": model.name,
                "size_bytes": model.stat().st_size,
                "scale_calibrated": True,
                "source_dense_cloud": "dense/fused.ply",
            }
        ),
        encoding="utf-8",
    )
    (navigation / "preparation.json").write_text(
        json.dumps({"plane": {"normal": [0, 0, 1], "offset": 0, "accepted": True}}),
        encoding="utf-8",
    )
    floor = [[x, y, 0.0] for x in np.arange(-1.0, 1.01, 0.1) for y in np.arange(-1.0, 1.01, 0.1)]
    wall = [[0.0, y, z] for y in np.arange(-0.8, 0.81, 0.1) for z in np.arange(0.2, 1.6, 0.1)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(floor + wall)))
    fused = dense / "fused.ply"
    assert o3d.io.write_point_cloud(str(fused), cloud)
    monkeypatch.setattr(
        navigation_service,
        "_camera_centers_and_up",
        lambda _reconstruction: (np.asarray([[0.8, 0.0, 1.0], [0.8, 0.2, 1.0], [0.8, -0.2, 1.0]]), np.asarray([0, 0, 1.0])),
    )
    monkeypatch.setattr(navigation_service.pycolmap, "Reconstruction", lambda _path: object())
    settings = Settings(
        navigation_cell_size_m=0.1,
        navigation_min_floor_points=10,
        navigation_min_floor_points_per_cell=1,
        navigation_grid_max_dimension=100,
        navigation_component_min_cells=5,
        agent_radius_meters=0.05,
        safety_margin_meters=0.0,
    )
    result = build_navigation_mesh(geometry, tmp_path / "sparse", settings)
    assert result["walkable_count"] > 0
    assert result["blocked_obstacle_count"] > 0
    assert result["navmesh_ready"] is False
    assert result["source_units_to_meters"] == 1.0
    assert len(result["spawn"]["source_world"]) == 3


def test_scale_recalibration_uses_absolute_target_scale(tmp_path: Path) -> None:
    collision = tmp_path / "collision"
    collision.mkdir()
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    mesh.export(collision / "collision.glb")
    (collision / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": "collision.glb",
                "preview_path": "missing.ply",
                "size_bytes": (collision / "collision.glb").stat().st_size,
                "voxel_size": 0.1,
                "scale_calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    calibrate_collision_scale(tmp_path, 4.0, 2.0, "first")
    result = calibrate_collision_scale(tmp_path, 6.0, 2.0, "corrected")
    calibrated = trimesh.load(collision / "collision.glb", force="mesh")
    assert result["scale_factor_meters_per_unit"] == 3.0
    assert calibrated.extents[0] == pytest.approx(6.0)
