import json
from pathlib import Path

import pytest
import trimesh

from app.config import Settings
from app.services.navigation import NavigationPreparationError, calibrate_collision_scale


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
    result = calibrate_collision_scale(tmp_path, 4.0, 2.0, "door width")
    assert result["scale_calibrated"] is True
    assert result["scale_factor_meters_per_unit"] == 2.0
    assert result["coordinate_units"] == "meters"
