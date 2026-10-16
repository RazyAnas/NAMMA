import json
from pathlib import Path

import pytest

from app.services.geometry import GeometryError, read_collision_manifest


def test_collision_manifest_requires_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not available"):
        read_collision_manifest(tmp_path)


def test_collision_manifest_validates_model_size(tmp_path: Path) -> None:
    output = tmp_path / "collision"
    output.mkdir()
    (output / "collision.glb").write_bytes(b"glTF")
    (output / "manifest.json").write_text(
        json.dumps({"model_path": "collision.glb", "size_bytes": 99}), encoding="utf-8"
    )
    with pytest.raises(GeometryError, match="size"):
        read_collision_manifest(tmp_path)


def test_collision_manifest_reads_verified_artifact(tmp_path: Path) -> None:
    output = tmp_path / "collision"
    output.mkdir()
    model = output / "collision.glb"
    model.write_bytes(b"glTF")
    payload = {"model_path": "collision.glb", "size_bytes": model.stat().st_size}
    (output / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_collision_manifest(tmp_path) == payload
