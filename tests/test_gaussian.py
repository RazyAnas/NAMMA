from pathlib import Path

import pytest

from app.config import Settings
from app.services.gaussian import (
    GaussianTrainingError,
    publish_gaussian_artifact,
    read_gaussian_manifest,
    run_gaussian_training,
)


def test_gaussian_training_requires_configured_command(tmp_path: Path) -> None:
    settings = Settings(gaussian_trainer_command=None)

    with pytest.raises(GaussianTrainingError, match="GAUSSIAN_TRAINER_COMMAND"):
        run_gaussian_training(tmp_path / "dataset", tmp_path / "output", tmp_path / "train.log", settings)


def test_publish_gaussian_artifact_creates_verified_manifest(tmp_path: Path) -> None:
    output = tmp_path / "gaussian"
    model = output / "export" / "splat.ply"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"ply\nexample")

    manifest = publish_gaussian_artifact(output)

    assert manifest["model_path"] == "export/splat.ply"
    assert manifest["size_bytes"] == len(b"ply\nexample")
    assert len(manifest["sha256"]) == 64
    assert read_gaussian_manifest(output) == manifest


def test_publish_gaussian_artifact_rejects_missing_ply(tmp_path: Path) -> None:
    with pytest.raises(GaussianTrainingError, match="no .ply"):
        publish_gaussian_artifact(tmp_path)
