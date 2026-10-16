from pathlib import Path

import pytest

from app.config import Settings
from app.services.gaussian import GaussianTrainingError, run_gaussian_training


def test_gaussian_training_requires_configured_command(tmp_path: Path) -> None:
    settings = Settings(gaussian_trainer_command=None)

    with pytest.raises(GaussianTrainingError, match="GAUSSIAN_TRAINER_COMMAND"):
        run_gaussian_training(tmp_path / "dataset", tmp_path / "output", tmp_path / "train.log", settings)
