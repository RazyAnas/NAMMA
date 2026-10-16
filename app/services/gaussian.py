from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

from app.config import Settings


class GaussianTrainingError(RuntimeError):
    pass


def run_gaussian_training(dataset_dir: Path, output_dir: Path, log_path: Path, settings: Settings) -> None:
    template = settings.gaussian_trainer_command
    if not template:
        raise GaussianTrainingError(
            "No Gaussian trainer is configured. Set GAUSSIAN_TRAINER_COMMAND with {dataset} and {output} placeholders."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = template.format(dataset=str(dataset_dir), output=str(output_dir))
    command = shlex.split(command_text, posix=os.name != "nt")
    if not command:
        raise GaussianTrainingError("GAUSSIAN_TRAINER_COMMAND resolved to an empty command.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        result = subprocess.run(
            command,
            cwd=dataset_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise GaussianTrainingError(
            f"Gaussian trainer failed with exit code {result.returncode}. Log tail: {tail}"
        )
