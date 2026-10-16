from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess

from app.config import PROJECT_ROOT, Settings


class GaussianTrainingError(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def publish_gaussian_artifact(output_dir: Path) -> dict:
    """Validate the trainer output and write a portable model manifest."""
    candidates = [path for path in output_dir.rglob("*.ply") if path.is_file()]
    if not candidates:
        raise GaussianTrainingError(
            "Gaussian training finished but no .ply model was exported under the output directory."
        )
    model = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.stat().st_size))
    manifest = {
        "schema_version": 1,
        "format": "gaussian-splat-ply",
        "model_path": model.relative_to(output_dir).as_posix(),
        "size_bytes": model.stat().st_size,
        "sha256": _file_sha256(model),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_gaussian_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Gaussian model is not available.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = manifest.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise GaussianTrainingError("Gaussian model manifest has an invalid model_path.")
    model = (output_dir / model_path).resolve()
    root = output_dir.resolve()
    if not model.is_relative_to(root) or not model.is_file():
        raise GaussianTrainingError("Gaussian model manifest points outside the artifact directory or is missing.")
    return manifest


def run_gaussian_training(
    dataset_dir: Path, output_dir: Path, log_path: Path, settings: Settings
) -> dict:
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
            cwd=PROJECT_ROOT,
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
    return publish_gaussian_artifact(output_dir)
