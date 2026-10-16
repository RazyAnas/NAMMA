"""Train Splatfacto from a COLMAP dataset and export a viewer-ready PLY."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def nerfstudio_command(name: str) -> str:
    """Resolve Nerfstudio entry points from the active virtual environment."""
    scripts_dir = Path(sys.executable).parent
    executable = scripts_dir / (f"{name}.exe" if sys.platform == "win32" else name)
    if not executable.is_file():
        raise RuntimeError(f"{name} is not installed beside {sys.executable}")
    return str(executable)


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    # Nerfstudio 1.1.5 predates Torch 2.6's weights_only=True default. The
    # checkpoint loaded here was created locally by this same wrapper.
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    torch_home = Path(__file__).resolve().parents[1] / "tmp" / "torch-cache"
    torch_home.mkdir(parents=True, exist_ok=True)
    environment.setdefault("TORCH_HOME", str(torch_home))
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=7000)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not (dataset / "images").is_dir() or not (dataset / "sparse" / "0").is_dir():
        parser.error(f"Expected COLMAP images and sparse/0 under {dataset}")
    output.mkdir(parents=True, exist_ok=True)

    run(
        [
            nerfstudio_command("ns-train"),
            "splatfacto",
            "--data",
            str(dataset),
            "--output-dir",
            str(output / "runs"),
            "--max-num-iterations",
            str(args.iterations),
            "--vis",
            "tensorboard",
            "colmap",
            "--colmap-path",
            "sparse/0",
        ]
    )
    configs = list((output / "runs").rglob("config.yml"))
    if not configs:
        raise RuntimeError("Nerfstudio training completed without producing config.yml")
    config = max(configs, key=lambda path: path.stat().st_mtime_ns)
    export_dir = output / "export"
    run(
        [
            nerfstudio_command("ns-export"),
            "gaussian-splat",
            "--load-config",
            str(config),
            "--output-dir",
            str(export_dir),
        ]
    )
    models = list(export_dir.rglob("*.ply"))
    if not models:
        raise RuntimeError("Nerfstudio export completed without producing a .ply model")
    print(f"Exported {max(models, key=lambda path: path.stat().st_mtime_ns)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
