"""Train Splatfacto from a COLMAP dataset and export a viewer-ready PLY."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


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
            "ns-train",
            "splatfacto",
            "--data",
            str(dataset),
            "--output-dir",
            str(output / "runs"),
            "--max-num-iterations",
            str(args.iterations),
            "--viewer.quit-on-train-completion",
            "True",
        ]
    )
    configs = list((output / "runs").rglob("config.yml"))
    if not configs:
        raise RuntimeError("Nerfstudio training completed without producing config.yml")
    config = max(configs, key=lambda path: path.stat().st_mtime_ns)
    export_dir = output / "export"
    run(
        [
            "ns-export",
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
