"""Train Gaussian splats with OpenSplat, the trainer for Macs and machines without CUDA.

OpenSplat (AGPL-3.0, https://github.com/pierotofy/OpenSplat) reads the exported COLMAP
dataset directly and runs on Apple silicon through Metal (MPS) or on the CPU. Build it once
following its README, then point NAMMA at it in `.env`:

    GAUSSIAN_TRAINER_COMMAND=python scripts/train_opensplat.py --dataset {dataset} --output {output}
    OPENSPLAT_BIN=/path/to/OpenSplat/build/opensplat
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--downscale", type=float, default=1.0, help="Train on images scaled down by this factor.")
    parser.add_argument("--opensplat", default=os.environ.get("OPENSPLAT_BIN", "opensplat"))
    args = parser.parse_args()

    executable = shutil.which(args.opensplat) or args.opensplat
    if not Path(executable).is_file():
        parser.error(f"OpenSplat executable not found: {args.opensplat} (set OPENSPLAT_BIN)")
    dataset = args.dataset.resolve()
    if not (dataset / "images").is_dir() or not (dataset / "sparse" / "0").is_dir():
        parser.error(f"Expected COLMAP images and sparse/0 under {dataset}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        str(dataset),
        "-n",
        str(args.iterations),
        "-d",
        str(args.downscale),
        "-o",
        str(output / "splat.ply"),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)
    print(f"Exported {output / 'splat.ply'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
