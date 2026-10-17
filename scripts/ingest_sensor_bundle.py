"""Unpack a phone sensor bundle saved by /record/ into reconstruction inputs.

    python scripts/ingest_sensor_bundle.py namma-ar-2026-10-17T10-00-00.zip out/lobby --stride 10

Writes frames/ (every --stride-th video frame), colmap_known_poses/ (COLMAP text model with the
ARCore poses, for colmap point_triangulator), pose_priors.jsonl, video.mp4 (or .webm) that the
normal project upload accepts, and report.json with measured rates, tracking loss and the
IMU-to-camera time offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.sensor_bundle import ingest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=10, help="keep every Nth video frame (default 10, 3 fps at 30 fps)")
    args = parser.parse_args()
    report = ingest(args.bundle, args.output, max(1, args.stride))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
