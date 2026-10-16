"""Evaluate a COLMAP model against a TUM RGB-D ground-truth trajectory.

The input video must have been created from every RGB image at a constant FPS.
The backend's fixed-rate extracted frame names are mapped back to source timestamps.
"""

from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
import pycolmap


def read_timed_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            values = line.split()
            rows.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
    data = np.asarray(rows)
    return data[:, 0], data[:, 1:4]


def read_rgb_timestamps(path: Path) -> np.ndarray:
    return np.asarray(
        [
            float(line.split()[0])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
    )


def similarity_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rgb-index", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    args = parser.parse_args()

    reconstruction = pycolmap.Reconstruction(str(args.model))
    rgb_timestamps = read_rgb_timestamps(args.rgb_index)
    gt_timestamps, gt_positions = read_timed_xyz(args.groundtruth)
    estimated = []
    expected = []
    for image in reconstruction.images.values():
        match = re.fullmatch(r"frame_(\d+)\.jpg", image.name)
        if not match or not image.has_pose:
            continue
        output_index = int(match.group(1)) - 1
        video_time = output_index / args.sample_fps
        source_index = min(round(video_time * args.video_fps), len(rgb_timestamps) - 1)
        timestamp = rgb_timestamps[source_index]
        gt_position = np.asarray(
            [np.interp(timestamp, gt_timestamps, gt_positions[:, axis]) for axis in range(3)]
        )
        camera_center = np.asarray(image.cam_from_world().inverse().translation)
        estimated.append(camera_center)
        expected.append(gt_position)

    estimated_array = np.asarray(estimated)
    expected_array = np.asarray(expected)
    if len(estimated_array) < 3:
        raise RuntimeError("At least three registered, timestamp-matched images are required.")
    aligned, scale = similarity_align(estimated_array, expected_array)
    errors = np.linalg.norm(aligned - expected_array, axis=1)
    print(f"matched_poses={len(errors)}")
    print(f"alignment_scale={scale:.8f}")
    print(f"ate_rmse_m={np.sqrt(np.mean(errors**2)):.6f}")
    print(f"ate_median_m={np.median(errors):.6f}")
    print(f"ate_max_m={np.max(errors):.6f}")


if __name__ == "__main__":
    main()
