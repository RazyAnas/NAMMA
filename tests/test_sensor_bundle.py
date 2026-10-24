import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np
import pytest

from app.services.sensor_bundle import (
    estimate_imu_alignment,
    gravity_in_camera,
    ingest,
    intrinsics,
    rotation_to_quaternion,
    so3_log,
    world_to_colmap,
)

ROOT = Path(__file__).resolve().parents[1]


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_colmap_pose_projects_like_webxr():
    # A camera 1.5 m up, turned 30 degrees left and tilted down, and a WebXR projection matrix.
    world_from_gl = np.eye(4)
    world_from_gl[:3, :3] = _rotation([0, 1, 0], np.radians(30)) @ _rotation([1, 0, 0], np.radians(-20))
    world_from_gl[:3, 3] = [0.4, 1.5, -1.0]
    width, height = 640, 480
    p = np.zeros((4, 4))
    p[0, 0], p[1, 1], p[0, 2], p[1, 2] = 1.6, 2.1, -0.03, 0.02  # column-major p0, p5, p8, p9
    p[2, 2], p[2, 3], p[3, 2] = -1.0, -0.02, -1.0
    point = np.array([0.1, 1.2, -3.0])

    # WebXR / OpenGL path: clip = P @ view @ point, then NDC to pixels with y down.
    clip = p @ np.linalg.inv(world_from_gl) @ np.append(point, 1.0)
    ndc = clip[:3] / clip[3]
    expected = [(ndc[0] + 1) / 2 * width, (1 - ndc[1]) / 2 * height]

    rotation, translation = world_to_colmap(world_from_gl)
    fx, fy, cx, cy = intrinsics([p[0, 0], p[1, 1], p[0, 2], p[1, 2]], width, height)
    camera = rotation @ point + translation
    assert camera[2] > 0
    actual = [fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_quaternion_round_trip_and_gravity():
    for axis, angle in [([1, 2, 3], 0.3), ([0, 0, 1], np.pi - 1e-3), ([1, 0, 0], 2.5)]:
        r = _rotation(axis, angle)
        np.testing.assert_allclose(_quaternion_to_matrix(rotation_to_quaternion(r)), r, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(so3_log(r)), angle, atol=1e-6)
    level = np.eye(4)  # camera looking along -z, level: gravity points down the image (+y in OpenCV)
    np.testing.assert_allclose(gravity_in_camera(level), [0, 1, 0], atol=1e-12)


def test_imu_alignment_recovers_offset_and_landscape_rotation():
    rng = np.random.default_rng(3)
    t = np.arange(0, 20, 1 / 30)

    def omega(times):
        return np.stack([0.9 * np.sin(1.1 * times), 0.6 * np.sin(0.7 * times + 1), 0.4 * np.sin(1.9 * times + 2)], axis=1)

    # Phone held in landscape: device axes are the camera axes turned 90 degrees about z.
    cam_from_imu = _rotation([0, 0, 1], np.pi / 2)
    offset = -0.031
    gyro_t = np.arange(0, 20, 1 / 60) + 0.004
    gyro = omega(gyro_t - offset) @ cam_from_imu + rng.normal(0, 0.01, (len(gyro_t), 3))
    result = estimate_imu_alignment(t, omega(t), gyro_t, gyro)
    assert result["ok"]
    assert abs(result["offset_ms"] - offset * 1000) < 2.0
    np.testing.assert_allclose(result["rotation_cam_from_imu"], cam_from_imu, atol=0.02)


def _write_bundle(path: Path, video: Path, frames: int, fps: float = 30.0) -> None:
    def table(header, rows):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        return buffer.getvalue()

    frame_rows, chunk_rows, imu_rows = [], [], []
    for i in range(frames):
        t_ms = 1000.0 + i * 1000.0 / fps
        ts = round(t_ms * 1000)
        world = np.eye(4)
        world[:3, 3] = [0.05 * i, 1.5, 0.0]
        frame_rows.append([i, t_ms, ts, 0, 0, *world.T.ravel(), 1.5, 2.0, 0.0, 0.0, 320, 240])
        chunk_rows.append([i, ts, int(i % 30 == 0), 0])
    for k in range(int(frames * 2)):
        imu_rows.append([1000.0 + k * 1000.0 / 60, "gyro", 0, 0, 0])
    meta = {"format": "namma-sensor-bundle", "version": 1, "mode": "ar", "duration_s": frames / fps}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("meta.json", json.dumps(meta))
        archive.writestr("frames.csv", table(
            ["frame", "t_ms", "video_ts_us", "tracking", "screen_angle", *[f"m{i}" for i in range(16)], "p0", "p5", "p8", "p9", "image_w", "image_h"],
            frame_rows,
        ))
        archive.writestr("video.csv", table(["chunk", "ts_us", "key", "bytes"], chunk_rows))
        archive.writestr("imu.csv", table(["t_ms", "sensor", "x", "y", "z"], imu_rows))
        archive.write(video, "video.h264")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ingest_h264_bundle(tmp_path):
    # Raw Annex B H.264 without B-frames, as Android's WebCodecs encoder produces.
    video = tmp_path / "video.h264"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-frames:v", "45",
         "-c:v", "libx264", "-bf", "0", "-g", "30", "-f", "h264", str(video)],
        check=True,
    )
    bundle = tmp_path / "bundle.zip"
    _write_bundle(bundle, video, 45)
    report = ingest(bundle, tmp_path / "out", stride=5)
    assert report["decoded_frames_matches_chunks"]
    assert report["frames_written"] == 9
    assert report["frames_with_pose"] == 9
    assert report["remuxed_video"] == "video.mp4"
    decoded = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(tmp_path / "out" / "video.mp4")],
        check=True, capture_output=True, text=True,
    )
    assert int(decoded.stdout.strip()) == 45
    assert report["path_length_m"] == pytest.approx(0.05 * 44, abs=1e-3)
    cameras = (tmp_path / "out" / "colmap_known_poses" / "cameras.txt").read_text().splitlines()[-1].split()
    assert cameras[:4] == ["1", "PINHOLE", "320", "240"]
    assert [float(v) for v in cameras[4:]] == pytest.approx([240.0, 240.0, 160.0, 120.0])
    priors = [json.loads(line) for line in (tmp_path / "out" / "pose_priors.jsonl").read_text().splitlines()]
    assert priors[1]["name"] == "frame_000005.jpg"
    assert priors[1]["position"] == pytest.approx([0.25, 1.5, 0.0])


NODE_TEST = ROOT / "tests" / "browser" / "record_e2e.mjs"


@pytest.mark.skipif(
    shutil.which("node") is None or not (ROOT / "tests" / "browser" / "node_modules" / "playwright").exists(),
    reason="needs node and playwright (tests/browser/node_modules/playwright)",
)
def test_recorder_page_end_to_end(tmp_path):
    """Drive /record/ in headless Chromium with a scripted AR session and a gyro 42 ms late."""
    subprocess.run(["node", str(NODE_TEST), str(tmp_path), "ar"], check=True, timeout=180, capture_output=True)
    report = ingest(tmp_path / "bundle-ar.zip", tmp_path / "out", stride=7)
    assert report["decoded_frames_matches_chunks"]
    assert report["tracked_fraction"] > 0.85
    alignment = report["imu_to_camera"]
    assert alignment["ok"] and abs(alignment["offset_ms"] - 42.0) < 5.0
    assert alignment["rotation_angle_deg"] < 1.0
    assert report["intrinsics_median"] == pytest.approx({"fx": 500.0, "fy": 500.0, "cx": 330.0, "cy": 236.0}, abs=0.01)

    # Every written frame must show the camera image of the XR frame its pose came from: the mock
    # paints the XR frame index as ten black/white blocks along the top edge.
    rows = {}
    with zipfile.ZipFile(tmp_path / "bundle-ar.zip") as archive:
        for row in csv.DictReader(io.TextIOWrapper(archive.open("frames.csv"), "utf-8")):
            if row["video_ts_us"]:
                rows[int(row["video_ts_us"])] = int(row["frame"])
        chunks = [int(r["ts_us"]) for r in csv.DictReader(io.TextIOWrapper(archive.open("video.csv"), "utf-8"))]
    checked = 0
    for image in sorted((tmp_path / "out" / "frames").glob("frame_*.jpg")):
        chunk = int(image.stem.split("_")[1])
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        painted = sum(1 << bit for bit in range(10) if np.median(gray[12:36, bit * 64 + 16 : bit * 64 + 48]) > 128)
        assert painted == rows[chunks[chunk]], image.name
        checked += 1
    assert checked >= 10
