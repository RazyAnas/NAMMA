"""Read phone sensor bundles from /record/ and turn them into pipeline inputs.

A bundle holds an encoded video stream, one CSV row per camera frame (WebXR / ARCore pose and
projection, or no pose in camera-only mode), raw IMU samples, and metadata, all timestamped on
the browser's performance.now() clock. This module converts it into what the reconstruction
already understands: JPEG frames, a COLMAP text model with the AR poses (for
`colmap point_triangulator` or as pose priors), KeyframeMeta-style priors, and a report that
measures stream rates and the IMU-to-camera time offset instead of assuming them.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np

# OpenGL camera (x right, y up, looking down -z) to OpenCV/COLMAP camera (x right, y down, +z).
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def _float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def _read_csv(archive: zipfile.ZipFile, name: str) -> list[dict]:
    if name not in archive.namelist():
        return []
    with archive.open(name) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, "utf-8")))


class Bundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            self.names = archive.namelist()
            self.meta = json.loads(archive.read("meta.json"))
            frames = _read_csv(archive, "frames.csv")
            chunks = _read_csv(archive, "video.csv")
            imu = _read_csv(archive, "imu.csv")
        self.video_name = next((n for n in ("video.h264", "video.ivf") if n in self.names), None)
        self.frame_t = np.array([_float(r["t_ms"]) for r in frames]) / 1000.0
        self.frame_video_ts = np.array([_float(r["video_ts_us"]) for r in frames])
        self.tracking = np.array([int(r["tracking"]) for r in frames], dtype=int)
        self.screen_angle = np.array([_float(r["screen_angle"]) for r in frames])
        self.matrices = np.array([[_float(r[f"m{i}"]) for i in range(16)] for r in frames]).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.projection = np.array([[_float(r[k]) for k in ("p0", "p5", "p8", "p9")] for r in frames])
        self.image_size = np.array([[_float(r["image_w"]), _float(r["image_h"])] for r in frames])
        self.chunk_ts = np.array([_float(r["ts_us"]) for r in chunks])
        self.imu: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        by_sensor: dict[str, list[list[float]]] = {}
        for row in imu:
            by_sensor.setdefault(row["sensor"], []).append(
                [_float(row[k]) for k in ("t_ms", "x", "y", "z")] + [_float(row.get("t_sensor_ms", ""))]
            )
        for sensor, values in by_sensor.items():
            array = np.array(values)
            sensor_clock = array[:, 4]
            if np.isfinite(sensor_clock).all():
                # Generic Sensor timestamps are exact sample times but, on Android Chrome, on a
                # different clock from performance.now() (measured 2026-10-17: about 230,000 s apart).
                # Map them onto the page clock with the smallest observed delivery latency.
                array[:, 0] = sensor_clock + np.min(array[:, 0] - sensor_clock)
            array = array[:, :4]
            array = array[np.argsort(array[:, 0])]
            self.imu[sensor] = (array[:, 0] / 1000.0, array[:, 1:])

    @property
    def has_poses(self) -> bool:
        return self.meta.get("mode") == "ar" and bool((self.tracking == 0).any())

    def extract_video(self, destination: Path) -> Path | None:
        if self.video_name is None:
            return None
        with zipfile.ZipFile(self.path) as archive, archive.open(self.video_name) as source:
            target = destination / self.video_name
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
        return target


# ---- geometry --------------------------------------------------------------------------------


def world_to_colmap(world_from_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WebXR world_from_camera (OpenGL axes) to COLMAP's cam_from_world rotation and translation."""
    rotation = world_from_gl[:3, :3] @ GL_TO_CV
    centre = world_from_gl[:3, 3]
    cam_from_world = rotation.T
    return cam_from_world, -cam_from_world @ centre


def rotation_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to unit quaternion (w, x, y, z), w >= 0."""
    m = r
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def so3_log(r: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-8:
        return axis / 2.0
    return axis * angle / (2.0 * np.sin(angle))


def intrinsics(projection: np.ndarray, width: float, height: float) -> tuple[float, float, float, float]:
    p0, p5, p8, p9 = projection
    return p0 * width / 2.0, p5 * height / 2.0, (1.0 - p8) * width / 2.0, (1.0 + p9) * height / 2.0


def body_angular_velocity(times: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity in the moving (camera) frame between consecutive orientations."""
    dt = np.diff(times)
    omega = np.array([so3_log(a.T @ b) for a, b in zip(rotations[:-1], rotations[1:])])
    mid = 0.5 * (times[:-1] + times[1:])
    return mid, omega / np.maximum(dt, 1e-6)[:, None]


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R source - target| over rows."""
    u, _, vt = np.linalg.svd(target.T @ source)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def estimate_imu_alignment(
    pose_t: np.ndarray,
    pose_omega: np.ndarray,
    gyro_t: np.ndarray,
    gyro: np.ndarray,
    max_offset_s: float = 0.25,
    step_s: float = 0.001,
) -> dict:
    """Find offset d and rotation R with R @ gyro(t + d) ~= pose angular velocity(t).

    A positive offset means the gyro's clock reads later than the camera's for the same motion.
    The pose angular velocity is a finite difference across one frame interval, so the gyro is
    averaged over the same interval before comparing.
    """
    if len(pose_t) < 20 or len(gyro_t) < 20:
        return {"ok": False, "reason": "not enough samples"}
    spacing = float(np.median(np.diff(pose_t)))
    # Box-filter the gyro over one frame interval (matches the finite-difference pose rate).
    cumulative = np.vstack([np.zeros(3), np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * np.diff(gyro_t)[:, None], axis=0)])

    def gyro_mean(times: np.ndarray) -> np.ndarray:
        lo = np.stack([np.interp(times - spacing / 2, gyro_t, cumulative[:, i]) for i in range(3)], axis=1)
        hi = np.stack([np.interp(times + spacing / 2, gyro_t, cumulative[:, i]) for i in range(3)], axis=1)
        return (hi - lo) / spacing

    valid = (pose_t - max_offset_s - spacing > gyro_t[0]) & (pose_t + max_offset_s + spacing < gyro_t[-1])
    t, target = pose_t[valid], pose_omega[valid]
    if len(t) < 20 or np.linalg.norm(target, axis=1).max() < 0.2:
        return {"ok": False, "reason": "too little rotation or overlap"}

    def residual(offset: float) -> tuple[float, np.ndarray]:
        source = gyro_mean(t + offset)
        rotation = _kabsch(source, target)
        error = target - source @ rotation.T
        return float(np.sqrt((error**2).sum(axis=1).mean())), rotation

    offsets = np.arange(-max_offset_s, max_offset_s + step_s / 2, step_s)
    errors = np.array([residual(o)[0] for o in offsets])
    best = int(np.argmin(errors))
    offset = float(offsets[best])
    if 0 < best < len(offsets) - 1:  # parabolic refinement between grid steps
        a, b, c = errors[best - 1 : best + 2]
        denominator = a - 2 * b + c
        if denominator > 0:
            offset += 0.5 * step_s * (a - c) / denominator
    rms, rotation = residual(offset)
    signal = float(np.sqrt((target**2).sum(axis=1).mean()))
    angle = np.degrees(np.linalg.norm(so3_log(rotation)))
    return {
        "ok": True,
        "offset_ms": round(offset * 1000.0, 2),
        "rotation_cam_from_imu": rotation.round(4).tolist(),
        "rotation_angle_deg": round(float(angle), 2),
        "residual_rms_dps": round(float(np.degrees(rms)), 3),
        "signal_rms_dps": round(float(np.degrees(signal)), 3),
        "at_search_edge": best in (0, len(offsets) - 1),
    }


def gravity_in_camera(world_from_gl: np.ndarray) -> np.ndarray:
    """Unit 'down' vector in OpenCV camera axes; local-floor worlds have +y up."""
    cam_from_world, _ = world_to_colmap(world_from_gl)
    return cam_from_world @ np.array([0.0, -1.0, 0.0])


# ---- report and export -----------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def bundle_report(bundle: Bundle) -> dict:
    t = bundle.frame_t
    report: dict = {"mode": bundle.meta.get("mode"), "duration_s": bundle.meta.get("duration_s"), "frames": int(len(t))}
    encoded = np.isfinite(bundle.frame_video_ts)
    report["frames_encoded"] = int(encoded.sum())
    report["chunks"] = int(len(bundle.chunk_ts))
    if encoded.sum() > 2:
        dt = np.diff(t[encoded])
        report["video_fps"] = round(1.0 / float(np.median(dt)), 2)
        report["video_gap_p95_ms"] = round(float(np.percentile(dt, 95) * 1000), 1)
        report["video_gap_max_ms"] = round(float(dt.max() * 1000), 1)
    report["imu"] = {}
    for sensor, (times, _values) in bundle.imu.items():
        dt = np.diff(times)
        report["imu"][sensor] = {
            "samples": int(len(times)),
            "rate_hz": round(1.0 / float(np.median(dt)), 1) if len(dt) else None,
            "gap_max_ms": round(float(dt.max() * 1000), 1) if len(dt) else None,
        }
    if bundle.meta.get("mode") == "ar":
        tracked = bundle.tracking == 0
        report["tracked_fraction"] = round(float(tracked.mean()), 4) if len(tracked) else 0.0
        lost = _runs(~tracked)
        report["tracking_lost_runs"] = len(lost)
        report["tracking_lost_longest_s"] = round(max((t[min(b, len(t) - 1)] - t[a] for a, b in lost), default=0.0), 3)
        centres = bundle.matrices[tracked][:, :3, 3]
        if len(centres) > 1:
            report["path_length_m"] = round(float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()), 3)
            report["extent_m"] = (centres.max(axis=0) - centres.min(axis=0)).round(3).tolist()
            sizes = bundle.image_size[tracked]
            fx = [intrinsics(p, *s) for p, s in zip(bundle.projection[tracked], sizes)]
            report["intrinsics_median"] = dict(zip(("fx", "fy", "cx", "cy"), np.median(np.array(fx), axis=0).round(2).tolist()))
            report["image_size"] = sizes[0].astype(int).tolist()
        # Prefer the Generic Sensor gyro, but only if its timestamps share the frames' clock (bundles
        # recorded before t_sensor_ms existed hold raw sensor-clock times that never overlap).
        overlapping = [
            key for key in ("gyro", "dm_gyro")
            if key in bundle.imu and bundle.imu[key][0][0] < t[-1] and bundle.imu[key][0][-1] > t[0]
        ]
        gyro_key = overlapping[0] if overlapping else None
        if gyro_key and tracked.sum() > 20:
            # GL-axes rotations; gyro device axes match GL camera axes up to the screen rotation.
            rotations = bundle.matrices[tracked][:, :3, :3]
            pose_t, omega = body_angular_velocity(t[tracked], rotations)
            # Drop rates across tracking gaps (finite differences over long jumps are meaningless).
            dt = np.diff(t[tracked])
            neighbours = (dt > 1e-4) & (dt < 3 * np.median(dt))
            gyro_t, gyro = bundle.imu[gyro_key]
            alignment = estimate_imu_alignment(pose_t[neighbours], omega[neighbours], gyro_t, gyro)
            alignment["sensor"] = gyro_key
            report["imu_to_camera"] = alignment
    return report


def _decode_frames(video: Path, frames_dir: Path, stride: int) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"select=not(mod(n\\,{stride}))", "-fps_mode", "passthrough", "-q:v", "2",
        str(frames_dir / "decoded_%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    return len(list(frames_dir.glob("decoded_*.jpg")))


def _remux(video: Path, fps: float, destination: Path) -> Path:
    if video.suffix == ".h264":
        target = destination / "video.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:.3f}", "-i", str(video), "-c", "copy", str(target)]
    else:
        target = destination / "video.webm"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-c", "copy", str(target)]
    subprocess.run(command, check=True)
    return target


def ingest(bundle_path: Path, output: Path, stride: int = 10) -> dict:
    """Unpack a bundle into frames/, colmap_known_poses/, pose_priors.jsonl, video.mp4, report.json."""
    bundle = Bundle(bundle_path)
    output.mkdir(parents=True, exist_ok=True)
    report = bundle_report(bundle)
    video = bundle.extract_video(output)
    if video is None or shutil.which("ffmpeg") is None:
        report["frames_written"] = 0
        (output / "report.json").write_text(json.dumps(report, indent=2))
        return report

    report["remuxed_video"] = _remux(video, report.get("video_fps") or 30.0, output).name
    frames_dir = output / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    decoded = _decode_frames(video, frames_dir, stride)
    report["decoded_frames_matches_chunks"] = decoded == len(range(0, len(bundle.chunk_ts), stride))
    row_by_ts = {int(ts): i for i, ts in enumerate(bundle.frame_video_ts) if np.isfinite(ts)}

    images, priors = [], []
    for k in range(decoded):
        chunk = k * stride
        source = frames_dir / f"decoded_{k + 1:06d}.jpg"
        row = row_by_ts.get(int(bundle.chunk_ts[chunk])) if chunk < len(bundle.chunk_ts) else None
        name = f"frame_{chunk:06d}.jpg"
        source.rename(frames_dir / name)
        if row is None:
            continue
        meta = {"name": name, "timestamp": float(bundle.frame_t[row]), "sharpness": 0.0, "position": None, "gravity": None}
        if bundle.tracking[row] == 0:
            world_from_gl = bundle.matrices[row]
            rotation, translation = world_to_colmap(world_from_gl)
            images.append((name, rotation, translation, row))
            meta["position"] = world_from_gl[:3, 3].round(5).tolist()
            meta["gravity"] = gravity_in_camera(world_from_gl).round(5).tolist()
        priors.append(meta)

    with (output / "pose_priors.jsonl").open("w") as handle:
        for meta in priors:
            handle.write(json.dumps(meta) + "\n")
    report["frames_written"] = decoded
    report["frames_with_pose"] = len(images)
    if images:
        _write_colmap_model(output / "colmap_known_poses", bundle, images)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _write_colmap_model(directory: Path, bundle: Bundle, images: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [row for *_rest, row in images]
    width, height = bundle.image_size[rows[0]]
    # The encoder drops an odd last column or row (H.264 needs even sizes); the principal point is
    # unaffected, but COLMAP must get the size of the decoded frames.
    width, height = width - width % 2, height - height % 2
    focal = np.median(np.array([intrinsics(bundle.projection[r], width, height) for r in rows]), axis=0)
    (directory / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {int(width)} {int(height)} {focal[0]:.4f} {focal[1]:.4f} {focal[2]:.4f} {focal[3]:.4f}\n"
    )
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# (empty POINTS2D line follows each image)"]
    for image_id, (name, rotation, translation, _row) in enumerate(images, start=1):
        q = rotation_to_quaternion(rotation)
        lines.append(" ".join([str(image_id), *(f"{v:.9f}" for v in q), *(f"{v:.9f}" for v in translation), "1", name]))
        lines.append("")
    (directory / "images.txt").write_text("\n".join(lines) + "\n")
    (directory / "points3D.txt").write_text("# 3D point list (empty: triangulate with colmap point_triangulator)\n")
