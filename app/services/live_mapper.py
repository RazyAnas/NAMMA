"""Streaming tracking and mapping for live phone capture.

Keyframes arrive one by one over a WebSocket. A background thread extracts SIFT for the
new frames, matches each against its recent predecessors (plus spatial neighbours when the
phone sends AR poses), and resumes COLMAP's incremental mapper from the current model so new
frames are registered and bundle-adjusted in seconds on a CPU. The design follows the
tracking/mapping split of PTAM and ORB-SLAM, but reuses COLMAP for every geometric step
instead of re-implementing them, so the live map and the final offline pass share one engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pycolmap

from app.config import Settings


@dataclass
class KeyframeMeta:
    name: str
    timestamp: float
    sharpness: float
    position: list[float] | None = None  # metres, from a phone AR session (WebXR / ARCore)
    gravity: list[float] | None = None  # unit vector in camera coordinates, from DeviceMotion


@dataclass
class LiveMapState:
    keyframes: int = 0
    registered: int = 0
    points: int = 0
    models: int = 0
    last_update_s: float = 0.0
    last_error: str | None = None
    camera_positions: list[list[float]] = field(default_factory=list)
    sample_points: list[list[float]] = field(default_factory=list)
    sample_colors: list[list[int]] = field(default_factory=list)
    lost_frames: list[str] = field(default_factory=list)


class LiveMapper:
    """One mapper per live session; thread-safe `add_keyframe` and `snapshot`."""

    def __init__(self, frames_dir: Path, workspace: Path, settings: Settings):
        self.frames_dir = frames_dir
        self.workspace = workspace
        self.settings = settings
        self.database_path = workspace / "database.db"
        self.model_dir = workspace / "live-model"
        self._pending: list[KeyframeMeta] = []
        self._ordered: list[str] = []
        self._meta: dict[str, KeyframeMeta] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._state = LiveMapState()
        self._thread: threading.Thread | None = None
        workspace.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="namma-live-mapper", daemon=True)
            self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=600)

    def add_keyframe(self, meta: KeyframeMeta) -> None:
        with self._lock:
            self._pending.append(meta)
            self._meta[meta.name] = meta
            self._state.keyframes += 1
        self._wake.set()

    def snapshot(self) -> dict:
        with self._lock:
            state = self._state
            return {
                "keyframes": state.keyframes,
                "registered": state.registered,
                "points": state.points,
                "models": state.models,
                "last_update_s": state.last_update_s,
                "last_error": state.last_error,
                "camera_positions": state.camera_positions,
                "sample_points": state.sample_points,
                "sample_colors": state.sample_colors,
                "lost_frames": state.lost_frames[-20:],
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            with self._lock:
                batch, self._pending = self._pending, []
            if not batch:
                continue
            try:
                started = time.monotonic()
                self._update(batch)
                with self._lock:
                    self._state.last_update_s = time.monotonic() - started
                    self._state.last_error = None
            except Exception as exc:  # keep the session alive; the next batch retries mapping
                with self._lock:
                    self._state.last_error = str(exc)[:500]
        # Final catch-up so every streamed frame is in the model before the offline pass.
        with self._lock:
            batch, self._pending = self._pending, []
        if batch:
            try:
                self._update(batch)
            except Exception as exc:
                with self._lock:
                    self._state.last_error = str(exc)[:500]

    def _update(self, batch: list[KeyframeMeta]) -> None:
        names = [meta.name for meta in batch]
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.sift.max_num_features = self.settings.live_max_features
        pycolmap.extract_features(
            self.database_path,
            self.frames_dir,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=extraction,
            device=pycolmap.Device.auto,
        )
        self._write_priors(batch)

        pairs = []
        for name in names:
            index = len(self._ordered)
            for previous in self._ordered[max(0, index - self.settings.live_match_window) :]:
                pairs.append((previous, name))
            pairs.extend((other, name) for other in self._spatial_neighbours(name))
            self._ordered.append(name)
        pairs_path = self.workspace / "pairs.txt"
        pairs_path.write_text("".join(f"{a} {b}\n" for a, b in dict.fromkeys(pairs)), encoding="utf-8")
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = str(pairs_path)
        pycolmap.match_image_pairs(self.database_path, pairing_options=pairing, device=pycolmap.Device.auto)

        options = pycolmap.IncrementalPipelineOptions()
        options.min_model_size = 3
        options.use_prior_position = any(self._meta[name].position is not None for name in self._ordered)
        output = self.workspace / "live-sparse"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir()
        input_path = self.model_dir if (self.model_dir / "images.bin").is_file() else ""
        maps = pycolmap.incremental_mapping(self.database_path, self.frames_dir, output, options=options, input_path=input_path)
        if not maps:
            with self._lock:
                self._state.lost_frames.extend(names)
            return
        best = max(maps.values(), key=lambda reconstruction: reconstruction.num_reg_images())
        staging = self.workspace / "live-model.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        best.write(str(staging))
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        staging.rename(self.model_dir)
        self._publish(best, len(maps), names)

    def _write_priors(self, batch: list[KeyframeMeta]) -> None:
        with_priors = [meta for meta in batch if meta.position is not None or meta.gravity is not None]
        if not with_priors:
            return
        database = pycolmap.Database.open(str(self.database_path))
        try:
            images = {image.name: image for image in database.read_all_images()}
            for meta in with_priors:
                image = images.get(meta.name)
                if image is None:
                    continue
                prior = pycolmap.PosePrior(corr_data_id=image.data_id)
                if meta.position is not None:
                    prior.position = np.asarray(meta.position, dtype=float)
                    prior.position_covariance = np.eye(3) * self.settings.live_prior_position_sigma_m**2
                    prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
                if meta.gravity is not None:
                    prior.gravity = np.asarray(meta.gravity, dtype=float)
                database.write_pose_prior(prior)
        finally:
            database.close()

    def _spatial_neighbours(self, name: str) -> list[str]:
        """Loop-closure candidates: earlier frames the AR pose says are nearby."""
        meta = self._meta[name]
        if meta.position is None:
            return []
        here = np.asarray(meta.position)
        recent = set(self._ordered[-self.settings.live_match_window :])
        candidates = [
            (float(np.linalg.norm(np.asarray(self._meta[other].position) - here)), other)
            for other in self._ordered
            if other not in recent and self._meta[other].position is not None
        ]
        return [other for distance, other in sorted(candidates)[:10] if distance < self.settings.live_loop_radius_m]

    def _publish(self, reconstruction: pycolmap.Reconstruction, model_count: int, batch_names: list[str]) -> None:
        registered = {image.name for image in reconstruction.images.values() if image.has_pose}
        positions = [
            np.asarray(image.projection_center()).round(4).tolist()
            for image in sorted(reconstruction.images.values(), key=lambda item: item.name)
            if image.has_pose
        ]
        points = list(reconstruction.points3D.values())
        step = max(1, len(points) // self.settings.live_preview_points)
        sample = points[::step]
        with self._lock:
            self._state.registered = len(registered)
            self._state.points = len(points)
            self._state.models = model_count
            self._state.camera_positions = positions
            self._state.sample_points = [np.asarray(point.xyz).round(4).tolist() for point in sample]
            self._state.sample_colors = [np.asarray(point.color).astype(int).tolist() for point in sample]
            # Frames older than this batch that the map still could not place.
            older = self._ordered[: len(self._ordered) - len(batch_names)]
            self._state.lost_frames = [name for name in older if name not in registered]


def write_keyframe_log(path: Path, meta: KeyframeMeta) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(meta.__dict__) + "\n")
