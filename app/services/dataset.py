from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pycolmap


def export_gaussian_dataset(model_dir: Path, frames_dir: Path, destination: Path) -> dict:
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    if not reconstruction.num_reg_images():
        raise RuntimeError("The selected COLMAP model contains no registered images.")

    images_dir = destination / "images"
    sparse_dir = destination / "sparse" / "0"
    if destination.exists():
        shutil.rmtree(destination)
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    reconstruction.write(str(sparse_dir))

    frames: list[dict] = []
    cameras: dict[str, dict] = {}
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        if not image.has_pose:
            continue
        source = frames_dir / image.name
        if not source.is_file():
            raise FileNotFoundError(f"Registered image is missing: {source}")
        shutil.copy2(source, images_dir / image.name)

        camera = reconstruction.cameras[image.camera_id]
        camera_key = str(image.camera_id)
        cameras[camera_key] = _camera_record(camera)
        world_from_camera = image.cam_from_world().inverse().matrix()
        transform = np.eye(4, dtype=float)
        transform[:3, :4] = world_from_camera
        # COLMAP/OpenCV uses +Y down and +Z forward; NeRF viewers use +Y up and +Z back.
        transform[:3, 1:3] *= -1
        frames.append(
            {
                "file_path": f"images/{image.name}",
                "camera_id": image.camera_id,
                "transform_matrix": transform.tolist(),
            }
        )

    transforms = {
        "schema_version": 1,
        "camera_model": "OPENCV",
        "cameras": cameras,
        "frames": frames,
    }
    (destination / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "format": "colmap+transforms_json",
        "registered_image_count": len(frames),
        "source_model": f"sparse/{model_dir.name}",
        "images": "images",
        "colmap_model": "sparse/0",
        "transforms": "transforms.json",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _camera_record(camera: pycolmap.Camera) -> dict:
    calibration = camera.calibration_matrix()
    record = {
        "model": camera.model_name,
        "width": camera.width,
        "height": camera.height,
        "fl_x": float(calibration[0, 0]),
        "fl_y": float(calibration[1, 1]),
        "cx": float(calibration[0, 2]),
        "cy": float(calibration[1, 2]),
        "params": [float(value) for value in camera.params],
    }
    return record
