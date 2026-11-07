from __future__ import annotations

import sys
import types

import numpy as np

from rig_adjust import enforce_panorama_rig_closure


class FakeRigid3d:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=np.float64).copy()

    def matrix(self):
        return self._matrix.copy()


class FakeFrame:
    def __init__(self, frame_id, camera_id, pose):
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.pose = pose

    def set_cam_from_world(self, camera_id, cam_from_world):
        assert camera_id == self.camera_id
        self.pose = cam_from_world


class FakeImage:
    def __init__(self, name, image_id, frame):
        self.name = name
        self.image_id = image_id
        self.frame_id = frame.frame_id
        self.camera_id = frame.camera_id
        self.frame = frame

    def cam_from_world(self):
        return self.frame.pose

    def num_points3D(self):
        return 10


class FakeReconstruction:
    def __init__(self, images):
        self.images = {image.image_id: image for image in images}


def make_camera_to_world(center, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    pose[:3, 3] = center
    return pose


def test_rig_closure_updates_colmap_frames_and_preserves_each_rotation(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pycolmap",
        types.SimpleNamespace(Rigid3d=FakeRigid3d),
    )

    images = []
    expected_rotations = {}
    image_id = 1
    frame_id = 1
    for pano_index, base_center in enumerate((
        np.array([1.0, 2.0, 3.0]),
        np.array([-4.0, 0.5, 8.0]),
    )):
        for view_index in range(6):
            # Deliberately inconsistent estimated centres simulate the SfM drift.
            drift = np.array([
                (view_index - 2.5) * 0.7,
                (view_index % 3 - 1) * 0.4,
                (view_index % 2) * 0.3,
            ])
            center = base_center + drift
            c2w = make_camera_to_world(center, view_index * np.pi / 3)
            expected_rotations[f"p{pano_index:05d}_v{view_index:02d}.jpg"] = c2w[:3, :3].copy()
            w2c = np.linalg.inv(c2w)
            frame = FakeFrame(frame_id, 1, FakeRigid3d(w2c[:3, :]))
            images.append(FakeImage(
                f"p{pano_index:05d}_v{view_index:02d}.jpg", image_id, frame
            ))
            image_id += 1
            frame_id += 1

    reconstruction = FakeReconstruction(images)
    report = enforce_panorama_rig_closure(reconstruction)

    assert report["panorama_groups"] == 2
    assert report["registered_views_constrained"] == 12
    assert report["post_closure_max_residual"] < 1e-10

    centers_by_pano = {}
    for image in reconstruction.images.values():
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = image.cam_from_world().matrix()
        c2w = np.linalg.inv(w2c)
        pano_id = image.name.split("_v")[0]
        centers_by_pano.setdefault(pano_id, []).append(c2w[:3, 3])
        assert np.allclose(
            c2w[:3, :3],
            expected_rotations[image.name],
            atol=1e-10,
        )

    for centers in centers_by_pano.values():
        assert np.allclose(centers, np.repeat([centers[0]], len(centers), axis=0))
