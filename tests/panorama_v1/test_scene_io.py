import math

import numpy as np
import pytest

import prepare_panorama as pp
import scene_io


def _spherical_px(ray, w, h):
    """pycolmap.panorama.spherical_img_from_cam for one ray (y down)."""
    yaw = math.atan2(ray[0], ray[2])
    pitch = -math.atan2(ray[1], math.hypot(ray[0], ray[2]))
    return (1 + yaw / math.pi) / 2 * w, (1 - pitch * 2 / math.pi) / 2 * h


@pytest.mark.parametrize("yaw", [0.0, 30.0, 150.0, 300.0])
def test_legacy_crop_rotation_matches_prepare_panorama(yaw):
    w, h, W, H = 64, 48, 400, 200
    mx, my, f = pp.perspective_maps(w, h, W, H, yaw, 60.0)
    R = scene_io.legacy_crop_rotation(yaw)  # cam_from_pano
    for (u, v) in [(5, 7), (32, 24), (60, 40)]:
        ray_cam = np.array([(u - w / 2) / f, (v - h / 2) / f, 1.0])
        ray_pano = R.T @ ray_cam
        x, y = _spherical_px(ray_pano, W, H)
        assert abs(((x - mx[v, u] + W / 2) % W) - W / 2) < 1e-2
        assert abs(y - my[v, u]) < 1e-2


def test_holdout_is_whole_panoramas_and_never_endpoints():
    ids = [f"{i:04d}" for i in range(40)]
    val = scene_io.holdout_panos(ids)
    assert "0000" not in val and "0039" not in val
    assert val == {"0004", "0012", "0020", "0028", "0036"}
    cams = [{"pano": p} for p in ids for _ in range(12)]
    scene_io.assign_splits(cams)
    per = {}
    for c in cams:
        per.setdefault(c["pano"], set()).add(c["split"])
    assert all(len(s) == 1 for s in per.values())


def test_openmvg_pose_convention_and_rig_views():
    R = np.eye(3)
    sfm = {"views": [{"key": 0, "value": {"ptr_wrapper": {"data": {"filename": "a.png", "id_pose": 7}}}}],
           "extrinsics": [{"key": 7, "value": {"rotation": R.tolist(), "center": [1.0, 2.0, 3.0]}}]}
    poses = scene_io.read_openmvg_poses(sfm)
    T = poses["a.png"]
    assert np.allclose(scene_io.camera_centre(T), [1, 2, 3])
    rots = [scene_io.legacy_crop_rotation(a) for a in (0, 90, 180)]
    K = np.array([[10.0, 0, 5], [0, 10, 5], [0, 0, 1]])
    cams = scene_io.rig_views_from_poses({"a.png": T}, "/v", rots, K, 10, 10)
    assert len(cams) == 3
    assert scene_io.within_pano_spread(cams) < 1e-9
    assert {c["pano"] for c in cams} == {"a"}


def test_world_up_from_symmetric_rig():
    rots = []
    for pitch in (-35, 0, 35):
        for yaw in range(0, 360, 90):
            a, b = math.radians(-pitch), math.radians(-yaw)
            Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
            rots.append(Rx @ Ry)
    cams = scene_io.rig_views_from_poses({"p.jpg": np.eye(4)}, "/v", rots, np.eye(3), 1, 1)
    up = scene_io.world_up_from_cameras(cams)
    assert np.allclose(up, [0, -1, 0], atol=1e-6)


def test_pano_of_name():
    assert scene_io.pano_of_name("pano_camera3/0012-0012.jpg") == "0012-0012"
    assert scene_io.pano_of_name("p00012_v03.jpg") == "p00012"
