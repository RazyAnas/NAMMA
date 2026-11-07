import numpy as np

import densify


def _route(n=6, spacing=1.0):
    cams = []
    for i in range(n):
        T = np.eye(4)
        T[:3, 3] = [-i * spacing, 0, 0]  # centres along +x
        cams.append({"pano": f"{i:04d}_x", "w2c": T})
    return cams


def test_metres_per_unit_from_gps_spacing():
    cams = _route(6, 1.0)
    man = {"panoramas": [{"lat": 48.86, "lon": 2.345 + i * 10 / 73300.0} for i in range(6)]}
    m = densify.metres_per_norm_unit(cams, np.zeros(3), 1.0, man)
    assert 9.0 < m < 11.0          # ~10 m between panoramas, 1 unit apart


def test_dense_paths_walk_the_route_at_eye_height():
    cams = _route(6, 1.0)
    up = np.array([0, 1.0, 0])
    clips = densify.dense_paths(cams, np.zeros(3), 1.0, up, eye_drop_units=0.05, step_units=0.02, frames=49)
    assert clips
    label, mats = clips[0]
    assert len(mats) == 49
    eyes = np.stack([-(V[:3, :3].T @ V[:3, 3]) for V in mats])
    assert np.allclose(eyes[:, 1], -0.05, atol=1e-6)          # lowered to walking height
    assert np.all(np.diff(eyes[:, 0]) > 0) or np.all(np.diff(eyes[:, 0]) >= -1e-9)
    fwd = mats[0][:3, :3].T @ np.array([0, 0, 1.0])
    assert fwd @ up < 0                                        # looking slightly down
