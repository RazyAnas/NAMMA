"""ground_seed: a flat road disk at the road height under every panorama, thin axis vertical."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "colab/real_scenes/panorama_v1"))
import ground_seed  # noqa: E402


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def street(up=np.array([0.0, 1, 0])):
    rng = np.random.default_rng(0)
    centres = np.stack([np.array([i * 10.0, 2.5, 0]) for i in range(5)])
    # road points at y=0 everywhere except a 2.5 m hole under each camera, plus wall points up high
    road = np.c_[rng.uniform(-10, 50, 20000), np.zeros(20000), rng.uniform(-6, 6, 20000)]
    d = np.min(np.linalg.norm(road[:, None, [0, 2]] - centres[None, :, [0, 2]], axis=-1), 1)
    road = road[d > 2.5]
    wall = np.c_[rng.uniform(-10, 50, 3000), rng.uniform(0, 15, 3000), np.full(3000, 6.0)]
    pts = np.concatenate([road, wall])
    rgb = np.concatenate([np.full((len(road), 3), 0.3), np.full((len(wall), 3), 0.8)])
    return centres, pts, rgb


def test_disk_sits_on_road_and_fills_hole():
    centres, pts, rgb = street()
    xyz, col, sc, q = ground_seed.ground_seeds(centres, pts, rgb, np.array([0.0, 1, 0]))
    assert len(xyz) > 1000
    assert np.abs(xyz[:, 1]).max() < 0.05              # on the road, not at camera height
    assert np.allclose(col, 0.3, atol=0.02)            # road colour, not wall colour
    # every camera's hidden disk (2.5 m) is covered
    for c in centres:
        hd = np.linalg.norm(xyz[:, [0, 2]] - c[[0, 2]], axis=1)
        assert (hd < 0.5).sum() > 5 and hd[hd < 3.4].max() > 3.0


def test_thin_axis_is_vertical():
    for up in (np.array([0.0, 1, 0]), np.array([0.0, 0, 1]), np.array([0.3, -0.9, 0.1])):
        q = ground_seed.quat_z_to(up)
        assert np.allclose(_rot(q) @ np.array([0, 0, 1.0]), up / np.linalg.norm(up), atol=1e-6)


def test_no_road_points_uses_default_height():
    centres, _, _ = street()
    xyz, col, _, _ = ground_seed.ground_seeds(centres, np.zeros((0, 3)), np.zeros((0, 3)), np.array([0.0, 1, 0]))
    assert np.allclose(xyz[:, 1], 2.5 - 2.5, atol=1e-6)   # 0.25 x 10 m spacing below each camera
    assert np.isfinite(col).all()
