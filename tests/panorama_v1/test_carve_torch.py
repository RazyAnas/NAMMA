import numpy as np
import pytest

torch = pytest.importorskip("torch")
import depth_memory as dm  # noqa: E402


def _look(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    down = np.array([0.0, -1.0, 0.0])
    down = down - f * down.dot(f)
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    R = np.stack([right, down, f])
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def test_carving_removes_floater_and_keeps_wall():
    # a wall at z = 5; three cameras at z = 0 looking at it
    grid = dm.CarveGrid(np.array([-3, -3, -1.0]), np.array([3, 3, 6.0]), max_voxels=200_000, device="cpu")
    K = np.array([[40.0, 0, 32], [0, 40.0, 32], [0, 0, 1]])
    for x in (-0.5, 0.0, 0.5):
        V = _look(np.array([x, 0, 0.0]), np.array([x, 0, 5.0]))
        depth = np.full((64, 64), 5.0, np.float32)
        grid.add_view(depth, K, V, stride=2)
    pts = torch.tensor([[0.0, 0.0, 2.5], [0.0, 0.0, 5.0], [2.9, 2.9, -0.9]])
    empty = grid.is_empty(pts)
    assert bool(empty[0])        # floater between cameras and wall
    assert not bool(empty[1])    # the wall itself
    assert not bool(empty[2])    # never observed: unknown, not empty


def test_grid_dims_follow_scene_shape():
    g = dm.CarveGrid(np.zeros(3), np.array([40.0, 4.0, 2.0]), max_voxels=100_000, device="cpu")
    assert g.dims[0] > 5 * g.dims[2]
    assert np.allclose(g.size, g.size[0], rtol=0.2)
