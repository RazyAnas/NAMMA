import numpy as np
from scipy.spatial.transform import Rotation

import export_game as eg


def _sh1_colour(sh, d):
    # 3DGS: SH_C1 * (-y c0 + z c1 - x c2)
    x, y, z = d
    return eg.SH_C1 * (-y * sh[0] + z * sh[1] - x * sh[2])


def test_sh1_rotation_preserves_view_dependent_colour():
    rng = np.random.default_rng(0)
    R = Rotation.random(random_state=1).as_matrix()
    sh = rng.normal(size=(5, 3, 3))
    sh2 = eg.rotate_sh1(sh, R)
    for _ in range(10):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for n in range(5):
            assert np.allclose(_sh1_colour(sh[n], d), _sh1_colour(sh2[n], R @ d), atol=1e-9)


def test_quaternion_rotation_rotates_covariance():
    rng = np.random.default_rng(2)
    R = Rotation.random(random_state=3).as_matrix()
    q = rng.normal(size=(4, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q2 = eg.quat_mul(np.repeat(eg.rot_to_quat(R)[None], 4, 0), q)
    S = np.diag([1.0, 2.0, 3.0])
    for a, b in zip(q, q2):
        Ra = Rotation.from_quat([a[1], a[2], a[3], a[0]]).as_matrix()
        Rb = Rotation.from_quat([b[1], b[2], b[3], b[0]]).as_matrix()
        assert np.allclose(R @ (Ra @ S @ Ra.T) @ R.T, Rb @ S @ Rb.T, atol=1e-9)


def test_heightfield_finds_floor_under_cameras():
    # free above y=1.0, solid below; cameras at y=3
    lo, vox = np.array([0.0, 0.0, 0.0]), 0.1
    ys = lo[1] + (np.arange(50) + 0.5) * vox
    t = np.where(ys < 1.0, -1.0, 1.0)[None, :, None].repeat(20, 0).repeat(20, 2)
    w = np.ones_like(t)
    g = eg.heightfield(t, w, lo, vox, np.array([[1.0, 3.0, 1.0]]), cell_vox=2, smooth=0.0)
    h = np.asarray(g["heights"])
    assert np.allclose(h, 1.0, atol=0.11) and g["coverage"] == 1.0


def test_game_dirs_axes():
    d = eg.game_dirs(8, 4)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1)
    assert d[0, :, 1].min() > 0.5 and d[-1, :, 1].max() < -0.5   # row 0 looks up
