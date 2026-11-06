import numpy as np

import depth_memory as dm


def test_robust_scale_recovers_scale_with_outliers():
    rng = np.random.default_rng(0)
    z_est = rng.uniform(1, 20, 500)
    z_ref = 2.5 * z_est * (1 + rng.normal(0, 0.01, 500))
    z_ref[:100] = rng.uniform(1, 80, 100)  # occluded / wrong references
    s, inl = dm.robust_scale(z_ref, z_est)
    assert abs(s - 2.5) < 0.03
    assert 0.75 < inl < 0.9


def test_backproject_then_project_roundtrip():
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    a = np.radians(20)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = [0.3, -0.1, 2.0]
    depth = np.full((48, 64), 5.0, np.float32)
    rgb = np.zeros((48, 64, 3), np.uint8)
    pts = dm.backproject(depth, rgb, K, w2c, stride=4)
    idx, uv, z = dm.project(pts[:, :3], K, w2c, 64, 48)
    assert len(idx) == len(pts)
    assert np.allclose(z, 5.0, atol=1e-4)


def test_align_view_uses_visible_reference_points():
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    w2c = np.eye(4)
    true = np.full((48, 64), 6.0, np.float32)
    rgb = np.zeros((48, 64, 3), np.uint8)
    ref = dm.backproject(true, rgb, K, w2c, stride=2)[:, :3]
    s, inl, n = dm.align_view(true / 3.0, K, w2c, ref)
    assert n > 100 and abs(s - 3.0) < 1e-3 and inl > 0.99


def test_depth_edge_mask_drops_jumps():
    d = np.ones((10, 10), np.float32)
    d[:, 5:] = 10.0
    m = dm.depth_edge_mask(d)
    assert not m[:, 4].any() and m[:, 0].all()


def test_panovggt_windows_cover_every_panorama_with_overlap():
    import pano_depth

    w = pano_depth.windows(20, size=8, overlap=3)
    assert w[0][0] == 0 and w[-1][-1] == 19
    covered = sorted({i for win in w for i in win})
    assert covered == list(range(20))
    assert all(len(x) == 8 for x in w)


def test_view_zdepth_turns_radial_into_z_depth():
    import pano_depth
    import scene_io

    radial = np.full((64, 128), 10.0, np.float32)
    K = np.array([[20.0, 0, 20], [0, 20.0, 20], [0, 0, 1]])
    z = pano_depth.view_zdepth(radial, scene_io.legacy_crop_rotation(30), K, 40, 40)
    assert abs(z[20, 20] - 10.0) < 0.05          # centre ray: z == radial
    assert z[0, 0] < z[20, 20]                    # corner ray: z = r cos(angle) < r
