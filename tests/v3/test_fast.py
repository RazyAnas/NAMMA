"""Fast training (plan of 26 Nov): the trainer patch, the variant recipes and the dense start. No GPU, no gsplat, no downloads."""
import os, sys
import numpy as np
import pytest

FIN = os.path.join(os.path.dirname(__file__), '..', '..', 'colab', 'real_scenes', 'final')
sys.path.insert(0, os.path.abspath(FIN))
import fast_patch, dense_init                                                             # noqa: E402

# the lines of gsplat 1.5.3's simple_trainer.py that the patch hooks into, in file order
TRAINER = '''import os
def create_splats_with_optimizers(parser, init_type="sfm"):
    if init_type == "sfm":
        points = 1
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    return points
class Runner:
    def __init__(self):
        self.trainset = Dataset(self.parser, split="train")
        self.valset = Dataset(self.parser, split="val")
    def train(self):
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                pass
    def eval(self):
        for i in range(3):
            height, width = pixels.shape[1:3]
def main(local_rank, world_rank, world_size, cfg):
    runner = Runner(local_rank, world_rank, world_size, cfg)
'''


def test_patch_applies_once_and_compiles(tmp_path):
    f = tmp_path / 'simple_trainer.py'; f.write_text(TRAINER)
    assert fast_patch.patch(str(f)) is True
    s = f.read_text(); compile(s, 'simple_trainer.py', 'exec')
    assert s.count('fast_patch.thin(self.trainset)') == 1 and s.count('fast_patch.shrink(') == 1 and s.count('fast_patch.dense_points(') == 1
    assert 'global BilateralGrid, color_correct, slice, total_variation_loss' in s      # the two-GPU import fix
    assert s.index('fast_patch.shrink(') < s.index('if cfg.pose_noise') and s.index('fast_patch.shrink(') < s.index('def eval')   # training loop only
    assert fast_patch.patch(str(f)) is True and f.read_text() == s                     # a second call changes nothing


def test_patch_leaves_an_unknown_trainer_alone(tmp_path):
    f = tmp_path / 'simple_trainer.py'; other = TRAINER.replace('if cfg.pose_noise:', 'if cfg.something_else:'); f.write_text(other)
    assert fast_patch.patch(str(f)) is False and f.read_text() == other


def test_recipe_tokens():
    assert fast_patch.recipe('gs', 30000) is None and fast_patch.recipe('prior+geo', 30000) is None      # stock recipes: untouched
    r = fast_patch.recipe('fbg+thin3+res0.5_0.6+dense', 15000, fused=True, dense='/x/dense.npz')
    assert r['grid'] == '--use-fused-bilagrid' and 'NM_TRAIN_EVERY=3' in r['env'] and 'NM_RES=0.5:0.6' in r['env'] and 'NM_DENSE=/x/dense.npz' in r['env']
    assert '--init-scale' in r['flags'] and r['notes'] == []
    r = fast_patch.recipe('fbg+dense', 15000, fused=False, dense=None)                                    # nothing installed / made:
    assert r['grid'] == '--use-bilateral-grid' and 'NM_DENSE' not in r['env'] and len(r['notes']) == 2     # falls back and says so
    assert fast_patch.recipe('nobg', 7000)['grid'] == ''
    assert fast_patch.recipe('fbg+2gpu+packed', 7000)['env'] == ''
    with pytest.raises(ValueError): fast_patch.recipe('fbg+tihn3', 7000)                                 # a typo must not pass silently


def test_scaler_gives_exactly_the_asked_steps():
    for steps in (500, 6999, 7000, 10000, 12345, 15000, 20000, 30000):
        assert int(fast_patch.NOMINAL * fast_patch.recipe('fbg', steps)['scaler']) == steps


def test_resolution_plan_ends_at_full_size():
    assert fast_patch.res_plan(15000, '') is None and fast_patch.res_plan(15000, '1.0:0.6') is None and fast_patch.res_plan(15000, '0.5:0') is None
    s, until = fast_patch.res_plan(15000, '0.5:0.6')
    assert s == 0.5 and until == 9000                                                    # steps 9000.. run at full size
    assert fast_patch.res_plan(700, '0.5:1')[1] == 700                                   # the speed check: small all the way


def test_thin_keeps_every_kth_training_frame_only(monkeypatch):
    class DS: pass
    allf = np.arange(924); tr, va = DS(), DS(); tr.indices = allf[allf % 8 != 0]; va.indices = allf[allf % 8 == 0]
    monkeypatch.setenv('NM_TRAIN_EVERY', '3'); fast_patch.thin(tr)
    assert len(tr.indices) == 270 and not set(tr.indices) & set(va.indices) and len(va.indices) == 116   # held-out frames untouched
    monkeypatch.delenv('NM_TRAIN_EVERY'); before = tr.indices.copy(); fast_patch.thin(tr); assert (tr.indices == before).all()


def test_shrink_keeps_the_picture_and_the_cameras_together():
    torch = pytest.importorskip('torch')
    H, W = 808, 1440; px = torch.rand(1, H, W, 3); K = torch.tensor([[[1100.0, 0, 720.0], [0, 1100.0, 404.0], [0, 0, 1]]])
    pts = torch.tensor([[[0.0, 0.0], [1439.0, 807.0], [719.5, 403.5]]])
    p2, K2, q = fast_patch.shrink(px, K, 0.5, pts)
    assert p2.shape == (1, 404, 720, 3) and abs(float(p2.mean() - px.mean())) < 1e-3     # area average: same brightness
    assert torch.allclose(K2[0, 0], K[0, 0] * 0.5) and torch.allclose(K2[0, 1], K[0, 1] * 0.5) and K[0, 0, 0] == 1100   # input not changed
    assert q.min() >= 0 and q[0, 1, 0] <= 719 and q[0, 1, 1] <= 403                       # sample positions stay inside
    assert torch.allclose(q[0, 2], torch.tensor([359.5, 201.5]))                          # the picture centre stays the centre
    same = fast_patch.shrink(px, K, 1.0, pts); assert same[0] is px


# ---- dense start: a made-up room (a box seen from inside) with exact depth maps
ROOM = np.array([4.0, 2.6, 3.0])                                                          # x, y (height), z in "metres"


def look(eye, yaw):
    """Camera-to-world, +z forward (OpenCV), level, turned by yaw about the vertical (y down in the camera)."""
    f = np.array([np.sin(yaw), 0.0, np.cos(yaw)]); r = np.array([np.cos(yaw), 0.0, -np.sin(yaw)]); d = np.cross(f, r)
    c = np.eye(4); c[:3, 0], c[:3, 1], c[:3, 2], c[:3, 3] = r, d, f, eye; return c


def box_depth(c2w, K, h, w):
    """Exact z-depth of the room's inside walls for every pixel."""
    v, u = np.mgrid[0:h, 0:w]; d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    dw = d @ c2w[:3, :3].T; o = c2w[:3, 3]; t = np.full((h, w), np.inf)
    for ax in range(3):
        for wall in (0.0, ROOM[ax]):
            with np.errstate(divide='ignore', invalid='ignore'): tt = (wall - o[ax]) / dw[..., ax]
            t = np.where((tt > 1e-6) & (tt < t), tt, t)
    return t.astype(np.float32)                                                           # d has z = 1, so t is the z-depth


def wall_distance(p): return np.minimum(p, ROOM - p).min(1)


def views(n=10, h=90, w=160):
    K = np.array([[110.0, 0, w / 2], [0, 110.0, h / 2], [0, 0, 1]]); cs = [look(np.array([2.0 + 0.5 * np.cos(a), 1.3, 1.5 + 0.4 * np.sin(a)]), a) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    return [box_depth(c, K, h, w) for c in cs], [K] * n, cs, [np.full((h, w, 3), 100 + 10 * i, np.uint8) for i in range(n)]


def test_dense_points_land_on_the_walls():
    D, K, C, I = views(); xyz, rgb = dense_init.cloud(D, K, C, I, voxel=0.04)
    assert len(xyz) > 2000 and rgb.dtype == np.uint8 and rgb.min() >= 100 and rgb.max() <= 190
    assert np.percentile(np.abs(wall_distance(xyz)), 99) < 0.03                           # on a wall, floor or ceiling


def test_one_wrong_depth_map_plants_nothing_in_the_air():
    D, K, C, I = views(); D[3] = D[3] * 0.6                                               # one frame's depth far too short
    xyz, _ = dense_init.cloud(D, K, C, I, voxel=0.04)
    assert np.mean(wall_distance(xyz) > 0.15) < 0.01                                      # its points had no second frame agreeing
    alone, _ = dense_init.cloud(D[3:4], K[3:4], C[3:4], I[3:4], voxel=0.04)
    assert len(alone) == 0                                                                # a single frame never passes on its own


def test_cap_and_edges():
    D, K, C, I = views(); xyz, _ = dense_init.cloud(D, K, C, I, voxel=0.04, cap=500); assert len(xyz) == 500
    z = np.ones((6, 8), np.float32); z[:, 4:] = 3.0; e = dense_init.edges(z)
    assert e[:, 3].all() and e[:, 4].all() and not e[:, :3].any() and not e[:, 5:].any()   # only the two sides of the jump


def test_sfm_scale_recovers_the_scale():
    D, K, C, _ = views(1, 90, 160); z = D[0]; c2w = C[0]; rng = np.random.default_rng(0)
    v, u = rng.integers(5, 85, 200), rng.integers(5, 155, 200); d = z[v, u]
    pc = np.stack([(u + 0.5 - K[0][0, 2]) / 110.0 * d, (v + 0.5 - K[0][1, 2]) / 110.0 * d, d], 1); pts = pc @ c2w[:3, :3].T + c2w[:3, 3]
    s, n, spread = dense_init.sfm_scale(z / 2.5, K[0], 160, 90, np.linalg.inv(c2w), pts)   # the network's depth is 2.5x too small
    assert n >= 150 and abs(s - 2.5) < 0.05 and spread < 0.02
    assert dense_init.sfm_scale(z, K[0], 160, 90, np.linalg.inv(c2w), pts[:10])[0] is None  # too few points to trust
