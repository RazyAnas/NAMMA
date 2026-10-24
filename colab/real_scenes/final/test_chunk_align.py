import numpy as np, chunk_align as CA
rng = np.random.default_rng(1); H, W = 48, 64; K = np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1.]])
# world: a box-ish room; true cameras on a walk; each chunk expressed in its own random Sim3 frame
def depth_of(c2w):   # ray-cast axis-aligned room [-3,3]x[-1.5,1.5]x[-4,4]
    v, u = np.mgrid[0:H, 0:W]; d = np.stack([(u - 32) / 50, (v - 24) / 50, np.ones_like(u, float)], -1) @ c2w[:3, :3].T
    lo, hi = np.array([-3, -1.5, -4.]), np.array([3, 1.5, 4.]); o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (hi - o) / d, (lo - o) / d)
    t = t.min(-1); return (t * (np.stack([(u - 32) / 50, (v - 24) / 50, np.ones_like(u, float)], -1) @ c2w[:3, :3].T @ c2w[:3, 2] / np.linalg.norm(d, axis=-1) * 0 + 1)) * 0 + t * d[..., 2:3].squeeze(-1) * 0 + t * (np.ones_like(t))
def rot(a, b, c):
    from scipy.spatial.transform import Rotation as Rr; return Rr.from_euler("xyz", [a, b, c]).as_matrix()
poses = []
for i in range(14):
    c = np.eye(4); c[:3, :3] = rot(0.1 * np.sin(i), 0.5 * i / 3, 0.05); c[:3, 3] = [-1.5 + 0.2 * i, 0.2 * np.cos(i), -2 + 0.15 * i]; poses.append(c)
def zdepth(c2w):
    v, u = np.mgrid[0:H, 0:W]; rd = np.stack([(u - 32) / 50, (v - 24) / 50, np.ones_like(u, float)], -1); dw = rd @ c2w[:3, :3].T; o = c2w[:3, 3]
    lo, hi = np.array([-3, -1.5, -4.]), np.array([3, 1.5, 4.])
    with np.errstate(divide="ignore", invalid="ignore"): t = np.where(dw > 0, (hi - o) / dw, (lo - o) / dw)
    return t.min(-1)                                   # rd has z=1, so t is the z-depth
chunks = []
for a in range(0, 14 - 5, 5):                          # chunks of 6 frames, 1 shared
    idx = list(range(a, a + 6)); s = rng.uniform(.5, 2); R = rot(*rng.uniform(-1, 1, 3)); t = rng.normal(size=3) * 3
    Tl = np.eye(4); Tl[:3, :3] = R / 1.0; Tl[:3, 3] = t                      # world -> local frame, plus scale s
    dep = np.array([zdepth(poses[g]) * s for g in idx]); c2w = []
    for g in idx:
        m = np.eye(4); m[:3, :3] = R @ poses[g][:3, :3]; m[:3, 3] = s * R @ poses[g][:3, 3] + t; c2w.append(m)
    chunks.append(dict(idx=idx, depth=dep, K=np.tile(K, (6, 1, 1)), c2w=np.array(c2w)))
D, Ks, C, have = CA.merge(chunks)
# compare to ground truth up to ONE global Sim3 (first chunk's frame): check relative geometry via pairwise camera distances / depth scale
g = np.where(have)[0]; gt = np.array([poses[i][:3, 3] for i in g]); est = C[g, :3, 3]
s, R, t = CA.umeyama(est, gt); err = np.linalg.norm((s * est @ R.T + t) - gt, axis=1)
print("frames merged", have.sum(), "| camera-centre error after one global Sim3: max %.4f m" % err.max(), "| depth scale ratio", round(float(s), 3))
