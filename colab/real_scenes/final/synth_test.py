"""CPU self-test: render depth from a known mesh at known poses, run fuse_nav, compare walkable area with the direct mesh result."""
import sys, time, numpy as np, open3d as o3d, trimesh, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent)); import fuse_nav as F, nav_from_mesh as N
src = sys.argv[1]; out = sys.argv[2]
m = trimesh.load(src, force="mesh"); R0 = N.analyse(src); up = R0["up"]; floor = R0["floor"]
t, b = N.floor_frame(up)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))))
Wd, Hd = 160, 120; f = 110.0; K = np.array([[f, 0, Wd/2], [0, f, Hd/2], [0, 0, 1]])
free = np.argwhere(R0["walk"]); gW, gH, u0, v0, c = R0["grid"]
rng = np.random.default_rng(0); D, C, RGB = [], [], []
for i in range(120):
    iu, iv = free[rng.integers(len(free))]
    pos = (u0 + (iu + .5) * c) * t + (v0 + (iv + .5) * c) * b + (floor + 1.4) * up   # eye height 1.4 m
    yaw = rng.uniform(0, 2 * np.pi); fwd = np.cos(yaw) * t + np.sin(yaw) * b
    z = fwd; y = -up; x = np.cross(y, z)                              # OpenCV: x right, y down, z forward
    c2w = np.eye(4); c2w[:3, :3] = np.stack([x, y, z], 1); c2w[:3, 3] = pos
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(K, np.linalg.inv(c2w), Wd, Hd)
    ans = scene.cast_rays(rays); th = ans["t_hit"].numpy()
    d = np.where(np.isfinite(th), th, 0).astype(np.float32)
    dirs = rays.numpy()[..., 3:]; zc = (dirs @ z)                   # t_hit is along unit ray; convert to z-depth
    D.append(d * zc); C.append(c2w); RGB.append(np.full((Hd, Wd, 3), 180, np.uint8))
np.savez(out + "/synth.npz", depth=np.array(D), K=np.tile(K, (len(D), 1, 1)), c2w=np.array(C), rgb=np.array(RGB))
t0 = time.time(); F.fuse(out + "/synth.npz", out, voxel=0.03); info = F.navigate(out + "/cloud.npz", out)
print(f"direct walkable {R0['walk'].sum()*c*c:.2f} m2 | from depth views {info['walkable_m2']:.2f} m2 | fuse+nav {time.time()-t0:.1f}s")
