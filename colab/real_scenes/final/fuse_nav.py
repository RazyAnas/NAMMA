"""Feed-forward depth + poses -> TSDF mesh -> collision/walkable grid/floor plan/A* path. CPU only.

Input npz: depth (N,H,W) metres, K (N,3,3), c2w (N,4,4) OpenCV camera (x right, y down, z forward), rgb (N,H,W,3) uint8.
Output dir: mesh.ply (vertex colours), collision.ply, nav.json, floorplan.png.
"""
import json, sys, pathlib
import numpy as np, open3d as o3d
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "experiments" / "nav_from_mesh"))
import nav_from_mesh as N

def fuse(npz, out, voxel=0.02, max_depth=8.0, stride=2):
    """Back-project every depth map into one world point cloud (voxel-merged, normals estimated). Saves cloud.ply + cloud.npz.
    (Open3D's TSDF fusion returned an empty mesh even for a flat wall in the test container, so it is not used here.)"""
    d = np.load(npz); out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    P, C = [], []
    for depth, K, c2w, rgb in zip(d["depth"], d["K"], d["c2w"], d["rgb"]):
        H, W = depth.shape; v, u = np.mgrid[0:H:stride, 0:W:stride]; z = depth[v, u]; ok = (z > 0) & (z < max_depth)
        x = (u[ok] - K[0, 2]) / K[0, 0] * z[ok]; y = (v[ok] - K[1, 2]) / K[1, 1] * z[ok]
        P.append((np.stack([x, y, z[ok]], 1) @ c2w[:3, :3].T) + c2w[:3, 3]); C.append(rgb[v, u][ok])
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.concatenate(P))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(C) / 255.0)
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pc)
    np.savez(out / "cloud.npz", pts=np.asarray(pc.points), nrm=np.asarray(pc.normals))
    return pc

def navigate(mesh_path, out, radius=0.25, cell=0.05):
    out = pathlib.Path(out); R = N.analyse(str(mesh_path), radius, cell)
    walk = R["walk"]; W, H, u0, v0, c = R["grid"]
    idx = np.argwhere(walk)
    s = tuple(idx[np.argmin(idx.sum(1))]); g = tuple(idx[np.argmax(idx.sum(1))])
    path = N.astar(walk, s, g)
    info = dict(up=R["up"].tolist(), floor=float(R["floor"]), grid=[W, H, float(u0), float(v0), c],
                walkable_m2=float(walk.sum() * c * c), floor_seen_m2=float(R["floor_hit"].sum() * c * c),
                demo_path_cells=None if path is None else [list(map(int, p)) for p in path])
    json.dump(info, open(out / "nav.json", "w"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = np.ones((H, W, 3)) * .08; img[R["floor_hit"].T] = [.25, .25, .3]; img[walk.T] = [.2, .55, .35]; img[R["obst"].T] = [.9, .9, .9]
    fig, ax = plt.subplots(figsize=(7, 7 * H / W)); ax.imshow(img, origin="lower"); ax.axis("off")
    if path: ax.plot([p[0] for p in path], [p[1] for p in path], "r-")
    fig.savefig(out / "floorplan.png", dpi=110, bbox_inches="tight")
    return info

if __name__ == "__main__":
    npz, out = sys.argv[1], sys.argv[2]
    fuse(npz, out); print(json.dumps({k: v for k, v in navigate(pathlib.Path(out) / "cloud.npz", out).items() if k != "demo_path_cells"}))
