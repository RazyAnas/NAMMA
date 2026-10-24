# Initial Gaussians from the fused mesh (mesh_offpath.py's mesh.ply): sample points uniformly on its surface with their
# vertex colours and write an xyz+rgb ply in the format gsplat_train.py reads, so splats start on a consistent surface
# instead of on per-frame depth points. Usage: python mesh_points.py <mesh.ply> <out.ply> [n=300000]
import sys, numpy as np, open3d as o3d
from plyfile import PlyData, PlyElement
m = o3d.io.read_triangle_mesh(sys.argv[1]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 300000
pc = m.sample_points_uniformly(n); X = np.asarray(pc.points); C = (np.asarray(pc.colors).clip(0, 1) * 255).round().astype(np.uint8)
a = np.empty(len(X), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
a['x'], a['y'], a['z'] = X.T; a['red'], a['green'], a['blue'] = C.T
PlyData([PlyElement.describe(a, 'vertex')]).write(sys.argv[2]); print(len(X), 'points')
