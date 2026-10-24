# Mesh for the walking demo's hybrid view: fuse every frame that has anchored depth (clean/boot_depth, uint16 mm) into a
# TSDF with its pose (same fusion as mesh_offpath.py, which measured 18.76 dB off the capture path against 15.3-16.6 for
# the splats), decimate it and write a vertex-coloured GLB in splat coordinates. The viewer shows it behind the splat and
# fades the splat out as the camera leaves the capture path.
# Usage: python make_demo_mesh.py <scene dir with clean/> <frames dir> <out.glb> [voxel_m=0.015] [triangles=300000] [depth_max_m=4]
import sys, os, json, numpy as np, cv2, open3d as o3d, open3d.core as oc, trimesh
S, FR, OUT = sys.argv[1:4]; vox = float(sys.argv[4]) if len(sys.argv) > 4 else 0.015
ntri = int(sys.argv[5]) if len(sys.argv) > 5 else 300000; dmax = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0
st = json.load(open(f'{S}/clean/stats.json')); P = json.load(open(f'{S}/clean/poses_w2c.json')); K = oc.Tensor(np.array(st['K']), oc.float64)
vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(oc.float32, oc.float32, oc.float32),
                                    attr_channels=((1), (1), (3)), voxel_size=vox, block_resolution=16, block_count=40000, device=oc.Device('CPU:0'))
used = 0
for name in sorted(os.listdir(f'{S}/clean/boot_depth')):
    f = int(name[:5]); c = cv2.imread(f'{FR}/{f:05d}.jpg')
    if str(f) not in P or c is None: continue
    d = cv2.imread(f'{S}/clean/boot_depth/{name}', -1)
    if c.shape[:2] != d.shape: c = cv2.resize(c, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
    D = o3d.t.geometry.Image(oc.Tensor(d)); C = o3d.t.geometry.Image(oc.Tensor(np.ascontiguousarray(c[:, :, ::-1])))
    E = oc.Tensor(np.array(P[str(f)]), oc.float64)
    blk = vbg.compute_unique_block_coordinates(D, K, E, 1000., dmax, 4.0)
    vbg.integrate(blk, D, C, K, K, E, 1000., dmax, 4.0); used += 1
m = vbg.extract_triangle_mesh().to_legacy(); m.remove_unreferenced_vertices(); full = len(m.triangles)
# drop small floating pieces, then decimate
labels, counts, _ = m.cluster_connected_triangles(); labels = np.asarray(labels); counts = np.asarray(counts)
m.remove_triangles_by_mask(counts[labels] < 200); m.remove_unreferenced_vertices()
if len(m.triangles) > ntri: m = m.simplify_quadric_decimation(ntri)
m.remove_unreferenced_vertices()
V = np.asarray(m.vertices, dtype=np.float32); F = np.asarray(m.triangles); Cv = (np.asarray(m.vertex_colors).clip(0, 1) * 255).astype(np.uint8)
tm = trimesh.Trimesh(V, F, vertex_colors=np.c_[Cv, np.full(len(Cv), 255, np.uint8)], process=False)
tm.export(OUT)
print(json.dumps(dict(frames=used, voxel_m=vox, triangles_full=full, triangles=len(F), vertices=len(V), mb=round(os.path.getsize(OUT) / 1e6, 1))))
