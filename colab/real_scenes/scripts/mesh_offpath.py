# Off-path test of the SkyeBrowse-style answer: a mesh instead of a splat. Fuses the training frames' depth (clean/boot_depth,
# uint16 mm) with their poses into a TSDF, extracts a coloured mesh, then ray-casts it from the held-out stretch's cameras
# (make_block_split.py) and scores PSNR / SSIM against the real frames exactly like score_whole.py. Pixels the mesh misses
# are filled with the image mean so holes cost what a grey guess costs, and coverage is reported.
# TEXTURE=k colours each hit from the k training frames nearest the test camera that see that point (depth test against their
# own depth maps), weighted by viewing-angle similarity: photogrammetry-style image texturing; vertex colour where none does.
# Usage: [TEXTURE=4] python mesh_offpath.py <work with block split> <out dir> [voxel_m=0.01] [trunc_vox=4] [depth_max_m=4]
import sys, os, json, numpy as np, cv2, open3d as o3d
from skimage.metrics import structural_similarity
W, O = sys.argv[1:3]; vox = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
tr = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0; dmax = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
os.makedirs(O, exist_ok=True)
st = json.load(open(f'{W}/clean/stats.json')); P = json.load(open(f'{W}/clean/poses_w2c.json')); K = np.array(st['K'])
Wd, H = st['width'], st['height']; intr = o3d.camera.PinholeCameraIntrinsic(Wd, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
col = lambda f: cv2.imread(f'{W}/data/color_raw/{f:05d}.jpg')
import open3d.core as oc
Kt = oc.Tensor(K, oc.float64)
vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(oc.float32, oc.float32, oc.float32),
                                    attr_channels=((1), (1), (3)), voxel_size=vox, block_resolution=16, block_count=30000, device=oc.Device('CPU:0'))
used = 0
for f in st['train_frames']:
    p = f'{W}/clean/boot_depth/{f:05d}.png'
    if not os.path.exists(p): continue
    d = cv2.imread(p, -1); c = col(f)
    if c.shape[:2] != d.shape: c = cv2.resize(c, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
    D = o3d.t.geometry.Image(oc.Tensor(d)); Ci = o3d.t.geometry.Image(oc.Tensor(np.ascontiguousarray(c[:, :, ::-1])))
    E = oc.Tensor(np.array(P[str(f)]), oc.float64)
    blk = vbg.compute_unique_block_coordinates(D, Kt, E, 1000., dmax, tr)
    vbg.integrate(blk, D, Ci, Kt, Kt, E, 1000., dmax, tr); used += 1
m = vbg.extract_triangle_mesh().to_legacy(); m.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(f'{O}/mesh.ply', m)
V = np.asarray(m.vertices); F = np.asarray(m.triangles); C = np.asarray(m.vertex_colors)
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
u, v = np.meshgrid(np.arange(Wd) + .5, np.arange(H) + .5); dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
KT = int(os.environ.get('TEXTURE', 0)); DT = float(os.environ.get('DEPTH_TOL', 0.03)); TR = {}; rows_tex = []
if KT:
    for f in st['train_frames']:
        p = f'{W}/clean/boot_depth/{f:05d}.png'
        if os.path.exists(p): Ew = np.array(P[str(f)]); TR[f] = (Ew, np.linalg.inv(Ew), col(f).astype(np.float32) / 255, cv2.imread(p, -1).astype(np.float64) / 1000.)
rows = []
for f in st['test_frames']:
    T = np.linalg.inv(np.array(P[str(f)])); d = dirs_c @ T[:3, :3].T
    rays = np.concatenate([np.broadcast_to(T[:3, 3], d.shape), d], -1).astype(np.float32)
    r = sc.cast_rays(o3d.core.Tensor(rays)); hit = np.isfinite(r['t_hit'].numpy())
    tid = r['primitive_ids'].numpy(); uv = r['primitive_uvs'].numpy()
    g = col(f).astype(np.float64) / 255
    if g.shape[:2] != (H, Wd): g = cv2.resize(g, (Wd, H), interpolation=cv2.INTER_AREA)
    img = np.broadcast_to(g.reshape(-1, 3).mean(0), g.shape).copy()   # holes cost a grey guess
    t = F[tid[hit]]; b1, b2 = uv[hit][:, :1], uv[hit][:, 1:]
    img[hit] = (C[t[:, 0]] * (1 - b1 - b2) + C[t[:, 1]] * b1 + C[t[:, 2]] * b2)[:, ::-1]
    if KT:
        X = rays[hit][:, :3] + rays[hit][:, 3:] * r['t_hit'].numpy()[hit][:, None]; vt = X - T[:3, 3]; vt /= np.linalg.norm(vt, axis=1, keepdims=True)
        near = sorted(TR, key=lambda q: np.linalg.norm(TR[q][1][:3, 3] - T[:3, 3]) + 0.3 * (1 - TR[q][1][:3, 2] @ T[:3, 2]))[:KT]
        acc = np.zeros((len(X), 3)); wsum = np.zeros(len(X))
        for q in near:
            Ew, Tq, Iq, Dq = TR[q]; Y = X @ Ew[:3, :3].T + Ew[:3, 3]; z = Y[:, 2]
            ok = z > 0.05; px = np.where(ok, K[0, 0] * Y[:, 0] / np.maximum(z, 1e-6) + K[0, 2], -1); py = np.where(ok, K[1, 1] * Y[:, 1] / np.maximum(z, 1e-6) + K[1, 2], -1)
            ok &= (px >= 0) & (px < Wd - 1) & (py >= 0) & (py < H - 1); xi, yi = px.astype(int).clip(0, Wd - 1), py.astype(int).clip(0, H - 1)
            ok &= np.abs(Dq[yi, xi] - z) < DT * z + 0.02            # the training frame sees this point, not something in front of it
            vq = X - Tq[:3, 3]; vq /= np.linalg.norm(vq, axis=1, keepdims=True); w = np.clip((vt * vq).sum(1), 0, 1) ** 8 * ok
            fx_, fy_ = (px - xi)[:, None], (py - yi)[:, None]; x1, y1 = (xi + 1).clip(0, Wd - 1), (yi + 1).clip(0, H - 1)
            c = (Iq[yi, xi] * (1 - fx_) + Iq[yi, x1] * fx_) * (1 - fy_) + (Iq[y1, xi] * (1 - fx_) + Iq[y1, x1] * fx_) * fy_
            acc += w[:, None] * c; wsum += w
        tex = wsum > 1e-6; sub = img[hit]; sub[tex] = acc[tex] / wsum[tex][:, None]; img[hit] = sub; rows_tex.append(float(tex.mean()))
    cv2.imwrite(f'{O}/test_{f:05d}.png', (img.clip(0, 1) * 255).round().astype(np.uint8))
    rows.append(dict(frame=f, psnr=10 * np.log10(1 / ((img - g) ** 2).mean()), ssim=structural_similarity(img, g, channel_axis=2, data_range=1),
                     coverage=float(hit.mean())))
p = np.array([x['psnr'] for x in rows])
res = dict(n_views=len(rows), resolution=f'{Wd}x{H}', voxel_m=vox, fused_frames=used, triangles=len(F), psnr_mean=float(p.mean()),
           psnr_median=float(np.median(p)), psnr_min=float(p.min()), psnr_max=float(p.max()),
           ssim_mean=float(np.mean([x['ssim'] for x in rows])), coverage_mean=float(np.mean([x['coverage'] for x in rows])), texture=KT, textured_frac=float(np.mean(rows_tex)) if rows_tex else 0., per_frame=rows)
json.dump(res, open(f'{O}/scores.json', 'w'), indent=1); print({k: v for k, v in res.items() if k != 'per_frame'})
