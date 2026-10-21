"""Stages 4-8 of run.py: depth, mesh, splat, score, strips."""
import os, json, time, glob
import numpy as np
import cv2


def _log(W, m):
    from run import log
    log(W, m)


def load_scene(W):
    S = json.load(open(f'{W}/scene.json'))
    S['K'] = np.array(S['K']); S['w2c'] = {int(k): np.array(v) for k, v in S['w2c'].items()}
    return S


def read_img(W, f):
    return cv2.imread(f'{W}/img/{f:05d}.png')[:, :, ::-1].astype(np.float32) / 255.


# ------------------------------------------------------------------------------------------------------------------ depth
def stage_depth(a, W):
    import onnxruntime as ort
    S = load_scene(W); K = S['K']; w, h = S['width'], S['height']
    P = np.load(f'{W}/scene_points.npz'); X = P['X']; good = (P['track'] >= 3) & (P['err'] < 2.0)
    sx, sy = w / S['size_full'][0], h / S['size_full'][1]
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(a.depth_model, so, providers=['CPUExecutionProvider'])
    # aspect-preserving network input: long side ~ 952 px (multiple of 14), short side scaled by the same factor
    s = 952 / max(w, h); iw, ih = int(round(w * s / 14)) * 14, int(round(h * s / 14)) * 14
    os.makedirs(f'{W}/depth', exist_ok=True)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    stats = []
    frames = S['train']
    for f in frames:
        out = f'{W}/depth/{f:05d}.png'
        if os.path.exists(out): continue
        im = read_img(W, f)
        x = ((cv2.resize(im, (iw, ih), interpolation=cv2.INTER_CUBIC) - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        disp = sess.run(None, {'image': x})[0][0]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
        uv = P[f'uv_{f}'] * [sx, sy]; ids = P[f'id_{f}']; keep = good[ids]
        uv, ids = uv[keep], ids[keep]
        M = S['w2c'][f]; z = (X[ids] @ M[:3, :3].T + M[:3, 3])[:, 2]
        ui = np.clip((uv[:, 0] - .5).round().astype(int), 0, w - 1); vi = np.clip((uv[:, 1] - .5).round().astype(int), 0, h - 1)
        d = disp[vi, ui]; ok = z > 0.05
        d, z = d[ok], z[ok]
        if len(z) < 20:
            stats.append(dict(frame=f, n=int(len(z)), skipped=True)); continue
        # robust fit of inverse depth = A * disparity + B (iteratively reweighted, Cauchy-like)
        A_ = np.stack([d, np.ones_like(d)], 1); y = 1 / z; wt = np.ones_like(y)
        for _ in range(10):
            coef = np.linalg.lstsq(A_ * wt[:, None], y * wt, rcond=None)[0]
            r = (A_ @ coef - y) / y; c = 1.4826 * np.median(np.abs(r)) + 1e-6; wt = 1 / (1 + (r / (2 * c)) ** 2)
        inv = coef[0] * disp + coef[1]
        dep = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), 0)
        dep[(dep > a.depth_max * 1.5)] = 0
        rel = np.abs(1 / np.maximum(A_ @ coef, 1e-3) - z) / z
        stats.append(dict(frame=f, n=int(len(z)), a=float(coef[0]), b=float(coef[1]), relerr_median=float(np.median(rel)),
                          relerr_p90=float(np.percentile(rel, 90))))
        cv2.imwrite(out, (dep * 1000).clip(0, 65535).astype(np.uint16))
    if stats: json.dump(stats, open(f'{W}/depth/fit_stats.json', 'w'), indent=1)
    stats = json.load(open(f'{W}/depth/fit_stats.json'))
    # multi-view consistency: keep a pixel if its 3D point lands within 3% depth in at least one of the 2 nearest train views
    D = {f: cv2.imread(f'{W}/depth/{f:05d}.png', -1).astype(np.float32) / 1000 for f in frames if os.path.exists(f'{W}/depth/{f:05d}.png')}
    os.makedirs(f'{W}/depth_cons', exist_ok=True)
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5); Ki = np.linalg.inv(K); kept = []
    C = {f: -S['w2c'][f][:3, :3].T @ S['w2c'][f][:3, 3] for f in D}
    for f in D:
        out = f'{W}/depth_cons/{f:05d}.png'
        z = D[f]; ray = np.stack([u, v, np.ones_like(u)], -1) @ Ki.T; Xc = ray * z[..., None]
        Tw = np.linalg.inv(S['w2c'][f]); Xw = Xc @ Tw[:3, :3].T + Tw[:3, 3]
        nb = sorted([g for g in D if g != f], key=lambda g: np.linalg.norm(C[g] - C[f]))[:2]
        ok = np.zeros(z.shape, bool)
        for g in nb:
            M = S['w2c'][g]; Y = Xw @ M[:3, :3].T + M[:3, 3]; zz = Y[..., 2]
            px = K[0, 0] * Y[..., 0] / np.maximum(zz, 1e-6) + K[0, 2] - .5; py = K[1, 1] * Y[..., 1] / np.maximum(zz, 1e-6) + K[1, 2] - .5
            inb = (zz > 0.05) & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
            zg = cv2.remap(D[g], px.astype(np.float32), py.astype(np.float32), cv2.INTER_NEAREST)
            ok |= inb & (zg > 0) & (np.abs(zg - zz) < 0.03 * zz)
        zc = np.where(ok & (z > 0), z, 0); kept.append(float((zc > 0).mean()))
        cv2.imwrite(out, (zc * 1000).astype(np.uint16))
    good_fits = [s_ for s_ in stats if not s_.get('skipped')]
    return dict(frames=len(D), net_input=[iw, ih], fit_relerr_median=float(np.median([s_['relerr_median'] for s_ in good_fits])),
                consistent_pixel_frac=float(np.mean(kept)))


# ------------------------------------------------------------------------------------------------------------------- mesh
def stage_mesh(a, W):
    import open3d as o3d, open3d.core as oc
    S = load_scene(W); K = S['K']; w, h = S['width'], S['height']
    Kt = oc.Tensor(K, oc.float64)
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'), attr_dtypes=(oc.float32, oc.float32, oc.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=a.voxel, block_resolution=16, block_count=60000,
                                        device=oc.Device('CPU:0'))
    used = 0
    for f in S['train']:
        p = f'{W}/{a.mesh_depth}/{f:05d}.png'
        if not os.path.exists(p): continue
        d = cv2.imread(p, -1); c = cv2.imread(f'{W}/img/{f:05d}.png')[:, :, ::-1]
        Dm = o3d.t.geometry.Image(oc.Tensor(d)); Ci = o3d.t.geometry.Image(oc.Tensor(np.ascontiguousarray(c)))
        E = oc.Tensor(S['w2c'][f], oc.float64)
        if not ((d > 0) & (d < a.depth_max * 1000)).any(): continue
        try:
            blk = vbg.compute_unique_block_coordinates(Dm, Kt, E, 1000., a.depth_max, a.trunc)
            vbg.integrate(blk, Dm, Ci, Kt, Kt, E, 1000., a.depth_max, a.trunc); used += 1
        except RuntimeError as e:          # a frame whose depth touches no voxel block
            _log(W, f'mesh: frame {f} skipped ({str(e)[-80:]})')
    m = vbg.extract_triangle_mesh(weight_threshold=a.weight_thresh).to_legacy()
    # drop small floating pieces
    cl, n, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
    m.remove_triangles_by_mask(n[cl] < max(200, 0.002 * len(cl))); m.remove_unreferenced_vertices()
    os.makedirs(f'{W}/mesh', exist_ok=True)
    o3d.io.write_triangle_mesh(f'{W}/mesh/mesh.ply', m)
    # compact copy for sharing (.glb)
    ms = m.simplify_quadric_decimation(min(len(m.triangles), 400000))
    try:
        import trimesh
        tm = trimesh.Trimesh(np.asarray(ms.vertices), np.asarray(ms.triangles),
                             vertex_colors=(np.asarray(ms.vertex_colors) * 255).astype(np.uint8), process=False)
        tm.export(f'{W}/mesh/mesh.glb')
    except Exception as e:
        _log(W, f'mesh: glb export failed {e}')
    return dict(fused_frames=used, vertices=len(m.vertices), triangles=len(m.triangles), voxel_m=a.voxel, depth=a.mesh_depth, weight_thresh=a.weight_thresh)


def stage_splat(a, W):
    from gs import train
    return train(a, W)


def stage_score(a, W):
    from evaluate import score
    return score(a, W)


def stage_strips(a, W):
    from evaluate import strips
    return strips(a, W)
