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
    tex = texture_median(W, S, m, a, apply=a.texture == 'median')   # always writes vcount.npy for the confidence blend
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
    return dict(fused_frames=used, vertices=len(m.vertices), triangles=len(m.triangles), voxel_m=a.voxel, texture=a.texture, **tex, depth=a.mesh_depth, weight_thresh=a.weight_thresh)


def texture_median(W, S, m, a, K=8, iters=2, apply=True):
    """Vertex colours from a weighted median over the K best training views per vertex. Weight = cos(view angle)^2 x image
    sharpness / depth; visibility by a depth test against the mesh itself. Each image gets an affine colour correction towards the
    consensus (exposure), then the median is recomputed. Also writes mesh/vcount.npy = number of training views that see each vertex."""
    import open3d as o3d
    m.compute_vertex_normals()
    V = np.asarray(m.vertices); Nv = np.asarray(m.vertex_normals); nV = len(V); Kc = S['K']; w, h = S['width'], S['height']
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(np.asarray(m.triangles).astype(np.uint32)))
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5); dc = np.stack([(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones_like(u)], -1)
    Wk = np.zeros((nV, K), np.float32); Ck = np.zeros((nV, K, 3), np.float32); Fk = np.full((nV, K), -1, np.int32); cnt = np.zeros(nV, np.int32)
    frames = S['train']; tol = max(0.02, 2 * a.voxel)
    for fi, f in enumerate(frames):
        im = read_img(W, f); g = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var() / (g.var() + 1e-6))
        M = S['w2c'][f]; T = np.linalg.inv(M); d = dc @ T[:3, :3].T
        rays = np.concatenate([np.broadcast_to(T[:3, 3], d.shape), d], -1).astype(np.float32)
        dz = sc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        Y = V @ M[:3, :3].T + M[:3, 3]; z = Y[:, 2]
        px = Kc[0, 0] * Y[:, 0] / np.maximum(z, 1e-6) + Kc[0, 2] - .5; py = Kc[1, 1] * Y[:, 1] / np.maximum(z, 1e-6) + Kc[1, 2] - .5
        ok = (z > 0.05) & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
        xi, yi = np.clip(np.round(px).astype(int), 0, w - 1), np.clip(np.round(py).astype(int), 0, h - 1)
        ok &= np.abs(dz[yi, xi] - z) < tol
        vd = T[:3, 3] - V; vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9
        cosang = np.abs((vd * Nv).sum(1))
        wv = np.where(ok, cosang ** 2 * sharp / np.maximum(z, 0.1), 0).astype(np.float32)
        cnt += ok
        idx = np.nonzero(wv > 0)[0]
        x0, y0 = np.floor(px[idx]).astype(int).clip(0, w - 2), np.floor(py[idx]).astype(int).clip(0, h - 2)
        fx_, fy_ = (px[idx] - x0)[:, None], (py[idx] - y0)[:, None]
        col = (im[y0, x0] * (1 - fx_) + im[y0, x0 + 1] * fx_) * (1 - fy_) + (im[y0 + 1, x0] * (1 - fx_) + im[y0 + 1, x0 + 1] * fx_) * fy_
        slot = Wk[idx].argmin(1); better = wv[idx] > Wk[idx, slot]
        ii, ss = idx[better], slot[better]
        Wk[ii, ss] = wv[ii]; Ck[ii, ss] = col[better]; Fk[ii, ss] = fi
    def wmedian():
        o = np.argsort(Ck, axis=1); out = np.zeros((nV, 3), np.float32)
        for c in range(3):
            vals = np.take_along_axis(Ck[:, :, c], o[:, :, c], 1); ww = np.take_along_axis(Wk, o[:, :, c], 1)
            cw = np.cumsum(ww, 1); half = cw[:, -1:] * 0.5; j = (cw < half).sum(1).clip(0, K - 1)
            out[:, c] = vals[np.arange(nV), j]
        return out
    med = wmedian(); gains = []
    for _ in range(iters):        # per-image affine colour towards the consensus, then re-median
        for fi in range(len(frames)):
            sel = Fk == fi
            if sel.sum() < 200: continue
            vi = np.nonzero(sel)[0]; src = Ck[sel]; dst = med[vi]
            A = np.concatenate([src, np.ones((len(src), 1))], 1); Mx = np.linalg.lstsq(A, dst, rcond=None)[0]
            Ck[sel] = (A @ Mx).clip(0, 1); gains.append(float(np.abs(Mx[:3] - np.eye(3)).mean()))
        med = wmedian()
    has = Wk.max(1) > 0
    if apply:
        C = np.asarray(m.vertex_colors).copy(); C[has] = med[has]
        m.vertex_colors = o3d.utility.Vector3dVector(C.astype(np.float64))
    np.save(f'{W}/mesh/vcount.npy', cnt)
    return dict(textured_vertex_frac=float(has.mean()), views_per_vertex_median=float(np.median(cnt)),
                exposure_fit_mean_abs_dev=float(np.mean(gains)) if gains else 0.0)


def stage_splat(a, W):
    from gs import train
    return train(a, W)


def stage_score(a, W):
    from evaluate import score
    return score(a, W)


def stage_strips(a, W):
    from evaluate import strips
    return strips(a, W)
