"""Scoring and pictures. Variants per held-out view:
  mesh    TSDF mesh ray-cast with vertex colours; pixels the mesh misses get the mean training colour
  splat   Gaussians composited over the same mean training colour
  hybrid  Gaussians composited over the mesh render (mesh fills wherever splat alpha < 1)
  flat    the mean training colour everywhere (what a blank guess scores)
Metrics: raw PSNR/SSIM on the full image, and 'cc' = PSNR after a per-image affine colour fit done fairly: fitted on the left half
and scored on the right half, and vice versa (no pixel is scored with a fit that saw it).
"""
import os, json, math
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity


def psnr(x, y):
    return float(10 * np.log10(1 / max(((x - y) ** 2).mean(), 1e-10)))


def affine_fit(src, dst):
    A = np.concatenate([src.reshape(-1, 3), np.ones((src.size // 3, 1))], 1)
    M = np.linalg.lstsq(A, dst.reshape(-1, 3), rcond=None)[0]
    return lambda im: (np.concatenate([im.reshape(-1, 3), np.ones((im.size // 3, 1))], 1) @ M).reshape(im.shape)


def psnr_cc_halves(img, gt):
    w = img.shape[1] // 2; out = img.copy()
    out[:, w:] = affine_fit(img[:, :w], gt[:, :w])(img[:, w:])
    out[:, :w] = affine_fit(img[:, w:], gt[:, w:])(img[:, :w])
    return psnr(out.clip(0, 1), gt)


class MeshRenderer:
    def __init__(self, path):
        import open3d as o3d
        self.o3d = o3d
        m = o3d.io.read_triangle_mesh(path)
        self.V = np.asarray(m.vertices); self.F = np.asarray(m.triangles); self.C = np.asarray(m.vertex_colors)
        self.sc = o3d.t.geometry.RaycastingScene()
        self.sc.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))

    def render(self, w2c, K, H, W):
        u, v = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
        dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        T = np.linalg.inv(w2c); d = dc @ T[:3, :3].T
        rays = np.concatenate([np.broadcast_to(T[:3, 3], d.shape), d], -1).astype(np.float32)
        r = self.sc.cast_rays(self.o3d.core.Tensor(rays)); th = r['t_hit'].numpy(); hit = np.isfinite(th)
        tid = r['primitive_ids'].numpy()[hit]; uv = r['primitive_uvs'].numpy()[hit]
        img = np.zeros((H, W, 3)); t = self.F[tid]; b1, b2 = uv[:, :1], uv[:, 1:]
        img[hit] = self.C[t[:, 0]] * (1 - b1 - b2) + self.C[t[:, 1]] * b1 + self.C[t[:, 2]] * b2
        return img, hit, np.where(hit, th, 0)


def setup(W):
    from stages import load_scene, read_img
    import gs
    S = load_scene(W)
    G = gs.load_ply(f'{W}/splat/splat.ply')
    fill = np.mean([read_img(W, f).reshape(-1, 3).mean(0) for f in S['train']], 0)
    return S, G, MeshRenderer(f'{W}/mesh/mesh.ply'), fill


def render_all(S, G, MR, fill, w2c, K, H, W):
    import gs
    with torch.no_grad():
        rgb, alpha, _, _, _ = gs.render(G, torch.tensor(w2c, dtype=torch.float32), K, H, W, int(os.environ.get("SH_DEG", 3)), torch.zeros(3))  # SH_DEG: render with fewer SH bands (debug)
    rgb = rgb.numpy().astype(np.float64); al = alpha.numpy()[..., None].astype(np.float64)
    mimg, hit, _ = MR.render(w2c, K, H, W)
    mesh = np.where(hit[..., None], mimg, fill)
    return dict(splat=(rgb + (1 - al) * fill).clip(0, 1), hybrid=(rgb + (1 - al) * mesh).clip(0, 1), mesh=mesh.clip(0, 1),
                flat=np.broadcast_to(fill, mesh.shape).copy()), float(al.mean()), float(hit.mean())


def score(a, W):
    from stages import read_img
    torch.set_num_threads(os.cpu_count())
    S, G, MR, fill = setup(W); K = S['K']; H, Wd = S['height'], S['width']
    os.makedirs(f'{W}/score', exist_ok=True)
    res = {}
    for split in ['test_offpath', 'test_near']:
        rows = []
        for f in S[split]:
            gt = read_img(W, f).astype(np.float64)
            R, am, cov = render_all(S, G, MR, fill, S['w2c'][f], K, H, Wd)
            row = dict(frame=f, splat_alpha_mean=am, mesh_coverage=cov)
            for k, im in R.items():
                row[f'{k}_psnr'] = psnr(im, gt)
                row[f'{k}_ssim'] = float(structural_similarity(im, gt, channel_axis=2, data_range=1))
                row[f'{k}_psnr_cc'] = psnr_cc_halves(im, gt)
            rows.append(row)
            if f in S[split][::max(1, len(S[split]) // 4)][:4]:
                cv2.imwrite(f'{W}/score/{split}_{f:05d}.jpg', (np.concatenate([gt, R['splat'], R['hybrid'], R['mesh']], 1)[:, :, ::-1] * 255).astype(np.uint8))
        summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'frame'} if rows else {}
        res[split] = dict(n_views=len(rows), mean=summ, per_frame=rows)
    json.dump(res, open(f'{W}/score/results.json', 'w'), indent=1)
    out = {}
    for split, r in res.items():
        m = r['mean']
        out[split] = dict(n=r['n_views'], **{k: round(m[f'{k}_psnr'], 2) for k in ['mesh', 'splat', 'hybrid', 'flat'] if m},
                          **{f'{k}_cc': round(m[f'{k}_psnr_cc'], 2) for k in ['mesh', 'splat', 'hybrid'] if m},
                          **{f'{k}_ssim': round(m[f'{k}_ssim'], 3) for k in ['mesh', 'splat', 'hybrid'] if m})
    return out


def label(im, text):
    im = (im.clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    s = im.shape[1] / 432
    cv2.rectangle(im, (0, 0), (im.shape[1], int(34 * s)), (0, 0, 0), -1)
    cv2.putText(im, text, (6, int(25 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return im


def strips(a, W):
    from stages import read_img
    torch.set_num_threads(os.cpu_count())
    S, G, MR, fill = setup(W); K = S['K']; H, Wd = S['height'], S['width']
    test = S['test_offpath'] or S['test_near']
    picks = [test[i] for i in np.linspace(0, len(test) - 1, 4).round().astype(int)]
    os.makedirs(f'{W}/strips', exist_ok=True); paths = []
    up = np.array([0, 1., 0])        # world +y is up (AR frame, or gravity estimate from camera orientations)
    for f in picks:
        M = S['w2c'][f]; Rc = M[:3, :3]; Cc = -Rc.T @ M[:3, 3]
        right = Rc.T @ np.array([1., 0, 0])
        def at(C, R):
            return np.vstack([np.hstack([R, (-R @ C)[:, None]]), [0, 0, 0, 1]])
        th = math.radians(30); kx = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
        Ry = np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * kx @ kx      # yaw about world up
        poses = [('held-out view', M), ('0.3 m right', at(Cc + 0.3 * right, Rc)), ('0.3 m up', at(Cc + 0.3 * up, Rc)),
                 ('rotated 30 deg', at(Cc, Rc @ Ry.T))]
        rows = []
        for name, P in poses:
            R, _, _ = render_all(S, G, MR, fill, P, K, H, Wd)
            first = label(read_img(W, f), f'photo {f}') if name == 'held-out view' else label(np.full((H, Wd, 3), 0.15), f'{name}: no photo')
            rows.append(np.concatenate([first, label(R['splat'], f'splat, {name}'), label(R['hybrid'], f'hybrid, {name}'), label(R['mesh'], f'mesh, {name}')], 1))
        img = np.concatenate(rows, 0)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        p = f'{W}/strips/strip_{f:05d}.jpg'; cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 88]); paths.append(p)
    return dict(strips=paths)
