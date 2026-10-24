# Small CPU Gaussian-splat renderer (EWA projection, front-to-back alpha compositing, SH up to degree 3) for fast local
# experiments on exported splats: no GPU, a few seconds per low-resolution view. Also loads Nerfstudio cameras.
#   load_ply(path) -> dict of arrays;  render(g, c2w_opengl, fx, fy, cx, cy, W, H) -> (H, W, 3) float image
#   ns_cameras(transforms.json) -> list of (c2w in the splat's frame, fx, fy, cx, cy, W, H, file_path)
import json, numpy as np
from numba import njit
from plyfile import PlyData

NEAR = 0.05
RADMAX = 1.0
SH0 = 0.28209479177387814
SH1 = 0.4886025119029199
SH2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
       1.445305721320277, -0.5900435899266435)


def load_ply(path):
    v = PlyData.read(path)['vertex']; names = [p.name for p in v.properties]
    get = lambda pre: np.stack([v[n] for n in sorted([n for n in names if n.startswith(pre)], key=lambda s: int(s.split('_')[-1]))], 1).astype(np.float32)
    rest = get('f_rest_') if any(n.startswith('f_rest_') for n in names) else np.zeros((v.count, 0), np.float32)
    q = get('rot_'); q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(xyz=np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32), dc=get('f_dc_'),
                rest=rest.reshape(v.count, 3, -1).transpose(0, 2, 1), opacity=1 / (1 + np.exp(-np.asarray(v['opacity'], np.float32))),
                scale=np.exp(get('scale_')), rot=q)


def quat_to_mat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def sh_color(g, campos):
    d = g['xyz'] - campos; d /= np.linalg.norm(d, axis=1, keepdims=True); x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    c = SH0 * g['dc']; r = g['rest']; k = r.shape[1]
    if k >= 3: c = c - SH1 * y * r[:, 0] + SH1 * z * r[:, 1] - SH1 * x * r[:, 2]
    if k >= 8:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        c = c + SH2[0] * xy * r[:, 3] + SH2[1] * yz * r[:, 4] + SH2[2] * (2 * zz - xx - yy) * r[:, 5] + SH2[3] * xz * r[:, 6] + SH2[4] * (xx - yy) * r[:, 7]
    if k >= 15:
        c = c + SH3[0] * y * (3 * xx - yy) * r[:, 8] + SH3[1] * xy * z * r[:, 9] + SH3[2] * y * (4 * zz - xx - yy) * r[:, 10] \
            + SH3[3] * z * (2 * zz - 3 * xx - 3 * yy) * r[:, 11] + SH3[4] * x * (4 * zz - xx - yy) * r[:, 12] \
            + SH3[5] * z * (xx - yy) * r[:, 13] + SH3[6] * x * (xx - 3 * yy) * r[:, 14]
    return np.clip(c + 0.5, 0, 1)


@njit(cache=True)
def _raster(u, v, ca, cb, cc, rad, op, col, order, W, H):
    img = np.zeros((H, W, 3)); T = np.ones((H, W))
    for i in order:
        x0, x1 = max(0, int(u[i] - rad[i])), min(W - 1, int(u[i] + rad[i]) + 1)
        y0, y1 = max(0, int(v[i] - rad[i])), min(H - 1, int(v[i] + rad[i]) + 1)
        for py in range(y0, y1 + 1):
            dy = py + 0.5 - v[i]
            for px in range(x0, x1 + 1):
                t = T[py, px]
                if t < 1e-4: continue
                dx = px + 0.5 - u[i]
                p = 0.5 * (ca[i] * dx * dx + 2 * cb[i] * dx * dy + cc[i] * dy * dy)
                if p > 4.5: continue
                a = min(0.99, op[i] * np.exp(-p))
                if a < 1 / 255: continue
                img[py, px, 0] += t * a * col[i, 0]; img[py, px, 1] += t * a * col[i, 1]; img[py, px, 2] += t * a * col[i, 2]
                T[py, px] = t * (1 - a)
    return img, T


def render(g, c2w, fx, fy, cx, cy, W, H, bg=0.0, keep=None):
    """c2w: 4x4 camera-to-world, OpenGL axes (x right, y up, z back), as Nerfstudio stores them."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    xyz = g['xyz'] if keep is None else g['xyz'][keep]
    sub = g if keep is None else {k: val[keep] for k, val in g.items()}
    pc = (xyz - t) @ R; pc[:, 1:] *= -1          # -> OpenCV camera axes (x right, y down, z forward)
    z = pc[:, 2]; m = z > NEAR
    u = fx * pc[:, 0] / np.where(m, z, 1) + cx; v = fy * pc[:, 1] / np.where(m, z, 1) + cy
    Rg = quat_to_mat(sub['rot']); S = sub['scale']
    cov = np.einsum('nij,nj,nkj->nik', Rg, S * S, Rg)                    # world covariance
    Rc = R.T.copy(); Rc[1:] *= -1                                         # world -> OpenCV camera rotation
    cc_ = np.einsum('ij,njk,lk->nil', Rc, cov, Rc)
    zz = np.where(m, z, 1)
    # as gsplat/3DGS: clamp x/z, y/z to 1.3x the field of view before the Jacobian, so huge Gaussians far outside the
    # view do not project to screen-filling blobs
    lx, ly = 1.3 * (W / 2) / fx, 1.3 * (H / 2) / fy
    tx = np.clip(pc[:, 0] / zz, -lx - (cx - W / 2) / fx, lx - (cx - W / 2) / fx) * zz
    ty = np.clip(pc[:, 1] / zz, -ly - (cy - H / 2) / fy, ly - (cy - H / 2) / fy) * zz
    J = np.zeros((len(z), 2, 3)); J[:, 0, 0] = fx / zz; J[:, 0, 2] = -fx * tx / zz ** 2; J[:, 1, 1] = fy / zz; J[:, 1, 2] = -fy * ty / zz ** 2
    c2 = np.einsum('nij,njk,nlk->nil', J, cc_, J) + 0.3 * np.eye(2)
    det = c2[:, 0, 0] * c2[:, 1, 1] - c2[:, 0, 1] ** 2
    rad = 3 * np.sqrt(np.maximum(c2[:, 0, 0], c2[:, 1, 1]))
    m &= (det > 0) & (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H) & (rad < RADMAX * max(W, H))
    col = sh_color(sub, t)
    order = np.where(m)[0][np.argsort(z[m])]
    img, T = _raster(u, v, c2[:, 1, 1] / det, -c2[:, 0, 1] / det, c2[:, 0, 0] / det, rad, sub['opacity'], col.astype(np.float64), order, W, H)
    return np.clip(img + T[..., None] * bg, 0, 1)


def ns_cameras(transforms_path, dataparser_transform=None, scale=1.0, downscale=1):
    """Cameras from transforms.json; if the splat lives in Nerfstudio's normalised frame, pass its 3x4 transform + scale."""
    t = json.load(open(transforms_path)); out = []
    for f in t['frames']:
        c2w = np.array(f['transform_matrix'], np.float64)
        if dataparser_transform is not None:
            c2w = np.vstack([np.asarray(dataparser_transform) @ c2w, [0, 0, 0, 1]]); c2w[:3, 3] *= scale
        d = downscale
        out.append((c2w, t['fl_x'] / d, t['fl_y'] / d, t['cx'] / d, t['cy'] / d, int(t['w'] / d), int(t['h'] / d), f['file_path']))
    return out


def ns_auto_transform(transforms_path):
    """Nerfstudio's default dataparser normalisation (orientation 'up', centre 'poses', auto-scale): returns (3x4, scale)."""
    t = json.load(open(transforms_path)); P = np.array([f['transform_matrix'] for f in t['frames']], np.float64)
    up = P[:, :3, 1].mean(0); up /= np.linalg.norm(up)
    # rotation taking `up` to +z (Rodrigues, as camera_utils.rotation_matrix_between)
    a, b = up, np.array([0, 0, 1.0]); v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-8: R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); s = np.linalg.norm(v)
        R = np.eye(3) + k + k @ k * ((1 - c) / s ** 2)
    T = np.hstack([R, (R @ -P[:, :3, 3].mean(0))[:, None]])
    pos = (T[:, :3] @ P[:, :3, 3].T).T + T[:, 3]
    return T, 1.0 / float(np.abs(pos).max())
