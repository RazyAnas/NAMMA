"""CPU Gaussian splatting trainer held to a mesh surface.

Rasteriser: raster_cpu.cpp (OpenSplat's CPU rasteriser generalised to RGB + depth channels), compiled on first use.
Projection, SH and losses are plain torch autograd. Densification follows 3DGS (clone/split on mean |dL/dxy|, absgrad),
plus two surface terms:
  * depth loss: rendered expected depth vs the TSDF mesh's depth in every training view (relative L1, where the mesh is hit)
  * surface pruning: every `prune_every` steps Gaussians farther than --prune-dist from the mesh are removed (during training)
Per-image exposure: a 3x3 colour matrix + bias per training image, applied to the render before the loss.
"""
import os, json, math, time
import numpy as np
import torch
import torch.nn.functional as Fnn

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _EXT = load('raster_cpu_ext', [f'{here}/raster_cpu.cpp'], extra_cflags=['-O3', '-march=native', '-ffast-math'],
                    build_directory=os.path.expanduser('~/.cache/raster_cpu_build') if _mk() else None, verbose=False)
    return _EXT


def _mk():
    os.makedirs(os.path.expanduser('~/.cache/raster_cpu_build'), exist_ok=True); return True


class Raster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xys, conics, colors, opac, bg, cov2d, depths, H, W, absacc):
        img, T, hdl = ext().forward(W, H, xys.detach(), conics.detach(), colors.detach(), opac.detach(), bg, cov2d.detach(), depths.detach())
        ctx.save_for_backward(xys, conics, colors, opac, bg, T)
        ctx.hdl, ctx.H, ctx.W, ctx.absacc = hdl, H, W, absacc
        ctx.mark_non_differentiable(T)
        return img, 1 - T

    @staticmethod
    def backward(ctx, gimg, galpha):
        xys, conics, colors, opac, bg, T = ctx.saved_tensors
        if galpha is None: galpha = torch.zeros_like(T)
        vxy, vcon, vcol, vop, vabs = ext().backward(ctx.W, ctx.H, xys.detach(), conics.detach(), colors.detach(), opac.detach(), bg, T,
                                                    ctx.hdl, gimg.contiguous(), galpha.contiguous())
        ctx.hdl = None
        if ctx.absacc is not None: ctx.absacc.append(vabs)
        return vxy, vcon, vcol, vop.view(-1), None, None, None, None, None, None


def quat_to_rot(q):
    q = Fnn.normalize(q, dim=-1); w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


C0 = 0.28209479177387814; C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_basis(deg, dirs):
    """dirs [N,3] unit (no grad) -> [N, (deg+1)^2] real SH basis (3DGS sign convention)."""
    x, y, z = dirs.unbind(-1); B = [torch.full_like(x, C0)]
    if deg >= 1: B += [-C1 * y, C1 * z, -C1 * x]
    if deg >= 2:
        xx, yy, zz = x * x, y * y, z * z
        B += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)]
    if deg >= 3:
        B += [C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy), C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
              C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return torch.stack(B, -1)


def sh_eval(deg, dirs, dc, rest):
    """dc [N,3], rest [N,15,3] -> rgb [N,3] (before +0.5); one fused einsum so backward is cheap."""
    with torch.no_grad():
        Bs = sh_basis(deg, dirs)
    if deg == 0: return C0 * dc
    n = Bs.shape[1] - 1
    return C0 * dc + torch.einsum('nk,nkc->nc', Bs[:, 1:], rest[:, :n])


def render(G, w2c, K, H, W, deg, bg, want_depth=False, absacc=None):
    """G: dict of tensors. Returns rgb [H,W,3], alpha [H,W], depth [H,W] or None, visible index tensor, radii."""
    R = w2c[:3, :3]; t = w2c[:3, 3]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    with torch.no_grad():
        pv = G['means'] @ R.T + t; z = pv[:, 2]
        u = fx * pv[:, 0] / z.clamp_min(1e-4) + cx; v = fy * pv[:, 1] / z.clamp_min(1e-4) + cy
        m = 0.15 * max(W, H)
        vis = (z > 0.05) & (u > -m) & (u < W + m) & (v > -m) & (v < H + m)
        idx = vis.nonzero().view(-1)
    means = G['means'][idx]; scales = torch.exp(G['scales'][idx]); quats = G['quats'][idx]
    pv = means @ R.T + t; z = pv[:, 2]
    Rm = quat_to_rot(quats); M = Rm * scales[:, None, :]; cov3 = M @ M.transpose(1, 2)
    tanx, tany = 0.5 * W / fx, 0.5 * H / fy
    tx = z * torch.clamp(pv[:, 0] / z, -1.3 * tanx, 1.3 * tanx); ty = z * torch.clamp(pv[:, 1] / z, -1.3 * tany, 1.3 * tany)
    rz = 1 / z; zero = torch.zeros_like(rz)
    J = torch.stack([fx * rz, zero, -fx * tx * rz * rz, zero, fy * rz, -fy * ty * rz * rz], -1).view(-1, 2, 3)
    T_ = J @ R; cov2 = T_ @ cov3 @ T_.transpose(1, 2)
    cov2 = cov2 + 0.3 * torch.eye(2)[None]
    a_, b_, c_ = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det = (a_ * c_ - b_ * b_).clamp_min(1e-6)
    conics = torch.stack([c_ / det, -b_ / det, a_ / det], -1)
    xys = torch.stack([fx * pv[:, 0] * rz + cx - 0.5, fy * pv[:, 1] * rz + cy - 0.5], -1)
    with torch.no_grad():
        bb = (a_ + c_) / 2; rad = torch.ceil(3 * torch.sqrt(torch.maximum(bb + torch.sqrt((bb * bb - det).clamp_min(0.1)), bb)))
    dirs = Fnn.normalize(means.detach() - (-R.T @ t), dim=-1)
    rgb = (sh_eval(deg, dirs, G['dc'][idx], G['rest'][idx]) + 0.5).clamp_min(0)
    cols = torch.cat([rgb, z[:, None]], 1) if want_depth else rgb
    bgc = torch.cat([bg, torch.zeros(1)]) if want_depth else bg
    op = torch.sigmoid(G['opac'][idx]).view(-1)
    img, alpha = Raster.apply(xys, conics, cols.contiguous(), op, bgc.contiguous(), cov2.detach(), z.detach().contiguous(), H, W, absacc)
    return img[..., :3], alpha, (img[..., 3] if want_depth else None), idx, rad


def ssim(x, y):
    """x,y [1,3,H,W]"""
    C1_, C2_ = 0.01 ** 2, 0.03 ** 2
    g = torch.exp(-((torch.arange(11) - 5.) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum(); w = (g[:, None] @ g[None]).expand(3, 1, 11, 11)
    mu1 = Fnn.conv2d(x, w, groups=3); mu2 = Fnn.conv2d(y, w, groups=3)
    s11 = Fnn.conv2d(x * x, w, groups=3) - mu1 ** 2; s22 = Fnn.conv2d(y * y, w, groups=3) - mu2 ** 2; s12 = Fnn.conv2d(x * y, w, groups=3) - mu1 * mu2
    return (((2 * mu1 * mu2 + C1_) * (2 * s12 + C2_)) / ((mu1 ** 2 + mu2 ** 2 + C1_) * (s11 + s22 + C2_))).mean()


def save_ply(G, path):
    from plyfile import PlyData, PlyElement
    N = G['means'].shape[0]
    rest = G['rest'].detach().transpose(1, 2).reshape(N, -1).numpy()
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(rest.shape[1])] + \
        ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    arr = np.concatenate([G['means'].detach().numpy(), np.zeros((N, 3)), G['dc'].detach().numpy(), rest, G['opac'].detach().numpy().reshape(N, 1),
                          G['scales'].detach().numpy(), Fnn.normalize(G['quats'].detach(), dim=-1).numpy()], 1).astype(np.float32)
    el = np.empty(N, dtype=[(n, 'f4') for n in names])
    for i, n in enumerate(names): el[n] = arr[:, i]
    PlyData([PlyElement.describe(el, 'vertex')]).write(path)


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)['vertex']; g = lambda n: np.asarray(v[n], np.float32)
    nrest = len([p.name for p in v.properties if p.name.startswith('f_rest_')])
    rest = np.stack([g(f'f_rest_{i}') for i in range(nrest)], 1).reshape(-1, 3, nrest // 3).transpose(0, 2, 1)
    T = lambda x: torch.tensor(np.ascontiguousarray(x))
    return dict(means=T(np.stack([g('x'), g('y'), g('z')], 1)), dc=T(np.stack([g(f'f_dc_{i}') for i in range(3)], 1)), rest=T(rest),
                opac=T(g('opacity')[:, None]), scales=T(np.stack([g(f'scale_{i}') for i in range(3)], 1)),
                quats=T(np.stack([g(f'rot_{i}') for i in range(4)], 1)))


def knn_scale(X):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(X).query(X, k=4)
    return np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-4, None)


def sharpness(img):
    import cv2
    g = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def blur_filter(W, frames, ratio=0.6, win=4):
    """Windowed blur filter: drop a frame only if its sharpness < ratio x median sharpness of its +-win neighbours
    (in path order). A global threshold would drop whole dim or plain stretches; this only drops local outliers."""
    from stages import read_img
    fr = sorted(frames); s = np.array([sharpness(read_img(W, f)) for f in fr]); keep = []
    for i, f in enumerate(fr):
        nb = np.r_[s[max(0, i - win):i], s[i + 1:i + 1 + win]]
        if len(nb) == 0 or s[i] >= ratio * np.median(nb): keep.append(f)
    return keep, [f for f in fr if f not in set(keep)]


def room_clusters(S, frames, k, overlap=1.3):
    """Split the camera path into k rooms: k-means on camera centres (path order breaks ties). A frame joins every room
    whose centre is within overlap x the distance to its own room centre, so neighbouring rooms share border views."""
    C = np.array([-S['w2c'][f][:3, :3].T @ S['w2c'][f][:3, 3] for f in frames])
    rng = np.random.default_rng(0); best = None
    for rep in range(8):
        cen = C[np.sort(rng.choice(len(C), k, replace=False))]
        for it in range(50):
            lab = np.argmin(((C[:, None] - cen[None]) ** 2).sum(-1), 1)
            cen = np.array([C[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        sse = ((C - cen[lab]) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, cen, lab)
    _, cen, lab = best; d = np.sqrt(((C[:, None] - cen[None]) ** 2).sum(-1))
    rooms = [[f for i, f in enumerate(frames) if d[i, j] <= overlap * d[i, lab[i]] + 1e-9] for j in range(k)]
    return cen, rooms


def train(a, W):
    """Train one splat, or (--rooms k) one splat per camera-path cluster from the same shared solve and mesh."""
    from stages import load_scene
    S = load_scene(W); frames = list(S['train']); info = {}
    if getattr(a, 'blur_ratio', 0) > 0:
        frames, dropped = blur_filter(W, frames, a.blur_ratio, a.blur_win); info['blur_dropped'] = dropped
    k = getattr(a, 'rooms', 1)
    if k <= 1:
        info.update(train_one(a, W, frames, f'{W}/splat')); return info
    cen, rooms = room_clusters(S, frames, k)
    info['rooms'] = []
    for j, fr in enumerate(rooms):
        r = train_one(a, W, fr, f'{W}/splat/room{j}')
        info['rooms'].append(dict(centre=cen[j].tolist(), frames=fr, **r))
        json.dump(dict(centres=cen.tolist(), rooms=[dict(frames=x['frames'], centre=x['centre']) for x in info['rooms']]),
                  open(f'{W}/splat/rooms.json', 'w'), indent=1)
    return info


def train_one(a, W, frames, out):
    import cv2, open3d as o3d
    from stages import load_scene, read_img
    torch.set_num_threads(os.cpu_count()); torch.manual_seed(0); np.random.seed(0)
    S = load_scene(W); os.makedirs(out, exist_ok=True)
    sc = a.train_scale; H0, W0 = S['height'], S['width']; Wt, Ht = int(round(W0 * sc)), int(round(H0 * sc))
    K = S['K'].copy(); K[:2] *= np.array([[Wt / W0], [Ht / H0]])
    imgs = {f: torch.tensor(cv2.resize(read_img(W, f), (Wt, Ht), interpolation=cv2.INTER_AREA)) for f in frames}
    # mesh: depth targets per train view + distance queries
    mesh = o3d.io.read_triangle_mesh(f'{W}/mesh/mesh.ply'); V = np.asarray(mesh.vertices).astype(np.float32); Fc = np.asarray(mesh.triangles).astype(np.uint32)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(Fc))
    u, v = np.meshgrid(np.arange(Wt) + .5, np.arange(Ht) + .5)
    dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    mdep = {}
    for f in frames:
        Tc = np.linalg.inv(S['w2c'][f]); d = dirs_c @ Tc[:3, :3].T
        rays = np.concatenate([np.broadcast_to(Tc[:3, 3], d.shape), d], -1).astype(np.float32)
        th = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()       # along un-normalised ray with z=1 -> camera depth
        mdep[f] = torch.tensor(np.where(np.isfinite(th), th, 0).astype(np.float32))
    if getattr(a, 'expo_init', 0):
        # photometric pre-solve: log I(view i, surface point v) = albedo_v + gain_i (per channel), from mesh vertices seen
        # in several views; training targets are divided by gain_i, so auto-exposure drift is removed before any
        # Gaussian can bake it into its colour
        rng0 = np.random.default_rng(1); Vs = V[rng0.choice(len(V), min(40000, len(V)), replace=False)]
        ii, vv, LL, XY = [], [], [], []
        for i, f in enumerate(frames):
            w2c = S['w2c'][f]; pc = Vs @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
            u = pc[:, 0] / np.maximum(z, 1e-3) * K[0, 0] + K[0, 2]; v = pc[:, 1] / np.maximum(z, 1e-3) * K[1, 1] + K[1, 2]
            ok = (z > 0.05) & (u >= 1) & (u < Wt - 1) & (v >= 1) & (v < Ht - 1); ui, vi = u.astype(int), v.astype(int); ui[~ok] = 0; vi[~ok] = 0
            md = mdep[f].numpy()[vi, ui]; ok &= np.abs(md - z) < 0.02 + 0.01 * z
            c = imgs[f].numpy()[vi, ui]; ok &= (c.max(-1) < 0.97) & (c.min(-1) > 0.03)
            j = np.nonzero(ok)[0]; ii.append(np.full(len(j), i)); vv.append(j); LL.append(np.log(c[j])); XY.append(np.c_[u[j] / Wt, v[j] / Ht])
        ii, vv, LL, XY = np.concatenate(ii), np.concatenate(vv), np.concatenate(LL), np.concatenate(XY)
        n_obs = np.bincount(vv, minlength=len(Vs)); sel = n_obs[vv] >= 3; ii, vv, LL, XY = ii[sel], vv[sel], LL[sel], XY[sel]
        gx, gy = (1, 1) if a.expo_init == 1 else (3, 5)     # 1: one gain per image; 2: smooth bilinear gain field per image
        def bil(xy):
            cx = xy[:, 0] * (gx - 1); cy = xy[:, 1] * (gy - 1); x0 = np.clip(cx.astype(int), 0, max(gx - 2, 0)); y0 = np.clip(cy.astype(int), 0, max(gy - 2, 0))
            fx, fy = cx - x0, cy - y0; B = np.zeros((len(xy), gx * gy))
            for dx, dy, w in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
                ok_ = (x0 + dx < gx) & (y0 + dy < gy); B[np.nonzero(ok_)[0], ((y0 + dy) * gx + x0 + dx)[ok_]] += w[ok_]
            return B
        B = bil(XY); g = np.zeros((len(frames), gx * gy, 3))
        D = [np.eye(gx * gy)[a_] - np.eye(gx * gy)[b_] for a_ in range(gx * gy) for b_ in range(gx * gy)
             if (b_ == a_ + 1 and (a_ + 1) % gx) or b_ == a_ + gx]          # neighbouring grid nodes
        lam = 1e-4 * np.eye(gx * gy) + (getattr(a, 'expo_smooth', 1.0) * np.array(D).T @ np.array(D) if D else 0)   # smooth field; global gain free
        for it in range(15):
            off = np.einsum('nk,nkc->nc', B, g[ii])
            alb = np.stack([np.bincount(vv, LL[:, c] - off[:, c], minlength=len(Vs)) for c in range(3)], 1) / np.maximum(np.bincount(vv, minlength=len(Vs)), 1)[:, None]
            r = LL - alb[vv]
            for i in range(len(frames)):
                s_ = ii == i
                if s_.sum() > 20: g[i] = np.linalg.solve(B[s_].T @ B[s_] / s_.sum() + lam, B[s_].T @ r[s_] / s_.sum())
            g -= g.mean(0)                                  # gauge: field shared by all images stays in the albedo
        uu, vv2 = np.meshgrid((np.arange(Wt) + .5) / Wt, (np.arange(Ht) + .5) / Ht); Bf = bil(np.c_[uu.ravel(), vv2.ravel()])
        for i, f in enumerate(frames): imgs[f] = imgs[f] * torch.tensor(np.exp(-(Bf @ g[i])).reshape(Ht, Wt, 3), dtype=torch.float32)
        expo_gain = {int(f): g[i].tolist() for i, f in enumerate(frames)}
    else:
        expo_gain = None
    # init: SfM points + mesh surface samples (vertex colours)
    P = np.load(f'{W}/scene_points.npz'); ok = (P['track'] >= 2) & (P['err'] < 4)
    Xs, Cs = P['X'][ok], P['C'][ok].astype(np.float32) / 255
    n_mesh = min(a.init_mesh_pts, len(V)) if hasattr(a, 'init_mesh_pts') else min(150000, len(V))
    mesh.compute_vertex_normals(); mesh.compute_triangle_normals(); tri_n = np.asarray(mesh.triangle_normals).astype(np.float32)
    if n_mesh > 0 and len(mesh.triangles):
        pc = mesh.sample_points_uniformly(n_mesh, use_triangle_normal=True)
        Xm = np.asarray(pc.points, np.float32); Cm = np.asarray(pc.colors, np.float32); Nm = np.asarray(pc.normals, np.float32)
    else:
        Xm = np.zeros((0, 3), np.float32); Cm = np.zeros((0, 3), np.float32); Nm = np.zeros((0, 3), np.float32)
    if len(frames) < len(S['train']) and getattr(a, 'rooms', 1) > 1:     # room: only points some room view sees
        def seen(X):
            m = np.zeros(len(X), bool)
            for f in frames:
                w2c = S['w2c'][f]; pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
                u = pc[:, 0] / np.maximum(z, 1e-3) * K[0, 0] + K[0, 2]; v = pc[:, 1] / np.maximum(z, 1e-3) * K[1, 1] + K[1, 2]
                ok = (z > 0.05) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht); ui, vi = u.astype(int), v.astype(int); ui[~ok] = 0; vi[~ok] = 0
                md = mdep[f].numpy()[vi, ui]; m |= ok & ((md == 0) | (z < md + 0.1))
            return m
        ms, mm = seen(Xs), seen(Xm); Xs, Cs, Xm, Cm, Nm = Xs[ms], Cs[ms], Xm[mm], Cm[mm], Nm[mm]
    X0 = np.concatenate([Xs, Xm]); C0_ = np.concatenate([Cs, Cm])
    N = len(X0); s0 = knn_scale(X0)
    S0 = np.log(np.repeat(s0[:, None], 3, 1)).astype(np.float32); Q0 = np.tile([1., 0, 0, 0], (N, 1)).astype(np.float32)
    if a.flat_init and len(Xm):
        # mesh samples start as flat discs lying in the surface: local z axis = triangle normal, z scale 10x thinner
        n = Nm * np.where(Nm[:, 2:3] < 0, -1, 1); q = np.stack([1 + n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))], 1)
        Q0[len(Xs):] = q / np.linalg.norm(q, axis=1, keepdims=True); S0[len(Xs):, 2] += math.log(0.1)
    G = dict(means=torch.tensor(X0), scales=torch.tensor(S0),
             quats=torch.tensor(Q0),
             opac=torch.full((N, 1), math.log(0.1 / 0.9)), dc=torch.tensor((C0_ - 0.5) / C0), rest=torch.zeros(N, 15, 3))
    for k in G: G[k].requires_grad_(True)
    cams = np.array([-S['w2c'][f][:3, :3].T @ S['w2c'][f][:3, 3] for f in frames])
    extent = 1.1 * float(np.linalg.norm(cams - cams.mean(0), axis=1).max()); extent = max(extent, 0.5)
    lr = dict(means=1.6e-4 * extent, scales=5e-3, quats=1e-3, opac=5e-2, dc=2.5e-3, rest=2.5e-3 / 20)
    def mkopt():
        return torch.optim.Adam([dict(params=[G[k]], lr=lr[k], name=k) for k in G], eps=1e-15)
    opt = mkopt()
    # per-image exposure: rgb' = A rgb + b
    expo = {f: torch.zeros(12, requires_grad=True) for f in frames}
    eopt = torch.optim.Adam([expo[f] for f in frames], lr=a.expo_lr)
    gg = [int(x) for x in str(getattr(a, 'gain_grid', '') or '0').split(',')]
    if len(gg) == 3:
        # per-image bilateral gain grid (BilaRF-style, reduced to a log-gain per cell and channel): x, y and rendered
        # luminance axes, so it can absorb vignetting and the phone's local tone mapping, which a global 3x3 cannot
        grid = {f: torch.zeros(1, 3, gg[2], gg[1], gg[0], requires_grad=True) for f in frames}
        gopt = torch.optim.Adam([grid[f] for f in frames], lr=a.grid_lr)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, Ht), torch.linspace(-1, 1, Wt), indexing='ij')
    else:
        grid = None
    bg = torch.zeros(3)
    steps = a.steps; dens_from, dens_until, dens_every = 300, int(0.6 * steps), 100
    gacc = torch.zeros(N); gcnt = torch.zeros(N); maxrad = torch.zeros(N)
    t0 = time.time(); hist = []; rng = np.random.default_rng(0)
    dmesh_ok = {f: (mdep[f] > 0) for f in frames}
    # edge-aware weights: high where the photo is flat (walls), ~0 on texture/edges
    tvw = {}
    if a.tv_w > 0:
        for f in frames:
            g = imgs[f]; gx = (g[:, 1:] - g[:, :-1]).abs().mean(-1); gy = (g[1:] - g[:-1]).abs().mean(-1)
            tvw[f] = (torch.exp(-a.tv_gamma * gx), torch.exp(-a.tv_gamma * gy))
    tn = None
    def mesh_normals(X):
        r = scene.compute_closest_points(o3d.core.Tensor(X.numpy()))
        return torch.tensor(tri_n[r['primitive_ids'].numpy().astype(np.int64)])
    order = []
    def prune(mask):
        nonlocal opt, gacc, gcnt, maxrad
        keep = ~mask
        for k in G:
            G[k] = G[k].detach()[keep].requires_grad_(True)
        # carry Adam state
        old = opt; opt = mkopt()
        for go, gn in zip(old.param_groups, opt.param_groups):
            st = old.state.get(go['params'][0], None)
            if st: opt.state[gn['params'][0]] = dict(step=st['step'], exp_avg=st['exp_avg'][keep], exp_avg_sq=st['exp_avg_sq'][keep])
        gacc, gcnt, maxrad = gacc[keep], gcnt[keep], maxrad[keep]
    def append(newG):
        nonlocal opt, gacc, gcnt, maxrad
        n = newG['means'].shape[0]
        old = opt
        for k in G:
            G[k] = torch.cat([G[k].detach(), newG[k]]).requires_grad_(True)
        opt = mkopt()
        for go, gn in zip(old.param_groups, opt.param_groups):
            st = old.state.get(go['params'][0], None)
            if st:
                z = torch.zeros((n,) + st['exp_avg'].shape[1:])
                opt.state[gn['params'][0]] = dict(step=st['step'], exp_avg=torch.cat([st['exp_avg'], z]), exp_avg_sq=torch.cat([st['exp_avg_sq'], z]))
        gacc, gcnt, maxrad = torch.cat([gacc, torch.zeros(n)]), torch.cat([gcnt, torch.zeros(n)]), torch.cat([maxrad, torch.zeros(n)])
    n_pruned_surface = 0; start = 1; t_prev = 0.0
    ck = f'{out}/checkpoint.pt'
    if os.path.exists(ck):      # resume (container may be reclaimed): params, exposure, Adam moments, counters
        c = torch.load(ck, weights_only=False)
        for k in G: G[k] = c['G'][k].requires_grad_(True)
        for f in frames: expo[f].data.copy_(c['expo'][f])
        opt = mkopt()
        for gp in opt.param_groups:
            st = c['adam'].get(gp['name'])
            if st is not None: opt.state[gp['params'][0]] = st
        gacc, gcnt, maxrad = c['gacc'], c['gcnt'], c['maxrad']
        start, hist, n_pruned_surface, t_prev = c['step'] + 1, c['hist'], c['pruned'], c['sec']
        rng = np.random.default_rng(c['step'])
        print(f'resumed at step {start}', flush=True)
    t0 = time.time() - t_prev
    for step in range(start, steps + 1):
        if not order: order = list(rng.permutation(frames))
        f = order.pop()
        # lr schedule for means (exponential decay to 1%)
        for g in opt.param_groups:
            if g['name'] == 'means': g['lr'] = lr['means'] * (0.01 ** (step / steps))
        deg = min(a.sh_degree, step // 500)
        w2c = torch.tensor(S['w2c'][f], dtype=torch.float32)
        absacc = []
        rgb, alpha, dep, idx, rad = render(G, w2c, K, Ht, Wt, deg, bg, want_depth=a.depth_w > 0, absacc=absacc)
        e = expo[f]; A = torch.eye(3) + e[:9].view(3, 3)
        if grid is not None:
            lum = rgb.detach().mean(-1).clamp(0, 1) * 2 - 1
            gmap = Fnn.grid_sample(grid[f], torch.stack([xx, yy, lum], -1)[None, None], align_corners=True)[0, :, 0].permute(1, 2, 0)
            rgb_c = (rgb * torch.exp(gmap)) @ A.T + e[9:]
        else:
            rgb_c = rgb @ A.T + e[9:]
        gt = imgs[f]
        l1 = (rgb_c - gt).abs().mean()
        ss = ssim(rgb_c.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        ld = torch.zeros(())
        if dep is not None:
            md = mdep[f]; m = dmesh_ok[f] & (alpha.detach() > 0.3)
            if m.any():
                ld = ((dep[m] - alpha.detach()[m] * md[m]).abs() / md[m]).mean()
                loss = loss + a.depth_w * ld
        lx = {}
        if a.flat_w > 0 or a.normal_w > 0:
            sc_v = torch.exp(G['scales'][idx])
            if a.flat_w > 0:     # thin discs: smallest axis small relative to the largest (2DGS / DN-Splatter style)
                lx['flat'] = (sc_v.min(-1).values / sc_v.max(-1).values.detach()).mean(); loss = loss + a.flat_w * lx['flat']
            if a.normal_w > 0:   # disc normal (smallest axis) aligned with the TSDF mesh normal at the closest surface point
                if tn is None or tn.shape[0] != G['means'].shape[0] or step % 100 == 0: tn = mesh_normals(G['means'].detach())
                Rv = quat_to_rot(G['quats'][idx]); amin = sc_v.detach().argmin(-1)
                nv = Rv[torch.arange(len(idx)), :, amin]
                lx['normal'] = (1 - (nv * tn[idx]).sum(-1).abs()).mean(); loss = loss + a.normal_w * lx['normal']
        if a.tv_w > 0:           # edge-aware total variation on colour and (relative) depth where the photo is flat
            wx, wy = tvw[f]
            tv = (wx[..., None] * (rgb_c[:, 1:] - rgb_c[:, :-1]).abs()).mean() + (wy[..., None] * (rgb_c[1:] - rgb_c[:-1]).abs()).mean()
            if dep is not None:
                dz = dep / alpha.detach().clamp_min(0.3); md = mdep[f].clamp_min(0.1)
                tv = tv + (wx * (dz[:, 1:] - dz[:, :-1]).abs() / md[:, 1:]).mean() + (wy * (dz[1:] - dz[:-1]).abs() / md[1:]).mean()
            lx['tv'] = tv; loss = loss + a.tv_w * tv
        if a.opac_reg > 0:       # MCMC-style: pay for opacity so unneeded smears fade and get pruned
            lx['opac'] = torch.sigmoid(G['opac'][idx]).mean(); loss = loss + a.opac_reg * lx['opac']
        if grid is not None:     # smooth grid: TV between neighbouring cells along every axis
            q = grid[f]; lx['grid_tv'] = sum((q.diff(dim=d) ** 2).mean() for d in (2, 3, 4) if q.shape[d] > 1)
            loss = loss + a.grid_tv * lx['grid_tv']; gopt.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True); eopt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); eopt.step()
        if grid is not None: gopt.step()
        with torch.no_grad():
            if step <= dens_until and absacc:
                ga = absacc[0] * torch.tensor([Wt / 2, Ht / 2]); gn = ga.norm(dim=-1)
                gacc[idx] += gn; gcnt[idx] += (rad > 0).float()
                maxrad[idx] = torch.maximum(maxrad[idx], rad / max(Wt, Ht))
            if step % 50 == 0:
                psnr = -10 * math.log10(((rgb_c.detach() - gt) ** 2).mean().item())
                hist.append(dict(step=step, loss=float(loss), psnr=psnr, depth=float(ld), **{k: round(float(v), 4) for k, v in lx.items()}, n=int(G['means'].shape[0]), sec=round(time.time() - t0, 1)))
                if step % 200 == 0: print(json.dumps(hist[-1]), flush=True)
            if dens_from <= step <= dens_until and step % dens_every == 0:
                avg = gacc / gcnt.clamp_min(1)
                sc_max = torch.exp(G['scales']).max(-1).values
                big = avg >= a.grad_thresh
                room_left = a.max_gauss - G['means'].shape[0]
                if big.sum() > room_left > 0:
                    th = torch.topk(avg, room_left).values[-1]; big = avg >= th
                elif room_left <= 0: big[:] = False
                clone = big & (sc_max <= 0.01 * extent); split = big & (sc_max > 0.01 * extent)
                newG = {k: G[k].detach()[clone].clone() for k in G}
                ns = int(split.sum())
                if ns:
                    sc = torch.exp(G['scales'].detach()[split]); Rm = quat_to_rot(G['quats'].detach()[split])
                    for rep in range(2):
                        smp = (Rm @ (torch.randn(ns, 3) * sc)[..., None])[..., 0] + G['means'].detach()[split]
                        d = {k: G[k].detach()[split].clone() for k in G}; d['means'] = smp; d['scales'] = torch.log(sc / 1.6)
                        newG = {k: torch.cat([newG[k], d[k]]) for k in G}
                n_before = G['means'].shape[0]
                append(newG)
                rm = torch.zeros(G['means'].shape[0], dtype=torch.bool); rm[:n_before] = split
                op = torch.sigmoid(G['opac'].detach()).view(-1)
                rm |= op < max(0.005, a.prune_opac)
                rm |= torch.exp(G['scales'].detach()).max(-1).values > 0.1 * extent
                prune(rm)
                gacc.zero_(); gcnt.zero_()
            if a.prune_opac > 0 and step > dens_until and step % 100 == 0 and step < steps:
                prune(torch.sigmoid(G['opac'].detach()).view(-1) < a.prune_opac)
            if a.prune_dist > 0 and step % a.prune_every == 0 and step < steps:
                dist = scene.compute_distance(o3d.core.Tensor(G['means'].detach().numpy())).numpy()
                far = torch.tensor(dist > a.prune_dist)
                n_pruned_surface += int(far.sum()); prune(far)
            if step == int(0.5 * steps) and a.opacity_reset:
                G['opac'].data.clamp_(max=math.log(0.3 / 0.7))
            if step % 250 == 0 and step < steps:
                adam = {gp['name']: opt.state.get(gp['params'][0]) for gp in opt.param_groups}
                torch.save(dict(G={k: G[k].detach() for k in G}, expo={f: expo[f].detach() for f in frames}, adam=adam, gacc=gacc,
                                gcnt=gcnt, maxrad=maxrad, step=step, hist=hist, pruned=n_pruned_surface, sec=time.time() - t0), ck + '.tmp')
                os.replace(ck + '.tmp', ck)
    save_ply(G, f'{out}/splat.ply')
    torch.save({f: expo[f].detach() for f in frames}, f'{out}/exposure.pt')
    if expo_gain: json.dump(expo_gain, open(f'{out}/expo_gain.json', 'w'))
    json.dump(hist, open(f'{out}/train_log.json', 'w'))
    return dict(steps=steps, train_res=[Wt, Ht], gaussians=int(G['means'].shape[0]), init_points=int(N), init_sfm=int(len(Xs)),
                init_mesh=int(len(Xm)), pruned_off_surface=n_pruned_surface, final_train_psnr=hist[-1]['psnr'] if hist else None,
                train_seconds=round(time.time() - t0, 1), sec_per_step=round((time.time() - t0) / steps, 3))
