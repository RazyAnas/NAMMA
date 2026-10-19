# Minimal CPU Gaussian-splat renderer (3DGS EWA projection, front-to-back alpha compositing) for scoring
# OpenSplat .ply files on a Mac without CUDA. Vectorised over (pixel, Gaussian) pairs: each Gaussian is
# expanded over its 3-sigma pixel box, pairs below 1/255 alpha are dropped, pairs are sorted by pixel then
# depth, and transmittance is a segmented cumulative sum of log(1 - alpha) in float64.
import numpy as np, torch
C0 = 0.28209479177387814; C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]

def gaussians(names, a):
    ix = {n: i for i, n in enumerate(names)}; col = lambda *k: torch.tensor(a[:, [ix[x] for x in k]])
    rest = sorted((n for n in names if n.startswith('f_rest_')), key=lambda n: int(n[7:]))
    nrest = len(rest) // 3
    sh = torch.cat([col('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None, :],
                    col(*rest).reshape(-1, 3, nrest).transpose(1, 2)], 1)   # [N, 1+nrest, 3]
    return dict(means=col('x', 'y', 'z'), scales=torch.exp(col('scale_0', 'scale_1', 'scale_2')),
                quats=col('rot_0', 'rot_1', 'rot_2', 'rot_3'), opac=torch.sigmoid(col('opacity')[:, 0]), sh=sh)

def quat_to_rot(q):
    q = q / q.norm(dim=1, keepdim=True); w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)

def sh_color(sh, d):
    x, y, z = d.unbind(1); deg = {1: 0, 4: 1, 9: 2, 16: 3}[sh.shape[1]]; c = C0 * sh[:, 0]
    if deg > 0:
        c = c - C1 * y[:, None] * sh[:, 1] + C1 * z[:, None] * sh[:, 2] - C1 * x[:, None] * sh[:, 3]
    if deg > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        c = (c + C2[0] * xy[:, None] * sh[:, 4] + C2[1] * yz[:, None] * sh[:, 5] + C2[2] * (2 * zz - xx - yy)[:, None] * sh[:, 6]
             + C2[3] * xz[:, None] * sh[:, 7] + C2[4] * (xx - yy)[:, None] * sh[:, 8])
    if deg > 2:
        c = (c + C3[0] * (y * (3 * xx - yy))[:, None] * sh[:, 9] + C3[1] * (xy * z)[:, None] * sh[:, 10]
             + C3[2] * (y * (4 * zz - xx - yy))[:, None] * sh[:, 11] + C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * sh[:, 12]
             + C3[4] * (x * (4 * zz - xx - yy))[:, None] * sh[:, 13] + C3[5] * (z * (xx - yy))[:, None] * sh[:, 14]
             + C3[6] * (x * (xx - 3 * yy))[:, None] * sh[:, 15])
    return (c + 0.5).clamp_min(0)

@torch.no_grad()
def render(g, w2c, K, W, H, max_pairs=60_000_000):
    w2c = torch.as_tensor(w2c, dtype=torch.float32); K = torch.as_tensor(K, dtype=torch.float32)
    Rw, tw = w2c[:3, :3], w2c[:3, 3]; p = g['means'] @ Rw.T + tw
    keep = p[:, 2] > 0.05; idx = keep.nonzero()[:, 0]; p = p[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; z = p[:, 2]
    u = fx * p[:, 0] / z + cx; v = fy * p[:, 1] / z + cy
    R = quat_to_rot(g['quats'][idx]); S = g['scales'][idx]
    M = R * S[:, None, :]; cov3 = M @ M.transpose(1, 2)
    lim = 1.3 * torch.tensor([0.5 * W / fx, 0.5 * H / fy])
    tx = (p[:, 0] / z).clamp(-lim[0], lim[0]) * z; ty = (p[:, 1] / z).clamp(-lim[1], lim[1]) * z
    J = torch.zeros(len(p), 2, 3); J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * tx / z ** 2; J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * ty / z ** 2
    T = J @ Rw; cov2 = T @ cov3 @ T.transpose(1, 2); cov2[:, 0, 0] += 0.3; cov2[:, 1, 1] += 0.3
    a_, b_, c_ = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]; det = a_ * c_ - b_ * b_
    ok = det > 0; mid = 0.5 * (a_ + c_); lam = mid + (mid * mid - det).clamp_min(0.1).sqrt(); r = (3 * lam.sqrt()).ceil()
    x0 = (u - r).clamp(0, W - 1).long(); x1 = (u + r).clamp(0, W - 1).long(); y0 = (v - r).clamp(0, H - 1).long(); y1 = (v + r).clamp(0, H - 1).long()
    ok &= (u + r >= 0) & (u - r < W) & (v + r >= 0) & (v - r < H)
    conic = torch.stack([c_, -b_, a_], 1) / det[:, None]
    camc = -Rw.T @ tw; d = g['means'][idx] - camc; d = d / d.norm(dim=1, keepdim=True)
    col = sh_color(g['sh'][idx], d); op = g['opac'][idx]
    order = torch.argsort(z); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
    sel = ok.nonzero()[:, 0]; bw = (x1 - x0 + 1)[sel]; bh = (y1 - y0 + 1)[sel]; cnt = bw * bh
    keys, logs, cols = [], [], []; start = 0; cum = torch.cumsum(cnt, 0)
    while start < len(sel):
        base = cum[start - 1] if start else 0
        end = int(torch.searchsorted(cum, base + max_pairs // 4, right=True)); end = max(end, start + 1)
        s = sel[start:end]; n = cnt[start:end]; gi = torch.repeat_interleave(torch.arange(start, end), n)
        off = torch.arange(int(n.sum())) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        gs = sel[gi]; px = x0[gs] + off % bw[gi]; py = y0[gs] + off // bw[gi]
        dx = px.float() - u[gs]; dy = py.float() - v[gs]; cn = conic[gs]
        power = -0.5 * (cn[:, 0] * dx * dx + cn[:, 2] * dy * dy) - cn[:, 1] * dx * dy
        al = (op[gs] * torch.exp(power)).clamp(max=0.99); m = (power <= 0) & (al >= 1 / 255)
        keys.append((py[m] * W + px[m]) * len(order) + rank[gs[m]]); logs.append(torch.log1p(-al[m]).double()); cols.append((gs[m], al[m]))
        start = end
    keys = torch.cat(keys); logs = torch.cat(logs); gsel = torch.cat([c[0] for c in cols]); al = torch.cat([c[1] for c in cols])
    o = torch.argsort(keys); keys, logs, gsel, al = keys[o], logs[o], gsel[o], al[o]
    pix = keys // len(order); cs = torch.cumsum(logs, 0)
    first = torch.ones_like(pix, dtype=torch.bool); first[1:] = pix[1:] != pix[:-1]
    seg = torch.cumsum(first.long(), 0) - 1; segstart = (cs - logs)[first]
    Tr = torch.exp(cs - logs - segstart[seg]).float()
    wgt = (Tr * al)[:, None] * col[gsel]
    img = torch.zeros(H * W, 3).index_add_(0, pix, wgt)
    return img.reshape(H, W, 3).clamp(0, 1).numpy()
