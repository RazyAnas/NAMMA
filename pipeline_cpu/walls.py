"""Plain-wall diagnostics: wall mask, wall-only PSNR, and the measurements used to find why walls smear.

python walls.py diag <work> [--views test_offpath|train|val] [--n N]
"""
import os, sys, json, math, argparse
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def wall_mask(gt, thr=0.012, lum=0.45, k=11):
    """Plain bright surfaces in the PHOTO: low local gradient after a light blur, bright, away from edges."""
    g = cv2.GaussianBlur(gt.mean(-1).astype(np.float32), (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8
    gm = cv2.blur(gm, (k, k))
    m = (gm < thr) & (g > lum)
    return cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def mpsnr(im, gt, m):
    if m.sum() < 50: return float('nan')
    return float(-10 * np.log10(((im - gt)[m] ** 2).mean() + 1e-12))


def blur_masked(e, m, s):
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), s)
    return np.stack([cv2.GaussianBlur((e[..., c] * m).astype(np.float32), (0, 0), s) for c in range(e.shape[-1])], -1) / np.maximum(w, 1e-3)[..., None]


def depth_normals(dep, K):
    H, W = dep.shape; u, v = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * dep, (v - K[1, 2]) / K[1, 1] * dep, dep], -1)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0)); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n


def diag(W, views='test_offpath', n=0, grads=True):
    import evaluate as E, gs
    from stages import read_img
    torch.set_num_threads(os.cpu_count())
    S, G, MR, fill = E.setup(W); K = S['K']; H, Wd = S['height'], S['width']
    a = json.load(open(f'{W}/args.json')); E.BLEND.update(mode=a.get('blend', 'hard'), p=a.get('blend_p', 2), n0=a.get('blend_n0', 4))
    ex = {int(k): v.numpy() for k, v in torch.load(f'{W}/splat/exposure.pt').items()} if os.path.exists(f'{W}/splat/exposure.pt') else {}
    P = np.load(f'{W}/scene_points.npz'); sx, sy = Wd / S['size_full'][0], H / S['size_full'][1]
    Kf = np.array(S['K_full'])
    frames = S[views] if views != 'train' else S['train'][::max(1, len(S['train']) // (n or 12))]
    if n: frames = frames[:n]
    Ga = G if 'means' in G else {k: torch.cat([g[k] for g in G['_rooms']]) for k in G['_rooms'][0]}
    Gm = Ga['means'].numpy(); Gs = np.exp(Ga['scales'].numpy()); Go = 1 / (1 + np.exp(-Ga['opac'].numpy().ravel()))
    rows = []
    for f in frames:
        gt = read_img(W, f).astype(np.float64); wm = wall_mask(gt)
        R, _, _ = E.render_all(S, G, MR, fill, S['w2c'][f], K, H, Wd)
        if views == 'train' and f in ex:      # training views: apply that view's learnt colour correction
            e = ex[f]; A = np.eye(3) + e[:9].reshape(3, 3)
            R = {k: (v @ A.T + e[9:]).clip(0, 1) for k, v in R.items()}
        _, hit, mdep = MR.render(S['w2c'][f], K, H, Wd); cnt = MR.last_count
        wm &= hit; tx = (~wm) & hit
        r = dict(frame=f, wall_frac=float(wm.mean()))
        for k in ['splat', 'hybrid', 'mesh']:
            r[f'{k}_wall'] = mpsnr(R[k], gt, wm); r[f'{k}_rest'] = mpsnr(R[k], gt, tx); r[f'{k}_all'] = mpsnr(R[k], gt, np.ones_like(wm))
        # error split on wall: smooth part (sigma 25 px: exposure / shading) vs the rest (smudges)
        eh = R['hybrid'] - gt; lo = blur_masked(eh, wm, 25)
        r['wall_mse'] = float((eh[wm] ** 2).mean()); r['wall_mse_lowfreq'] = float((lo[wm] ** 2).mean())
        r['wall_mse_highfreq'] = float(((eh - lo)[wm] ** 2).mean()); r['wall_signed'] = float(eh[wm].mean())
        # global affine colour fit on the whole image (in-sample: an upper bound on what exposure can explain)
        A = np.c_[R['hybrid'].reshape(-1, 3), np.ones(H * Wd)]; M = np.linalg.lstsq(A, gt.reshape(-1, 3), rcond=None)[0]
        # blotchiness: RMS of the band-pass (3-30 px) luminance on wall pixels, render vs photo (plain wall: photo ~ low)
        def band(im):
            l = im.mean(-1); return blur_masked(l[..., None], wm, 3)[..., 0] - blur_masked(l[..., None], wm, 30)[..., 0]
        r['blotch_render'] = float(np.sqrt((band(R['hybrid'])[wm] ** 2).mean())) if wm.sum() > 50 else float('nan')
        r['blotch_photo'] = float(np.sqrt((band(gt)[wm] ** 2).mean())) if wm.sum() > 50 else float('nan')
        r['hybrid_wall_affine'] = mpsnr((A @ M).reshape(H, Wd, 3).clip(0, 1), gt, wm)
        r['hybrid_all_affine'] = mpsnr((A @ M).reshape(H, Wd, 3).clip(0, 1), gt, np.ones_like(wm))
        # affine fitted on NON-wall pixels only, scored on wall (exposure that textured content can predict)
        tr_ = tx.ravel(); M2 = np.linalg.lstsq(A[tr_], gt.reshape(-1, 3)[tr_], rcond=None)[0]
        ha = (A @ M2).reshape(H, Wd, 3).clip(0, 1); r['hybrid_wall_affine_fromrest'] = mpsnr(ha, gt, wm)
        e2 = ha - gt; lo2 = blur_masked(e2, wm, 25)
        r['wall_mse_after_affine_lowfreq'] = float((lo2[wm] ** 2).mean()); r['wall_mse_after_affine_highfreq'] = float(((e2 - lo2)[wm] ** 2).mean())
        if os.environ.get('ERRMAP') and f == frames[len(frames) // 2]:
            em = np.abs(eh).mean(-1); em[~wm] *= 0.25
            cv2.imwrite(os.environ['ERRMAP'], np.concatenate([gt[..., ::-1] * 255, R['hybrid'][..., ::-1] * 255, cv2.applyColorMap((np.clip(em / 0.25, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)], 1).astype(np.uint8))
        # residual shift between render and photo on textured pixels (pose error shows up as a coherent shift)
        g1 = (R['hybrid'].mean(-1) * 255).astype(np.uint8); g2 = (gt.mean(-1) * 255).astype(np.uint8)
        fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g2, g1, None)
        tex = tx & ~wall_mask(gt, thr=0.04, lum=0)
        r['flow_px_textured'] = float(np.median(np.linalg.norm(fl[tex], axis=-1))) if tex.sum() > 100 else float('nan')
        r['flow_px_wall'] = float(np.median(np.linalg.norm(fl[wm], axis=-1))) if wm.sum() > 100 else float('nan')
        # multi-view support of the mesh surface under wall vs textured pixels
        r['views_wall'] = float(np.median(cnt[wm])) if wm.any() else float('nan'); r['views_rest'] = float(np.median(cnt[tx]))
        # mesh roughness: angle between per-pixel mesh normal and its 15 px average
        nrm = depth_normals(mdep, K); na = cv2.blur(nrm, (15, 15)); na /= np.linalg.norm(na, axis=-1, keepdims=True) + 1e-9
        ang = np.degrees(np.arccos(np.clip((nrm * na).sum(-1), -1, 1)))
        r['mesh_rough_deg_wall'] = float(np.median(ang[wm])) if wm.any() else float('nan'); r['mesh_rough_deg_rest'] = float(np.median(ang[tx]))
        # SfM: points per 10k pixels and reprojection residual, wall vs rest
        if f'uv_{f}' in P:
            uv = P[f'uv_{f}']; ids = P[f'id_{f}']; X = P['X'][ids]; w2c = S['w2c'][f]
            pc = X @ w2c[:3, :3].T + w2c[:3, 3]; pr = pc[:, :2] / pc[:, 2:] * [Kf[0, 0], Kf[1, 1]] + Kf[:2, 2]
            res = np.linalg.norm(pr - uv, axis=1); q = np.round(uv * [sx, sy]).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < Wd) & (q[:, 1] >= 0) & (q[:, 1] < H)
            onw = np.zeros(len(uv), bool); onw[ok] = wm[q[ok, 1], q[ok, 0]]; ont = np.zeros(len(uv), bool); ont[ok] = tx[q[ok, 1], q[ok, 0]]
            r['sfm_pts_per10k_wall'] = float(onw.sum() / max(wm.sum(), 1) * 1e4); r['sfm_pts_per10k_rest'] = float(ont.sum() / max(tx.sum(), 1) * 1e4)
            r['sfm_res_px_wall'] = float(np.median(res[onw])) if onw.sum() > 5 else float('nan'); r['sfm_res_px_rest'] = float(np.median(res[ont]))
            # mesh depth error at SfM points (depth prior / TSDF vs triangulated points)
            dm = np.zeros(len(uv)); dm[ok] = mdep[q[ok, 1], q[ok, 0]]; good = ok & (dm > 0) & (P['track'][ids] >= 3)
            rel = np.abs(dm - pc[:, 2]) / pc[:, 2]
            r['mesh_vs_sfm_rel_wall'] = float(np.median(rel[good & onw])) if (good & onw).sum() > 5 else float('nan')
            r['mesh_vs_sfm_rel_rest'] = float(np.median(rel[good & ont]))
        # Gaussians whose centre lands on wall vs textured pixels, in front of the mesh
        w2c = S['w2c'][f]; pc = Gm @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
        u = np.round(pc[:, 0] / np.maximum(z, 1e-3) * K[0, 0] + K[0, 2] - .5).astype(int); v = np.round(pc[:, 1] / np.maximum(z, 1e-3) * K[1, 1] + K[1, 2] - .5).astype(int)
        ok = (z > 0.05) & (u >= 0) & (u < Wd) & (v >= 0) & (v < H); u, v, zz = u[ok], v[ok], z[ok]; dmv = mdep[v, u]
        front = (dmv > 0) & (zz < dmv + 0.03)          # visible-ish: not hidden behind the surface
        gw = front & wm[v, u]; gt_ = front & tx[v, u]; sc = Gs[ok]; op = Go[ok]
        r['gauss_per1k_px_wall'] = float(gw.sum() / max(wm.sum(), 1) * 1e3); r['gauss_per1k_px_rest'] = float(gt_.sum() / max(tx.sum(), 1) * 1e3)
        r['gauss_maxscale_cm_wall'] = float(np.median(sc[gw].max(1)) * 100) if gw.any() else float('nan'); r['gauss_maxscale_cm_rest'] = float(np.median(sc[gt_].max(1)) * 100)
        off = np.abs(zz - dmv)
        r['gauss_off_surface_frac_wall'] = float((off[gw] > 0.03).mean()) if gw.any() else float('nan')
        r['gauss_off_surface_frac_rest'] = float((off[gt_] > 0.03).mean())
        r['gauss_floater_opac_wall'] = float(op[gw][off[gw] > 0.03].sum() / max(op[gw].sum(), 1e-6)) if gw.any() else float('nan')
        rows.append(r)
        print(f, {k: round(v, 3) for k, v in r.items() if k in ('hybrid_wall', 'hybrid_rest', 'wall_frac')}, flush=True)
    keys = [k for k in rows[0] if k != 'frame']
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    out = dict(work=W, views=views, n=len(rows), mean=mean, per_frame=rows)
    if grads: out['grads'] = grad_stats(W, S, G, ex, n=8)
    return out


def grad_stats(W, S, G, ex, n=8):
    """Densification signal on wall vs textured Gaussians: mean screen-space absgrad (as the trainer accumulates it)."""
    import gs
    from stages import read_img
    K = S['K']; H, Wd = S['height'], S['width']; fr = S['train'][::max(1, len(S['train']) // n)][:n]
    Gd = {k: v.clone().requires_grad_(True) for k, v in G.items()}
    gw, gr = [], []
    for f in fr:
        gt = read_img(W, f); wm = wall_mask(gt.astype(np.float64))
        absacc = []
        rgb, al, _, idx, rad = gs.render(Gd, torch.tensor(S['w2c'][f], dtype=torch.float32), K, H, Wd, 1, torch.zeros(3), absacc=absacc)
        e = torch.tensor(ex.get(f, np.zeros(12)), dtype=torch.float32); A = torch.eye(3) + e[:9].view(3, 3)
        loss = (rgb @ A.T + e[9:] - torch.tensor(gt)).abs().mean(); loss.backward()
        g = (absacc[0] * torch.tensor([Wd / 2, H / 2])).norm(dim=-1).detach().numpy()
        w2c = S['w2c'][f]; pc = G['means'].numpy()[idx.numpy()] @ w2c[:3, :3].T + w2c[:3, 3]
        u = np.clip((pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).astype(int), 0, Wd - 1); v = np.clip((pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).astype(int), 0, H - 1)
        vis = rad.numpy() > 0; onw = vis & wm[v, u]; onr = vis & ~wm[v, u]
        gw.append(g[onw]); gr.append(g[onr])
        for k in Gd: Gd[k].grad = None
    gw, gr = np.concatenate(gw), np.concatenate(gr)
    return dict(thresh=8e-4, wall_mean=float(gw.mean()), rest_mean=float(gr.mean()), wall_frac_over=float((gw > 8e-4).mean()),
                rest_frac_over=float((gr > 8e-4).mean()), n_wall=int(len(gw)), n_rest=int(len(gr)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd'); p.add_argument('work'); p.add_argument('--views', default='test_offpath')
    p.add_argument('--n', type=int, default=0); p.add_argument('--out', default=''); p.add_argument('--no-grads', action='store_true')
    a = p.parse_args()
    if a.cmd == 'diag':
        o = diag(a.work, a.views, a.n, grads=not a.no_grads)
        if a.out: json.dump(o, open(a.out, 'w'), indent=1)
        print(json.dumps({k: round(v, 4) for k, v in o['mean'].items()}, indent=0))
        if 'grads' in o: print(json.dumps(o['grads']))
