"""Wall error vs training-view coverage: for each wall pixel of a held-out view, how many training views see that surface
point (depth-checked) and the smallest angle between the held-out ray and a training ray. python coverage.py <work>"""
import os, sys, json
import numpy as np, torch, open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from stages import read_img
from walls import wall_mask

def run(W, views='test_offpath', step=1):
    S, G, MR, fill = E.setup(W); K = S['K']; H, Wd = S['height'], S['width']
    a = json.load(open(f'{W}/args.json')); E.BLEND.update(mode=a.get('blend', 'hard'), p=a.get('blend_p', 2), n0=a.get('blend_n0', 4))
    tr = S['train']; ex = {int(k): v.numpy() for k, v in torch.load(f'{W}/splat/exposure.pt').items()}
    can = {}
    for g in tr:      # training photo in the splat's canonical colour space (undo its learnt 3x3+bias)
        e = ex[g]; A = np.eye(3) + e[:9].reshape(3, 3); can[g] = (read_img(W, g).astype(np.float64) - e[9:]) @ np.linalg.inv(A).T
    CS = []; dt = {f: MR.render(S['w2c'][f], K, H, Wd)[2] for f in tr}
    C = {f: -S['w2c'][f][:3, :3].T @ S['w2c'][f][:3, 3] for f in tr}
    u, v = np.meshgrid(np.arange(Wd) + .5, np.arange(H) + .5); dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    E_, N_, A_, M_ = [], [], [], []
    for f in (S[views][::step] if not os.environ.get('CONS_FRAME') else [int(os.environ['CONS_FRAME'])]):
        gt = read_img(W, f).astype(np.float64); R = E.render_all(S, G, MR, fill, S['w2c'][f], K, H, Wd)[0]['hybrid']
        _, hit, dep = MR.render(S['w2c'][f], K, H, Wd); wm = wall_mask(gt) & hit; tx = hit & ~wm
        Am = np.c_[R.reshape(-1, 3), np.ones(H * Wd)]; Mx = np.linalg.lstsq(Am[tx.ravel()], gt.reshape(-1, 3)[tx.ravel()], rcond=None)[0]
        err = (((Am @ Mx).reshape(H, Wd, 3).clip(0, 1) - gt) ** 2).mean(-1)
        T = np.linalg.inv(S['w2c'][f]); P = (dc * dep[..., None]) @ T[:3, :3].T + T[:3, 3]; P = P[wm]; ct = T[:3, 3]
        rt = P - ct; rt /= np.linalg.norm(rt, axis=1, keepdims=True); cnt = np.zeros(len(P)); amin = np.full(len(P), 180.); smp = []
        for g in tr:
            w2c = S['w2c'][g]; pc = P @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
            x = pc[:, 0] / np.maximum(z, 1e-3) * K[0, 0] + K[0, 2]; y = pc[:, 1] / np.maximum(z, 1e-3) * K[1, 1] + K[1, 2]
            ok = (z > 0.05) & (x >= 0) & (x < Wd) & (y >= 0) & (y < H); xi, yi = x.astype(int), y.astype(int); xi[~ok] = 0; yi[~ok] = 0
            ok &= np.abs(dt[g][yi, xi] - z) < 0.03 + 0.01 * z
            rg = P - C[g]; rg /= np.linalg.norm(rg, axis=1, keepdims=True); ang = np.degrees(np.arccos(np.clip((rt * rg).sum(1), -1, 1)))
            smp.append(np.where(ok[:, None], can[g][yi, xi], np.nan))
            cnt += ok; amin = np.where(ok, np.minimum(amin, ang), amin)
        cons = np.nanmedian(np.stack(smp), 0); okc = np.isfinite(cons).all(1)   # consensus of training photos at this surface point
        cm = (np.c_[cons, np.ones(len(cons))] @ Mx).clip(0, 1); CS.append(np.where(okc, ((cm - gt[wm]) ** 2).mean(1), np.nan))
        if os.environ.get('CONS_OUT'):     # picture: photo | consensus of training photos (wall only) | model hybrid
            ci = gt * 0.35; ci[wm] = np.where(okc[:, None], cm, 0); hi = gt * 0.35; hi[wm] = (Am @ Mx).reshape(H, Wd, 3).clip(0, 1)[wm]
            import cv2
            cv2.imwrite(os.environ['CONS_OUT'], np.concatenate([E.label(gt, f'photo {f}'), E.label(ci, 'median of train photos'), E.label(hi, 'R2 hybrid (wall)')], 1))
        E_.append(err[wm]); N_.append(cnt); A_.append(amin); M_.append(np.full(wm.sum(), f))
    e, n, an = map(np.concatenate, (E_, N_, A_))
    out = {}
    for name, x, bins in [('views', n, [0, 1, 3, 6, 11, 21, 999]), ('min_angle_deg', an, [0, 5, 10, 20, 40, 181])]:
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (x >= lo) & (x < hi)
            if s.sum(): rows.append(dict(bin=f'{lo}-{hi}', frac=round(float(s.mean()), 3), wall_psnr=round(float(-10 * np.log10(e[s].mean())), 2)))
        out[name] = rows
    good = (n >= 6) & (an < 10); out['well_covered'] = dict(frac=round(float(good.mean()), 3), wall_psnr=round(float(-10 * np.log10(e[good].mean())), 2),
                                                             rest_psnr=round(float(-10 * np.log10(e[~good].mean())), 2))
    cs = np.concatenate(CS); k = np.isfinite(cs)
    out['consensus_vs_model_on_same_px'] = dict(frac=round(float(k.mean()), 3), consensus_wall_psnr=round(float(-10 * np.log10(cs[k].mean())), 2), model_wall_psnr=round(float(-10 * np.log10(e[k].mean())), 2))
    out['all_wall_psnr_after_affine'] = round(float(-10 * np.log10(e.mean())), 2)
    return out

if __name__ == '__main__':
    o = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'test_offpath', int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    if len(sys.argv) > 4: json.dump(o, open(sys.argv[4], 'w'), indent=1)
    print(json.dumps(o))
