"""Image-based rendering (IBR) proof: build a view from the nearest REAL frames, re-projected through the SfM cameras + depth.
This is the Matterport / "8K re-projection" principle: real pixels wherever a real frame saw the surface, nothing invented.
Test: for held-out frames (ids 3, 11, 19, ...) synthesise the frame from its nearest source frames (never itself) and compare
with the real photo (PSNR / SSIM / coverage). Also synthesise views shifted 30 cm sideways (no ground truth: coverage only).
CPU only (numpy). Usage: python ibr.py <cache_dir> <out_dir> [--n 12] [--k 4] [--keep good_frames.json]"""
import argparse, glob, json, os, time, numpy as np
from PIL import Image

def keep_filter(d, keep):
    """Drop frames whose camera failed the real-pixel check (keep = path to good_frames.json, or None for all frames)."""
    if not keep: return d
    ok = np.isin(d['idx'], json.load(open(keep))['keep']); print(f'keep list {keep}: {ok.sum()}/{len(ok)} frames', flush=True)
    return {k: (v[ok] if len(v) == len(ok) else v) for k, v in d.items()}

def load(cache, scale=2, keep=None):
    z = np.load(f'{cache}/da3_sfm.npz'); d = keep_filter({k: z[k] for k in z.files}, keep); files = sorted(glob.glob(f'{cache}/frames/*.jpg'))   # in memory: safe to share with worker processes
    Hd, Wd = d['depth'].shape[1:]; H, W = Hd * scale, Wd * scale
    K = d['K'].copy(); K[:, :2] *= scale
    has = d['depth'].reshape(len(d['idx']), -1).max(1) > 0                      # frames with dense depth (sources)
    return d, files, K, has, H, W

def image(files, g, W, H):
    return np.asarray(Image.open(files[g]).convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255

def warp(src_img, src_depth, Ks, c2w_s, Kt, w2c_t, H, W, zbuf, out, wsum, srcw):
    """Forward-warp one source into the target with a z-buffer: the nearer surface wins, samples of the same surface blend.
    src_depth is on a coarser grid than the images; Ks/Kt are intrinsics at image (H x W) resolution."""
    Hs, Ws = src_depth.shape; s = W / Ws
    v, u = np.mgrid[0:Hs, 0:Ws]; z = src_depth.ravel(); ok = z > 0
    x = (u.ravel() + .5) * s - .5; y = (v.ravel() + .5) * (H / Hs) - .5           # depth-grid cell centre in image pixels
    x, y, z = x[ok], y[ok], z[ok]
    P = np.stack([(x - Ks[0, 2]) / Ks[0, 0] * z, (y - Ks[1, 2]) / Ks[1, 1] * z, z], 1)
    Pt = (P @ c2w_s[:3, :3].T + c2w_s[:3, 3]) @ w2c_t[:3, :3].T + w2c_t[:3, 3]
    vis = Pt[:, 2] > 0.05; Pt, x, y = Pt[vis], x[vis], y[vis]; zt = Pt[:, 2]
    ut = Pt[:, 0] / zt * Kt[0, 0] + Kt[0, 2]; vt = Pt[:, 1] / zt * Kt[1, 1] + Kt[1, 2]
    col = src_img[np.clip(np.round(y).astype(int), 0, H - 1), np.clip(np.round(x).astype(int), 0, W - 1)]
    n = int(np.ceil(s)) + 1                                                      # each sample covers ~s x s target pixels
    for dy in range(n):
        for dx in range(n):
            pu = np.floor(ut - s / 2 + .5).astype(int) + dx; pv = np.floor(vt - s / 2 + .5).astype(int) + dy   # floor, not np.round: round-half-to-even left every other pixel empty at integer positions
            m = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
            idx = pv[m] * W + pu[m]; zz = zt[m]; cc = col[m]
            o = np.argsort(-zz); idx, zz, cc = idx[o], zz[o], cc[o]              # far first: for repeated pixels the nearest is written last
            cur = zbuf[idx]; nearer = zz < cur * 0.97; same = (~nearer) & (zz < cur * 1.03)
            zbuf[idx[nearer]] = zz[nearer]; out[idx[nearer]] = cc[nearer] * srcw; wsum[idx[nearer]] = srcw
            np.add.at(out, idx[same], cc[same] * srcw); np.add.at(wsum, idx[same], srcw)

def synth(d, files, K, has, H, W, tgt_K, tgt_c2w, exclude, k=4, cache_imgs={}):
    centers = d['c2w'][:, :3, 3]; fwd = d['c2w'][:, :3, 2]
    tc, tf = tgt_c2w[:3, 3], tgt_c2w[:3, 2]
    dist = np.linalg.norm(centers - tc, axis=1); ang = np.degrees(np.arccos(np.clip(fwd @ tf, -1, 1)))
    score = dist + ang / 30.0                                                     # 30 deg of view difference ~ 1 m of distance
    score[~has] = np.inf; score[list(exclude)] = np.inf
    srcs = np.argsort(score)[:k]
    zbuf = np.full(H * W, np.inf, np.float32); out = np.zeros((H * W, 3), np.float32); wsum = np.zeros(H * W, np.float32)
    w2c = np.linalg.inv(tgt_c2w)
    for rank, j in enumerate(srcs):
        g = int(d['idx'][j])
        if g not in cache_imgs: cache_imgs[g] = image(files, g, W, H)
        warp(cache_imgs[g], d['depth'][j], K[j], d['c2w'][j], tgt_K, w2c, H, W, zbuf, out, wsum, 1.0 / (1 + score[j]))
    filled = wsum > 0; img = np.zeros((H * W, 3), np.float32); img[filled] = out[filled] / wsum[filled, None]
    return img.reshape(H, W, 3), filled.reshape(H, W), srcs, score[srcs]

def psnr(a, b, m=None):
    e = (a - b) ** 2; e = e[m] if m is not None else e
    return float(-10 * np.log10(e.mean() + 1e-12))

def psnr_gain(a, b, m):
    """PSNR after matching a's per-channel brightness to b on the covered pixels: photos from different passes differ in
    auto-exposure, which is not a placement error."""
    g = b[m].mean(0) / np.maximum(a[m].mean(0), 1e-3); return psnr(np.clip(a * g, 0, 1), b, m)

def ssim(a, b):
    from scipy.ndimage import uniform_filter
    a, b = a.mean(2), b.mean(2); f = lambda x: uniform_filter(x, 7)
    ma, mb = f(a), f(b); va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return float((((2 * ma * mb + 1e-4) * (2 * cab + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4))).mean())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cache'); ap.add_argument('out'); ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--keep'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    d, files, K, has, H, W = load(a.cache, keep=a.keep); fid = d['idx']
    tests = [i for i in range(len(fid)) if fid[i] % 8 == 3][:: max(1, len([1 for x in fid if x % 8 == 3]) // a.n)][:a.n]
    rows, res = [], []
    for i in tests:
        img, m, srcs, sc = synth(d, files, K, has, H, W, K[i], d['c2w'][i], exclude={i}, k=a.k)
        gt = image(files, int(fid[i]), W, H)
        filled_img = img.copy(); filled_img[~m] = gt.mean((0, 1))                 # holes scored as flat grey, not hidden
        r = dict(frame=int(fid[i]), coverage=float(m.mean()), psnr_covered=psnr(img, gt, m), psnr_full=psnr(filled_img, gt),
                 ssim_full=ssim(filled_img, gt), nearest_source_m=float(np.linalg.norm(d['c2w'][srcs[0], :3, 3] - d['c2w'][i, :3, 3])))
        res.append(r); print(json.dumps(r), flush=True)
        err = np.clip(np.abs(filled_img - gt).mean(2, keepdims=True).repeat(3, 2) * 3, 0, 1)
        rows.append(np.concatenate([gt, filled_img, err], 1))
    Image.fromarray((np.concatenate(rows[:6], 0) * 255).astype(np.uint8)).resize((3 * W // 2, len(rows[:6]) * H // 2)).save(f'{a.out}/ibr_heldout.jpg', quality=90)
    # off-path: same held-out cameras moved 30 cm to the right (no ground truth exists there)
    offs = []
    for i in tests[:6]:
        c = d['c2w'][i].copy(); c[:3, 3] += 0.30 * c[:3, 0]
        img, m, _, _ = synth(d, files, K, has, H, W, K[i], c, exclude=set(), k=a.k); img[~m] = [1, 0, 1]   # magenta = never filmed from here
        offs.append(img); res.append(dict(offpath_frame=int(fid[i]), offpath_coverage=float(m.mean())))
        Image.fromarray((img * 255).astype(np.uint8)).save(f'{a.out}/offpath_{int(fid[i]):04d}.png')            # for the Difix demo (P3)
        Image.fromarray((image(files, int(fid[i]), W, H) * 255).astype(np.uint8)).save(f'{a.out}/offpath_{int(fid[i]):04d}_reference_real.jpg', quality=95)
    Image.fromarray((np.concatenate(offs, 0) * 255).astype(np.uint8)).resize((W // 2, len(offs) * H // 2)).save(f'{a.out}/ibr_offpath30cm.jpg', quality=90)
    hd = [r for r in res if 'psnr_full' in r]; op = [r['offpath_coverage'] for r in res if 'offpath_coverage' in r]
    summary = dict(views=len(hd), psnr_full=float(np.mean([r['psnr_full'] for r in hd])), psnr_covered=float(np.mean([r['psnr_covered'] for r in hd])),
                   ssim_full=float(np.mean([r['ssim_full'] for r in hd])), coverage=float(np.mean([r['coverage'] for r in hd])),
                   offpath30_coverage=float(np.mean(op)), seconds=round(time.time() - t0))
    json.dump(dict(summary=summary, views=res), open(f'{a.out}/ibr_report.json', 'w'), indent=1)
    print('IBR SUMMARY', json.dumps(summary), flush=True)
    import shutil; os.makedirs(f'{a.out}/proof_frames', exist_ok=True)                 # 32 evenly spaced real frames for the Tencent demo (P2)
    for g in np.linspace(0, len(files) - 1, 32).astype(int): shutil.copy(files[g], f'{a.out}/proof_frames/')
