"""Dense start for the splat trainer (fast_patch NM_DENSE; plan of 26 Nov). gsplat starts from the camera solve's sparse points
(tens of thousands) and spends most of its steps growing splats onto the surfaces. Here a depth network gives a depth map for
every few training frames; each map is put on the camera solve's scale with that frame's own SfM points (the rule of
geo_prior.py: frames whose points disagree with the map are dropped), every pixel is pushed out to its depth, and the points
are merged on a small grid. Only grid cells that at least two different frames agree on are kept, so one frame's wrong depth
cannot plant points in the air. Colours are the real photo's.
Depth model: DA3_MODEL (default depth-anything/DA3-BASE, Apache-2.0; the giant model of geo_prior.py is non-commercial).
  python dense_init.py <data dir (images/ + sparse/0)> <out dir> [every n-th training frame = 4]
(run from gsplat's examples dir; writes dense.npz (xyz, rgb: the trainer's world) + dense_report.json)"""
import json, os, sys, time, numpy as np

LONG = 480                                                                                     # depth map size used (long side)
MODELS = ('depth-anything/DA3-BASE', 'depth-anything/DA3-SMALL')                              # Apache-2.0 sizes, best first


def edges(z, rel=0.04):
    """True where the depth jumps (more than `rel` of the depth to a neighbour): pixels there float between two surfaces."""
    bad = np.zeros(z.shape, bool); dx = np.abs(np.diff(z, axis=1)) > rel * np.minimum(z[:, 1:], z[:, :-1])
    dy = np.abs(np.diff(z, axis=0)) > rel * np.minimum(z[1:], z[:-1])
    bad[:, 1:] |= dx; bad[:, :-1] |= dx; bad[1:] |= dy; bad[:-1] |= dy; return bad


def unproject(z, K, c2w):
    """Depth map [h, w] (0 = none; K at this size; c2w with +z forward) -> world points [n, 3] and their pixel mask [h, w]."""
    h, w = z.shape; ok = (z > 0) & np.isfinite(z) & ~edges(np.where(z > 0, z, 1e9))
    v, u = np.nonzero(ok); d = z[v, u].astype(np.float64)
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * d, (v + 0.5 - K[1, 2]) / K[1, 1] * d, d], 1)
    return (pc @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32), ok


def key(ijk):
    """One integer per grid cell (cells within +-2^20 of the origin on each axis)."""
    q = ijk.astype(np.int64) + (1 << 20); return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def cloud(depths, Ks, c2ws, imgs, voxel, min_views=2, cap=800_000, seed=0):
    """Merged start points. depths: [h, w] maps in world units; Ks at map size; c2ws 4x4; imgs: [h, w, 3] uint8 at map size.
    One point per grid cell of size `voxel` (mean position and colour), kept only if >= min_views different frames put a
    point there. More than `cap` cells: a random `cap` of them (fixed seed). Returns xyz float32 [n, 3], rgb uint8 [n, 3]."""
    ks, xs, cs = [], [], []
    for z, K, c2w, im in zip(depths, Ks, c2ws, imgs):
        p, ok = unproject(np.asarray(z, np.float32), K, c2w)
        if not len(p): continue
        c = np.asarray(im, np.float32)[ok]; k = key(np.floor(p / voxel))
        u, inv = np.unique(k, return_inverse=True); n = np.bincount(inv, minlength=len(u)).astype(np.float32)[:, None]   # this frame votes
        ks.append(u); xs.append(np.stack([np.bincount(inv, p[:, i], len(u)) for i in range(3)], 1) / n)              # once per cell
        cs.append(np.stack([np.bincount(inv, c[:, i], len(u)) for i in range(3)], 1) / n)
    if not ks: return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    k, x, c = np.concatenate(ks), np.concatenate(xs), np.concatenate(cs)
    u, inv, votes = np.unique(k, return_inverse=True, return_counts=True); n = votes.astype(np.float64)[:, None]
    xyz = np.stack([np.bincount(inv, x[:, i], len(u)) for i in range(3)], 1) / n
    rgb = np.stack([np.bincount(inv, c[:, i], len(u)) for i in range(3)], 1) / n
    keep = np.nonzero(votes >= min_views)[0]
    if len(keep) > cap: keep = np.sort(np.random.default_rng(seed).choice(keep, cap, replace=False))
    return xyz[keep].astype(np.float32), np.clip(rgb[keep] + 0.5, 0, 255).astype(np.uint8)


def sfm_scale(z, K, W, H, w2c, pts):
    """Scale that puts depth map z (any size) on the SfM points this frame sees, and how much they disagree (geo_prior's rule)."""
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]; pc = pc[pc[:, 2] > 1e-6]
    if not len(pc): return None, 0, None
    hz, wz = z.shape; u = (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]) * wz / W; v = (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]) * hz / H
    ins = (u >= 0) & (u < wz - 1) & (v >= 0) & (v < hz - 1); d = z[v[ins].astype(int), u[ins].astype(int)]; r = pc[ins, 2][d > 0] / d[d > 0]
    if len(r) < 30: return None, int(len(r)), None
    s = float(np.median(r)); return s, int(len(r)), float(np.median(np.abs(np.log(r / s))))


def load_model():
    import densify
    names = [os.environ['DA3_MODEL']] if os.environ.get('DA3_MODEL') else list(MODELS); last = None
    for n in names:
        try: m = densify.da3(n); print('dense start: depth model', n, flush=True); return n, m
        except Exception as e: print('dense start: model', n, 'not usable:', repr(e)[:200], flush=True); last = e
    raise RuntimeError(f'no depth model could be loaded: {last!r}')


def prepare(data_dir, out, every=4, chunk=8):
    import cv2, gc, torch
    from datasets.colmap import Parser
    t0 = time.time(); p = Parser(data_dir, factor=1, normalize=True, test_every=8); os.makedirs(f'{out}/und', exist_ok=True)
    train = [i for i in range(len(p.image_names)) if i % 8 != 0][::max(1, int(os.environ.get('NM_TRAIN_EVERY') or 1))]
    pick = train[::max(1, every)]; paths = []; small = []
    for i in pick:
        cid = p.camera_ids[i]; im = cv2.imread(p.image_paths[i])
        if cid in p.mapx_dict:                                                                 # SIMPLE_RADIAL -> pinhole, cropped
            im = cv2.remap(im, p.mapx_dict[cid], p.mapy_dict[cid], cv2.INTER_LINEAR); x, y, w, h = p.roi_undist_dict[cid]; im = im[y:y + h, x:x + w]
        q = f'{out}/und/{len(paths):05d}.jpg'; cv2.imwrite(q, im, [cv2.IMWRITE_JPEG_QUALITY, 95]); paths.append(q)
        k = LONG / max(im.shape[:2]); small.append(cv2.cvtColor(cv2.resize(im, (round(im.shape[1] * k), round(im.shape[0] * k)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    name, model = load_model(); rep = dict(model=name, frames=len(pick), kept=0, dropped=[], spread=[]); D, KS, C, IM = [], [], [], []
    for a in range(0, len(paths), chunk):
        with torch.no_grad(): pred = model.inference(paths[a:a + chunk])
        for j, z in zip(range(a, a + chunk), np.asarray(pred.depth)):
            i = pick[j]; cid = p.camera_ids[i]; W, H = p.imsize_dict[cid]; K = p.Ks_dict[cid]; c2w = p.camtoworlds[i]
            s, n, spread = sfm_scale(z, K, W, H, np.linalg.inv(c2w), p.points[p.point_indices.get(p.image_names[i], np.zeros(0, int))])
            if s is None or spread > 0.15: rep['dropped'].append([p.image_names[i], n if s is None else round(spread, 3)]); continue
            h, w = small[j].shape[:2]; zz = cv2.resize((z * s).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)   # no mixed depths
            Ks = K.copy(); Ks[0] *= w / W; Ks[1] *= h / H
            D.append(zz); KS.append(Ks); C.append(c2w); IM.append(small[j]); rep['kept'] += 1; rep['spread'].append(spread)
        del pred; gc.collect(); torch.cuda.empty_cache()
    del model; gc.collect(); torch.cuda.empty_cache()
    assert D, 'dense start: no frame agreed with its SfM points'
    foot = float(np.median([np.median(z[z > 0]) / K[0, 0] for z, K in zip(D, KS)]))                # one depth pixel at the typical depth
    voxel = float(os.environ.get('DENSE_VOXEL') or 3.0) * foot
    xyz, rgb = cloud(D, KS, C, IM, voxel, cap=int(os.environ.get('DENSE_CAP') or 800_000))
    np.savez(f'{out}/dense.npz', xyz=xyz, rgb=rgb)
    rep.update(median_spread=round(float(np.median(rep.pop('spread') or [0])), 4), voxel=round(voxel, 5), points=int(len(xyz)), sfm_points=int(len(p.points)),
               minutes=round((time.time() - t0) / 60, 2), dropped=rep['dropped'][:20])
    json.dump(rep, open(f'{out}/dense_report.json', 'w'), indent=1); print('DENSE', json.dumps({k: v for k, v in rep.items() if k != 'dropped'}), flush=True)
    return rep


if __name__ == '__main__':
    prepare(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
