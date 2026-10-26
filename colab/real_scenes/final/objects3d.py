"""Object layer for a finished run (approved plan 26 Oct, 'NAMMA WorldGen'): which splats belong to which object, named or not,
with a photo, a size in metres and a box for walking into / clicking / search. The splats themselves are not changed or retrained.
  masks      SAM 2.1 automatic masks ("segment everything", no names needed) on ~40 training frames (held-out frames never used).
             Per pixel the largest mask under 25 % of the frame wins, so a sofa is one mask, not a cushion and not the wall.
  voting     every opaque splat centre that is visible in a frame (in front of the rendered depth) votes for the mask under it.
             Masks of different frames that share >= 50 % of their splats are the same object (union-find); each splat takes the
             object it voted for most (purity = that share). Faint splats inherit their nearest opaque neighbour's object.
  structure  floor and ceiling bands (room_prior.planes) and the room box sides (v3_qa.dollhouse) are walls, not objects.
  geometry   opaque splats standing clear of floor, ceiling and walls that no mask claimed are clustered in 3D (DBSCAN, 5 cm):
             anything physically there becomes an object even if no model recognises it.
  names      Qwen2.5-VL-3B (Apache) names each object's best crop in 1-4 words; generic answers -> "object N". A CLIP ViT-B/32 image
             embedding per object lets the tour search by appearance with the same model's text encoder in the browser.
  out        objects.json (id, label, metres box/centre in tour coordinates, splat count, purity, frames, crop, embedding),
             splat_object.npy (uint16 object id per splat, 0 = room), objects_sheet.jpg (crops + labels), crops/*.jpg
Usage (from gsplat's examples dir): python objects3d.py <data dir> <model.ply> <level.json> <metres_per_unit> <out dir> [n_frames]"""
import json, os, sys, time, numpy as np, torch
from PIL import Image, ImageDraw
T0 = time.time()
GENERIC = {'object', 'thing', 'item', 'wall', 'floor', 'ceiling', 'room', 'unknown', 'nothing', 'background', 'surface', 'image', 'photo'}


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def load_ply(path, dev):
    from plyfile import PlyData
    v = PlyData.read(path)['vertex'].data; t = lambda *k: torch.tensor(np.stack([v[x] for x in k], 1), dtype=torch.float32, device=dev)
    nr = len([n for n in v.dtype.names if n.startswith('f_rest_')]) // 3
    return dict(means=t('x', 'y', 'z'), scales=t('scale_0', 'scale_1', 'scale_2'), quats=t('rot_0', 'rot_1', 'rot_2', 'rot_3'),
                opacities=torch.tensor(np.asarray(v['opacity']), dtype=torch.float32, device=dev),
                colors=torch.cat([t('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None], t(*[f'f_rest_{i}' for i in range(3 * nr)]).reshape(-1, 3, nr).transpose(1, 2)], 1))


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def merge_masks(node_frame, node_splats, n_splats, min_share=0.5, min_common=30):
    """node_splats: list of int arrays (splat ids per (frame, mask) node). Links nodes of different frames sharing >= min_share of
    the smaller one; returns component id per node."""
    import scipy.sparse as sp
    rows = np.concatenate([np.full(len(s), i) for i, s in enumerate(node_splats)]) if node_splats else np.zeros(0, int)
    cols = np.concatenate(node_splats) if node_splats else np.zeros(0, int)
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(node_splats), n_splats))
    O = (M @ M.T).tocoo(); size = np.array([len(s) for s in node_splats]); uf = UF(len(node_splats))
    for a, b, c in zip(O.row, O.col, O.data):
        if a < b and node_frame[a] != node_frame[b] and c >= min_common and c >= min_share * min(size[a], size[b]): uf.union(a, b)
    return np.array([uf.find(i) for i in range(len(node_splats))])


def main(data, ply, level_json, mpu, out, nframes=40):
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser, Dataset
    from gsplat.rendering import rasterization
    import room_prior, v3_qa
    dev = 'cuda'; os.makedirs(f'{out}/crops', exist_ok=True)
    p = Parser(data, factor=1, normalize=True, test_every=8); C = p.camtoworlds; cid = p.camera_ids[0]
    K0 = p.Ks_dict[cid].copy(); W0, H0 = p.imsize_dict[cid]; s = load_ply(ply, dev); N = len(s['means'])
    ds = Dataset(p, split='train'); pick = np.linspace(0, len(ds) - 1, min(nframes, len(ds))).astype(int)
    k = 1024 / max(W0, H0); W, H = round(W0 * k), round(H0 * k); K = K0.copy(); K[:2] *= k
    op = torch.sigmoid(s['opacities']); solid = (op > 0.3).nonzero()[:, 0]; X = s['means'][solid]
    say(N, 'splats,', len(solid), 'opaque; frames', len(pick), f'at {W}x{H}')
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator.from_pretrained('facebook/sam2.1-hiera-large', points_per_side=24, pred_iou_thresh=0.8,
                                                     stability_score_thresh=0.9, min_mask_region_area=400)
    node_frame, node_splats, frames = [], [], []
    for fi, j in enumerate(pick):
        d = ds[int(j)]; img = np.asarray(Image.fromarray(d['image'].numpy().astype(np.uint8)).resize((W, H), Image.LANCZOS))
        c2w = d['camtoworld'].numpy(); frames.append((int(ds.indices[int(j)]), img, c2w))
        with torch.no_grad():
            vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]; Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]
            rd, _, _ = rasterization(s['means'], s['quats'], torch.exp(s['scales']), op, s['colors'], vm, Kt, W, H,
                                     sh_degree=int(round(np.sqrt(s['colors'].shape[1]))) - 1, render_mode='ED')
            ed = rd[0, ..., 0]; pc = X @ vm[0, :3, :3].T + vm[0, :3, 3]; z = pc[:, 2]
            u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).round().long(); v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ok[ok.clone()] &= z[ok] <= ed[v[ok], u[ok]] * 1.03 + 0.01                         # in front of the surface = visible
        with torch.inference_mode(): ms = gen.generate(img)                                  # fp32: T4, no bf16 kernels
        lab = np.zeros((H, W), np.int32)                                                    # largest mask < 25 % of the frame wins
        ms = sorted([m for m in ms if m['area'] < 0.25 * W * H], key=lambda m: m['area'])
        for mi, m in enumerate(ms): lab[m['segmentation']] = mi + 1
        idx = ok.nonzero()[:, 0].cpu().numpy(); mid = lab[v[idx].cpu().numpy(), u[idx].cpu().numpy()]
        for mi in np.unique(mid[mid > 0]):
            sel = idx[mid == mi]
            if len(sel) >= 30: node_frame.append(fi); node_splats.append(sel)
        say(f'frame {fi + 1}/{len(pick)}: {len(ms)} masks, {int(ok.sum())} visible splats')
    del gen; torch.cuda.empty_cache()
    comp = merge_masks(node_frame, node_splats, len(solid))
    votes = {}                                                                              # (splat, component) -> votes
    for c, sp_ in zip(comp, node_splats):
        for x in sp_: votes[(int(x), int(c))] = votes.get((int(x), int(c)), 0) + 1
    best, tot = {}, {}
    for (x, c), n in votes.items():
        tot[x] = tot.get(x, 0) + n
        if n > best.get(x, (0, -1))[0]: best[x] = (n, c)
    owner = np.full(len(solid), -1); pure = np.zeros(len(solid))
    for x, (n, c) in best.items(): owner[x], pure[x] = c, n / tot[x]
    # structure: floor / ceiling bands and the room box sides belong to the room
    Xn = X.cpu().numpy(); f, ceil, band, up = room_prior.planes(C, p.points); h = Xn @ up
    near_plane = np.abs(h - f) < band
    if ceil is not None: near_plane |= np.abs(h - ceil) < band
    inside, _, _ = v3_qa.dollhouse(Xn, p.points, C, up)
    flat = Xn - np.outer(h, up); m0 = flat.mean(0); ax = np.linalg.svd(flat - m0, full_matrices=False)[2][:2]; q = (flat - m0) @ ax.T
    lo, hi = np.percentile(q, 3, 0), np.percentile(q, 97, 0); wall = np.any((q < lo + 0.05 * (hi - lo)) | (q > hi - 0.05 * (hi - lo)), 1)
    room = near_plane | wall | ~inside
    objs = []
    for c in np.unique(owner[owner >= 0]):
        mem = np.where(owner == c)[0]
        if len(mem) < 300 or room[mem].mean() > 0.6: continue
        objs.append(dict(members=mem[~room[mem]], how='masks', purity=float(pure[mem].mean())))
    # geometry: what no mask claimed but stands clear of floor, ceiling and walls
    free = np.where((owner < 0) & ~room)[0]
    if len(free) > 500:
        import open3d as o3d
        if len(free) > 400_000: free = free[np.random.default_rng(0).permutation(len(free))[:400_000]]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xn[free].astype(np.float64)))
        cl = np.asarray(pcd.cluster_dbscan(eps=0.05 / mpu, min_points=20))
        for c in np.unique(cl[cl >= 0]):
            mem = free[cl == c]
            if len(mem) >= 500 and np.ptp(h[mem]) > 0.1 / mpu: objs.append(dict(members=mem, how='geometry', purity=1.0))
    say(len(objs), 'objects', sum(o['how'] == 'geometry' for o in objs), 'of them found only by geometry')
    # per splat id (faint splats: nearest opaque splat's object)
    sid = np.zeros(len(solid), np.uint16)
    for i, o in enumerate(objs): sid[o['members']] = i + 1
    from scipy.spatial import cKDTree
    allid = np.zeros(N, np.uint16); sol = solid.cpu().numpy(); allid[sol] = sid
    faint = np.setdiff1d(np.arange(N), sol); _, nn = cKDTree(Xn).query(s['means'][faint].cpu().numpy(), k=1, workers=-1)
    near = np.linalg.norm(s['means'][faint].cpu().numpy() - Xn[nn], axis=1) < 0.03 / mpu; allid[faint[near]] = sid[nn[near]]
    np.save(f'{out}/splat_object.npy', allid)
    # best crop per object: the frame where most of its splats are visible and its box is not cut by the frame edge
    L = np.array(json.load(open(level_json))['matrix'])
    for i, o in enumerate(objs):
        P = torch.tensor(Xn[o['members']], device=dev); bestf = None
        for fi, (gi, img, c2w) in enumerate(frames):
            w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev); pc = P @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]; v = K[1, 1] * pc[:, 1] / z + K[1, 2]; vis = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if vis.sum() < 50: continue
            box = [float(u[vis].quantile(0.02)), float(v[vis].quantile(0.02)), float(u[vis].quantile(0.98)), float(v[vis].quantile(0.98))]
            cut = box[0] < 3 or box[1] < 3 or box[2] > W - 3 or box[3] > H - 3; sc = float(vis.float().mean()) * (0.4 if cut else 1.0) * min(1, (box[2] - box[0]) * (box[3] - box[1]) / (0.04 * W * H))
            if bestf is None or sc > bestf[0]: bestf = (sc, fi, box)
        if bestf is None: o['crop'] = None; continue
        _, fi, (x0, y0, x1, y1) = bestf; mx, my = 0.12 * (x1 - x0) + 8, 0.12 * (y1 - y0) + 8
        Image.fromarray(frames[fi][1]).crop((max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))).save(f'{out}/crops/{i + 1}.jpg', quality=90)
        o['crop'], o['frame'] = f'crops/{i + 1}.jpg', int(frames[fi][0])
        Pl = (np.c_[Xn[o['members']], np.ones(len(o['members']))] @ L.T)[:, :3] * mpu       # tour coordinates, metres
        lo_, hi_ = np.percentile(Pl, 2, 0), np.percentile(Pl, 98, 0); o['box_m'] = [lo_.round(3).tolist(), hi_.round(3).tolist()]
        o['centre_m'] = ((lo_ + hi_) / 2).round(3).tolist(); o['size_m'] = (hi_ - lo_).round(2).tolist()
    del s; torch.cuda.empty_cache()
    # names (best effort) + appearance embedding for search
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    vp = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    vl = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.bfloat16).to(dev).eval()
    cm = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval(); cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'What is the main object in this photo of a room? '
            'Answer with 1 to 4 words only, the object name, no sentence.'}]}]
    text = vp.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for i, o in enumerate(objs):
        o['id'] = i + 1
        if not o.get('crop'): o['label'] = f'object {i + 1}'; continue
        im = Image.open(f'{out}/{o["crop"]}').convert('RGB')
        with torch.no_grad():
            enc = vp(text=[text], images=[im], return_tensors='pt').to(dev)
            ans = vp.batch_decode(vl.generate(**enc, max_new_tokens=12, do_sample=False)[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            e = cm.get_image_features(**cp(images=im, return_tensors='pt').to(dev)); e = (e / e.norm(dim=-1, keepdim=True))[0]
        ans = ans.strip().strip('.').lower()[:40]; o['vlm'] = ans
        o['label'] = ans if ans and ans not in GENERIC and len(ans.split()) <= 5 else f'object {i + 1}'
        o['embedding'] = [round(float(x), 4) for x in e.cpu()]
    # sheet + json
    tiles = []
    for o in objs:
        if not o.get('crop'): continue
        t = Image.open(f'{out}/{o["crop"]}').convert('RGB'); t.thumbnail((220, 220)); c = Image.new('RGB', (230, 260), (24, 24, 24)); c.paste(t, (5, 5))
        ImageDraw.Draw(c).text((6, 230), f"{o['id']} {o['label'][:26]}  {o['how'][0]} {o['purity']:.2f}", fill=(255, 255, 255)); tiles.append(c)
    if tiles:
        cols = 6; sheet = Image.new('RGB', (cols * 230, ((len(tiles) + cols - 1) // cols) * 260), (0, 0, 0))
        for n, t in enumerate(tiles): sheet.paste(t, ((n % cols) * 230, (n // cols) * 260))
        sheet.save(f'{out}/objects_sheet.jpg', quality=88)
    res = [{k: v for k, v in o.items() if k != 'members'} | dict(splats=int((allid == o['id']).sum())) for o in objs]
    json.dump(dict(objects=res, frames=len(pick), splats=N, metres_per_unit=mpu, minutes=round((time.time() - T0) / 60, 1)),
              open(f'{out}/objects.json', 'w'), indent=1)
    say('OBJECTS DONE', json.dumps([(o['id'], o['label'], o['how'], o.get('size_m')) for o in res]))


if __name__ == '__main__':
    main(*sys.argv[1:4], float(sys.argv[4]), sys.argv[5], *(int(x) for x in sys.argv[6:7]))
