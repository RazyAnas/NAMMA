"""AI find (plan of 16 Nov): type any thing, the model's own frames are searched for it, and it is marked in 3D as a POI.
Two tiers, both free and open models only:
  index   (objects3d.py calls region_index) every SAM mask of every frame becomes a region: CLIP image embedding of its crop +
          its place in the room (median of the splats under it, tour metres) + the frame. regions.json + regions.f16 go into the
          tour; the page embeds the typed words with CLIP's text encoder and runs `search` below (poi_core.js has the same
          rule): a region counts when the words beat generic words on it, a hit only when regions from two different frames
          agree on the place within 40 cm.
  agent   (this file as a script, on Kaggle after objects3d) for what the index missed. A loop of tools with Qwen2.5-VL as
          the planner: words -> names of visible things (phrases) -> the frames most likely to show them (rank, CLIP) -> boxes
          (detect, Grounding DINO) -> is it really that? (verify, Qwen on the crop) -> its place (lift, the model's rendered
          depth) -> do two frames agree? (agree). Nothing found: the planner is told what was tried and names it differently,
          once more. Every step is written to the trace; nothing is marked that was not seen and confirmed.
Pure functions first (tests/v3/test_ai_find.py, no GPU); the models are only loaded in region_index and main.
  python ai_find.py <data dir> <model.ply> <level.json> <metres_per_unit> <out dir> '<json list of things to find>' [frames=60]"""
import json, os, re, sys, time
import numpy as np

CLIP = 'openai/clip-vit-base-patch32'            # the same model in the browser (Xenova/clip-vit-base-patch32, text half)
DINO = 'IDEA-Research/grounding-dino-tiny'       # Apache 2.0
QWEN = 'Qwen/Qwen2.5-VL-3B-Instruct'             # Apache 2.0 (objects3d names objects with it)
T0 = time.time()


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ---- geometry
def to_tour(X, L, mpu):
    """Model world -> tour coordinates in metres (the level matrix, then metres per unit), as objects3d's boxes."""
    X = np.atleast_2d(np.asarray(X, float)); return (np.c_[X, np.ones(len(X))] @ np.asarray(L, float).T)[:, :3] * mpu


def lift(depth, K, c2w, box, inner=0.5):
    """The 3D point (model world) a box in a frame looks at: the median rendered depth in the middle `inner` of the box, then
    out along the ray of the box centre. depth [H, W] = z in the camera (0 = nothing there); OpenCV camera (x right, y down,
    z forward), c2w as gsplat's loader. None if the middle of the box sees nothing."""
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; hw, hh = (x1 - x0) * inner / 2, (y1 - y0) * inner / 2
    H, W = depth.shape; a, b = max(0, int(cy - hh)), min(H, int(np.ceil(cy + hh)) + 1); c, d = max(0, int(cx - hw)), min(W, int(np.ceil(cx + hw)) + 1)
    z = depth[a:b, c:d]; z = z[z > 1e-4]
    if len(z) < 4: return None
    z = float(np.median(z)); p = np.array([(cx - K[0, 2]) * z / K[0, 0], (cy - K[1, 2]) * z / K[1, 1], z, 1.0])
    return (np.asarray(c2w, float) @ p)[:3]


def agree(cands, radius=0.4, min_frames=2):
    """Candidates (dicts with pos [3] in metres, frame, score) grouped by place: strongest first, each joins the group whose
    centre is within `radius`, else starts one. A group is `ok` when at least `min_frames` different frames are in it: one view
    can be a reflection, a picture of the thing, or a wrong box; two views landing on the same 40 cm are the thing.
    Its score is the mean of its two best frames. Returns the groups, ok ones first, best first."""
    out = []
    for c in sorted(cands, key=lambda c: -c['score']):
        p = np.asarray(c['pos'], float); best = None
        for g in out:
            d = float(np.linalg.norm(p - g['pos']))
            if d <= radius and (best is None or d < best[0]): best = (d, g)
        if best: g = best[1]; g['members'].append(c); w = np.array([m['score'] for m in g['members']]); g['pos'] = (np.array([m['pos'] for m in g['members']], float) * w[:, None]).sum(0) / w.sum()
        else: out.append(dict(pos=p, members=[c]))
    for g in out:
        per = {}
        for m in g['members']: per[m['frame']] = max(per.get(m['frame'], -1e9), m['score'])
        top = sorted(per.values(), reverse=True)[:2]; g.update(frames=sorted(per), score=float(np.mean(top)), ok=len(per) >= min_frames)
    return sorted(out, key=lambda g: (not g['ok'], -g['score']))


CANON = ('an object', 'things', 'stuff', 'a texture')   # generic words every region is compared with (the rule LERF uses)


def search(q, E, items, top=24, radius=0.4, min_frames=2, floor=0.2, margin=0.03):
    """The page's instant find, here as the reference (poi_core.js `aiSearch` must return the same): q = text embedding (unit),
    E = region embeddings [N, D] (unit), items[i] = dict(p=[x, y, z] metres, f=frame, c=that region's best score against the
    generic words CANON). A region counts when the words beat the generic ones on it by `margin` (score - c) and the score is
    above `floor`. Raw CLIP scores sit in a narrow band (0.20-0.30) that says little on its own: on a real house's photos
    (16 Nov) the raw best was the right place for 4 of 10 things, the best margin for 9 of 10. An index without `c` falls back to
    "stands out in this room" (mean + 2.5 x std). The `top` best by margin are grouped by place (agree). found = the best group
    seen from >= min_frames frames; else `near` = the best single region, for "not found, the closest thing is there".
    On a real region index (SAM crops, Kaggle test 2, 16 Nov, 577 regions of a hotel room): 13 of 15 things found, 3 of 3
    absent ones refused; the two misses (television, coat) had the right region first but only one frame."""
    E = np.asarray(E, np.float32); s = E @ np.asarray(q, np.float32)
    if len(s) == 0: return dict(found=None, near=None, threshold=None)
    c = np.array([it['c'] for it in items], np.float32) if all('c' in it for it in items) else np.full(len(s), float(s.mean() + 2.5 * s.std()) - margin, np.float32)
    m = s - c; order = np.argsort(-m, kind='stable')[:top]
    cand = [dict(pos=items[i]['p'], frame=items[i]['f'], score=float(m[i]), region=int(i)) for i in order if m[i] >= margin and s[i] >= floor]
    g = [x for x in agree(cand, radius, min_frames) if x['ok']]; i0 = int(order[0])
    hit = lambda x: dict(pos=[round(float(v), 3) for v in x['pos']], score=round(x['score'], 4), frames=x['frames'], regions=[k['region'] for k in x['members']])
    return dict(found=hit(g[0]) if g else None, near=dict(pos=items[i0]['p'], score=round(float(m[i0]), 4), frame=items[i0]['f'], region=i0),
                threshold=margin)


# ---- the agent loop (tools are passed in: real models on Kaggle, stand-ins in the tests)
def find_one(query, tools, rounds=2, max_frames=12, radius=0.4):
    """Find `query` with the tools: phrases(query, feedback) -> [names]; rank(names) -> [frame ids, best first];
    detect(frame, names) -> [(box, score, name)]; verify(frame, box, query) -> bool; lift(frame, box) -> pos metres | None.
    Returns dict(query, found, pos, confidence, frames, score, box, frame, trace). confidence: high (>= 3 frames agree),
    medium (2), low (one confirmed view only: shown to the admin as 'check this'), none."""
    trace = []; step = lambda tool, **kv: trace.append(dict(tool=tool, **kv)); feedback = None; best_single = None
    for rnd in range(rounds):
        names = [n for n in tools['phrases'](query, feedback) if n][:5] or [query]; step('phrases', round=rnd + 1, names=names)
        frames = list(tools['rank'](names))[:max_frames]; step('rank', frames=frames); cands = []; rejected = 0
        for f in frames:
            boxes = tools['detect'](f, names)
            for box, score, name in boxes:
                if not tools['verify'](f, box, query): rejected += 1; continue
                p = tools['lift'](f, box)
                if p is None: continue
                cands.append(dict(pos=[float(x) for x in p], frame=f, score=float(score), box=[float(x) for x in box], name=name))
            step('detect', frame=f, boxes=len(boxes), kept=sum(c['frame'] == f for c in cands))
        groups = agree(cands, radius); step('agree', candidates=len(cands), rejected_by_verify=rejected,
                                           groups=[dict(frames=g['frames'], score=round(g['score'], 3), ok=g['ok']) for g in groups[:4]])
        if groups and groups[0]['ok']:
            g = groups[0]; m = max(g['members'], key=lambda c: c['score'])
            return dict(query=query, found=True, pos=[round(float(x), 3) for x in g['pos']], confidence='high' if len(g['frames']) >= 3 else 'medium',
                        frames=g['frames'], score=round(g['score'], 3), box=m['box'], frame=m['frame'], name=m['name'], trace=trace)
        if cands:
            c = max(cands, key=lambda c: c['score'])
            if best_single is None or c['score'] > best_single['score']: best_single = c
        feedback = dict(tried=names, frames_looked_at=len(frames), confirmed_views=len(cands))
    if best_single:
        c = best_single
        return dict(query=query, found=True, pos=[round(x, 3) for x in c['pos']], confidence='low', frames=[c['frame']], score=round(c['score'], 3),
                    box=c['box'], frame=c['frame'], name=c['name'], trace=trace)
    return dict(query=query, found=False, pos=None, confidence='none', frames=[], score=0.0, box=None, frame=None, name=None, trace=trace)


def parse_names(text, limit=5):
    """The planner's answer as a list of short names: one per line or comma separated, numbering and quotes removed."""
    out = []
    for ln in re.split(r'[\n,;]', text):
        ln = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', ln).strip().strip('"\'.').lower()
        if ln and len(ln) <= 40 and len(ln.split()) <= 5 and ln not in out: out.append(ln)
    return out[:limit]


def slug(s): return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:40] or 'thing'


# ---- the region index (called by objects3d.main; needs torch + transformers)
def region_index(regions, frames, Xn, sid, L, mpu, out, dev='cuda', batch=64, max_spread_m=1.2):
    """regions: [(frame number in `frames`, mask box [x, y, w, h] px, indices of the opaque splats seen under the mask)].
    Writes regions.json (items: f = dataset frame, p = place in tour metres, o = object id or 0, n = splats, c = its best CLIP
    score against the generic words CANON, which `search` subtracts) and regions.f16
    (float16 [N, 512], unit length). A mask whose splats spread over more than max_spread_m (it spans near and far) is left out."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(CLIP).to(dev).eval(); cp = CLIPProcessor.from_pretrained(CLIP); items, crops = [], []
    for fi, (x, y, w, h), sel in regions:
        if w < 24 or h < 24 or len(sel) < 30: continue
        P = to_tour(Xn[sel], L, mpu); c = np.median(P, 0); spread = float(np.percentile(np.linalg.norm(P - c, axis=1), 80))
        if spread > max_spread_m: continue
        img = frames[fi][1]; H, W = img.shape[:2]; mx, my = 0.1 * w + 4, 0.1 * h + 4
        crops.append(Image.fromarray(img).crop((int(max(0, x - mx)), int(max(0, y - my)), int(min(W, x + w + mx)), int(min(H, y + h + my)))))
        ids = sid[sel]; ids = ids[ids > 0]; obj = int(np.bincount(ids).argmax()) if len(ids) > 0.5 * len(sel) else 0
        items.append(dict(f=int(frames[fi][0]), p=[round(float(v), 3) for v in c], o=obj, n=int(len(sel)), r=round(spread, 2)))
    E = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            e = cm.visual_projection(cm.vision_model(pixel_values=cp(images=crops[i:i + batch], return_tensors='pt')['pixel_values'].to(dev)).pooler_output)
            E.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    E = np.concatenate(E) if E else np.zeros((0, 512), np.float32); E.astype('<f2').tofile(f'{out}/regions.f16')
    with torch.no_grad():                                                                   # each region's best score against the generic words
        tk = cp.tokenizer([f'a photo of {c}' for c in CANON], padding=True, return_tensors='pt').to(dev)
        tf = cm.text_projection(cm.text_model(**tk).pooler_output); tf = (tf / tf.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    if len(items):
        for it, c in zip(items, (E @ tf.T).max(1)): it['c'] = round(float(c), 4)
    json.dump(dict(model=CLIP, dim=int(E.shape[1]), count=len(items), prompt='a photo of a {}', items=items), open(f'{out}/regions.json', 'w'))
    del cm; torch.cuda.empty_cache(); say('region index:', len(items), 'regions of', len(regions), 'masks'); return len(items)


# ---- the agent's real tools (Kaggle, one T4)
class Tools:
    def __init__(self, data, ply, level_json, mpu, out, nframes=60, dev='cuda'):
        import torch
        from PIL import Image
        sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser, Dataset
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor, Qwen2_5_VLForConditionalGeneration)
        import objects3d
        self.torch, self.Image, self.dev, self.out, self.mpu = torch, Image, dev, out, mpu; self.L = np.array(json.load(open(level_json))['matrix'])
        p = Parser(data, factor=1, normalize=True, test_every=8); cid = p.camera_ids[0]; K0 = p.Ks_dict[cid].copy(); W0, H0 = p.imsize_dict[cid]
        ds = Dataset(p, split='train'); pick = np.linspace(0, len(ds) - 1, min(nframes, len(ds))).astype(int)
        k = 1024 / max(W0, H0); self.W, self.H = round(W0 * k), round(H0 * k); self.K = K0.copy(); self.K[:2] *= k
        self.s = objects3d.load_ply(ply, dev); self.fr = {}
        for j in pick:
            d = ds[int(j)]; img = np.asarray(Image.fromarray(d['image'].numpy().astype(np.uint8)).resize((self.W, self.H), Image.LANCZOS))
            self.fr[int(ds.indices[int(j)])] = (img, d['camtoworld'].numpy())
        self.depths = {}; say(len(self.fr), 'frames at', self.W, 'x', self.H)
        self.cm = CLIPModel.from_pretrained(CLIP).to(dev).eval(); self.cp = CLIPProcessor.from_pretrained(CLIP)
        ids, tiles = [], []                                                                # each frame whole + as 2 x 2 tiles: a small thing fills a tile
        for f, (img, _) in self.fr.items():
            im = Image.fromarray(img); ids.append(f); tiles.append(im)
            for a in range(2):
                for b in range(2): ids.append(f); tiles.append(im.crop((b * self.W // 2, a * self.H // 2, (b + 1) * self.W // 2, (a + 1) * self.H // 2)))
        E = []
        with torch.no_grad():
            for i in range(0, len(tiles), 64):
                e = self.cm.visual_projection(self.cm.vision_model(pixel_values=self.cp(images=tiles[i:i + 64], return_tensors='pt')['pixel_values'].to(dev)).pooler_output)
                E.append((e / e.norm(dim=-1, keepdim=True)).float())
        self.tile_f, self.tile_e = np.array(ids), torch.cat(E)
        self.gp = AutoProcessor.from_pretrained(DINO); self.gm = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(dev).eval()
        self.vp = AutoProcessor.from_pretrained(QWEN)
        self.vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN, torch_dtype=torch.bfloat16).to(dev).eval()

    def ask(self, text, image=None, tokens=48):
        """One question to Qwen2.5-VL (with or without a picture), its answer as text."""
        content = ([{'type': 'image'}] if image is not None else []) + [{'type': 'text', 'text': text}]
        t = self.vp.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        enc = (self.vp(text=[t], images=[image], return_tensors='pt') if image is not None else self.vp(text=[t], return_tensors='pt')).to(self.dev)
        with self.torch.no_grad(): o = self.vl.generate(**enc, max_new_tokens=tokens, do_sample=False)
        return self.vp.batch_decode(o[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()

    def phrases(self, query, feedback=None):
        q = (f'Someone walking through a home or building asks to be taken to: "{query}". Which visible things should a camera look for? '
             'Give up to 4 short names of physical objects (1 to 3 words each), one per line, most likely first. Names only.')
        if feedback: q += f" Looking for {', '.join(feedback['tried'])} found nothing. Give other names: a part of it, what it is attached to, or another word for it."
        return parse_names(self.ask(q)) or [query.lower()[:40]]

    def rank(self, names):
        tk = self.cp.tokenizer([f'a photo of a {n}' for n in names], padding=True, return_tensors='pt').to(self.dev)
        with self.torch.no_grad(): t = self.cm.text_projection(self.cm.text_model(**tk).pooler_output); t = t / t.norm(dim=-1, keepdim=True)
        s = (self.tile_e @ t.float().T).max(1).values.cpu().numpy(); best = {}
        for f, v in zip(self.tile_f, s): best[int(f)] = max(best.get(int(f), -1), float(v))
        return sorted(best, key=lambda f: -best[f])

    def detect(self, frame, names, thr=0.3):
        im = self.Image.fromarray(self.fr[frame][0]); text = ' '.join(n.strip('.') + ' .' for n in names)
        inp = self.gp(images=im, text=text, return_tensors='pt').to(self.dev)
        with self.torch.no_grad(): o = self.gm(**inp)
        try: r = self.gp.post_process_grounded_object_detection(o, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[(self.H, self.W)])[0]
        except TypeError: r = self.gp.post_process_grounded_object_detection(o, inp.input_ids, box_threshold=thr, text_threshold=0.25, target_sizes=[(self.H, self.W)])[0]
        lab = r.get('text_labels', r.get('labels')); out = []
        for b, sc, l in zip(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), lab):
            if (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * self.W * self.H: out.append(([float(x) for x in b], float(sc), str(l)))   # not the whole frame
        return sorted(out, key=lambda x: -x[1])[:4]

    def crop(self, frame, box, margin=0.2):
        x0, y0, x1, y1 = box; mx, my = margin * (x1 - x0) + 8, margin * (y1 - y0) + 8
        return self.Image.fromarray(self.fr[frame][0]).crop((int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(self.W, x1 + mx)), int(min(self.H, y1 + my))))

    def verify(self, frame, box, query):
        a = self.ask(f'Does this picture show: {query}? Answer yes or no only.', self.crop(frame, box), tokens=4)
        return a.lower().startswith('yes')

    def lift(self, frame, box):
        if frame not in self.depths:
            from gsplat.rendering import rasterization
            torch, s = self.torch, self.s; c2w = self.fr[frame][1]
            with torch.no_grad():
                vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=self.dev)[None]; Kt = torch.tensor(self.K, dtype=torch.float32, device=self.dev)[None]
                rd, al, _ = rasterization(s['means'], s['quats'], torch.exp(s['scales']), torch.sigmoid(s['opacities']), s['colors'], vm, Kt, self.W, self.H,
                                          sh_degree=int(round(np.sqrt(s['colors'].shape[1]))) - 1, render_mode='ED')
            d = rd[0, ..., 0].cpu().numpy(); d[al[0, ..., 0].cpu().numpy() < 0.5] = 0; self.depths[frame] = d
        p = lift(self.depths[frame], self.K, self.fr[frame][1], box)
        return None if p is None else to_tour(p, self.L, self.mpu)[0]


def main(data, ply, level_json, mpu, out, queries, nframes=60):
    os.makedirs(f'{out}/find', exist_ok=True); t = Tools(data, ply, level_json, mpu, out, nframes)
    tools = dict(phrases=t.phrases, rank=t.rank, detect=t.detect, verify=t.verify, lift=t.lift); pois, missing = [], []
    for q in queries:
        t1 = time.time(); r = find_one(q, tools); r['seconds'] = round(time.time() - t1, 1)
        say(f"'{q}': {'found' if r['found'] else 'NOT found'} ({r['confidence']}, frames {r['frames']}, {r['seconds']} s)")
        if not r['found']: missing.append(dict(query=q, trace=r['trace'])); continue
        thumb = f'find/{slug(q)}.jpg'; t.crop(r['frame'], r['box']).convert('RGB').save(f'{out}/{thumb}', quality=88)
        pois.append(dict(id=f'ai-{slug(q)}', name=q[:1].upper() + q[1:], query=q, pos=r['pos'], confidence=r['confidence'], frames=r['frames'],
                         score=r['score'], seen_as=r['name'], thumb=thumb, source='ai', review=r['confidence'] == 'low', trace=r['trace']))
    json.dump(dict(pois=pois, not_found=missing, frames=len(t.fr), minutes=round((time.time() - T0) / 60, 1),
                   models=dict(planner=QWEN, detector=DINO, ranker=CLIP)), open(f'{out}/pois_ai.json', 'w'), indent=1)
    say('AI FIND DONE', json.dumps([(p['name'], p['confidence'], p['pos']) for p in pois]), 'not found:', [m['query'] for m in missing])


if __name__ == '__main__':
    main(*sys.argv[1:4], float(sys.argv[4]), sys.argv[5], json.loads(sys.argv[6]), *(int(x) for x in sys.argv[7:8]))
