"""Split a whole-home capture into rooms from its SfM model alone (CPU; per-room plan of 30 Oct, docs/HANDOFF_V3.md):
  1 rooms = communities of the co-visibility graph (cameras linked by the 3D points both see; doorways cut most links, the walk
    order does not matter). v1 (watershed of a wall map) split single rooms: furniture looked like walls (30 Oct)
  2 each 3D point -> the room whose cameras see it most; floor plan (5 cm cells, metres from the phone ~1.45 m above the floor):
    each cell -> the room with most points / cameras there (smoothed) -> outline, area, doors (where two rooms touch)
  3 'extra' cameras of a room = other rooms' cameras that see > 30 % of their points in it (views through doorways)
Writes rooms.json (rooms with outline in SfM coordinates, area, cameras, doors) and rooms.png (coloured plan, path, doors).
  python rooms.py <sparse/0 dir> <out dir>"""
import json, os, sys
import numpy as np

CELL, DOOR_MAX, CAM_H = 0.05, 1.3, 1.45


def load(sparse):
    from pycolmap import SceneManager                          # the fork gsplat uses (rmbrualla/pycolmap)
    m = SceneManager(sparse); m.load_cameras(); m.load_images(); m.load_points3D()
    ks = sorted(m.images, key=lambda k: m.images[k].name)
    R = np.stack([m.images[k].R() for k in ks]); t = np.stack([m.images[k].tvec for k in ks])
    C = -np.einsum('nji,nj->ni', R, t)                         # camera centres; viewing direction = R[2], camera y (down) = R[1]
    row = {pid: i for i, pid in enumerate(m.point3D_ids)}    # point id -> row of m.points3D
    obs = [np.array([row[p] for p in m.images[k].point3D_ids if p in row], np.int64) for k in ks]
    return [m.images[k].name for k in ks], C, R, np.asarray(m.points3D), obs


def covis_rooms(C, obs, npts, min_cams=40, resolution=1.0, seed=0):
    """Rooms as communities of the co-visibility graph: two cameras are linked by how many 3D points both see; doorways cut
    most of those links, the walk order does not matter (a room filmed twice is still one community). Louvain, then pieces with
    fewer than min_cams cameras are merged into the room they share most points with. Returns a room id per camera."""
    import networkx as nx
    from scipy import sparse
    rows = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)]); cols = np.concatenate(obs)
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(obs), npts))
    S = (A @ A.T).tocoo(); n = np.asarray(A.sum(1)).ravel() + 1
    g = nx.Graph(); g.add_nodes_from(range(len(obs)))
    for i, j, w in zip(S.row, S.col, S.data):
        if i < j and w >= 15: g.add_edge(int(i), int(j), weight=float(w) / np.sqrt(n[i] * n[j]))
    lab = np.zeros(len(obs), int)
    for k, cset in enumerate(nx.community.louvain_communities(g, weight='weight', resolution=resolution, seed=seed)): lab[list(cset)] = k
    while True:                                                 # small pieces -> the neighbour they share most with
        cnt = np.bincount(lab); small = [k for k in range(len(cnt)) if 0 < cnt[k] < min_cams]
        if not small: break
        k = min(small, key=lambda x: cnt[x]); mem = np.nonzero(lab == k)[0]; share = np.zeros(len(cnt))
        for i in mem:
            for j, d in g[i].items():
                if lab[j] != k: share[lab[j]] += d['weight']
        if share.max() <= 0: share[np.argmax(np.where(np.arange(len(cnt)) != k, cnt, -1))] = 1
        lab[mem] = int(share.argmax())
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def split(names, C, R, P, obs):
    from scipy import ndimage, sparse
    from skimage.measure import find_contours
    up = -np.median(R[:, 1, :], 0); up /= np.linalg.norm(up)
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    hp, hc = P @ up, C @ up
    floor = np.percentile(hp[hp < np.median(hc)], 3); scale = CAM_H / max(np.median(hc) - floor, 1e-6)   # metres per SfM unit
    ceil = np.percentile(hp[hp > np.median(hc)], 97)
    lab = covis_rooms(C, obs, len(P)); nr = lab.max() + 1
    rows = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)]); cols = np.concatenate(obs)
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(obs), len(P)))
    votes = (A.T @ sparse.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), nr))).toarray()
    seen = votes.sum(1) > 0; prow = np.where(seen, votes.argmax(1), -1)          # each point -> the room whose cameras see it most
    to2 = lambda X: np.c_[X @ e1, X @ e2] * scale
    p2, c2 = to2(P), to2(C); hm = (hp - floor) * scale; inb = seen & (hm > -0.2) & (hm < (ceil - floor) * scale + 0.2)
    lo = np.percentile(np.r_[p2[inb], c2], 0.5, 0) - 1; hi = np.percentile(np.r_[p2[inb], c2], 99.5, 0) + 1
    shape = np.ceil((hi - lo) / CELL).astype(int); cell = lambda q: np.clip(((q - lo) / CELL).astype(int), 0, shape - 1)
    vote = np.zeros(tuple(shape) + (nr,), np.int32); ci = cell(p2[inb]); np.add.at(vote, (ci[:, 0], ci[:, 1], prow[inb]), 1)
    cc = cell(c2); np.add.at(vote, (cc[:, 0], cc[:, 1], lab), 5)
    vote = ndimage.uniform_filter(vote.astype(np.float32), size=(9, 9, 1))
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(vote.sum(-1) > 0.05, iterations=6))
    reg = np.where(foot, vote.argmax(-1), -1)
    hiw = inb & (hm > 1.9) & (hm < (ceil - floor) * scale - 0.15)             # tall things only: walls, not furniture (for the plan)
    wall = np.zeros(shape, np.int32); np.add.at(wall, tuple(cell(p2[hiw]).T), 1)
    wallm = wall >= max(2, np.percentile(wall[wall > 0], 70)) if (wall > 0).any() else wall > 0
    fwd2 = np.c_[R[:, 2] @ e1, R[:, 2] @ e2]; fwd2 /= np.linalg.norm(fwd2, axis=1, keepdims=True) + 1e-9
    rooms = []
    for i in range(nr):
        m = reg == i
        if not m.any(): continue
        cs = max(find_contours(np.pad(m, 1).astype(float), 0.5), key=len) - 1; poly = (cs[::max(1, len(cs) // 200)] * CELL + lo) / scale
        doors = []
        for j in range(nr):
            sh = ndimage.binary_dilation(m, iterations=1) & (reg == j) if j != i else None
            if sh is not None and sh.sum():
                cen = np.argwhere(sh).mean(0) * CELL + lo; doors.append(dict(to=j, centre=(cen / scale).tolist(), width_m=round(float(sh.sum() * CELL), 2)))
        own = np.nonzero(lab == i)[0]; mine = prow == i
        share = np.array([mine[o].mean() if len(o) else 0 for o in obs])        # other rooms' cameras that see a lot of this room
        extra = np.nonzero((lab != i) & (share > 0.3))[0]
        rooms.append(dict(id=i, area_m2=round(float(m.sum() * CELL ** 2), 1), cameras=own.tolist(), extra=extra.tolist(), points=int(mine.sum()),
                          outline=poly.tolist(), doors=doors, first_frame=names[own.min()] if len(own) else None))
    order = sorted(range(len(rooms)), key=lambda i: min(rooms[i]['cameras'] or [1 << 30]))
    meta = dict(metres_per_unit=1 / scale, up=up.tolist(), e1=e1.tolist(), e2=e2.tolist(), origin_m=lo.tolist(), cell_m=CELL,
                floor=float(floor), ceiling=float(ceil), cameras=len(C), points=len(P), point_room=prow.tolist())
    return meta, rooms, order, (reg, wallm, c2, lo)


def draw(path, reg, wallm, c2, lo, rooms):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(3); pal = (rng.random((max(1, reg.max() + 1), 3)) * 160 + 70).astype(np.uint8)
    img = np.full(reg.shape + (3,), 30, np.uint8); img[reg >= 0] = pal[reg[reg >= 0]]; img[wallm] = 235
    im = Image.fromarray(np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1])); dr = ImageDraw.Draw(im); H = im.size[1]
    xy = lambda q: ((q[0] - lo[0]) / CELL, H - 1 - (q[1] - lo[1]) / CELL)
    dr.line([xy(p) for p in c2], fill=(0, 0, 0), width=2)
    for r in rooms:
        cen = np.argwhere(reg == r['id']).mean(0) * CELL + lo
        dr.text(xy(cen), f"room {r['id']}\n{r['area_m2']} m2\n{len(r['cameras'])} cams", fill=(0, 0, 0))
    k = max(1, 1200 // max(im.size)); im = im.resize((im.size[0] * k, im.size[1] * k), Image.NEAREST) if k > 1 else im
    im.save(path)


if __name__ == '__main__':
    sp, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    names, C, R, P, obs = load(sp); meta, rooms, order, (reg, wallm, c2, lo) = split(names, C, R, P, obs)
    json.dump(dict(meta, order=order, rooms=rooms), open(f'{out}/rooms.json', 'w'))
    draw(f'{out}/rooms.png', reg, wallm, c2, lo, rooms)
    print(len(rooms), 'rooms:', [(r['id'], r['area_m2'], len(r['cameras']), len(r['extra']), [d['to'] for d in r['doors']]) for r in rooms], flush=True)
