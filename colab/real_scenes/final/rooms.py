"""Split a whole-home capture into rooms from its SfM model alone (CPU; per-room plan of 30 Oct, docs/HANDOFF_V3.md):
  1 up = the cameras' median up axis; heights -> floor (low percentile of points), metres from the phone's ~1.45 m above the floor
  2 floor plan grid (5 cm): wall cells = many points in the wall band (0.3 m above the floor .. 0.3 m below the ceiling);
    free space = floor seen (floor points) or walked (camera path), minus walls
  3 rooms = watershed of the free space's distance-to-wall map (doorways are narrow necks), seeded at its wide spots; neighbours
    joined when the opening between them is wider than a door (DOOR_MAX m); tiny or camera-less pieces merged into a neighbour
  4 each camera -> the room it stands in; plus 'extra' cameras of other rooms that stand near a shared doorway and look into it
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
    return [m.images[k].name for k in ks], C, R, np.asarray(m.points3D)


def split(names, C, R, P):
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    up = -np.median(R[:, 1, :], 0); up /= np.linalg.norm(up)
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    hp, hc = P @ up, C @ up
    near = np.linalg.norm((P - np.median(C, 0)) - np.outer(hp - np.median(hc), up), axis=1) < np.percentile(np.linalg.norm(C - np.median(C, 0), axis=1), 99) * 1.5 + 1e-9
    floor = np.percentile(hp[near & (hp < np.median(hc))], 3); scale = CAM_H / max(np.median(hc) - floor, 1e-6)   # metres per unit
    ceil = np.percentile(hp[near & (hp > np.median(hc))], 97)
    to2 = lambda X: np.c_[X @ e1, X @ e2] * scale
    p2, c2 = to2(P), to2(C); lo = np.percentile(np.r_[p2[near], c2], 0.5, 0) - 1; hi = np.percentile(np.r_[p2[near], c2], 99.5, 0) + 1
    shape = np.ceil((hi - lo) / CELL).astype(int); cell = lambda q: np.clip(((q - lo) / CELL).astype(int), 0, shape - 1)
    hm = (hp - floor) * scale; band = near & (hm > 0.3) & (hm < (ceil - floor) * scale - 0.3); flo = near & (hm < 0.15)
    wall = np.zeros(shape, np.int32); np.add.at(wall, tuple(cell(p2[band]).T), 1)
    thr = max(3, np.percentile(wall[wall > 0], 60)) if (wall > 0).any() else 3
    wallm = ndimage.binary_closing(wall >= thr, iterations=1)
    seen = np.zeros(shape, bool); seen[tuple(cell(p2[flo]).T)] = True
    walked = np.zeros(shape, bool); walked[tuple(cell(c2).T)] = True
    free = (ndimage.binary_closing(seen, iterations=4) | ndimage.binary_dilation(walked, iterations=int(0.4 / CELL))) & ~wallm
    free = ndimage.binary_opening(free, iterations=2)
    lab, n = ndimage.label(free); keep = np.isin(lab, np.unique(lab[tuple(cell(c2).T)])); free &= keep & (lab > 0)
    dist = ndimage.distance_transform_edt(free) * CELL
    pk = peak_local_max(dist, min_distance=int(1.2 / CELL), threshold_abs=0.5, labels=free.astype(int))
    mk = np.zeros(shape, np.int32)
    for i, (a, b) in enumerate(pk): mk[a, b] = i + 1
    reg = watershed(-dist, mk, mask=free)

    def merge_pass():                                          # join neighbours whose opening is wider than a door
        nonlocal reg
        changed = False
        for a in [x for x in np.unique(reg) if x]:
            ma = reg == a; ring = ndimage.binary_dilation(ma, iterations=1) & ~ma & (reg > 0)
            for b in np.unique(reg[ring]):
                if b == a: continue
                width = (ring & (reg == b)).sum() * CELL
                if width > DOOR_MAX or ma.sum() * CELL ** 2 < 2.0: reg[reg == a] = b; changed = True; break
            if changed: return True
        return False
    while merge_pass(): pass
    ci = cell(c2); camroom = reg[ci[:, 0], ci[:, 1]]
    for r in [x for x in np.unique(reg) if x and not (camroom == x).any()]:      # no camera stood here: give it to its widest neighbour
        ma = reg == r; ring = ndimage.binary_dilation(ma, iterations=1) & ~ma & (reg > 0); nb = reg[ring]
        if len(nb): reg[ma] = np.bincount(nb).argmax()
    ids = [x for x in np.unique(reg) if x]; remap = {x: i for i, x in enumerate(ids)}
    reg = np.vectorize(lambda v: remap.get(v, -1))(reg); camroom = reg[ci[:, 0], ci[:, 1]]
    if (camroom < 0).any():                                    # a camera off the free space (on a wall line): nearest room cell
        d = ndimage.distance_transform_edt(reg < 0, return_indices=True)[1]
        for j in np.nonzero(camroom < 0)[0]: camroom[j] = reg[d[0][tuple(ci[j])], d[1][tuple(ci[j])]]
    from skimage.measure import find_contours
    rooms = []
    fwd2 = np.c_[R[:, 2] @ e1, R[:, 2] @ e2]; fwd2 /= np.linalg.norm(fwd2, axis=1, keepdims=True) + 1e-9
    for i in range(len(ids)):
        m = reg == i; cs = max(find_contours(np.pad(m, 1).astype(float), 0.5), key=len) - 1
        poly = (cs[::max(1, len(cs) // 200)] * CELL + lo) / scale
        doors = []
        for j in range(len(ids)):
            if j == i: continue
            sh = ndimage.binary_dilation(m, iterations=2) & (reg == j)
            if sh.sum():
                cen = np.argwhere(sh).mean(0) * CELL + lo; doors.append(dict(to=j, centre=(cen / scale).tolist(), width_m=round(float(sh.sum() * CELL / 2), 2)))
        own = np.nonzero(camroom == i)[0]; extra = []
        for dd in doors:                                       # cameras of other rooms near this doorway, looking into it
            dc = np.array(dd['centre']) * scale; v = dc - c2; dist2 = np.linalg.norm(v, axis=1)
            look = ((v / (dist2[:, None] + 1e-9)) * fwd2).sum(1) > 0.5
            extra += np.nonzero((camroom == dd['to']) & (dist2 < 2.0) & look)[0].tolist()
        rooms.append(dict(id=i, area_m2=round(float(m.sum() * CELL ** 2), 1), cameras=own.tolist(), extra=sorted(set(extra)),
                          outline=poly.tolist(), doors=doors, first_frame=names[own.min()] if len(own) else None))
    order = sorted(range(len(rooms)), key=lambda i: min(rooms[i]['cameras'] or [1 << 30]))      # room 0 = where the walk starts
    meta = dict(metres_per_unit=1 / scale, up=up.tolist(), e1=e1.tolist(), e2=e2.tolist(), origin_m=lo.tolist(), cell_m=CELL,
                floor=float(floor), ceiling=float(ceil), cameras=len(C), points=len(P))
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
    names, C, R, P = load(sp); meta, rooms, order, (reg, wallm, c2, lo) = split(names, C, R, P)
    json.dump(dict(meta, order=order, rooms=rooms), open(f'{out}/rooms.json', 'w'))
    draw(f'{out}/rooms.png', reg, wallm, c2, lo, rooms)
    print(len(rooms), 'rooms:', [(r['id'], r['area_m2'], len(r['cameras']), len(r['extra']), [d['to'] for d in r['doors']]) for r in rooms], flush=True)
