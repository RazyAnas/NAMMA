"""Walkable grid of a tour, for routes to a place (make_tour.navgrid -> <tour>/nav.json; A* runs in the browser, poi_core.js).
  obstacles  every triangle of collision.glb (room box + inner walls + furniture mesh) between 0.15 m and 1.7 m above the floor
             under it, sampled onto a 10 cm grid over the room box: what a standing person would bump into. The floor itself,
             what can be stepped on (up to 0.15 m) and anything above head height never block.
  body       obstacles grown by a 0.25 m body radius.
  walked     the filmed camera path is walkable by definition (someone stood there); its cells are forced free, one cell
             wider, so the path stays one 4-connected chain even where the smoothed track clips a table corner.
  reachable  only the floor joined to the filmed path is kept: sealed pockets, the far side of a wall never filmed and the
             space outside the room box cannot be routed into.
The collision file itself is never changed. Grid: row-major uint8 (1 = walkable), index = iz * w + ix, cell centre at
(x0 + (ix + 0.5) * cell, z0 + (iz + 0.5) * cell), room metres (the same frame as settings.json / overlay.json).
Usage: python navgrid.py <tour dir>   (writes nav.json; needs collision.glb and plan.json or tour.json)"""
import base64, json, os, sys, zlib
import numpy as np

CELL, RADIUS, Y_LO, Y_HI, POI_CELL = 0.10, 0.25, 0.15, 1.70, 0.5
DY, EYE = 0.05, 1.35                                    # height bin; make_tour.plan's phone height above the floor


def morton3(ix, iy, iz):
    """30-bit Morton (Z-order) key of a 3D cell, 10 bits an axis, offset 512 so cells from -256 m to +256 m at 0.5 m fit.
    The same function is in poi_core.js (the browser's spatial index); tests check both give the same keys."""
    def spread(v):
        v = (int(v) + 512) & 0x3ff
        v = (v | (v << 16)) & 0x30000ff; v = (v | (v << 8)) & 0x300f00f; v = (v | (v << 4)) & 0x30c30c3; v = (v | (v << 2)) & 0x9249249
        return v
    return spread(ix) | (spread(iy) << 1) | (spread(iz) << 2)


def poi_cell(pos, cell=POI_CELL):
    """Morton key of the 0.5 m cell that holds a point in room metres (pois.json 'cell')."""
    return morton3(*(int(np.floor(c / cell)) for c in pos))


def sample_triangles(tri, step):
    """Points on every triangle no further apart than `step` (corners, edges and inside): tri (n, 3, 3) -> (m, 3).
    Triangles are grouped by how finely they must be cut, so 150k small furniture faces and a 12 m wall both stay cheap."""
    e = np.maximum.reduce([np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1) for i in range(3)])
    n = np.clip(np.ceil(e / step).astype(int), 1, 4096); out = []
    for k in np.unique(n):
        t = tri[n == k]; i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij'); m = i + j <= k
        a, b = i[m] / k, j[m] / k; w = np.stack([1 - a - b, a, b], 1)                       # barycentric weights (p, 3)
        out.append(np.einsum('pc,ncd->npd', w, t).reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


def camera_path(tour):
    """The filmed walk in room metres, (n, 3): the guided-walk keyframes of settings.json (the smoothed camera track) and the
    start view. The camera height matters: it says which floor level the person stood on (a house with steps or stairs)."""
    pts = []
    try:
        s = json.load(open(f'{tour}/settings.json'))
        for tr in s.get('animTracks', []): pts.append(np.array(tr['keyframes']['values']['position'], float).reshape(-1, 3))
        for c in s.get('cameras', []):
            p = c.get('initial', {}).get('position')
            if p: pts.append(np.array([p], float))
    except (OSError, KeyError, ValueError): pass
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def dense(path, step):
    """The polyline through `path` sampled every `step` metres (consecutive points only; a jump longer than 3 m is a new
    place, e.g. the start view after the walk, and is not joined)."""
    out = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        d = float(np.linalg.norm((b - a)[[0, 2]]))
        if d > 3.0: out.append(b[None]); continue
        k = max(1, int(np.ceil(d / step))); out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.concatenate(out)


def build(tri, box, path, cell=CELL, radius=RADIUS, y_lo=Y_LO, y_hi=Y_HI, dy=DY, eye=EYE):
    """tri (n, 3, 3) collision triangles, box [x0, y0, z0, x1, y1, z1], path (m, 3) filmed camera positions, room metres ->
    dict(x0, z0, cell, w, h, grid uint8 (h, w) 1 = walkable, floor float32 (h, w) metres (nan where not walkable)).
    The floor is not y = 0 everywhere (the 1 Nov house: rooms 0.4 m up, an upper floor), so "0.15 m to 1.7 m above the floor"
    is measured from the surface under each cell, in a column of 5 cm height bins:
      surface   a bin with collision geometry in it and none more than 0.15 m and up to 1.7 m above it, in the cell or within
                the body radius of it (so a wall or a table edge 20 cm away blocks);
      walk      from the filmed cells outwards, to a neighbour whose surface is at most 0.15 m up or down (what does not block
                can be stepped on): floors, ramps and low sills join; a sofa seat, a table top, a 0.18 m stair riser do not.
                Stairs are therefore walkable only where they were filmed (below)."""
    from collections import deque
    from scipy import ndimage
    x0, z0, x1, z1 = box[0] - 0.5, box[2] - 0.5, box[3] + 0.5, box[5] + 0.5                 # half a metre around the room box
    w, h = int(np.ceil((x1 - x0) / cell)), int(np.ceil((z1 - z0) / cell))
    assert 4 <= w <= 2048 and 4 <= h <= 2048, f'room box gives a {w} x {h} grid'
    ylo = min(-0.3, float(box[1]) - 0.2); ny = int(np.clip(np.ceil((max(float(box[4]), 2.5) + 0.5 - ylo) / dy), 8, 400))
    occ = np.zeros((h, w, ny), bool)
    if len(tri):
        p = sample_triangles(np.asarray(tri, float), cell / 2)
        ix, iz = np.floor((p[:, 0] - x0) / cell).astype(int), np.floor((p[:, 2] - z0) / cell).astype(int)
        iy = np.floor((p[:, 1] - ylo) / dy + 1e-6).astype(int)
        ok = (ix >= 0) & (ix < w) & (iz >= 0) & (iz < h) & (iy >= 0) & (iy < ny); occ[iz[ok], ix[ok], iy[ok]] = True
    sb, b = int(round(y_lo / dy)), int(round(y_hi / dy))                                      # step / top of the band, in bins
    cs = np.concatenate([np.zeros((h, w, 1), np.int32), np.cumsum(occ, 2, dtype=np.int32)], 2)
    k = np.arange(ny); band = (cs[:, :, np.minimum(k + b + 1, ny)] - cs[:, :, np.minimum(k + sb + 1, ny)]) > 0   # something in the way
    r = int(np.ceil(radius / cell)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disc = (np.hypot(xx, yy) * cell <= radius)[:, :, None]
    stand = occ & ~ndimage.binary_dilation(band, structure=disc)                             # body radius: nothing in the way near
    for s_ in range(1, sb + 1): stand[:, :, :-s_] &= ~stand[:, :, s_:]                        # a thick slab: its top face only
    # filmed cells are walkable whatever the mesh says (the 1 Nov house mesh has blobs right on the walk). Their floor level,
    # from two clues: the level of the cell before (a walk goes up and down smoothly) and the camera height, which is eye
    # height above the floor give or take how the phone was held (median over 2 m of walk; +-0.3 m). In this order:
    #  - the surface (at least 0.7 m under the camera) within one step of the cell before;
    #  - else the surface within 0.35 m of eye height under the camera: a platform or stair tread the mesh has no clean
    #    steps to (a sofa seat the smoothed track clips is further off than that and stays an obstacle);
    #  - else the level of the cell before; when even that is 0.6 m off eye height under the camera, the surface nearest
    #    to eye height under it (no surface: that height).
    lev = np.full((h, w), -1, np.int32); forced = np.zeros((h, w), bool); on = np.zeros((h, w), bool); q = deque()   # on: the track itself
    if len(path):
        d = dense(np.asarray(path, float).reshape(-1, 3), cell / 4)
        ix, iz = np.floor((d[:, 0] - x0) / cell).astype(int), np.floor((d[:, 2] - z0) / cell).astype(int)
        kcam = np.round((d[:, 1] - ylo) / dy).astype(int); last = None
        wants = np.round((ndimage.median_filter(d[:, 1], size=min(len(d), 81), mode='nearest') - eye - ylo) / dy).astype(int)
        for cx, cz, kc, want in zip(ix, iz, kcam, wants):
            if not (0 <= cx < w and 0 <= cz < h): continue
            if not on[cz, cx]:                                # (a cell first freed as a neighbour gets its own level here)
                ks = np.nonzero(stand[cz, cx, :max(0, kc - int(0.7 / dy))])[0]; k_ = None
                if len(ks) and last is not None and np.abs(ks - last).min() <= sb: k_ = int(ks[np.argmin(np.abs(ks - last))])
                elif len(ks) and np.abs(ks - want).min() <= int(0.35 / dy): k_ = int(ks[np.argmin(np.abs(ks - want))])
                elif last is not None and abs(last - want) <= int(0.6 / dy): k_ = last
                if k_ is None: k_ = int(ks[np.argmin(np.abs(ks - want))]) if len(ks) else int(np.clip(want, int(-ylo / dy), ny - 1))   # never under y = 0
                lev[cz, cx] = k_; on[cz, cx] = forced[cz, cx] = True; q.append((cx, cz))
            last = int(lev[cz, cx])
            for ex, ez in ((1, 0), (-1, 0), (0, 1), (0, -1)):                                 # + 4 neighbours: one 4-connected chain
                jx, jz = cx + ex, cz + ez
                if 0 <= jx < w and 0 <= jz < h and not forced[jz, jx]: lev[jz, jx] = last; forced[jz, jx] = True; q.append((jx, jz))
    else:                                                    # no walk known (never in a built tour): the biggest floor level
        top = np.where(stand.any(2), stand.argmax(2), -1); vals = top[top >= 0]
        if len(vals):
            k0 = int(np.bincount(vals).argmax()); lab, n = ndimage.label(top == k0)
            if n: big = lab == 1 + int(np.argmax(ndimage.sum(top == k0, lab, range(1, n + 1)))); lev[big] = k0; q.extend((int(x_), int(z_)) for z_, x_ in zip(*np.nonzero(big)))
    while q:                                                 # 4-connected, as the browser's A* (no diagonal through a corner)
        cx, cz = q.popleft(); kc = lev[cz, cx]
        for ex, ez in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jx, jz = cx + ex, cz + ez
            if not (0 <= jx < w and 0 <= jz < h) or lev[jz, jx] >= 0: continue
            ks = np.nonzero(stand[jz, jx, max(0, kc - sb):kc + sb + 1])[0]
            if len(ks): lev[jz, jx] = max(0, kc - sb) + int(ks[np.argmin(np.abs(ks + max(0, kc - sb) - kc))]); q.append((jx, jz))
    grid = lev >= 0; floor = np.where(grid, ylo + (lev + 0.5) * dy, np.nan).astype(np.float32)   # bin centre
    return dict(x0=float(x0), z0=float(z0), cell=float(cell), w=w, h=h, grid=grid.astype(np.uint8), floor=floor,
                walked=int(forced.sum()), stand=stand.any(2))


def cell_of(nav, x, z):
    return int(np.floor((x - nav['x0']) / nav['cell'])), int(np.floor((z - nav['z0']) / nav['cell']))


def snap(nav, x, z, max_m=None):
    """The centre of the walkable cell nearest to (x, z) (a place inside a sofa or on a wall is reached from beside it), or
    None when nothing is walkable (within max_m)."""
    g = nav['grid']; iz, ix = np.nonzero(g)
    if not len(ix): return None
    cx, cz = nav['x0'] + (ix + 0.5) * nav['cell'], nav['z0'] + (iz + 0.5) * nav['cell']; d = np.hypot(cx - x, cz - z); k = int(np.argmin(d))
    if max_m is not None and d[k] > max_m: return None
    return float(cx[k]), float(cz[k])


def reachable(nav, a, b):
    """Whether cell b (ix, iz) can be walked to from cell a: 4-connected flood fill, the same reach as the browser's A*."""
    from scipy import ndimage
    g = nav['grid'].astype(bool); h, w = g.shape
    for ix, iz in (a, b):
        if not (0 <= ix < w and 0 <= iz < h and g[iz, ix]): return False
    lab, _ = ndimage.label(g); return bool(lab[a[1], a[0]] == lab[b[1], b[0]])


def encode(nav):
    """nav.json: the grid, and the floor height under each cell in cm (uint8, 0 = not walkable, else cm above floor_base + 1;
    the route is drawn on it and a place's approach point gets its height), both zlib-compressed and base64-encoded
    (a 20 x 20 m home is 40,000 cells -> about 1 kB)."""
    g = np.ascontiguousarray(nav['grid'], np.uint8); z = lambda a: base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()
    out = dict(version=1, x0=round(nav['x0'], 4), z0=round(nav['z0'], 4), cell=nav['cell'], w=int(nav['w']), h=int(nav['h']),
               radius=RADIUS, band=[Y_LO, Y_HI], walkable=int(g.sum()), grid=z(g))
    if nav.get('floor') is not None and g.any():
        f = nav['floor']; base = float(np.floor(np.nanmin(f) * 100) / 100)
        out.update(floor_base=base, floor=z(np.where(g > 0, np.clip(np.round((f - base) * 20), 0, 254) + 1, 0).astype(np.uint8)))   # 5 cm steps
    return out


def decode(d):
    un = lambda s: np.frombuffer(zlib.decompress(base64.b64decode(s)), np.uint8).reshape(d['h'], d['w'])
    nav = dict(x0=d['x0'], z0=d['z0'], cell=d['cell'], w=d['w'], h=d['h'], grid=un(d['grid']))
    if 'floor' in d: f = un(d['floor']).astype(np.float32); nav['floor'] = np.where(f > 0, d['floor_base'] + (f - 1) / 20, np.nan)
    return nav


def write(tour):
    """<tour>/collision.glb -> <tour>/nav.json. Returns the summary, or None (and removes a stale nav.json) when there is no
    collision file or no room box: the tour then has no routes, everything else works."""
    out = f'{tour}/nav.json'
    if os.path.exists(out): os.remove(out)
    if not os.path.exists(f'{tour}/collision.glb'): return None
    import trimesh
    if os.path.exists(f'{tour}/plan.json'): box = json.load(open(f'{tour}/plan.json'))['box']
    else: rb = json.load(open(f'{tour}/tour.json'))['room_box']; box = list(rb[0]) + list(rb[1])     # a published tour (no plan.json)
    m = trimesh.load(f'{tour}/collision.glb', force='mesh'); nav = build(np.asarray(m.triangles, float), box, camera_path(tour))
    if not nav['grid'].any(): return None
    json.dump(encode(nav), open(out, 'w'))
    return dict(cells=[nav['w'], nav['h']], walkable_m2=round(float(nav['grid'].sum()) * nav['cell'] ** 2, 1), walked_cells=nav['walked'],
                floor_m=[round(float(np.nanmin(nav['floor'])), 2), round(float(np.nanmax(nav['floor'])), 2)], kb=round(os.path.getsize(out) / 1e3, 1))


if __name__ == '__main__':
    print('nav grid:', write(sys.argv[1]), flush=True)
