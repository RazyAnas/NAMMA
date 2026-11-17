"""Read a published tour's splat (scene.compressed.ply, the PlayCanvas packed format splat-transform writes) back into room
coordinates: metres, y up, floor at y = 0, the frame of walls.json / nav.json / pois.json / settings.json.

    read_cply(path)  -> dict(xyz (N,3) float64, opacity (N,), size (N,) largest axis in metres, rgb (N,3) 0-1)
    cloud(path, ...) -> (M,3) float32 point cloud for localisation (solid, small splats; one point per grid cell)
    python tour_cloud.py <scene.compressed.ply> [out.npy]

The scene file holds F(Rp + t) * s with F = diag(-1, -1, 1) (make_tour.py header: the viewer turns every splat 180 deg about
z), everything else in the tour is the un-flipped Rp + t, so x and y are negated here. Only numpy is needed, no token: the
file is in the public tour folder. Positions are packed to 11/10/11 bits inside each 256-splat chunk's own box, which is
far finer than a splat is wide."""
import sys
import numpy as np


def _unpack(p, bits):
    out, shift = [], sum(bits)
    for b in bits:
        shift -= b; out.append(((p >> shift) & ((1 << b) - 1)) / float((1 << b) - 1))
    return np.stack(out, 1)


def read_cply(path):
    raw = path if isinstance(path, (bytes, bytearray)) else open(path, "rb").read(); e = raw.index(b'end_header\n') + 11; head = raw[:e].decode('ascii', 'replace').split('\n')
    assert any(l.startswith('property uint packed_position') for l in head), 'not a compressed splat ply (scene.compressed.ply)'
    count, props, cur = {}, {}, None
    for l in head:
        w = l.split()
        if w[:1] == ['element']: cur = w[1]; count[cur] = int(w[2]); props[cur] = []
        elif w[:1] == ['property'] and cur: props[cur].append(w[-1])
    nc, nv, k = count['chunk'], count['vertex'], len(props['chunk'])
    ch = np.frombuffer(raw, np.float32, nc * k, e).reshape(nc, k).astype(np.float64); col = {n: i for i, n in enumerate(props['chunk'])}
    v = np.frombuffer(raw, np.uint32, nv * 4, e + nc * k * 4).reshape(nv, 4); vc = {n: i for i, n in enumerate(props['vertex'])}
    ci = np.arange(nv) // 256
    box = lambda a, b: (ch[ci][:, [col[f'{a}_x'], col[f'{a}_y'], col[f'{a}_z']]], ch[ci][:, [col[f'{b}_x'], col[f'{b}_y'], col[f'{b}_z']]])
    lo, hi = box('min', 'max'); xyz = lo + _unpack(v[:, vc['packed_position']], (11, 10, 11)) * (hi - lo)
    lo, hi = box('min_scale', 'max_scale'); size = np.exp(lo + _unpack(v[:, vc['packed_scale']], (11, 10, 11)) * (hi - lo)).max(1)
    c = _unpack(v[:, vc['packed_color']], (8, 8, 8, 8)); rgb = c[:, :3]
    if 'min_r' in col:
        lo, hi = ch[ci][:, [col['min_r'], col['min_g'], col['min_b']]], ch[ci][:, [col['max_r'], col['max_g'], col['max_b']]]
        rgb = np.clip(0.5 + 0.28209479 * (lo + rgb * (hi - lo)), 0, 1)                       # stored as the SH dc term
    return dict(xyz=xyz * np.array([-1.0, -1.0, 1.0]), opacity=c[:, 3], size=size, rgb=rgb)


def cloud(path, grid=0.02, opacity=0.3, size=0.25, box=None, max_points=None, seed=0):
    """Point cloud of the room's surfaces: splats that are solid (opacity) and small (size, metres), inside `box`
    ([x0, y0, z0, x1, y1, z1], room metres, e.g. walls.json 'box') if given, one point (the mean) per `grid` cell."""
    s = read_cply(path); m = (s['opacity'] >= opacity) & (s['size'] <= size)
    if box is not None:
        b = np.asarray(box, float); m &= np.all((s['xyz'] >= b[:3] - 0.3) & (s['xyz'] <= b[3:] + 0.3), 1)
    p = s['xyz'][m]
    if grid:
        key = np.floor(p / grid).astype(np.int64); _, inv, n = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1); p = np.stack([np.bincount(inv, p[:, j]) / n for j in range(3)], 1)
    if max_points and len(p) > max_points: p = p[np.random.default_rng(seed).choice(len(p), max_points, replace=False)]
    return p.astype(np.float32)


if __name__ == '__main__':
    pts = cloud(sys.argv[1]); print(len(pts), 'points; min', pts.min(0).round(2), 'max', pts.max(0).round(2))
    if len(sys.argv) > 2: np.save(sys.argv[2], pts)
