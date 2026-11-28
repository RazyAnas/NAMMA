"""Shape lock: where is the phone in a scanned room, from the room's SHAPE alone. Reference implementation and the map writer;
the browser runs the same steps in ar_geo.js (kept in step by tests/v3/test_ar_shape.py). No server, no GPU, no image matching.

An AR phone knows which way is down, how far it has moved in real metres, and (depth API) how far away the surfaces in front
of it are. Collected while the person looks around, those depth points are a rough 3D sketch of the part of the room they
have looked at, in the phone's own coordinates. The scan gives the same room as a mesh. Four numbers are unknown: one turn
about the vertical, a shift on the floor, and the scale between the scan's guessed metres and real ones. That is the 2D scan
matching robots do with a laser scanner and a floor plan, small enough for a phone's CPU:
  map    (ShapeMap, written as <tour>/ar_shape.bin) three height bands of the mesh as "how close is a surface here" fields on
         a 5 cm grid, a free-space field (where a person can be), and the floor-to-ceiling height.
  live   (prepare) the depth points in the same bands, the phone's own path, the floor and, when seen, the ceiling: floor to
         ceiling measures the scale directly, so it is not searched.
  match  (locate) for every turn (5 deg) and scale one FFT cross-correlation per band gives the best shift; the best few
         different places are polished; the phone's path must lie in free space.
One match is not to be believed (measured: no score tells a right one from a wrong one); a position is believed when three
matches in a row, a second apart, agree within 15 cm. Measured on the switch room's real mesh with simulated depth (5 % noise,
30 % of pixels missing, 5 % wrong; scripts/ar/exp_geo.py): believed after 4-6 s, right in 31 of 36 visits (12 of 12 when
standing and looking round), 7-11 cm off at the room's corners. Not measured: real ARCore depth, which is weak on blank walls.

File ar_shape.bin: float32 x0, z0 (grid corner), cell, floor-to-ceiling height (0: unknown); uint32 N, bands; float32 lo, hi
per band; float32 second level above head height (0: none); then per band N*N uint8 (field * 255, row = z), then N*N
uint8 free space (1 = free)."""
import struct
import numpy as np
from scipy import ndimage

BANDS = ((0.25, 0.9), (0.9, 1.6), (1.6, 2.3)); CELL = 0.05; N = 256; SIG = 0.10; COARSE = 2


def ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def peaks(y, lo, hi, least=0.0, k=2):
    """The heights between lo and hi where most points are (floor, ceiling, a level of trays or shelves): up to k of them, at
    least 20 cm apart, strongest first; none with fewer than `least` of all points."""
    h, e = np.histogram(y, bins=np.arange(lo, hi + 0.04, 0.04)); out = []
    for i in np.argsort(-h, kind='stable'):
        if h[i] < max(20, least * len(y)) or len(out) == k: break
        m = (y > e[i] - 0.04) & (y < e[i + 1] + 0.04); v = float(y[m].mean())
        if all(abs(v - o) >= 0.2 for o in out): out.append(v)
    return out


def peak(y, lo, hi, least=0.0):
    p = peaks(y, lo, hi, least, 1); return p[0] if p else None


def scale_choices(live, scan, default):
    """Scales worth trying. A level seen above head height by the phone (real metres) over a level of the scan (its guessed
    metres) is the scale, if they are the same level: the ceiling, or the cable trays under it. Which is which is not known
    (from below, trays hide the ceiling), so every pairing is tried, each with 2.5 % either side; the match decides."""
    c = sorted(h / H for h in live for H in scan if H and 0.8 < h / H < 1.25); keep = []
    for v in c:
        if not keep or v / keep[-1] > 1.02: keep.append(v)
    return tuple(round(v * f, 4) for v in keep for f in (0.975, 1.0, 1.025)) if keep else tuple(default)


class ShapeMap:
    """The scan side: per height band a field 'how well does a surface point fit here' (1 on a surface, falling off over
    SIG metres), and a free-space field for the phone's own path (1 where a person can be, -3 inside things or outside)."""
    def __init__(self, P):
        self.g0 = (P[:, [0, 2]].min(0) + P[:, [0, 2]].max(0)) / 2 - N * CELL / 2                # grid corner (x, z); the room in the middle
        ij = lambda p: np.clip(((p[:, [0, 2]] - self.g0) / CELL).astype(int), 0, N - 1); self.L = []
        for a, b in BANDS:
            occ = np.zeros((N, N), bool); k = ij(P[(P[:, 1] > a) & (P[:, 1] < b)]); occ[k[:, 1], k[:, 0]] = True
            self.L.append(np.exp(-0.5 * (ndimage.distance_transform_edt(~occ) * CELL / SIG) ** 2).astype(np.float32))
        body = np.zeros((N, N), bool); k = ij(P[(P[:, 1] > 0.25) & (P[:, 1] < 1.7)]); body[k[:, 1], k[:, 0]] = True
        seen = np.zeros((N, N), bool); k = ij(P); seen[k[:, 1], k[:, 0]] = True
        inside = ndimage.binary_fill_holes(ndimage.binary_closing(seen, iterations=6)) & ~ndimage.binary_dilation(body, iterations=2)
        self.free = np.where(inside, 1.0, -3.0).astype(np.float32)
        # coarse copies for the search, 10 cm cells: the best of each 2 x 2 block (a thin wall must not vanish), free space the mean
        pool = lambda x, f: f(x.reshape(N // COARSE, COARSE, N // COARSE, COARSE), axis=(1, 3))
        self.C = [pool(x, np.max) for x in self.L] + [pool(self.free, np.mean)]; self.F = [np.fft.rfft2(x) for x in self.C]
        self.Hs = peaks(P[:, 1], 1.8, 3.8); self.H = self.Hs[0] if self.Hs else None          # levels above head height: ceiling, trays

    def write(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<4f2I', self.g0[0], self.g0[1], CELL, self.H or 0.0, N, len(BANDS)) + np.array(BANDS, '<f4').tobytes())
            f.write(struct.pack('<f', self.Hs[1] if len(self.Hs) > 1 else 0.0))
            for x in self.L: f.write(np.round(x * 255).astype(np.uint8).tobytes())
            f.write((self.free > 0).astype(np.uint8).tobytes())

    def look(self, field, xz):                                                               # bilinear lookup, metres -> field value
        f = (xz - self.g0) / CELL - 0.5; i = np.floor(f).astype(int); w = f - i; ok = np.all((i >= 0) & (i < N - 1), 1); i = np.clip(i, 0, N - 2)
        v = (field[i[:, 1], i[:, 0]] * (1 - w[:, 0]) + field[i[:, 1], i[:, 0] + 1] * w[:, 0]) * (1 - w[:, 1]) + \
            (field[i[:, 1] + 1, i[:, 0]] * (1 - w[:, 0]) + field[i[:, 1] + 1, i[:, 0] + 1] * w[:, 0]) * w[:, 1]
        return np.where(ok, v, 0.0)


def prepare(pts, path):
    """The phone side: depth points and the phone's path in its own (AR) coordinates -> floor height, points thinned to one
    per 5 cm cell. The floor is the low end of what was seen (the phone sees the floor almost always when held upright)."""
    low = np.percentile(pts[:, 1], 25); floor = peak(pts[:, 1], low - 1.5, low + 0.04)       # the lowest big level, not a stray low point
    if floor is None: floor = float(np.percentile(pts[:, 1], 2.0))
    up = peaks(pts[:, 1], floor + 1.8, floor + 3.8, least=0.02)                             # ceiling / trays, when the phone has looked up
    _, first = np.unique(np.floor(pts / CELL).astype(np.int64), axis=0, return_index=True); pts = pts[first]
    c = np.r_[np.median(pts[:, 0]), floor, np.median(pts[:, 2])]
    return dict(p=pts - c, path=path - c, c=c, heights=[u - floor for u in up])


def score(M, L, yaw, s, t):
    """How well the phone's sketch fits the scan when turned by yaw, scaled by 1/s and shifted by t (tour x, z)."""
    R = ry(yaw); q = L['p'] @ R.T / s; tot, n = 0.0, 0
    for (a, b), fld in zip(BANDS, M.L):
        m = (q[:, 1] > a) & (q[:, 1] < b)
        if m.any(): tot += M.look(fld, q[m][:, [0, 2]] + t).sum(); n += m.sum()
    pq = L['path'] @ R.T / s
    return tot / max(n, 1) + 0.5 * float(M.look(M.free, pq[:, [0, 2]] + t).mean())


def locate(M, L, scales=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15), step=5, keep=6):
    """The join tour <- phone: (yaw, s, t, score, second-best score). Coarse search by FFT, then a local polish."""
    n = N // COARSE; c = COARSE * CELL; cands = []
    scales = scale_choices(L.get('heights') or [], M.Hs, scales)                            # a level above head height seen: scale measured, not searched
    for s in scales:
        for yaw in np.radians(np.arange(0, 360, step)):
            R = ry(yaw); q = L['p'] @ R.T / s; pq = L['path'] @ R.T / s; corr = 0.0; cnt = 0
            for k, band in enumerate(BANDS + (None,)):
                src = pq if band is None else q[(q[:, 1] > band[0]) & (q[:, 1] < band[1])]
                if not len(src): continue
                ij = np.round(src[:, [0, 2]] / c).astype(int) + n // 2; ok = np.all((ij >= 0) & (ij < n), 1); A = np.zeros((n, n), np.float32)
                np.add.at(A, (ij[ok, 1], ij[ok, 0]), 1.0)
                w = (0.5 / max(len(src), 1)) if band is None else 1.0; cnt += 0 if band is None else len(src)
                corr = corr + w * np.fft.irfft2(M.F[k] * np.conj(np.fft.rfft2(A)), s=(n, n))
            corr = corr / max(cnt, 1) if np.ndim(corr) else corr
            dz, dx = np.unravel_index(int(np.argmax(corr)), corr.shape); d = np.array([dx, dz]); d = np.where(d >= n // 2, d - n, d)
            cands.append((float(corr.max()), yaw, s, M.g0 + (n // 2 + d) * c))
    cands.sort(key=lambda x: -x[0]); out = []; picked = []
    apart = lambda a, b: np.linalg.norm(a[3] - b[3]) > 0.4 or abs(np.angle(np.exp(1j * (a[1] - b[1])))) > np.radians(15)
    for cnd in cands:                                                                        # the best few places that are really different
        if all(apart(cnd, q) for q in picked): picked.append(cnd)
        if len(picked) == keep: break
    for _, yaw, s, t in picked:                                                              # polish: shrinking steps around each candidate
        best = score(M, L, yaw, s, t)
        for dy, ds, dt in ((3.0, 0.02, 0.10), (1.5, 0.01, 0.05), (0.75, 0.005, 0.025), (0.4, 0.0025, 0.0125), (0.2, 0.001, 0.006)):
            for _ in range(4):
                moved = False
                for d in ((np.radians(dy), 0, 0, 0), (-np.radians(dy), 0, 0, 0), (0, ds, 0, 0), (0, -ds, 0, 0), (0, 0, dt, 0), (0, 0, -dt, 0), (0, 0, 0, dt), (0, 0, 0, -dt)):
                    v = score(M, L, yaw + d[0], s + d[1], t + np.array(d[2:]))
                    if v > best: best, yaw, s, t, moved = v, yaw + d[0], s + d[1], t + np.array(d[2:]), True
                if not moved: break
        out.append((best, yaw, s, t))
    out.sort(key=lambda x: -x[0]); b = out[0]
    other = [o[0] for o in out[1:] if apart(o, b)]; second = max(other) if other else 0.0
    return dict(yaw=b[1], s=b[2], t=b[3], score=b[0], second=second, margin=(b[0] - second) / max(b[0], 1e-6))   # margin: how much better than the next place
