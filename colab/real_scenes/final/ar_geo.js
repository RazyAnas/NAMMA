// Shape lock in the browser: where is the phone in the scanned room, from the room's shape alone (the phone's depth points
// against the scan's mesh). No server, no GPU, no image matching. This is ar_shape.py step for step (that file explains the
// idea and what was measured); tests/v3/test_ar_shape.py runs both on the same made-up visit and compares the answers.
// The AR page runs this file in a Web Worker (one match takes about a second of CPU), Node tests import it.
//
//   readShape(buf)            the tour's ar_shape.bin -> the map (fields per height band, free space, their FFTs)
//   prepare(pts, path, cap)   depth points and the phone's path, flat x y z in the phone's AR world -> the live sketch
//   locate(M, L)              -> {yaw, s, tx, tz, score, margin}: p_tour = Ry(yaw) (p_phone - L.c) / s + (tx, ., tz)
//   toJoin(J, L)              the same as ar_core.js's join {yaw, s, t}: p_phone = s Ry(yaw) p_tour + t

const COARSE = 2;

function makeFft(n) {                                                                        // n x n complex FFT, in place (n a power of 2)
  const bits = Math.log2(n), rev = new Uint16Array(n), co = new Float64Array(n / 2), si = new Float64Array(n / 2), tr = new Float64Array(n), ti = new Float64Array(n);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  for (let i = 0; i < n / 2; i++) { co[i] = Math.cos(2 * Math.PI * i / n); si[i] = Math.sin(2 * Math.PI * i / n); }
  const line = (re, im, off, stride, inv) => {
    for (let i = 0; i < n; i++) { tr[rev[i]] = re[off + i * stride]; ti[rev[i]] = im[off + i * stride]; }
    for (let len = 2; len <= n; len <<= 1) { const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) for (let j = 0; j < half; j++) { const wr = co[j * step], wi = inv ? si[j * step] : -si[j * step], a = i + j, b = a + half;
        const xr = tr[b] * wr - ti[b] * wi, xi = tr[b] * wi + ti[b] * wr; tr[b] = tr[a] - xr; ti[b] = ti[a] - xi; tr[a] += xr; ti[a] += xi; } }
    for (let i = 0; i < n; i++) { re[off + i * stride] = tr[i]; im[off + i * stride] = ti[i]; }
  };
  return (re, im, inv) => { for (let r = 0; r < n; r++) line(re, im, r * n, 1, inv); for (let c = 0; c < n; c++) line(re, im, c, n, inv);
    if (inv) { const k = 1 / (n * n); for (let i = 0; i < n * n; i++) { re[i] *= k; im[i] *= k; } } };
}

export function readShape(buf) {
  const dv = new DataView(buf), x0 = dv.getFloat32(0, true), z0 = dv.getFloat32(4, true), cell = dv.getFloat32(8, true), H = dv.getFloat32(12, true), N = dv.getUint32(16, true), nb = dv.getUint32(20, true);
  const bands = []; for (let k = 0; k < nb; k++) bands.push([dv.getFloat32(24 + 8 * k, true), dv.getFloat32(28 + 8 * k, true)]);
  const H2 = dv.getFloat32(24 + 8 * nb, true), Hs = [H, H2].filter(v => v > 0), u8 = new Uint8Array(buf); let off = 28 + 8 * nb; const L = [];
  for (let k = 0; k < nb; k++) { const f = new Float32Array(N * N); for (let i = 0; i < N * N; i++) f[i] = u8[off + i] / 255; L.push(f); off += N * N; }
  const free = new Float32Array(N * N); for (let i = 0; i < N * N; i++) free[i] = u8[off + i] ? 1 : -3;
  // coarse copies for the search: the best of each block for surfaces (a thin wall must not vanish), the mean for free space
  const n = N / COARSE, fft = makeFft(n), F = [];
  [...L, free].forEach((f, k) => { const re = new Float64Array(n * n), im = new Float64Array(n * n);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { let m = k < nb ? -Infinity : 0;
      for (let a = 0; a < COARSE; a++) for (let b = 0; b < COARSE; b++) { const v = f[(r * COARSE + a) * N + c * COARSE + b]; m = k < nb ? Math.max(m, v) : m + v / (COARSE * COARSE); }
      re[r * n + c] = m; }
    fft(re, im, false); F.push({ re, im }); });
  return { x0, z0, cell, H, Hs, N, bands, L, free, n, c: COARSE * cell, F, fft };
}

function peaks(y, lo, hi, least = 0, k = 2) {                                                // heights where most points are, strongest first
  const nb = Math.max(1, Math.ceil((hi + 0.04 - lo) / 0.04 - 1e-9) - 1), h = new Int32Array(nb), out = [];
  for (let i = 0; i < y.length; i++) { const b = Math.floor((y[i] - lo) / 0.04); if (b >= 0 && b < nb) h[b]++; }
  const order = Array.from(h.keys()).sort((a, b) => h[b] - h[a] || a - b);
  for (const b of order) { if (h[b] < Math.max(20, least * y.length) || out.length === k) break;
    const a0 = lo + b * 0.04 - 0.04, a1 = lo + (b + 1) * 0.04 + 0.04; let s = 0, m = 0;
    for (let i = 0; i < y.length; i++) if (y[i] > a0 && y[i] < a1) { s += y[i]; m++; }
    if (m && out.every(o => Math.abs(s / m - o) >= 0.2)) out.push(s / m); }
  return out;
}
const peak = (y, lo, hi, least = 0) => { const p = peaks(y, lo, hi, least, 1); return p.length ? p[0] : null; };
export function scaleChoices(live, scan, dflt) {                                             // see ar_shape.py scale_choices
  const c = []; for (const h of live) for (const H of scan) if (H && h / H > 0.8 && h / H < 1.25) c.push(h / H);
  c.sort((a, b) => a - b); const keep = []; for (const v of c) if (!keep.length || v / keep[keep.length - 1] > 1.02) keep.push(v);
  return keep.length ? keep.flatMap(v => [0.975, 1, 1.025].map(f => Math.round(v * f * 1e4) / 1e4)) : dflt;
}
const sorted = (v) => Float64Array.from(v).sort(), median = (v) => { const s = sorted(v), n = s.length; return n ? (n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0; };
const pct = (v, p) => { const s = sorted(v), f = (s.length - 1) * p / 100, i = Math.floor(f); return s.length ? s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (f - i) : 0; };

// The phone side. cap: at most this many points are matched (one per 5 cm cell first, then every k-th), for speed.
export function prepare(pts, path, cap = Infinity, cell = 0.05) {
  const n = Math.floor(pts.length / 3); if (n < 50) return null;
  const ys = new Float64Array(n); for (let i = 0; i < n; i++) ys[i] = pts[3 * i + 1];
  const low = pct(ys, 25); let floor = peak(ys, low - 1.5, low + 0.04); if (floor === null) floor = pct(ys, 2);
  const up = peaks(ys, floor + 1.8, floor + 3.8, 0.02), seen = new Set(), keep = [];
  for (let i = 0; i < n; i++) { const k = ((Math.floor(pts[3 * i] / cell) + 4096) * 8192 + Math.floor(pts[3 * i + 1] / cell) + 4096) * 8192 + Math.floor(pts[3 * i + 2] / cell) + 4096;
    if (!seen.has(k)) { seen.add(k); keep.push(i); } }
  const cx = median(keep.map(i => pts[3 * i])), cz = median(keep.map(i => pts[3 * i + 2])), step = Math.max(1, Math.ceil(keep.length / cap)), p = [];
  for (let j = 0; j < keep.length; j += step) { const i = keep[j]; p.push(pts[3 * i] - cx, pts[3 * i + 1] - floor, pts[3 * i + 2] - cz); }
  const q = new Float64Array(path.length); for (let i = 0; i < path.length; i += 3) { q[i] = path[i] - cx; q[i + 1] = path[i + 1] - floor; q[i + 2] = path[i + 2] - cz; }
  return { p: Float64Array.from(p), path: q, c: [cx, floor, cz], heights: up.map(u => u - floor) };
}

function look(M, f, x, z) {                                                                  // bilinear lookup, metres -> field value (0 off the grid)
  const fx = (x - M.x0) / M.cell - 0.5, fz = (z - M.z0) / M.cell - 0.5, ix = Math.floor(fx), iz = Math.floor(fz), N = M.N;
  if (ix < 0 || iz < 0 || ix >= N - 1 || iz >= N - 1) return 0;
  const wx = fx - ix, wz = fz - iz, o = iz * N + ix;
  return (f[o] * (1 - wx) + f[o + 1] * wx) * (1 - wz) + (f[o + N] * (1 - wx) + f[o + N + 1] * wx) * wz;
}

export function score(M, L, yaw, s, tx, tz) {
  const co = Math.cos(yaw) / s, si = Math.sin(yaw) / s, p = L.p, B = M.bands; let tot = 0, n = 0;
  for (let i = 0; i < p.length; i += 3) { const y = p[i + 1] / s; let k = -1; for (let b = 0; b < B.length; b++) if (y > B[b][0] && y < B[b][1]) { k = b; break; }
    if (k < 0) continue; tot += look(M, M.L[k], co * p[i] + si * p[i + 2] + tx, -si * p[i] + co * p[i + 2] + tz); n++; }
  let fr = 0; const q = L.path; for (let i = 0; i < q.length; i += 3) fr += look(M, M.free, co * q[i] + si * q[i + 2] + tx, -si * q[i] + co * q[i + 2] + tz);
  return tot / Math.max(n, 1) + 0.5 * fr / Math.max(q.length / 3, 1);
}

export function locate(M, L, o = {}) {
  const n = M.n, c = M.c, nn = n * n, B = M.bands, cands = [], step = o.step || 5, keep = o.keep || 6;
  const scales = scaleChoices(L.heights || [], M.Hs || [], o.scales || [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]);   // a level above head height seen: scale measured
  const are = new Float64Array(nn), aim = new Float64Array(nn), sre = new Float64Array(nn), sim = new Float64Array(nn);
  for (const s of scales) for (let deg = 0; deg < 360; deg += step) {
    const yaw = deg * Math.PI / 180, co = Math.cos(yaw) / s, si = Math.sin(yaw) / s; sre.fill(0); sim.fill(0); let cnt = 0, any = false;
    for (let k = 0; k <= B.length; k++) { const src = k < B.length ? L.p : L.path; are.fill(0); aim.fill(0); let m = 0, tot = 0;
      for (let i = 0; i < src.length; i += 3) { if (k < B.length) { const y = src[i + 1] / s; if (!(y > B[k][0] && y < B[k][1])) continue; }
        tot++; const ix = Math.round((co * src[i] + si * src[i + 2]) / c) + n / 2, iz = Math.round((-si * src[i] + co * src[i + 2]) / c) + n / 2;
        if (ix >= 0 && iz >= 0 && ix < n && iz < n) { are[iz * n + ix] += 1; m++; } }
      if (!tot) continue; any = true; if (k < B.length) cnt += tot; const w = k < B.length ? 1 : 0.5 / tot, F = M.F[k];
      if (!m) continue; M.fft(are, aim, false);
      for (let i = 0; i < nn; i++) { sre[i] += w * (F.re[i] * are[i] + F.im[i] * aim[i]); sim[i] += w * (F.im[i] * are[i] - F.re[i] * aim[i]); } }
    if (!any) continue; M.fft(sre, sim, true); let b = 0; for (let i = 1; i < nn; i++) if (sre[i] > sre[b]) b = i;
    let dz = Math.floor(b / n), dx = b % n; if (dx >= n / 2) dx -= n; if (dz >= n / 2) dz -= n;
    cands.push([sre[b] / Math.max(cnt, 1), yaw, s, M.x0 + (n / 2 + dx) * c, M.z0 + (n / 2 + dz) * c]);
  }
  if (!cands.length) return null;
  cands.sort((a, b) => b[0] - a[0]);
  const apart = (a, b) => Math.hypot(a[3] - b[3], a[4] - b[4]) > 0.4 || Math.abs(Math.atan2(Math.sin(a[1] - b[1]), Math.cos(a[1] - b[1]))) > 15 * Math.PI / 180, picked = [];
  for (const cd of cands) { if (picked.every(q => apart(cd, q))) picked.push(cd); if (picked.length === keep) break; }
  const out = picked.map(([, yaw, s, tx, tz]) => { let best = score(M, L, yaw, s, tx, tz);       // polish: shrinking steps around each place
    for (const [dy, ds, dt] of [[3, 0.02, 0.10], [1.5, 0.01, 0.05], [0.75, 0.005, 0.025], [0.4, 0.0025, 0.0125], [0.2, 0.001, 0.006]]) { const r = dy * Math.PI / 180;
      for (let it = 0; it < 4; it++) { let moved = false;
        for (const d of [[r, 0, 0, 0], [-r, 0, 0, 0], [0, ds, 0, 0], [0, -ds, 0, 0], [0, 0, dt, 0], [0, 0, -dt, 0], [0, 0, 0, dt], [0, 0, 0, -dt]]) {
          const v = score(M, L, yaw + d[0], s + d[1], tx + d[2], tz + d[3]); if (v > best) { best = v; yaw += d[0]; s += d[1]; tx += d[2]; tz += d[3]; moved = true; } }
        if (!moved) break; } }
    return [best, yaw, s, tx, tz]; }).sort((a, b) => b[0] - a[0]);
  const b = out[0], other = out.slice(1).filter(q => apart(q, b)).map(q => q[0]), second = other.length ? Math.max(...other) : 0;
  return { yaw: b[1], s: b[2], tx: b[3], tz: b[4], score: b[0], second, margin: (b[0] - second) / Math.max(b[0], 1e-6) };
}

export function toJoin(J, L) {                                                               // p_phone = s Ry(-yaw) (p_tour - T) + c
  const a = -J.yaw, s = J.s, x = Math.cos(a) * J.tx + Math.sin(a) * J.tz, z = -Math.sin(a) * J.tx + Math.cos(a) * J.tz;
  return { yaw: a, s, t: [L.c[0] - s * x, L.c[1], L.c[2] - s * z] };
}
