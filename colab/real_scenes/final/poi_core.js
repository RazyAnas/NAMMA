// Places (points of interest) in a tour: spatial index, walkable grid, route. No DOM in this file: make_tour.py puts it into the
// tour page next to poi_overlay.js (the `export` words are dropped there), and tests/browser/poi_e2e.mjs imports it in Node.
// Coordinates are room metres as in settings.json / overlay.json: x, y (up, floor near 0), z.

// ---- spatial index: a 3D hash grid keyed by Morton (Z-order) code, 0.5 m cells -------------------------------------------------
// 10 bits an axis, offset 512: cells from -256 m to +256 m. The same key is written to pois.json as `cell` (navgrid.morton3).
export const POI_CELL = 0.5;
const spread = (v) => { v = (v + 512) & 0x3ff; v = (v | (v << 16)) & 0x30000ff; v = (v | (v << 8)) & 0x300f00f; v = (v | (v << 4)) & 0x30c30c3; return (v | (v << 2)) & 0x9249249; };
export const morton3 = (ix, iy, iz) => (spread(ix) | (spread(iy) << 1) | (spread(iz) << 2)) >>> 0;
export const cellKey = (p, cell = POI_CELL) => morton3(Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell));

// Items are points ({id, pos}) or boxes ({id, lo, hi}: the object layer). A box is listed in every cell it touches.
export class SpatialIndex {
  constructor(cell = POI_CELL) { this.cell = cell; this.cells = new Map(); this.items = new Map(); }
  _span(it) { const c = this.cell, lo = it.lo || it.pos, hi = it.hi || it.pos; return [0, 1, 2].map(k => [Math.floor(lo[k] / c), Math.floor(hi[k] / c)]); }
  _each(it, fn) { const [X, Y, Z] = this._span(it); for (let x = X[0]; x <= X[1]; x++) for (let y = Y[0]; y <= Y[1]; y++) for (let z = Z[0]; z <= Z[1]; z++) fn(morton3(x, y, z)); }
  insert(it) { this.remove(it.id); this.items.set(it.id, it);
    this._each(it, (k) => { const a = this.cells.get(k); if (a) a.push(it); else this.cells.set(k, [it]); }); return it; }
  remove(id) { const it = this.items.get(id); if (!it) return false; this.items.delete(id);
    this._each(it, (k) => { const a = this.cells.get(k); if (!a) return; const i = a.indexOf(it); if (i >= 0) a.splice(i, 1); if (!a.length) this.cells.delete(k); }); return true; }
  clear() { this.cells.clear(); this.items.clear(); }
  get size() { return this.items.size; }
  // distance from a point to an item: to the point, or to the box surface (0 inside)
  static dist(it, p) { if (it.pos) return Math.hypot(it.pos[0] - p[0], it.pos[1] - p[1], it.pos[2] - p[2]);
    let s = 0; for (let k = 0; k < 3; k++) { const d = Math.max(it.lo[k] - p[k], 0, p[k] - it.hi[k]); s += d * d; } return Math.sqrt(s); }
  // every item within r metres of p, nearest first: only the cells the sphere touches are read
  within(p, r, filter) { const c = this.cell, seen = new Set(), out = [];
    const a = [0, 1, 2].map(k => Math.floor((p[k] - r) / c)), b = [0, 1, 2].map(k => Math.floor((p[k] + r) / c));
    for (let x = a[0]; x <= b[0]; x++) for (let y = a[1]; y <= b[1]; y++) for (let z = a[2]; z <= b[2]; z++) {
      const l = this.cells.get(morton3(x, y, z)); if (!l) continue;
      for (const it of l) { if (seen.has(it.id)) continue; seen.add(it.id); if (filter && !filter(it)) continue;
        const d = SpatialIndex.dist(it, p); if (d <= r) out.push({ item: it, d }); } }
    return out.sort((u, v) => u.d - v.d); }
  // the nearest item: rings of cells outwards, stopping once no closer cell is left
  nearest(p, maxDist = Infinity, filter) { if (!this.items.size) return null;
    const c = this.cell; let r = c;
    for (;;) { const h = this.within(p, Math.min(r, maxDist), filter); if (h.length) return h[0];
      if (r >= maxDist || r > 600) return null; r *= 2; } }
  // first item a ray hits: the ray walks the cells in order (3D DDA); a point counts as a ball of `ball` metres
  rayPick(o, d, maxT = 30, ball = 0.25, filter) { const L = Math.hypot(d[0], d[1], d[2]) || 1, u = [d[0] / L, d[1] / L, d[2] / L], c = this.cell;
    const hitT = (it) => { if (it.pos) { const m = [o[0] - it.pos[0], o[1] - it.pos[1], o[2] - it.pos[2]], b = m[0] * u[0] + m[1] * u[1] + m[2] * u[2],
          q = m[0] * m[0] + m[1] * m[1] + m[2] * m[2] - ball * ball, D = b * b - q; if (D < 0) return -1; const t = -b - Math.sqrt(D); return t >= 0 ? t : (q <= 0 ? 0 : -1); }
      let t0 = 0, t1 = maxT; for (let k = 0; k < 3; k++) { const inv = 1 / (u[k] || 1e-12); let a = (it.lo[k] - o[k]) * inv, b = (it.hi[k] - o[k]) * inv;
        if (a > b) [a, b] = [b, a]; t0 = Math.max(t0, a); t1 = Math.min(t1, b); } return t0 <= t1 ? t0 : -1; };
    const g = [0, 1, 2].map(k => Math.floor(o[k] / c)), st = u.map(v => (v > 0 ? 1 : -1)), seen = new Set();
    const tm = [0, 1, 2].map(k => u[k] ? ((g[k] + (u[k] > 0 ? 1 : 0)) * c - o[k]) / u[k] : Infinity), td = u.map(v => (v ? Math.abs(c / v) : Infinity));
    let best = null, t = 0;
    for (let n = 0; n < 4096 && t <= maxT; n++) {
      for (let ex = -1; ex <= 1; ex++) for (let ey = -1; ey <= 1; ey++) for (let ez = -1; ez <= 1; ez++) {      // + neighbours: a ball reaches over a cell border
        const l = this.cells.get(morton3(g[0] + ex, g[1] + ey, g[2] + ez)); if (!l) continue;
        for (const it of l) { if (seen.has(it.id)) continue; seen.add(it.id); if (filter && !filter(it)) continue;
          const h = hitT(it); if (h >= 0 && h <= maxT && (!best || h < best.t)) best = { item: it, t: h }; } }
      const k = tm[0] <= tm[1] ? (tm[0] <= tm[2] ? 0 : 2) : (tm[1] <= tm[2] ? 1 : 2);
      if (best && best.t <= tm[k] - 2 * c) break;                 // nothing in a later cell can be nearer
      t = tm[k]; g[k] += st[k]; tm[k] += td[k]; }
    return best; }
}

// ---- walkable grid (nav.json, navgrid.py) --------------------------------------------------------------------------------------
const b64 = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
const inflate = async (s) => new Uint8Array(await new Response(new Blob([b64(s)]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());

// grid: 1 = walkable, index iz * w + ix; clear: cells to the nearest non-walkable cell (chamfer, capped), for routes that keep
// off walls; floor: metres under each cell (multi-level homes), or null
export function makeNav(j, grid, floorU8) { const { w, h } = j, n = w * h;
  if (grid.length !== n) throw new Error('nav grid size'); const clear = new Float32Array(n), INF = 1e6;
  for (let i = 0; i < n; i++) clear[i] = grid[i] ? INF : 0;
  const at = (x, z) => (x < 0 || z < 0 || x >= w || z >= h ? 0 : clear[z * w + x]);
  for (let z = 0; z < h; z++) for (let x = 0; x < w; x++) { const i = z * w + x; if (!clear[i]) continue;
    clear[i] = Math.min(clear[i], at(x - 1, z) + 1, at(x, z - 1) + 1, at(x - 1, z - 1) + 1.414, at(x + 1, z - 1) + 1.414); }
  for (let z = h - 1; z >= 0; z--) for (let x = w - 1; x >= 0; x--) { const i = z * w + x; if (!clear[i]) continue;
    clear[i] = Math.min(clear[i], at(x + 1, z) + 1, at(x, z + 1) + 1, at(x + 1, z + 1) + 1.414, at(x - 1, z + 1) + 1.414); }
  let floor = null;
  if (floorU8 && floorU8.length === n) { floor = new Float32Array(n); for (let i = 0; i < n; i++) floor[i] = floorU8[i] ? j.floor_base + (floorU8[i] - 1) / 20 : 0; }
  return { x0: j.x0, z0: j.z0, cell: j.cell, w, h, grid, clear, floor }; }
export async function decodeNav(j) { return makeNav(j, await inflate(j.grid), j.floor ? await inflate(j.floor) : null); }

export const cellOf = (nav, x, z) => [Math.floor((x - nav.x0) / nav.cell), Math.floor((z - nav.z0) / nav.cell)];
export const centre = (nav, ix, iz) => [nav.x0 + (ix + 0.5) * nav.cell, nav.z0 + (iz + 0.5) * nav.cell];
export const walkable = (nav, ix, iz) => ix >= 0 && iz >= 0 && ix < nav.w && iz < nav.h && nav.grid[iz * nav.w + ix] === 1;
export const floorAt = (nav, ix, iz) => (nav.floor && walkable(nav, ix, iz) ? nav.floor[iz * nav.w + ix] : 0);

// the walkable cell nearest to (x, z): a place inside a sofa or on a wall is reached from beside it. Rings outwards.
export function snap(nav, x, z, maxM = Infinity) { const [cx, cz] = cellOf(nav, x, z); if (walkable(nav, cx, cz)) return { ix: cx, iz: cz, d: 0 };
  const R = Math.min(Math.ceil(maxM / nav.cell), nav.w + nav.h); let best = null;
  for (let r = 1; r <= R; r++) { if (best && best.d <= (r - 1) * nav.cell) break;
    for (let dz = -r; dz <= r; dz++) for (let dx = -r; dx <= r; dx += (Math.abs(dz) === r ? 1 : 2 * r)) {
      const ix = cx + dx, iz = cz + dz; if (!walkable(nav, ix, iz)) continue; const [px, pz] = centre(nav, ix, iz), d = Math.hypot(px - x, pz - z);
      if (d <= maxM && (!best || d < best.d)) best = { ix, iz, d }; } }
  return best; }

// A*: 8 directions, never diagonally past a corner (both side cells must be walkable), each step dearer near obstacles so the
// route keeps to the middle of a corridor where it can. Returns the cells from start to goal, or null (no way).
const NEAR = 4, NEAR_W = 1.5;                                     // cells of extra clearance wanted (0.4 m); how much it is worth
const dear = (nav, i) => 1 + NEAR_W * Math.max(0, NEAR - nav.clear[i]) / NEAR;
export function astar(nav, s, g) { const { w, h, grid } = nav, n = w * h, S = s[1] * w + s[0], G = g[1] * w + g[0];
  if (!walkable(nav, s[0], s[1]) || !walkable(nav, g[0], g[1])) return null; if (S === G) return [s];
  const gs = new Float32Array(n).fill(Infinity), from = new Int32Array(n).fill(-1), shut = new Uint8Array(n), heap = [], hk = [];
  const push = (i, f) => { let k = heap.length; heap.push(i); hk.push(f);
    while (k > 0) { const p = (k - 1) >> 1; if (hk[p] <= f) break; heap[k] = heap[p]; hk[k] = hk[p]; k = p; } heap[k] = i; hk[k] = f; };
  const pop = () => { const top = heap[0], li = heap.pop(), lf = hk.pop(), m = heap.length; if (m) { let k = 0;
      for (;;) { let c = 2 * k + 1; if (c >= m) break; if (c + 1 < m && hk[c + 1] < hk[c]) c++; if (hk[c] >= lf) break; heap[k] = heap[c]; hk[k] = hk[c]; k = c; }
      heap[k] = li; hk[k] = lf; } return top; };
  const gx = g[0], gz = g[1], H = (i) => { const dx = Math.abs(i % w - gx), dz = Math.abs(((i / w) | 0) - gz); return dx + dz - 0.5858 * Math.min(dx, dz); };
  gs[S] = 0; push(S, H(S));
  while (heap.length) { const c = pop(); if (c === G) break; if (shut[c]) continue; shut[c] = 1; const cx = c % w, cz = (c / w) | 0;
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) { if (!dx && !dz) continue; const x = cx + dx, z = cz + dz;
      if (x < 0 || z < 0 || x >= w || z >= h) continue; const i = z * w + x; if (!grid[i] || shut[i]) continue;
      if (dx && dz && (!grid[cz * w + x] || !grid[z * w + cx])) continue;                        // no cutting a corner
      const t = gs[c] + (dx && dz ? 1.4142 : 1) * 0.5 * (dear(nav, c) + dear(nav, i));
      if (t < gs[i]) { gs[i] = t; from[i] = c; push(i, t + H(i)); } } }
  if (from[G] < 0) return null; const out = []; for (let i = G; i >= 0; i = from[i]) { out.push([i % w, (i / w) | 0]); if (i === S) break; }
  return out.reverse(); }

// straight sight between two cells over walkable cells only that keep `min` cells clear of obstacles (never past a corner)
export function sight(nav, a, b, min = 0) { const dx = b[0] - a[0], dz = b[1] - a[1], n = Math.max(1, Math.ceil(Math.hypot(dx, dz) * 3)); let px = a[0], pz = a[1];
  for (let k = 0; k <= n; k++) { const x = Math.round(a[0] + dx * k / n), z = Math.round(a[1] + dz * k / n);
    if (!walkable(nav, x, z) || nav.clear[z * nav.w + x] < min) return false;
    if (x !== px && z !== pz && (!walkable(nav, x, pz) || !walkable(nav, px, z))) return false; px = x; pz = z; }
  return true; }

// fewer, straighter legs: from each kept cell jump to the furthest later cell in sight. The clearance asked of a shortcut is
// what the stretch it replaces had (capped), so smoothing never pulls the route against a wall it kept away from.
export function smooth(nav, cells) { if (cells.length < 3) return cells.slice(); const out = [cells[0]]; let i = 0;
  const cl = (c) => nav.clear[c[1] * nav.w + c[0]], LOOK = 150;                           // at most 15 m a leg: keeps it quick
  while (i < cells.length - 1) { const end = Math.min(cells.length - 1, i + LOOK), pm = [];
    for (let k = i, m = Infinity; k <= end; k++) { m = Math.min(m, cl(cells[k])); pm.push(m); }
    let j = end; for (; j > i + 1; j--) if (sight(nav, cells[i], cells[j], Math.min(pm[j - i], NEAR))) break;
    out.push(cells[j]); i = j; }
  return out; }

export const pathLength = (pts) => { let s = 0; for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][2] - pts[i - 1][2]); return s; };

// from (x, z) to (x, z): {ok, points [[x, y, z] ...] on the floor, length m, from/to snap distances} or {ok: false, reason}
export function route(nav, a, b, maxSnap = 3) { const s = snap(nav, a[0], a[1]), g = snap(nav, b[0], b[1], maxSnap);
  if (!s) return { ok: false, reason: 'no-floor' }; if (!g) return { ok: false, reason: 'target-off-floor' };
  const cells = astar(nav, [s.ix, s.iz], [g.ix, g.iz]); if (!cells) return { ok: false, reason: 'unreachable' };
  const pts = smooth(nav, cells).map(([ix, iz]) => { const [x, z] = centre(nav, ix, iz); return [x, floorAt(nav, ix, iz), z]; });
  return { ok: true, points: pts, length: pathLength(pts), from: s.d, to: g.d, cells: cells.length }; }

// ---- places ------------------------------------------------------------------------------------------------------------------
export const CATEGORIES = ['place', 'room', 'exit', 'washroom', 'kitchen', 'furniture', 'appliance', 'safety', 'other'];
const r3 = (v) => Math.round(v * 1000) / 1000;
// a place as pois.json stores it. approach = the walkable spot beside it a route ends at (null: none within 3 m, or no grid).
export function makePoi(p, nav, now = Date.now()) { const pos = [r3(p.pos[0]), r3(p.pos[1]), r3(p.pos[2])]; let approach = null;
  if (nav) { const s = snap(nav, pos[0], pos[2], 3); if (s) approach = centre(nav, s.ix, s.iz).map(r3); }
  return { id: p.id || 'p' + now.toString(36) + Math.random().toString(36).slice(2, 6), name: String(p.name || 'Place').slice(0, 80),
    category: CATEGORIES.includes(p.category) ? p.category : 'place', note: String(p.note || '').slice(0, 400), pos, approach, cell: cellKey(pos),
    source: p.source === 'ai' ? 'ai' : 'admin', confidence: p.source === 'ai' ? Math.max(0, Math.min(1, +p.confidence || 0)) : 1,
    thumb: p.thumb || null, created: p.created || new Date(now).toISOString() }; }
// a pois.json read from anywhere: bad entries dropped, never a throw (a broken file must not break the tour)
export function readPois(j) { const l = j && Array.isArray(j.pois) ? j.pois : [];
  return l.filter(p => p && typeof p.id === 'string' && Array.isArray(p.pos) && p.pos.length === 3 && p.pos.every(Number.isFinite) && typeof p.name === 'string'); }
// `removed`: ids the admin deleted, so a later publish of the tour does not bring a place the AI find made back (tour_publish.merge_pois)
export const poisFile = (pois, now = Date.now(), removed = []) => ({ version: 1, frame: 'room-metres', updated: new Date(now).toISOString(), pois, removed });
// search words of a place, as the object layer's `words`
export const poiWords = (p) => [p.name, p.category, p.note].filter(Boolean).join(' ').toLowerCase();
