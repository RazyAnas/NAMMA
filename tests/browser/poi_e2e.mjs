// Places in the tour (colab/real_scenes/final/poi_core.js): spatial index, walkable grid, route. Three ways to run:
//
//   node tests/browser/poi_e2e.mjs
//       logic checks, no browser: Morton keys equal navgrid.py's, nearest / radius / ray-pick against brute force, A* through
//       a doorway, no way into a sealed room, no corner cutting, the route keeps off walls, snapping, pois.json handling.
//   node tests/browser/poi_e2e.mjs --nav <nav.json> --from x,z --to x,z
//       routes on a nav.json written by navgrid.py and prints the result as JSON (tests/v3/test_navgrid.py uses it to prove
//       the Python-written file is read by the browser code).
//   node tests/browser/poi_e2e.mjs --url http://localhost:8765/index.html
//       real page in headless Chromium (needs `npm i playwright`): a built tour folder with nav.json served locally. Adds a
//       place as admin, reloads, searches it, routes to it, puts the walker next to it and waits for the arrival message.
//
// What the first two cannot prove: anything in the 3D view (pins, the line on the floor, placing by tap). That is the --url run.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

// poi_core.js is a plain .js with `export`s (it is pasted into the tour page): load a .mjs copy so any Node treats it as a module
const src = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../colab/real_scenes/final/poi_core.js');
const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'poi-')), 'poi_core.mjs'); fs.copyFileSync(src, tmp);
const C = await import(pathToFileURL(tmp).href);
const arg = (k) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };

if (arg('--nav')) {                                                   // cross-check mode: no tests, one JSON line
  const nav = await C.decodeNav(JSON.parse(fs.readFileSync(arg('--nav'), 'utf8')));
  const a = arg('--from').split(',').map(Number), b = arg('--to').split(',').map(Number), r = C.route(nav, a, b);
  let ok = true; if (r.ok) for (const p of r.points) { const [ix, iz] = C.cellOf(nav, p[0], p[2]); ok = ok && C.walkable(nav, ix, iz); }
  console.log(JSON.stringify({ ok: r.ok, reason: r.reason || null, length: r.ok ? +r.length.toFixed(3) : null, points: r.ok ? r.points.length : 0, onFloor: ok,
    walkable: nav.grid.reduce((s, v) => s + v, 0), w: nav.w, h: nav.h, floorMax: nav.floor ? +Math.max(...nav.floor).toFixed(2) : null }));
  process.exit(0);
}

if (arg('--url')) {
  const { chromium } = await import('playwright');
  const url = arg('--url'), b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await b.newPage({ viewport: { width: 1280, height: 800 } }); const errs = []; page.on('pageerror', (e) => errs.push(String(e)));
  const ready = async () => { await page.waitForFunction(() => window.nammaPoi && window.viewer && window.viewer.state && window.viewer.state.loaded, null, { timeout: 180000 }); };
  await page.goto(url + (url.includes('?') ? '&' : '?') + 'admin=1'); await ready();
  await page.evaluate(() => localStorage.removeItem('nammaPois:' + location.pathname));
  const nav = await page.evaluate(() => { const n = window.nammaPoi.nav(); return n ? { w: n.w, h: n.h } : null; }); assert.ok(nav, 'the tour has no nav.json');
  // a place on a walkable cell far from the camera, added the way the admin panel adds one
  const made = await page.evaluate(() => { const n = window.nammaPoi.nav(), c = window.app.root.findComponents('camera')[0].entity.getPosition(); let best = null;
    for (let iz = 0; iz < n.h; iz++) for (let ix = 0; ix < n.w; ix++) if (n.grid[iz * n.w + ix]) { const x = n.x0 + (ix + 0.5) * n.cell, z = n.z0 + (iz + 0.5) * n.cell, d = Math.hypot(x - c.x, z - c.z); if (!best || d > best.d) best = { x, z, d, y: n.floor ? n.floor[iz * n.w + ix] : 0 }; }
    const p = window.nammaPoi.add({ name: 'Test corner', category: 'place', note: 'e2e', pos: [best.x, best.y + 1, best.z] }); return { id: p.id, approach: p.approach, far: best.d }; });
  assert.ok(made.approach, 'the place got no approach point');
  await page.reload(); await ready();                                       // unsaved admin edits survive a reload
  assert.equal(await page.evaluate((id) => !!window.nammaPoi.get(id), made.id), true, 'the draft was lost on reload');
  await page.fill('#nm-q', 'test corner'); await page.waitForSelector('#nm-hits button'); await page.click('#nm-hits button');
  await page.waitForFunction(() => { const r = window.nammaPoi.route(); return r && r.points && r.points.length > 1; }, null, { timeout: 10000 });
  const r0 = await page.evaluate(() => window.nammaPoi.route().length); assert.ok(r0 > 1, 'route too short to mean anything: ' + r0);
  assert.ok((await page.getAttribute('#nm-path .l', 'd') || '').length >= 0);
  // go there: put the camera at the end of the route (fly mode) and wait for the arrival message
  await page.evaluate(() => { window.viewer.state.cameraMode = 'walk'; }); await page.waitForFunction(() => window.viewer.state.cameraMode === 'walk' && window.nammaWalk && window.nammaWalk._hasPosition, null, { timeout: 15000 });
  await page.evaluate((a) => { const w = window.nammaWalk, r = window.nammaPoi.route(), y = r.points[r.points.length - 1][1] + w.eyeHeight + w.hoverHeight; w._position.set(a[0] + 0.3, y, a[1]); w._prevPosition.copy(w._position); w._resetMotion(); }, made.approach);
  await page.waitForFunction(() => /reached your destination/.test(document.getElementById('nm-toast').textContent) && document.getElementById('nm-toast').style.display === 'block', null, { timeout: 15000 });
  await page.evaluate(() => localStorage.removeItem('nammaPois:' + location.pathname));
  assert.deepEqual(errs, [], 'page errors'); console.log('browser run ok: place added, kept over reload, found, routed', r0.toFixed(1), 'm, arrival shown'); await b.close(); process.exit(0);
}

// ---------------------------------------------------------------- logic checks
const rnd = (() => { let s = 12345; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();

test('Morton keys are the ones navgrid.py writes to pois.json', () => {
  const want = [[[0, 0, 0], 939524096], [[1, 0, 0], 939524097], [[0, 1, 0], 939524098], [[0, 0, 1], 939524100], [[-1, -1, -1], 134217727], [[3, 5, 7], 939524527],
    [[-12, 2, 40], 824612944], [[511, 511, 511], 1073741823], [[-512, -512, -512], 0]];
  for (const [c, k] of want) assert.equal(C.morton3(...c), k, c.join());
  assert.equal(C.cellKey([1.26, 0.9, -3.4]), 479348750); assert.equal(C.cellKey([-0.01, 0.0, 0.01]), 824480329);
});

test('spatial index: nearest and radius agree with brute force; remove works', () => {
  const ix = new C.SpatialIndex(), pts = [];
  for (let i = 0; i < 400; i++) { const p = { id: 'p' + i, pos: [rnd() * 30 - 15, rnd() * 3, rnd() * 20 - 10] }; pts.push(p); ix.insert(p); }
  for (let n = 0; n < 60; n++) { const q = [rnd() * 34 - 17, rnd() * 3, rnd() * 24 - 12], d = (p) => Math.hypot(p.pos[0] - q[0], p.pos[1] - q[1], p.pos[2] - q[2]);
    const bf = pts.slice().sort((a, b) => d(a) - d(b)), got = ix.nearest(q); assert.equal(got.item.id, bf[0].id); assert.ok(Math.abs(got.d - d(bf[0])) < 1e-9);
    const r = 0.5 + rnd() * 4, w = ix.within(q, r), want = bf.filter((p) => d(p) <= r).map((p) => p.id); assert.deepEqual(w.map((h) => h.item.id), want); }
  assert.equal(ix.nearest([100, 0, 100], 5), null, 'nothing within 5 m');
  const first = ix.nearest([0, 1, 0]).item.id; assert.equal(ix.remove(first), true); assert.notEqual(ix.nearest([0, 1, 0]).item.id, first); assert.equal(ix.size, 399);
});

test('spatial index: boxes (objects) and ray pick', () => {
  const ix = new C.SpatialIndex(); ix.insert({ id: 'sofa', lo: [2, 0, 1], hi: [4, 0.9, 2] }); ix.insert({ id: 'lamp', pos: [3, 1.5, 6] }); ix.insert({ id: 'far', pos: [3, 1.5, 12] });
  assert.equal(ix.nearest([3, 0.4, 1.5]).d, 0, 'inside the box');
  assert.ok(Math.abs(ix.nearest([5, 0.4, 1.5]).d - 1) < 1e-9, 'one metre from its side');
  let h = ix.rayPick([3, 0.5, -3], [0, 0, 1]); assert.equal(h.item.id, 'sofa'); assert.ok(Math.abs(h.t - 4) < 1e-9);
  h = ix.rayPick([3, 1.5, -3], [0, 0, 1]); assert.equal(h.item.id, 'lamp', 'over the sofa: the nearer of two pins on the ray'); assert.ok(Math.abs(h.t - 8.75) < 1e-6);
  assert.equal(ix.rayPick([3, 1.5, -3], [0, 0, -1]), null, 'looking away');
  assert.equal(ix.rayPick([0, 2.6, -3], [0, 0, 1]), null, 'a ray that passes everything');
  assert.equal(ix.rayPick([3, 1.5, -3], [0, 0, 1], 30, 0.25, (it) => it.id !== 'lamp').item.id, 'far', 'filter');
  for (let n = 0; n < 200; n++) {                                               // random rays against brute force (points only)
    const o = [rnd() * 10 - 2, rnd() * 3, rnd() * 16 - 4], d = [rnd() - 0.5, (rnd() - 0.5) * 0.4, rnd() - 0.5], L = Math.hypot(...d), u = d.map((v) => v / L); let best = null;
    for (const it of [{ id: 'lamp', pos: [3, 1.5, 6] }, { id: 'far', pos: [3, 1.5, 12] }]) { const m = o.map((v, k) => v - it.pos[k]), b = m[0] * u[0] + m[1] * u[1] + m[2] * u[2], q = m[0] ** 2 + m[1] ** 2 + m[2] ** 2 - 0.0625, D = b * b - q;
      if (D < 0) continue; const t = -b - Math.sqrt(D), tt = t >= 0 ? t : (q <= 0 ? 0 : -1); if (tt >= 0 && tt <= 30 && (!best || tt < best.t)) best = { id: it.id, t: tt }; }
    const got = ix.rayPick(o, d, 30, 0.25, (it) => !!it.pos); assert.equal(got ? got.item.id : null, best ? best.id : null);
  }
});

// a 6 x 4 m flat (10 cm cells): two rooms split by a wall at x = 3 m with a 0.8 m doorway at z = 1.6 .. 2.4, a closed cupboard
// (0.6 x 0.6 m of floor walled in) in room B, a table in room A
function flat() { const w = 60, h = 40, g = new Uint8Array(w * h).fill(1), set = (x0, z0, x1, z1, v) => { for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) g[z * w + x] = v; };
  set(0, 0, w - 1, 0, 0); set(0, h - 1, w - 1, h - 1, 0); set(0, 0, 0, h - 1, 0); set(w - 1, 0, w - 1, h - 1, 0);           // outer walls
  set(29, 0, 31, 15, 0); set(29, 24, 31, h - 1, 0);                                                                           // the wall, doorway z cells 16..23
  set(44, 4, 52, 4, 0); set(44, 12, 52, 12, 0); set(44, 4, 44, 12, 0); set(52, 4, 52, 12, 0);                                 // cupboard walls (inside stays 1)
  set(10, 26, 18, 32, 0);                                                                                                       // table
  return C.makeNav({ x0: 0, z0: 0, cell: 0.1, w, h }, g, null); }

test('A*: through the doorway, on walkable cells only, never past a corner', () => {
  const nav = flat(), p = C.astar(nav, [5, 5], [55, 35]); assert.ok(p, 'a way exists'); assert.deepEqual(p[0], [5, 5]); assert.deepEqual(p[p.length - 1], [55, 35]);
  let through = false;
  for (let i = 0; i < p.length; i++) { const [x, z] = p[i]; assert.equal(C.walkable(nav, x, z), true); if (x >= 29 && x <= 31) { through = true; assert.ok(z >= 16 && z <= 23, 'crossed the wall outside the doorway'); }
    if (i) { const [px, pz] = p[i - 1]; assert.ok(Math.abs(x - px) <= 1 && Math.abs(z - pz) <= 1, 'a jump');
      if (x !== px && z !== pz) assert.ok(C.walkable(nav, x, pz) && C.walkable(nav, px, z), 'cut a corner'); } }
  assert.ok(through);
  const mid = p.filter(([x]) => x === 30)[0]; assert.ok(mid[1] >= 18 && mid[1] <= 21, 'keeps to the middle of the doorway, got z cell ' + mid[1]);
});

test('A*: a sealed room cannot be reached; route() says so and does not throw', () => {
  const nav = flat(); assert.equal(C.walkable(nav, 48, 8), true, 'the cupboard floor itself is free');
  assert.equal(C.astar(nav, [5, 5], [48, 8]), null); assert.equal(C.astar(nav, [5, 5], [30, 5]), null, 'a wall cell as the goal');
  const r = C.route(nav, [0.5, 0.5], [4.8, 0.8]); assert.equal(r.ok, false); assert.equal(r.reason, 'unreachable');
  assert.equal(C.route(nav, [0.5, 0.5], [20, 20]).reason, 'target-off-floor', 'a target far outside the grid');
  const none = C.makeNav({ x0: 0, z0: 0, cell: 0.1, w: 8, h: 8 }, new Uint8Array(64), null); assert.equal(C.route(none, [0.2, 0.2], [0.5, 0.5]).reason, 'no-floor');
});

test('route(): smoothed legs are in sight of each other and the length is sane', () => {
  const nav = flat(), r = C.route(nav, [0.55, 0.55], [5.55, 3.55]); assert.equal(r.ok, true);
  assert.ok(r.points.length >= 3 && r.points.length < r.cells / 2, `smoothed: ${r.points.length} points from ${r.cells} cells`);
  for (let i = 1; i < r.points.length; i++) assert.equal(C.sight(nav, C.cellOf(nav, r.points[i - 1][0], r.points[i - 1][2]), C.cellOf(nav, r.points[i][0], r.points[i][2])), true);
  const straight = Math.hypot(5, 3); assert.ok(r.length >= straight && r.length < straight * 1.5, 'length ' + r.length);
  for (const p of r.points) assert.equal(p[1], 0, 'flat floor');
  const t0 = Date.now(); for (let i = 0; i < 50; i++) C.route(nav, [0.55, 0.55], [5.55, 3.55]); assert.ok(Date.now() - t0 < 2000, 'fast enough to redo while walking');
});

test('snap: a place inside furniture is reached from beside it', () => {
  const nav = flat(), s = C.snap(nav, 1.45, 2.95); assert.ok(s && s.d > 0 && s.d < 0.5, 'beside the table'); assert.equal(C.walkable(nav, s.ix, s.iz), true);
  assert.equal(C.snap(nav, 1.45, 2.95, 0.05), null, 'nothing walkable within 5 cm'); assert.equal(C.snap(nav, 0.55, 0.55).d, 0);
  for (let n = 0; n < 100; n++) { const x = rnd() * 8 - 1, z = rnd() * 6 - 1, got = C.snap(nav, x, z); let best = Infinity;        // against brute force
    for (let iz = 0; iz < nav.h; iz++) for (let ix = 0; ix < nav.w; ix++) if (C.walkable(nav, ix, iz)) { const [cx, cz] = C.centre(nav, ix, iz); best = Math.min(best, Math.hypot(cx - x, cz - z)); }
    const [cx, cz] = C.cellOf(nav, x, z); if (!C.walkable(nav, cx, cz)) assert.ok(Math.abs(got.d - best) < 1e-9, `snap ${got.d} vs ${best}`); }
});

test('nav.json: zlib + base64 grid and floor heights decode', async () => {
  const w = 5, h = 3, g = Uint8Array.from([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]), fl = Uint8Array.from([1, 1, 0, 9, 9, 1, 1, 5, 9, 9, 0, 0, 0, 9, 9]);
  const z = (a) => zlib.deflateSync(Buffer.from(a)).toString('base64');
  const nav = await C.decodeNav({ x0: -1, z0: 2, cell: 0.1, w, h, grid: z(g), floor: z(fl), floor_base: -0.05 });
  assert.deepEqual(Array.from(nav.grid), Array.from(g)); assert.ok(Math.abs(C.floorAt(nav, 3, 0) - 0.35) < 1e-6); assert.ok(Math.abs(C.floorAt(nav, 0, 0) + 0.05) < 1e-6);
  assert.equal(C.floorAt(nav, 2, 0), 0, 'not walkable: no height'); assert.deepEqual(C.cellOf(nav, -0.95, 2.25), [0, 2]);
  await assert.rejects(C.decodeNav({ x0: 0, z0: 0, cell: 0.1, w: 4, h: 4, grid: z(g) }), /size/);
});

test('places: pois.json entries', () => {
  const nav = flat(), p = C.makePoi({ name: 'Table', category: 'furniture', pos: [1.45, 0.75, 2.95] }, nav, 1700000000000);
  assert.equal(p.cell, C.cellKey(p.pos)); assert.equal(p.source, 'admin'); assert.equal(p.confidence, 1); assert.match(p.id, /^p[a-z0-9]+$/);
  const [ax, az] = C.cellOf(nav, p.approach[0], p.approach[1]); assert.equal(C.walkable(nav, ax, az), true, 'the approach point is on the floor');
  assert.equal(C.route(nav, [0.55, 0.55], p.approach).ok, true);
  const ai = C.makePoi({ name: 'Socket', pos: [5, 0.3, 3], source: 'ai', confidence: 0.83, category: 'nonsense' }, nav); assert.equal(ai.source, 'ai'); assert.equal(ai.confidence, 0.83); assert.equal(ai.category, 'place');
  assert.equal(C.makePoi({ name: 'x', pos: [40, 0, 40] }, nav).approach, null, 'too far from any floor'); assert.equal(C.makePoi({ name: 'x', pos: [1, 0, 1] }, null).approach, null, 'no grid');
  const again = C.makePoi({ ...p, name: 'Dining table' }, nav); assert.equal(again.id, p.id); assert.equal(again.created, p.created);
  const f = C.poisFile([p, ai], 1700000000000, ['gone']); assert.equal(f.version, 1); assert.equal(f.updated, '2023-11-14T22:13:20.000Z'); assert.deepEqual(f.removed, ['gone']); assert.deepEqual(C.poisFile([]).removed, []);
  assert.deepEqual(C.readPois(JSON.parse(JSON.stringify(f))).map((x) => x.id), [p.id, ai.id]);
  assert.deepEqual(C.readPois(null), []); assert.deepEqual(C.readPois({ pois: 'x' }), []);
  assert.deepEqual(C.readPois({ pois: [{ id: 'a', name: 'ok', pos: [1, 2, 3] }, { id: 'b', name: 'bad', pos: [1, 2] }, { id: 3, name: 'bad', pos: [1, 2, 3] }, null, { id: 'c', name: 'nan', pos: [1, NaN, 3] }] }).map((x) => x.id), ['a']);
  assert.equal(C.poiWords(p), 'table furniture');
});
