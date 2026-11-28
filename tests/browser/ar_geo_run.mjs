// Runs the browser's shape lock (colab/real_scenes/final/ar_geo.js) on files written by tests/v3/test_ar_shape.py:
//   node tests/browser/ar_geo_run.mjs <ar_shape.bin> <live.json: {pts: [x,y,z,...], path: [x,y,z,...], cap}>   -> one JSON line
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const src = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'colab', 'real_scenes', 'final', 'ar_geo.js');
const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'argeo-')), 'ar_geo.mjs'); fs.copyFileSync(src, tmp);
const G = await import(pathToFileURL(tmp).href);
const raw = fs.readFileSync(process.argv[2]), M = G.readShape(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const live = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')), t0 = performance.now();
const L = G.prepare(Float32Array.from(live.pts), Float32Array.from(live.path), live.cap || Infinity), t1 = performance.now(), J = G.locate(M, L, live.opts || {}), t2 = performance.now();
console.log(JSON.stringify({ J, join: G.toJoin(J, L), c: L.c, heights: L.heights, points: L.p.length / 3, prepare_ms: t1 - t0, locate_ms: t2 - t1 }));
