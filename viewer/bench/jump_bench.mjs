// Can the walker jump onto furniture? From each standable cell next to an
// observed furniture top, run at it while jumping and see where the feet land.
//
//   node bench/jump_bench.mjs <scene-dir> [--old]
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import RAPIER from '@dimforge/rapier3d-compat';
import { Walker, decodeMovementLayers, readGlbTriangles } from '../src/walker.js';

const args = process.argv.slice(2);
const sceneDir = args.find((arg) => !arg.startsWith('--'));
const old = args.includes('--old');
await RAPIER.init({});
const read = async (file) => readFile(path.join(sceneDir, file));
const navigation = JSON.parse(await read('geometry/navigation/navmesh.json'));
const buffer = await read('geometry/collision/collision.glb');
const collision = readGlbTriangles(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
const layers = await decodeMovementLayers(navigation);
if (!layers.top || layers.top === layers.support) throw new Error('navmesh has no climb layer');
const { width, height, cell_size_m: cell, origin_floor_coordinates: [x0, z0] } = navigation;
// --old: the previous rules (no climb layer, 3.2 m/s jump).
const walker = new Walker(RAPIER, {
  navigation,
  layers: old ? { ...layers, top: layers.support } : layers,
  collision,
  options: old ? { jumpSpeed: 3.2 } : {},
});
const centre = (gx, gy) => [x0 + (gx + 0.5) * cell, z0 + (gy + 0.5) * cell];
const at = (gx, gy) => gy * width + gx;
const bands = { 'low (0.35-0.6 m, tables, seats)': [0.35, 0.6], 'high (0.6-0.85 m, arms, desks)': [0.6, 0.85] };
const result = {};
for (const [name, [lo, hi]] of Object.entries(bands)) {
  let tries = 0;
  let landed = 0;
  for (let gy = 2; gy < height - 2; gy += 1) {
    for (let gx = 2; gx < width - 2; gx += 1) {
      if (!layers.standable[at(gx, gy)]) continue;
      // Target: furniture top two cells away in a straight line.
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const tx = gx + dx * 3;
        const ty = gy + dy * 3;
        if (tx < 0 || ty < 0 || tx >= width || ty >= height) continue;
        const rise = layers.top[at(tx, ty)] - layers.support[at(tx, ty)];
        if (layers.standable[at(tx, ty)] || rise < lo || rise >= hi) continue;
        tries += 1;
        const [sx, sz] = centre(gx, gy);
        walker.placeFeet(sx, sz);
        for (let i = 0; i < 10; i += 1) walker.step([0, 0], 1 / 60);
        // Push until the centre is half a metre in, then let go, as a player would.
        for (let i = 0; i < 90; i += 1) {
          const [px, , pz] = walker.feet;
          const push = Math.hypot(px - sx, pz - sz) < 0.5 ? 1.4 : 0;
          walker.step([dx * push, dy * push], 1 / 60, i === 2);
        }
        for (let i = 0; i < 30; i += 1) walker.step([0, 0], 1 / 60);
        const [fx, fy, fz] = walker.feet;
        const [cx, cy] = [Math.floor((fx - x0) / cell), Math.floor((fz - z0) / cell)];
        const floor = layers.support[at(Math.min(width - 1, Math.max(0, cx)), Math.min(height - 1, Math.max(0, cy)))];
        if (walker.grounded && fy - floor > 0.3) landed += 1;
        else if (process.env.DEBUG) {
          const travelled = Math.hypot(fx - sx, fz - sz);
          const peak = layers.top[at(tx, ty)] - layers.support[at(tx, ty)];
          console.log(`miss ${gx},${gy} -> ${tx},${ty} top ${peak.toFixed(2)} moved ${travelled.toFixed(2)} feet ${(fy - floor).toFixed(2)} grounded ${walker.grounded}`);
        }
      }
    }
  }
  result[name] = { tries, landed, pct: tries ? +((100 * landed) / tries).toFixed(1) : null };
}
console.log(JSON.stringify({ scene: path.basename(sceneDir), rules: old ? 'old' : 'new', ...result }, null, 1));
