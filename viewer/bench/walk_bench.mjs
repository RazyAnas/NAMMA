// Compares the current viewer movement with the Rapier walker on a scene with
// known ground truth (see scripts/build_walk_test_scene.py).
//
//   node bench/walk_bench.mjs <scene-dir> [--no-mesh] [--json out.json]
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import RAPIER from '@dimforge/rapier3d-compat';
import { NavFrame, Walker, decodeMovementLayers, inflate, readGlbTriangles } from '../src/walker.js';

const args = process.argv.slice(2);
const sceneDir = args.find((arg) => !arg.startsWith('--'));
const useMesh = !args.includes('--no-mesh');
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;
if (!sceneDir) throw new Error('usage: node bench/walk_bench.mjs <scene-dir>');

await RAPIER.init({});
const read = async (file) => readFile(path.join(sceneDir, file));
const navigation = JSON.parse(await read('geometry/navigation/navmesh.json'));
const scene = JSON.parse(await read('scene.json'));
const toArrayBuffer = (buffer) => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const collision = readGlbTriangles(toArrayBuffer(await read('geometry/collision/collision.glb')));
const layers = await decodeMovementLayers(navigation);
const oldWalkable = await inflate(navigation.walkable_grid_base64, navigation.width * navigation.height);
const frame = new NavFrame(navigation);
const { width, height, cell_size_m: cell, origin_floor_coordinates: [x0, z0] } = navigation;

// ---- ground truth in the nav frame -------------------------------------
const w2s = scene.world_to_source;
const worldToNav = ([x, y, z]) => {
  const source = [0, 1, 2].map((r) => w2s[r][0] * x + w2s[r][1] * y + w2s[r][2] * z + w2s[r][3]);
  return frame.metersToNav(source.map((value) => value * frame.scale));
};
const gtToNav = ({ positions, indices }) => {
  const vertices = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) vertices.set(worldToNav([positions[i], positions[i + 1], positions[i + 2]]), i);
  return RAPIER.ColliderDesc.trimesh(vertices, indices);
};
const gtObstacles = new RAPIER.World({ x: 0, y: 0, z: 0 });
gtObstacles.createCollider(gtToNav(readGlbTriangles(toArrayBuffer(await read('ground_truth_obstacles.glb')))));
gtObstacles.step();
const gtFull = new RAPIER.World({ x: 0, y: 0, z: 0 });
gtFull.createCollider(gtToNav(readGlbTriangles(toArrayBuffer(await read('ground_truth.glb')))));
gtFull.step();

const R = navigation.agent_radius_m;
const H = navigation.agent_height_m;
const EYE = 1.6;
const HALF = H / 2 - R;

// Ground height below a point (true geometry), searching down from `fromY`.
function gtGround(x, z, fromY) {
  const hit = gtFull.castRay(new RAPIER.Ray({ x, y: fromY, z }, { x: 0, y: -1, z: 0 }), 10, true);
  return hit ? fromY - hit.timeOfImpact : -Infinity;
}

// Deepest overlap (m) between a capsule standing with feet at `feet` and the
// true obstacles. Points along the axis are checked against the true surface.
function gtPenetration([x, feetY, z]) {
  let worst = 0;
  for (let k = 0; k <= 8; k += 1) {
    const y = feetY + R + (k / 8) * (H - 2 * R);
    const projection = gtObstacles.projectPoint({ x, y, z }, false);
    if (!projection) continue;
    const dx = projection.point.x - x;
    const dy = projection.point.y - y;
    const dz = projection.point.z - z;
    const distance = Math.hypot(dx, dy, dz);
    worst = Math.max(worst, projection.isInside ? R + distance : R - distance);
  }
  return worst;
}

const CLIP = 0.03; // 3 cm overlap counts as walking into something

// ---- ground-truth reachability -----------------------------------------
const cellCenter = (gx, gy) => [x0 + (gx + 0.5) * cell, z0 + (gy + 0.5) * cell];
const gtSupport = new Float32Array(width * height).fill(NaN);
const gtStand = new Uint8Array(width * height);
const roomCorners = [[0, 0, 0], [scene.room.x, 0, 0], [0, 0, scene.room.z], [scene.room.x, 0, scene.room.z]].map(worldToNav);
for (let gy = 0; gy < height; gy += 1) {
  for (let gx = 0; gx < width; gx += 1) {
    const [x, z] = cellCenter(gx, gy);
    const ground = gtGround(x, z, 2.4);
    if (!Number.isFinite(ground) || ground > 0.55) continue; // sofa/table tops are not reachable floors
    gtSupport[gy * width + gx] = ground;
    if (gtPenetration([x, ground + 0.005, z]) <= 0.005) gtStand[gy * width + gx] = 1;
  }
}
function bfs(start, canMove) {
  const seen = new Uint8Array(width * height);
  const queue = [start];
  seen[start[1] * width + start[0]] = 1;
  while (queue.length) {
    const [gx, gy] = queue.pop();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = gx + dx;
      const ny = gy + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height || seen[ny * width + nx]) continue;
      if (!canMove(gx, gy, nx, ny)) continue;
      seen[ny * width + nx] = 1;
      queue.push([nx, ny]);
    }
  }
  return seen;
}
const spawn = navigation.spawn.grid;
const gtReach = bfs(spawn, (ax, ay, bx, by) => gtStand[by * width + bx]
  && Math.abs(gtSupport[by * width + bx] - gtSupport[ay * width + ax]) <= 0.25);

// ---- old viewer: flat grid, binary stop, free-flying camera -------------
class OldViewer {
  constructor() { this.eye = [0, 0, 0]; }
  placeFeet(x, z, feetY = 0) { this.eye = [x, feetY + EYE, z]; }
  walkable(x, z) {
    const [gx, gy] = frame.cellOf(x, z);
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) return false;
    return oldWalkable[gy * width + gx] === 1;
  }
  get feet() { return [this.eye[0], this.eye[1] - EYE, this.eye[2]]; }
  // Mirrors updateMovement() in main.js: move along the camera direction,
  // substep at half a cell, stop at the first blocked cell.
  move(yaw, pitch, speed, dt) {
    const dir = [Math.cos(pitch) * Math.cos(yaw), Math.sin(pitch), Math.cos(pitch) * Math.sin(yaw)];
    const distance = speed * dt;
    const steps = Math.max(1, Math.ceil(distance / (cell * 0.5)));
    for (let i = 0; i < steps; i += 1) {
      const next = this.eye.map((value, k) => value + (dir[k] * distance) / steps);
      if (!this.walkable(next[0], next[2])) break;
      this.eye = next;
    }
  }
}

const oldReach = bfs(spawn, (ax, ay, bx, by) => oldWalkable[ay * width + ax] && oldWalkable[by * width + bx]);

const walker = new Walker(RAPIER, { navigation, layers, collision: useMesh ? collision : null });
// The walker moves itself between neighbouring cells; a move succeeds when
// it ends within a third of a cell of the target centre.
function walkerMove(ax, ay, bx, by) {
  const [sx, sz] = cellCenter(ax, ay);
  const [tx, tz] = cellCenter(bx, by);
  walker.placeFeet(sx, sz, layers.support[ay * width + ax]);
  for (let i = 0; i < 4; i += 1) walker.step([0, 0], 1 / 60); // settle
  const startY = walker.feet[1];
  for (let i = 0; i < 12; i += 1) {
    const c = walker.center;
    const vx = (tx - c.x) * 60 / 3;
    const vz = (tz - c.z) * 60 / 3;
    walker.step([vx, vz], 1 / 60);
  }
  const c = walker.center;
  return Math.hypot(c.x - tx, c.z - tz) < cell / 3 && Math.abs(walker.feet[1] - startY) < 0.4;
}
const t0 = performance.now();
const walkerReach = layers.standable[spawn[1] * width + spawn[0]] ? bfs(spawn, walkerMove) : new Uint8Array(width * height);
const bfsMs = performance.now() - t0;

// Room-space (x, z) of a nav cell centre, to say which object it overlaps.
const s2w = invert4(w2s);
function invert4(m) {
  const r = [0, 1, 2].map((i) => [0, 1, 2].map((j) => m[i][j]));
  const det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
  const inv = [
    [(r[1][1] * r[2][2] - r[1][2] * r[2][1]) / det, (r[0][2] * r[2][1] - r[0][1] * r[2][2]) / det, (r[0][1] * r[1][2] - r[0][2] * r[1][1]) / det],
    [(r[1][2] * r[2][0] - r[1][0] * r[2][2]) / det, (r[0][0] * r[2][2] - r[0][2] * r[2][0]) / det, (r[0][2] * r[1][0] - r[0][0] * r[1][2]) / det],
    [(r[1][0] * r[2][1] - r[1][1] * r[2][0]) / det, (r[0][1] * r[2][0] - r[0][0] * r[2][1]) / det, (r[0][0] * r[1][1] - r[0][1] * r[1][0]) / det],
  ];
  const t = [0, 1, 2].map((i) => -(inv[i][0] * m[0][3] + inv[i][1] * m[1][3] + inv[i][2] * m[2][3]));
  return inv.map((row, i) => [...row, t[i]]);
}
function navToWorld([x, z]) {
  const source = frame.navToMeters([x, 0, z]).map((value) => value / frame.scale);
  const w = [0, 1, 2].map((i) => s2w[i][0] * source[0] + s2w[i][1] * source[1] + s2w[i][2] * source[2] + s2w[i][3]);
  return [w[0], w[2]];
}
const FOOTPRINTS = {
  glass_table: [2.8, 3.8, 2.0, 2.6], dining_table: [4.5, 6.0, 0.5, 1.4], lamp: [3.35, 3.65, 4.35, 4.65],
  sofa: [0.3, 2.3, 0.2, 1.1], stairs: [0.0, 1.0, 3.0, 6.0], ramp: [1.5, 3.5, 5.0, 6.0], raised_floor: [5.2, 7.0, 3.5, 6.0],
};
function countByObject(cells) {
  const counts = {};
  for (const [x, z, pen] of cells) {
    let name = 'walls';
    let best = Infinity;
    for (const [key, [xa, xb, za, zb]] of Object.entries(FOOTPRINTS)) {
      const d = Math.hypot(Math.max(xa - x, 0, x - xb), Math.max(za - z, 0, z - zb));
      if (d < 0.45 && d < best) { best = d; name = key; }
    }
    const entry = counts[name] || (counts[name] = { cells: 0, worst_cm: 0 });
    entry.cells += 1;
    entry.worst_cm = Math.max(entry.worst_cm, +(100 * pen).toFixed(1));
  }
  return counts;
}

function reachStats(reach, feetFor) {
  let truth = 0; let hit = 0; let total = 0; let clipping = 0; let worstClip = 0;
  const clipCells = [];
  for (let i = 0; i < width * height; i += 1) {
    if (gtReach[i]) truth += 1;
    if (!reach[i]) continue;
    total += 1;
    if (gtReach[i]) hit += 1;
    const gx = i % width; const gy = Math.floor(i / width);
    const pen = gtPenetration(feetFor(gx, gy));
    if (pen > CLIP) { clipping += 1; clipCells.push([...navToWorld(cellCenter(gx, gy)), pen]); }
    worstClip = Math.max(worstClip, pen);
  }
  return {
    reachable_cells: total,
    true_reachable_cells: truth,
    coverage_pct: +(100 * hit / truth).toFixed(1),
    cells_inside_furniture_pct: total ? +(100 * clipping / total).toFixed(1) : 0,
    worst_overlap_cm: +(100 * worstClip).toFixed(1),
    overlap_by_object: countByObject(clipCells),
  };
}
const oldStats = reachStats(oldReach, (gx, gy) => { const [x, z] = cellCenter(gx, gy); return [x, 0, z]; });
const walkerStats = reachStats(walkerReach, (gx, gy) => {
  const [x, z] = cellCenter(gx, gy);
  walker.placeFeet(x, z, layers.support[gy * width + gx]);
  for (let i = 0; i < 6; i += 1) walker.step([0, 0], 1 / 60);
  return walker.feet;
});

// ---- scripted scenarios (room coordinates, metres) ----------------------
function headingTo(a, b) { return Math.atan2(b[2] - a[2], b[0] - a[0]); }
function runScenario({ from, to, pitchDeg = 0, seconds = 4, speed = 1.5, approachDeg = 0 }) {
  const a = worldToNav([from[0], 0, from[1]]);
  const b = worldToNav([to[0], 0, to[1]]);
  const yaw = headingTo(a, b) + (approachDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const dt = 1 / 60;
  const result = {};
  for (const [name, agent] of [['old', new OldViewer()], ['new', walker]]) {
    agent.placeFeet(a[0], a[2], name === 'new' ? walker.supportAt(a[0], a[2]) : 0);
    if (name === 'new') for (let i = 0; i < 10; i += 1) walker.step([0, 0], dt);
    const start = agent.feet;
    let maxPen = 0; let clipFrames = 0; let eyeErr = 0;
    const frames = Math.round(seconds / dt);
    for (let f = 0; f < frames; f += 1) {
      if (name === 'old') agent.move(yaw, pitch, speed, dt);
      else walker.step([Math.cos(yaw) * speed, Math.sin(yaw) * speed], dt);
      const feet = agent.feet;
      const pen = gtPenetration(feet);
      maxPen = Math.max(maxPen, pen);
      if (pen > CLIP) clipFrames += 1;
      const ground = gtGround(feet[0], feet[2], feet[1] + 0.3);
      if (Number.isFinite(ground)) eyeErr = Math.max(eyeErr, Math.abs(feet[1] + EYE - (ground + EYE)));
    }
    const end = agent.feet;
    const travelled = Math.hypot(end[0] - start[0], end[2] - start[2]);
    const goal = Math.hypot(end[0] - b[0], end[2] - b[2]);
    result[name] = {
      travelled_m: +travelled.toFixed(2),
      left_from_goal_m: +goal.toFixed(2),
      end_feet_height_m: +(end[1]).toFixed(2),
      max_eye_height_error_m: +eyeErr.toFixed(2),
      max_overlap_cm: +(100 * maxPen).toFixed(1),
      frames_inside_something_pct: +(100 * clipFrames / frames).toFixed(0),
    };
  }
  return result;
}

const scenarios = {
  'slide along a wall (walk 45 deg into it)': { from: [3.0, 1.1], to: [4.3, -0.2], seconds: 3 },
  'walk while looking down 35 deg': { from: [3.0, 3.4], to: [5.0, 3.4], pitchDeg: -35, seconds: 1.3 },
  'step up onto 15 cm raised floor': { from: [4.4, 4.75], to: [6.5, 4.75], seconds: 2 },
  'climb three 16 cm stairs': { from: [0.5, 2.3], to: [0.5, 5.2], seconds: 3 },
  'walk up the 14 deg ramp': { from: [1.8, 5.5], to: [3.3, 5.5], seconds: 2 },
  'walk into the 3 cm lamp pole': { from: [3.5, 3.6], to: [3.5, 5.4], seconds: 2 },
  'walk into the glass coffee table': { from: [3.3, 1.5], to: [3.3, 3.2], seconds: 2 },
  'walk through the blank wall': { from: [6.0, 2.7], to: [8.0, 2.7], seconds: 2 },
  'walk under the dining table': { from: [5.2, 1.9], to: [5.2, 0.2], seconds: 2 },
};
const scenarioResults = Object.fromEntries(Object.entries(scenarios).map(([name, spec]) => [name, runScenario(spec)]));

// ---- per-frame cost ------------------------------------------------------
walker.spawn();
const t1 = performance.now();
for (let i = 0; i < 600; i += 1) walker.step([Math.cos(i / 50), Math.sin(i / 50)], 1 / 60);
const stepMs = (performance.now() - t1) / 600;

const report = {
  scene: path.basename(path.resolve(sceneDir)),
  collision_mesh_triangles: collision.indices.length / 3,
  walker_uses_collision_mesh: useMesh,
  reachability: { old_viewer: oldStats, new_walker: walkerStats },
  scenarios: scenarioResults,
  walker_step_ms: +stepMs.toFixed(3),
  bfs_seconds: +(bfsMs / 1000).toFixed(1),
};
console.log(JSON.stringify(report, null, 2));
if (jsonOut) await writeFile(jsonOut, JSON.stringify(report, null, 2));
