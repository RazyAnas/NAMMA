// On-site AR maths (colab/real_scenes/final/ar_core.js): the join between the phone's AR world and the scan from pictures the
// server located, some of them wrongly.   node tests/browser/ar_core_test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const src = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'colab', 'real_scenes', 'final', 'ar_core.js');
const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'arcore-')), 'ar_core.mjs'); fs.copyFileSync(src, tmp);
const A = await import(pathToFileURL(tmp).href);

let seed = 7; const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296, gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.2831853 * rnd());
const Ry = (a) => [[Math.cos(a), 0, Math.sin(a), 0], [0, 1, 0, 0], [-Math.sin(a), 0, Math.cos(a), 0], [0, 0, 0, 1]];
const Rx = (a) => [[1, 0, 0, 0], [0, Math.cos(a), -Math.sin(a), 0], [0, Math.sin(a), Math.cos(a), 0], [0, 0, 0, 1]];
const FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]];
const toCm = (M) => { const o = []; for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o.push(M[r][c]); return o; };
const at = (M, p) => { const T = M.map(r => [...r]); T[0][3] = p[0]; T[1][3] = p[1]; T[2][3] = p[2]; return T; };

// a phone at `pos` in the scan, heading `head`, pitched `pitch`; what the server and the AR session say about that picture
function picture(J, pos, head, pitch, o = {}) {
  const glTour = at(A.mul(Ry(head), Rx(pitch)), pos);                                       // camera (GL axes) -> tour
  const found = at(A.mul(A.mul(Ry(head + (o.dyaw || 0)), Rx(pitch + (o.dtilt || 0))), FLIP), [pos[0] + (o.dx || 0), pos[1] + (o.dy || 0), pos[2] + (o.dz || 0)]);
  const glXr = at(A.mul(Ry(J.yaw), A.mul(Ry(head), Rx(pitch))), A.apply(J, pos));          // the phone's own pose, real metres
  return A.fix(A.invRigid(found), toCm(glXr));
}
const walk = (J, n, o = () => ({})) => Array.from({ length: n }, (_, k) => picture(J, [-2.5 + 5 * rnd(), 1.3 + 0.2 * rnd(), -1.4 + 2.8 * rnd()], 6.28 * rnd(), 0.3 * (rnd() - 0.5), o(k)));
const noise = () => ({ dx: 0.02 * gauss(), dy: 0.01 * gauss(), dz: 0.02 * gauss(), dyaw: 0.012 * gauss(), dtilt: 0.008 * gauss() });
const wrong = () => ({ dx: 3 * (rnd() - 0.5), dz: 3 * (rnd() - 0.5), dyaw: 3 * (rnd() - 0.5) });

test('one exact fix gives the join', () => {
  const J = { yaw: 0.7, s: 1, t: [1.5, -1.2, 0.4] }, f = picture(J, [1, 1.35, -0.5], 2.1, -0.2), S = A.solve([f]);
  assert.ok(f.tilt < 1e-4 && Math.abs(f.yaw - 0.7) < 1e-6);
  assert.ok(S && S.used === 1 && !S.scaled && Math.abs(S.yaw - 0.7) < 1e-6 && Math.hypot(...S.t.map((v, k) => v - J.t[k])) < 1e-6);
  const p = [2, 0, 1], q = A.apply(S, p), w = A.apply(J, p); assert.ok(Math.hypot(q[0] - w[0], q[1] - w[1], q[2] - w[2]) < 1e-6);
});

test('noisy fixes with a third of them wrong: the join, and the scale the scan was off by', () => {
  const J = { yaw: -2.2, s: 1.13, t: [-0.8, -1.25, 2.0] };
  const F = walk(J, 14, (k) => k % 3 === 2 ? wrong() : noise()), S = A.solve(F);
  assert.ok(S.used >= 8 && S.used <= 10, `used ${S.used}`); assert.ok(S.scaled && Math.abs(S.s - 1.13) < 0.03, `scale ${S.s}`);
  assert.ok(Math.abs(Math.atan2(Math.sin(S.yaw - J.yaw), Math.cos(S.yaw - J.yaw))) < 0.02 && S.cm < 6 && S.deg < 2, JSON.stringify(S));
  for (const p of [[2.5, 0, 1.4], [-2.5, 0, -1.4], [0, 2.7, 0]]) { const q = A.apply(S, p), w = A.apply(J, p); assert.ok(Math.hypot(q[0] - w[0], q[1] - w[1], q[2] - w[2]) < 0.08, 'a room corner is more than 8 cm off'); }
  assert.ok(S.cm1 > S.cm, 'the error with the scale held at 1 is reported and is larger here');
});

test('two fixes close together do not set the scale', () => {
  const J = { yaw: 0.3, s: 1.1, t: [0, -1.3, 0] }, S = A.solve([picture(J, [0, 1.35, 0], 0.5, 0), picture(J, [0.5, 1.35, 0.2], 1.5, 0)]);
  assert.ok(S && !S.scaled && S.s === 1);
});

test('a fix that tilts the room is thrown out', () => {
  const J = { yaw: 1.0, s: 1, t: [0.2, -1.3, 0.1] }, good = picture(J, [0, 1.35, 0], 0.5, 0), bad = picture(J, [1, 1.35, 1], 2.5, 0, { dtilt: 0.2 });
  assert.ok(bad.tilt > 10 && good.tilt < 0.01); const S = A.solve([bad, good]); assert.equal(S.used, 1); assert.ok(Math.abs(S.yaw - 1.0) < 1e-6);
  assert.equal(A.solve([bad]), null);
});

test('lens from the WebXR projection matrix, and the join as a matrix', () => {
  const w = 867, h = 1920, fx = 1331, fy = 1329, cx = 430.2, cy = 965.5, P = new Array(16).fill(0);
  P[0] = 2 * fx / w; P[5] = 2 * fy / h; P[8] = 1 - 2 * cx / w; P[9] = 2 * cy / h - 1; P[10] = -1; P[11] = -1; P[14] = -0.02;
  const K = A.intrinsics(P, w, h); assert.ok(Math.abs(K[0][0] - fx) < 1e-6 && Math.abs(K[1][1] - fy) < 1e-6 && Math.abs(K[0][2] - cx) < 1e-6 && Math.abs(K[1][2] - cy) < 1e-6);
  const J = { yaw: 0.9, s: 1.2, t: [1, 2, 3] }, m = A.joinCm(J), p = [0.4, 1.1, -2], q = A.apply(J, p);
  for (let r = 0; r < 3; r++) assert.ok(Math.abs(m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] - q[r]) < 1e-9);
});

test('shape-lock joins are believed only when three in a row agree', () => {
  const refs = [[0, 0, 0], [6, 0, 0], [6, 0, 4], [0, 0, 4]], J = (yaw, tx) => ({ yaw, s: 1, t: [tx, -1.3, 0.5] });
  assert.equal(A.steady([J(0.5, 1), J(0.5, 1.02)], refs), false, 'two are not enough');
  assert.equal(A.steady([J(0.5, 1), J(0.505, 1.02), J(0.5, 1.03)], refs), true);
  assert.equal(A.steady([J(2.0, 4), J(0.5, 1), J(0.505, 1.02), J(0.5, 1.03)], refs), true, 'an old wrong one does not count');
  assert.equal(A.steady([J(0.5, 1), J(0.5, 1.02), J(0.56, 1.03)], refs), false, 'turned 3.4 deg: a corner 7 m away moves 40 cm');
  assert.equal(A.steady([J(0.5, 1), J(0.5, 1.4), J(0.5, 1.02)], refs), false);
});
