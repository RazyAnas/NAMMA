// On-site AR: joining the phone's own AR world to the scan. No DOM, no WebXR calls: the AR page pastes this file in, Node tests
// import it (tests/browser/ar_core_test.mjs).
//
// Two worlds, both upright (y up, metres):
//   tour  the scan's room coordinates (walls.json, nav.json, pois.json; scale is a guess: phone height taken as 1.35 m)
//   xr    the phone's AR session (WebXR 'local': origin where the session started, real metres)
// A "fix" is one camera picture located in the scan by the server (DRS-VPT: world-to-camera T_w2c in tour coordinates, camera
// axes x right, y down, z forward) together with the phone's own pose for that same picture (WebXR view.transform.matrix:
// camera-to-xr, camera axes x right, y up, looking down -z). One fix already gives the join; the server is right only for
// about half to two thirds of pictures and says nothing about its confidence, so the join is the one most fixes agree on:
//   p_xr = s * Ry(yaw) * p_tour + t        (one turn about the vertical, one shift, one scale)
// A fix that would need the scan tilted is wrong by exactly that tilt (gravity is known on both sides): thrown out above 4 deg.
// Scale is held at 1 until the fixes span 2.5 m, because two fixes 1 m apart with 5 cm noise put it off by 7 %.
// All matrices here are row-major arrays of 4 rows unless a name says `cm` (column-major, 16 numbers, as WebXR gives them).

export const mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
export const fromCm = (m) => [0, 1, 2, 3].map(r => [0, 1, 2, 3].map(c => m[c * 4 + r]));
export const invRigid = (T) => { const R = [0, 1, 2].map(i => [0, 1, 2].map(j => T[j][i])), t = R.map(r => -(r[0] * T[0][3] + r[1] * T[1][3] + r[2] * T[2][3]));
  return [[...R[0], t[0]], [...R[1], t[1]], [...R[2], t[2]], [0, 0, 0, 1]]; };
const FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]];                       // camera axes: OpenCV <-> OpenGL

// Lens of the AR camera picture from the WebXR projection matrix (column-major) and the picture's own size (view.camera
// width / height, not the screen): pixels, origin top left, y down. Measured on the test phone: 867 x 1920, f 1331 (SENSOR_CAPTURE.md).
export function intrinsics(P, w, h) { return [[P[0] * w / 2, 0, (1 - P[8]) * w / 2], [0, P[5] * h / 2, (1 + P[9]) * h / 2], [0, 0, 1]]; }

// One fix: where the scan would have to sit in the AR world for this picture to be right.
export function fix(T_w2c, viewCm, at = 0) {
  const tourFromGl = mul(invRigid(T_w2c), FLIP), xrFromGl = fromCm(viewCm), S = mul(xrFromGl, invRigid(tourFromGl));   // xr <- tour
  const up = [S[0][1], S[1][1], S[2][1]], tilt = Math.acos(Math.max(-1, Math.min(1, up[1]))) * 180 / Math.PI;
  return { at, tilt, yaw: Math.atan2(S[0][2] - S[2][0], S[0][0] + S[2][2]),                 // the turn about y nearest to S's rotation
    tour: [tourFromGl[0][3], tourFromGl[1][3], tourFromGl[2][3]], xr: [xrFromGl[0][3], xrFromGl[1][3], xrFromGl[2][3]] };
}

const ry = (a, p) => [Math.cos(a) * p[0] + Math.sin(a) * p[2], p[1], -Math.sin(a) * p[0] + Math.cos(a) * p[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], len = (a) => Math.hypot(a[0], a[1], a[2]);
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a)), med = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; };
export const apply = (J, p) => { const q = ry(J.yaw, p); return [J.s * q[0] + J.t[0], J.s * q[1] + J.t[1], J.s * q[2] + J.t[2]]; };
// The same join as a column-major 4x4 for the renderer (model matrix of everything drawn in tour coordinates).
export const joinCm = (J) => { const c = Math.cos(J.yaw) * J.s, s = Math.sin(J.yaw) * J.s; return [c, 0, -s, 0, 0, J.s, 0, 0, s, 0, c, 0, J.t[0], J.t[1], J.t[2], 1]; };

// The join most fixes agree on. Returns null without a usable fix, else {yaw, s, t, used, of, cm, deg, span, scaled, cm1}:
// used / of = fixes that agree / fixes given, cm and deg = how far the agreeing fixes are from the join (median), span = how
// far apart they are (m), scaled = whether s was measured (else 1), cm1 = the same error with s held at 1 (a free scale can hide error).
export function solve(fixes, o = {}) {
  const tol = o.tol ?? 0.15, tolDeg = (o.tolDeg ?? 5) * Math.PI / 180, F = fixes.filter(f => f.tilt <= (o.tilt ?? 4));
  if (!F.length) return null;
  const err = (J, f) => [len(sub(apply(J, f.tour), f.xr)), Math.abs(wrap(f.yaw - J.yaw))];
  const fit = (yaw, s, set) => { const t = [0, 0, 0]; set.forEach(f => { const q = ry(yaw, f.tour); for (let k = 0; k < 3; k++) t[k] += (f.xr[k] - s * q[k]) / set.length; }); return { yaw, s, t }; };
  const agree = (J) => F.filter(f => { const e = err(J, f); return e[0] <= tol && e[1] <= tolDeg; });
  let best = null;
  const tryJ = (J) => { const a = agree(J); if (!a.length) return; const e = med(a.map(f => err(J, f)[0]));
    if (!best || a.length > best.a.length || (a.length === best.a.length && e < best.e)) best = { J, a, e }; };
  for (const f of F) tryJ(fit(f.yaw, 1, [f]));                                             // one fix, scale 1
  for (let i = 0; i < F.length; i++) for (let j = i + 1; j < F.length; j++) {              // two fixes far enough apart: scale too
    const dt = len(sub(F[j].tour, F[i].tour)), dx = len(sub(F[j].xr, F[i].xr));
    if (dt < 1 || dx / dt < 0.7 || dx / dt > 1.4) continue;
    const yaw = Math.atan2(Math.sin(F[i].yaw) + Math.sin(F[j].yaw), Math.cos(F[i].yaw) + Math.cos(F[j].yaw)); tryJ(fit(yaw, dx / dt, [F[i], F[j]]));
  }
  if (!best) return null;
  let a = best.a, span = 0; const ratios = [];
  for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) { const dt = len(sub(a[j].tour, a[i].tour)); span = Math.max(span, dt);
    if (dt >= 1) ratios.push(len(sub(a[j].xr, a[i].xr)) / dt); }
  const yaw = Math.atan2(a.reduce((s, f) => s + Math.sin(f.yaw), 0), a.reduce((s, f) => s + Math.cos(f.yaw), 0));
  const scaled = a.length >= 3 && span >= 2.5 && ratios.length > 0, s = scaled ? Math.max(0.8, Math.min(1.25, med(ratios))) : 1;
  const J = fit(yaw, s, a), J1 = fit(yaw, 1, a);
  return { ...J, used: a.length, of: fixes.length, cm: med(a.map(f => err(J, f)[0])) * 100, deg: med(a.map(f => err(J, f)[1])) * 180 / Math.PI,
    span, scaled, cm1: med(a.map(f => err(J1, f)[0])) * 100 };
}

// The shape lock (ar_geo.js) gives a join about once a second. One alone is not to be believed (measured: no score tells a right
// match from a wrong one); the last k that put the same points (the room's corners, tour coordinates) within tol metres are.
export function steady(joins, refs, k = 3, tol = 0.15) {
  if (joins.length < k) return false; const last = joins[joins.length - 1];
  for (let i = joins.length - k; i < joins.length - 1; i++) for (const p of refs) { const a = apply(joins[i], p), b = apply(last, p);
    if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > tol) return false; }
  return true;
}
