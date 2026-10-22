// Synthetic check of coverage.js: a 4 x 5 m room, 2.5 m high; the camera turns around at 1.4 m, then walks a loop.
import { Coverage } from '../app/static/record/coverage.js';
const W = 64, H = 48, f = 1.2;   // p0 = p5 = f (normalised), ~80 deg fov
const room = { x0: -2, x1: 2, z0: -2.5, z1: 2.5, h: 2.5 };
function depthFrame(cx, cy, cz, yaw) {
  const d = new Uint16Array(W * H);
  const c = Math.cos(yaw), s = Math.sin(yaw);
  for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) {
    const vx = (q + 0.5) / W, vy = (r + 0.5) / H;
    const X = (2 * vx - 1) / f, Y = (1 - 2 * vy) / f;   // camera ray (x right, y up, -z forward), z = -1
    const dx = c * X - s * 1, dz = -s * X - c * 1, dy = Y;   // rotate about y by yaw
    let t = Infinity;
    for (const [p, o, v] of [[room.x0, cx, dx], [room.x1, cx, dx], [room.z0, cz, dz], [room.z1, cz, dz], [0, cy, dy], [room.h, cy, dy]]) {
      if (Math.abs(v) < 1e-9) continue; const tt = (p - o) / v; if (tt > 0.01 && tt < t) t = tt;
    }
    d[r * W + q] = Math.round(t * 1000);   // depth along -z axis equals t (ray has z component 1 in camera)
  }
  const pose = [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, cx, cy, cz, 1];
  return { bytes: new Uint8Array(d.buffer), width: W, height: H, rawToMeters: 0.001, normDepthFromView: null, pose, p0: f, p5: f, p8: 0, p9: 0 };
}
const cov = new Coverage();
let t = 0;
for (let i = 0; i < 24; i++) { const yaw = (i / 24) * 2 * Math.PI; cov.add(depthFrame(0, 1.4, 0, yaw), [0, 1.4, 0]); }
cov.findWalls((t += 3000)); console.log('after one turn on the spot:', cov.stats());
for (let i = 0; i < 48; i++) { const a = (i / 48) * 2 * Math.PI, x = 1.2 * Math.cos(a), z = 1.6 * Math.sin(a); cov.add(depthFrame(x, 1.4, z, a + Math.PI), [x, 1.4, z]); cov.add(depthFrame(x, 0.8, z, a + Math.PI / 2), [x, 0.8, z]); }
cov.findWalls((t += 3000)); console.log('after a walking loop at two heights:', cov.stats());
for (const w of cov.walls) console.log(' wall', w.a.map((v) => v.toFixed(2)), w.b.map((v) => v.toFixed(2)), 'top', w.top.toFixed(2));
