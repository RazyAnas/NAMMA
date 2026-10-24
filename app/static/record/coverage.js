// Live scan coverage for AR recording: ARCore depth frames are fused into a coarse voxel map while recording,
// and the map is drawn over the camera image so the user sees what is already scanned and what is left.
//   - each 8 cm surface voxel remembers from which directions it was seen (24 bins: 12 headings of 30 deg x 2 heights);
//     red = one direction, yellow = two, green = three or more (good for a photoreal model)
//   - unpainted surfaces have not been seen at all
//   - walls: every couple of seconds the map is searched for large vertical planes; each one is outlined from the
//     floor up, the outline drawing itself in from the corners when the wall is first found
// Pure math plus one small WebGL program; nothing here touches the recording itself.
import { depthFrameToWorld } from './quick-model.js';

const VOXEL = 0.08;
const WALL_BAND = 0.06;   // a point within this distance of a plane belongs to it
const GOOD_DIRECTIONS = 3;   // seen from three 30-degree headings (or two headings and two heights)
const WALL_EVERY_MS = 2000;
const WALL_MIN_VOXELS = 40;

const VS = `attribute vec3 p; attribute vec4 c; uniform mat4 vp; uniform float size; varying vec4 vc;
void main() { gl_Position = vp * vec4(p, 1.0); gl_PointSize = clamp(size / gl_Position.w, 2.0, 18.0); vc = c; }`;
const FS = `precision mediump float; varying vec4 vc; void main() { gl_FragColor = vc; }`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

function multiply(a, b) {   // column-major 4x4: a * b
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

function directionBin(dx, dy, dz) {   // where the camera looked from, seen from the voxel
  const heading = Math.floor(((Math.atan2(dz, dx) + Math.PI) / (2 * Math.PI)) * 12) % 12;
  const high = dy > 0.35 * Math.hypot(dx, dy, dz) ? 1 : 0;
  return heading + 12 * high;
}

const bits = (m) => { let n = 0; while (m) { n += m & 1; m >>>= 1; } return n; };

export class Coverage {
  constructor() {
    this.voxels = new Map();   // key -> { x, y, z, dirs, hits }
    this.walls = [];
    this.lastWalls = 0;
    this.dirty = true;
  }

  // One ARCore depth frame (same fields depthFrameToWorld takes) plus the camera position.
  add(frame, cameraPos) {
    const { positions } = depthFrameToWorld(frame, { stride: 4, minM: 0.25, maxM: 4 });
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i], y = positions[i + 1], z = positions[i + 2];
      const kx = Math.floor(x / VOXEL), ky = Math.floor(y / VOXEL), kz = Math.floor(z / VOXEL);
      const key = `${kx},${ky},${kz}`;
      let v = this.voxels.get(key);
      if (!v) { v = { x: (kx + 0.5) * VOXEL, y: (ky + 0.5) * VOXEL, z: (kz + 0.5) * VOXEL, dirs: 0, hits: 0 }; this.voxels.set(key, v); }
      v.hits++;
      v.dirs |= 1 << directionBin(cameraPos[0] - x, cameraPos[1] - y, cameraPos[2] - z);
    }
    this.dirty = true;
  }

  // Share of seen surface that is covered well enough (voxels seen at least twice, to ignore depth noise).
  stats() {
    let seen = 0, good = 0;
    for (const v of this.voxels.values()) {
      if (v.hits < 2) continue;
      seen++;
      if (bits(v.dirs) >= GOOD_DIRECTIONS) good++;
    }
    return { seen, good, percent: seen ? Math.round((100 * good) / seen) : 0, walls: this.walls.length };
  }

  // Large vertical planes: RANSAC on voxel centres, two points define a vertical plane (normal horizontal).
  findWalls(now) {
    if (now - this.lastWalls < WALL_EVERY_MS) return;
    this.lastWalls = now;
    let pts = [...this.voxels.values()].filter((v) => v.hits >= 2 && v.y > 0.3);
    const walls = [];
    for (let w = 0; w < 6 && pts.length > WALL_MIN_VOXELS; w++) {
      let best = null;
      // Score candidates on a sample so a wall search stays a few milliseconds on a phone.
      const sample = pts.length > 4000 ? Array.from({ length: 4000 }, () => pts[(Math.random() * pts.length) | 0]) : pts;
      const scale = pts.length / sample.length;
      for (let it = 0; it < 120; it++) {
        const a = pts[(Math.random() * pts.length) | 0], b = pts[(Math.random() * pts.length) | 0];
        const tx = b.x - a.x, tz = b.z - a.z, len = Math.hypot(tx, tz);
        if (len < 0.4) continue;
        const nx = -tz / len, nz = tx / len, d = -(nx * a.x + nz * a.z);
        let count = 0;
        for (const p of sample) if (Math.abs(nx * p.x + nz * p.z + d) < WALL_BAND) count += scale;
        if (!best || count > best.count) best = { nx, nz, d, count };
      }
      if (!best || best.count < WALL_MIN_VOXELS) break;
      const inliers = pts.filter((p) => Math.abs(best.nx * p.x + best.nz * p.z + best.d) < WALL_BAND);
      const tx = -best.nz, tz = best.nx;   // along the wall
      const ts = inliers.map((p) => p.x * tx + p.z * tz).sort((a, b) => a - b);
      const lo = ts[Math.floor(ts.length * 0.02)], hi = ts[Math.floor(ts.length * 0.98)];
      const ys = inliers.map((p) => p.y).sort((a, b) => a - b);
      const top = ys[ys.length - 1];
      const upright = ys[Math.floor(ys.length * 0.9)] - ys[Math.floor(ys.length * 0.1)] > 0.8;   // a real wall rises; a line across the ceiling does not
      const duplicate = walls.some((o) => Math.abs(o.nx * best.nx + o.nz * best.nz) > 0.95 && Math.abs(Math.sign(o.nx * best.nx + o.nz * best.nz) * o.d - best.d) < 0.2);
      if (hi - lo > 0.5 && upright && !duplicate) {
        const old = this.walls.find((o) => Math.abs(o.nx * best.nx + o.nz * best.nz) > 0.97 && Math.abs(o.d - best.d) < 0.15);
        const ox = -best.d * best.nx, oz = -best.d * best.nz;   // point on the wall closest to the origin
        walls.push({ nx: best.nx, nz: best.nz, d: best.d, top,
          a: [ox + tx * lo, oz + tz * lo], b: [ox + tx * hi, oz + tz * hi], born: old ? old.born : now });
      }
      pts = pts.filter((p) => Math.abs(best.nx * p.x + best.nz * p.z + best.d) >= 2.5 * WALL_BAND);
    }
    this.walls = walls;
  }

  // Draw into the XR layer's framebuffer, over the camera image, for one view.
  draw(gl, view, now) {
    if (!this.program) {
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VS));
      gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(program);
      this.program = program;
      this.loc = { p: gl.getAttribLocation(program, 'p'), c: gl.getAttribLocation(program, 'c'), vp: gl.getUniformLocation(program, 'vp'), size: gl.getUniformLocation(program, 'size') };
      this.points = gl.createBuffer();
      this.lines = gl.createBuffer();
    }
    if (this.dirty) {   // interleaved x y z r g b a
      const data = new Float32Array(this.voxels.size * 7);
      let i = 0;
      for (const v of this.voxels.values()) {
        if (v.hits < 2) continue;
        const n = bits(v.dirs);
        const [r, g, b] = n >= GOOD_DIRECTIONS ? [0.25, 0.9, 0.45] : n === 2 ? [1.0, 0.8, 0.2] : [1.0, 0.3, 0.25];
        data.set([v.x, v.y, v.z, r, g, b, 0.55], i);
        i += 7;
      }
      this.pointCount = i / 7;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.points);
      gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, i), gl.DYNAMIC_DRAW);
      this.dirty = false;
    }
    // Wall outlines: floor edge, two verticals, top edge, drawn in over 1.2 s from the bottom corners.
    const seg = [];
    for (const w of this.walls) {
      const k = Math.min(1, (now - w.born) / 1200);
      const pulse = 0.55 + 0.45 * Math.sin(now / 300);
      const col = [0.35, 0.85, 1.0, k < 1 ? 1 : pulse];
      const [ax, az] = w.a, [bx, bz] = w.b, mx = (ax + bx) / 2, mz = (az + bz) / 2;
      const lerp = (p, q, t) => p + (q - p) * t;
      seg.push(ax, 0, az, ...col, lerp(ax, mx, k), 0, lerp(az, mz, k), ...col);
      seg.push(bx, 0, bz, ...col, lerp(bx, mx, k), 0, lerp(bz, mz, k), ...col);
      seg.push(ax, 0, az, ...col, ax, w.top * k, az, ...col);
      seg.push(bx, 0, bz, ...col, bx, w.top * k, bz, ...col);
      if (k >= 1) seg.push(ax, w.top, az, ...col, bx, w.top, bz, ...col);
    }
    const vp = multiply(view.projectionMatrix, view.transform.inverse.matrix);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.loc.vp, false, vp);
    gl.uniform1f(this.loc.size, 40.0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const bind = (buffer) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(this.loc.p);
      gl.vertexAttribPointer(this.loc.p, 3, gl.FLOAT, false, 28, 0);
      gl.enableVertexAttribArray(this.loc.c);
      gl.vertexAttribPointer(this.loc.c, 4, gl.FLOAT, false, 28, 12);
    };
    if (this.pointCount) { bind(this.points); gl.drawArrays(gl.POINTS, 0, this.pointCount); }
    if (seg.length) {
      bind(this.lines);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(seg), gl.DYNAMIC_DRAW);
      gl.lineWidth(3);
      gl.drawArrays(gl.LINES, 0, seg.length / 7);
    }
    gl.disable(gl.BLEND);
    gl.disableVertexAttribArray(this.loc.c);
  }
}
