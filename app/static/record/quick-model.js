// On-phone quick 3D model: ARCore poses + ARCore depth (WebXR depth-sensing) fused into a coloured
// voxel point cloud, in seconds and with no network. No browser APIs beyond Blob, so Node can test it.
//
// Coordinate chain for one depth pixel (all conventions as recorded by record.js):
//   depth buffer uv (origin top-left, row-major data)  --inverse(normDepthBufferFromNormView)-->
//   normalised view xy (origin top-left)  --projection p0 p5 p8 p9-->  OpenGL camera (x right, y up, -z forward)
//   --frames.csv m0..m15 (world_from_camera)-->  WebXR local-floor world (metres, y up, floor at y = 0).
// The laptop's splat is trained on the same ARCore poses, so both models share this world frame.

const decoder = new TextDecoder();

// ---- Bundle (store-only ZIP written by bundle.js) -------------------------------------------------

export async function readZipIndex(blob) {
  // End of central directory sits in the last 22 bytes (no archive comment is ever written).
  const tail = new DataView(await blob.slice(Math.max(0, blob.size - 22)).arrayBuffer());
  if (tail.getUint32(0, true) !== 0x06054b50) throw new Error('Not a NAMMA bundle (zip end record missing).');
  const count = tail.getUint16(10, true);
  const size = tail.getUint32(12, true);
  const offset = tail.getUint32(16, true);
  const dir = new DataView(await blob.slice(offset, offset + size).arrayBuffer());
  const entries = new Map();
  let p = 0;
  for (let i = 0; i < count; i++) {
    const method = dir.getUint16(p + 10, true);
    const compressed = dir.getUint32(p + 20, true);
    const nameLength = dir.getUint16(p + 28, true);
    const extra = dir.getUint16(p + 30, true);
    const comment = dir.getUint16(p + 32, true);
    const local = dir.getUint32(p + 42, true);
    const name = decoder.decode(new Uint8Array(dir.buffer, dir.byteOffset + p + 46, nameLength));
    entries.set(name, { method, size: compressed, local });
    p += 46 + nameLength + extra + comment;
  }
  return entries;
}

export async function readEntry(blob, entries, name) {
  const entry = entries.get(name);
  if (!entry) return null;
  const header = new DataView(await blob.slice(entry.local, entry.local + 30).arrayBuffer());
  const start = entry.local + 30 + header.getUint16(26, true) + header.getUint16(28, true);
  const bytes = new Uint8Array(await blob.slice(start, start + entry.size).arrayBuffer());
  if (entry.method === 0) return bytes;
  if (entry.method === 8 && typeof DecompressionStream !== 'undefined') {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  throw new Error(`${name}: zip method ${entry.method} is not supported.`);
}

export function parseCsv(text) {
  const lines = text.split('\n').filter((line) => line.length);
  const header = lines.shift().split(',');
  return lines.map((line) => {
    const values = line.split(',');
    const row = {};
    header.forEach((key, i) => {
      const value = values[i];
      row[key] = value === undefined || value === '' ? null : Number.isNaN(Number(value)) ? value : Number(value);
    });
    return row;
  });
}

// ---- Geometry -------------------------------------------------------------------------------------

// 2D affine part of a column-major 4x4 that maps (x, y, 0, 1) -> (u, v); returns its inverse.
function invertUvMatrix(n) {
  const a = n[0], b = n[4], c = n[12];
  const d = n[1], e = n[5], f = n[13];
  const det = a * e - b * d;
  if (Math.abs(det) < 1e-12) return null;
  return { a: e / det, b: -b / det, c: (b * f - c * e) / det, d: -d / det, e: a / det, f: (c * d - a * f) / det };
}

function depthReader(bytes, width, height) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length === width * height * 2) return (i) => view.getUint16(i * 2, true);
  if (bytes.length === width * height * 4) return (i) => view.getFloat32(i * 4, true);
  return null;
}

// Back-project one depth frame into world points. `color` is an optional camera snapshot
// { width, height, rgb } read with gl.readPixels, so its first row is the bottom of the view.
export function depthFrameToWorld(frame, options = {}) {
  const { stride = 2, minM = 0.15, maxM = 5 } = options;
  const { bytes, width, height, rawToMeters, normDepthFromView, pose, p0, p5, p8, p9, color } = frame;
  const read = depthReader(bytes, width, height);
  const inv = normDepthFromView && normDepthFromView[0] != null ? invertUvMatrix(normDepthFromView) : { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };
  if (!read || !inv || !pose || !p0 || !p5) return { positions: new Float32Array(0), colors: new Uint8Array(0) };
  const positions = [];
  const colors = [];
  for (let row = 0; row < height; row += stride) {
    for (let col = 0; col < width; col += stride) {
      const d = read(row * width + col) * rawToMeters;
      if (!(d >= minM && d <= maxM)) continue;
      const u = (col + 0.5) / width;
      const v = (row + 0.5) / height;
      const vx = inv.a * u + inv.b * v + inv.c;
      const vy = inv.d * u + inv.e * v + inv.f;
      if (vx < 0 || vx > 1 || vy < 0 || vy > 1) continue;
      const ndcX = 2 * vx - 1;
      const ndcY = 1 - 2 * vy;
      const X = (d * (ndcX + p8)) / p0;
      const Y = (d * (ndcY + p9)) / p5;
      const Z = -d;
      positions.push(
        pose[0] * X + pose[4] * Y + pose[8] * Z + pose[12],
        pose[1] * X + pose[5] * Y + pose[9] * Z + pose[13],
        pose[2] * X + pose[6] * Y + pose[10] * Z + pose[14]
      );
      if (color) {
        const cx = Math.min(color.width - 1, Math.floor(vx * color.width));
        const cy = Math.min(color.height - 1, Math.floor((1 - vy) * color.height));
        const k = (cy * color.width + cx) * 3;
        colors.push(color.rgb[k], color.rgb[k + 1], color.rgb[k + 2]);
      } else {
        colors.push(0, 0, 0);
      }
    }
  }
  return { positions: new Float32Array(positions), colors: new Uint8Array(colors), coloured: !!color };
}

// Averages points into a sparse voxel grid; a voxel seen by fewer than `minHits` points is dropped
// as depth noise. Keys pack (ix, iy, iz) in [-1024, 1023] voxels each, so +-30 m at 3 cm.
export class VoxelFusion {
  constructor(voxel = 0.03) {
    this.voxel = voxel;
    this.index = new Map();
    this.sum = [];
    this.hits = [];
  }

  add(positions, colors) {
    const s = 1 / this.voxel;
    for (let i = 0; i < positions.length; i += 3) {
      const ix = Math.floor(positions[i] * s) + 1024;
      const iy = Math.floor(positions[i + 1] * s) + 1024;
      const iz = Math.floor(positions[i + 2] * s) + 1024;
      if (ix < 0 || iy < 0 || iz < 0 || ix > 2047 || iy > 2047 || iz > 2047) continue;
      const key = (ix * 2048 + iy) * 2048 + iz;
      let slot = this.index.get(key);
      if (slot === undefined) {
        slot = this.hits.length;
        this.index.set(key, slot);
        this.hits.push(0);
        this.sum.push(0, 0, 0, 0, 0, 0);
      }
      this.hits[slot]++;
      const o = slot * 6;
      this.sum[o] += positions[i];
      this.sum[o + 1] += positions[i + 1];
      this.sum[o + 2] += positions[i + 2];
      this.sum[o + 3] += colors[i];
      this.sum[o + 4] += colors[i + 1];
      this.sum[o + 5] += colors[i + 2];
    }
  }

  result(minHits = 2) {
    const keep = [];
    for (let slot = 0; slot < this.hits.length; slot++) if (this.hits[slot] >= minHits) keep.push(slot);
    const positions = new Float32Array(keep.length * 3);
    const colors = new Uint8Array(keep.length * 3);
    keep.forEach((slot, j) => {
      const n = this.hits[slot];
      for (let k = 0; k < 3; k++) {
        positions[j * 3 + k] = this.sum[slot * 6 + k] / n;
        colors[j * 3 + k] = Math.round(this.sum[slot * 6 + 3 + k] / n);
      }
    });
    return { positions, colors };
  }
}

// Colour by height when the bundle has no camera snapshots (older recordings): floor blue, ceiling warm.
export function heightColours(positions) {
  const colors = new Uint8Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    const t = Math.max(0, Math.min(1, positions[i + 1] / 2.6));
    colors[i] = Math.round(60 + 190 * t);
    colors[i + 1] = Math.round(140 + 60 * Math.sin(Math.PI * t));
    colors[i + 2] = Math.round(230 - 170 * t);
  }
  return colors;
}

// ---- Whole bundle ---------------------------------------------------------------------------------

export async function buildQuickModel(blob, options = {}) {
  const { voxel = 0.03, stride = 2, minHits = 2, onProgress = () => {} } = options;
  const entries = await readZipIndex(blob);
  const meta = JSON.parse(decoder.decode(await readEntry(blob, entries, 'meta.json')));
  const frames = parseCsv(decoder.decode(await readEntry(blob, entries, 'frames.csv')));
  const byIndex = new Map(frames.map((row) => [row.frame, row]));
  const path = [];
  for (const row of frames) if (row.tracking === 0) path.push(row.m12, row.m13, row.m14);
  const depthCsv = await readEntry(blob, entries, 'depth.csv');
  const depthRows = depthCsv ? parseCsv(decoder.decode(depthCsv)) : [];
  const colourSize = meta.depth_color_size || null;
  const fusion = new VoxelFusion(voxel);
  let used = 0;
  let coloured = 0;
  for (let i = 0; i < depthRows.length; i++) {
    const d = depthRows[i];
    const pose = byIndex.get(d.frame);
    onProgress((i + 1) / Math.max(1, depthRows.length));
    if (!pose || pose.tracking !== 0) continue;
    const bytes = await readEntry(blob, entries, d.file);
    if (!bytes) continue;
    let color = null;
    if (colourSize) {
      const rgb = await readEntry(blob, entries, d.file.replace(/\.bin$/, '.rgb'));
      if (rgb && rgb.length === colourSize[0] * colourSize[1] * 3) color = { width: colourSize[0], height: colourSize[1], rgb };
    }
    const points = depthFrameToWorld(
      {
        bytes,
        width: d.width,
        height: d.height,
        rawToMeters: d.raw_to_m,
        normDepthFromView: Array.from({ length: 16 }, (_, k) => d[`n${k}`]),
        pose: Array.from({ length: 16 }, (_, k) => pose[`m${k}`]),
        p0: pose.p0,
        p5: pose.p5,
        p8: pose.p8,
        p9: pose.p9,
        color,
      },
      { stride }
    );
    fusion.add(points.positions, points.colors);
    used++;
    if (points.coloured) coloured++;
  }
  const cloud = fusion.result(used > 3 ? minHits : 1);
  if (!coloured) cloud.colors = heightColours(cloud.positions);
  return {
    ...cloud,
    path: new Float32Array(path),
    stats: {
      mode: meta.mode,
      depth_frames: depthRows.length,
      depth_frames_used: used,
      coloured_frames: coloured,
      points: cloud.positions.length / 3,
      voxel_m: voxel,
      tracked_frames: path.length / 3,
      frames: frames.length,
    },
  };
}

// Binary little-endian PLY of the quick model, so the laptop or any viewer can open it.
export function quickModelPly({ positions, colors }) {
  const n = positions.length / 3;
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const head = new TextEncoder().encode(header);
  const body = new DataView(new ArrayBuffer(n * 15));
  for (let i = 0; i < n; i++) {
    body.setFloat32(i * 15, positions[i * 3], true);
    body.setFloat32(i * 15 + 4, positions[i * 3 + 1], true);
    body.setFloat32(i * 15 + 8, positions[i * 3 + 2], true);
    body.setUint8(i * 15 + 12, colors[i * 3]);
    body.setUint8(i * 15 + 13, colors[i * 3 + 1]);
    body.setUint8(i * 15 + 14, colors[i * 3 + 2]);
  }
  return new Blob([head, body.buffer], { type: 'application/octet-stream' });
}
