// Capture-bundle writing helpers with no browser dependencies, so Node can test them.
//
// A bundle is an uncompressed ZIP (video is already compressed; storing keeps the phone's CPU
// free for the encoder) holding the encoded video stream, per-frame camera poses, raw IMU
// samples and a metadata file. scripts/ingest_sensor_bundle.py reads it on the laptop.

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes, crc = 0) {
  let c = ~crc >>> 0;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

const encoder = new TextEncoder();

// One growing file inside the bundle. Chunks are kept as they arrive and the CRC is updated
// incrementally, so the video never has to be copied into one large buffer on the phone.
export class BundleFile {
  constructor(name) {
    this.name = name;
    this.parts = [];
    this.size = 0;
    this.crc = 0;
  }

  write(data) {
    const bytes = typeof data === 'string' ? encoder.encode(data) : data instanceof Uint8Array ? data : new Uint8Array(data);
    if (!bytes.length) return;
    this.parts.push(bytes);
    this.size += bytes.length;
    this.crc = crc32(bytes, this.crc);
  }
}

// Store-only ZIP (method 0, no ZIP64): every member and the whole archive must stay under 4 GiB,
// far above a few minutes of phone video.
export function zipParts(files) {
  const out = [];
  const central = [];
  let offset = 0;
  const { time, date } = dosDateTime(new Date());
  for (const file of files) {
    const name = encoder.encode(file.name);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true); // UTF-8 names
    local.setUint16(8, 0, true);
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, file.crc, true);
    local.setUint32(18, file.size, true);
    local.setUint32(22, file.size, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    out.push(new Uint8Array(local.buffer), name, ...file.parts);

    const entry = new DataView(new ArrayBuffer(46));
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);
    entry.setUint16(6, 20, true);
    entry.setUint16(8, 0x0800, true);
    entry.setUint16(10, 0, true);
    entry.setUint16(12, time, true);
    entry.setUint16(14, date, true);
    entry.setUint32(16, file.crc, true);
    entry.setUint32(20, file.size, true);
    entry.setUint32(24, file.size, true);
    entry.setUint16(28, name.length, true);
    entry.setUint32(42, offset, true);
    central.push(new Uint8Array(entry.buffer), name);
    offset += 30 + name.length + file.size;
  }
  const centralSize = central.reduce((sum, part) => sum + part.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);
  return [...out, ...central, new Uint8Array(end.buffer)];
}

function dosDateTime(d) {
  return {
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
  };
}

// IVF is the simplest container FFmpeg reads for VP8/VP9 elementary streams (32-byte file header,
// 12-byte frame headers). H.264 is written as a raw Annex B stream instead, which needs no container.
export function ivfHeader(fourcc, width, height, timebaseDen = 1_000_000) {
  const view = new DataView(new ArrayBuffer(32));
  'DKIF'.split('').forEach((c, i) => view.setUint8(i, c.charCodeAt(0)));
  view.setUint16(4, 0, true);
  view.setUint16(6, 32, true);
  fourcc.split('').forEach((c, i) => view.setUint8(8 + i, c.charCodeAt(0)));
  view.setUint16(12, width, true);
  view.setUint16(14, height, true);
  view.setUint32(16, timebaseDen, true); // rate: timestamps are in microseconds
  view.setUint32(20, 1, true); // scale
  view.setUint32(24, 0, true); // frame count, unknown while streaming; FFmpeg ignores it
  return new Uint8Array(view.buffer);
}

export function ivfFrameHeader(size, timestampUs) {
  const view = new DataView(new ArrayBuffer(12));
  view.setUint32(0, size, true);
  view.setBigUint64(4, BigInt(Math.max(0, Math.round(timestampUs))), true);
  return new Uint8Array(view.buffer);
}

export function csvRow(values) {
  return `${values.map((v) => (v == null || Number.isNaN(v) ? '' : typeof v === 'number' ? +v.toPrecision(9) : v)).join(',')}\n`;
}

// WebXR projection matrix (column-major, OpenGL clip space) to pixel intrinsics of a width x height
// image with the origin at the top-left pixel corner and y pointing down (OpenCV / COLMAP).
export function intrinsicsFromProjection(p, width, height) {
  return {
    fx: (p[0] * width) / 2,
    fy: (p[5] * height) / 2,
    cx: ((1 - p[8]) * width) / 2,
    cy: ((1 + p[9]) * height) / 2,
  };
}
