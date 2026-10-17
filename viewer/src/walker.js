// Game-style walking against reconstructed geometry.
//
// Everything here works in the "nav frame": metres, Y up, floor plane at
// y = 0, x/z along the NavMesh grid axes. A Rapier kinematic character
// controller moves a capsule against the calibrated collision mesh plus a
// terraced height field built from the NavMesh support heights, which gives
// gravity, step-up, slope limits and wall sliding. The NavMesh "standable"
// mask is applied on top so unseen space and thin obstacles that Poisson
// meshing dropped still block movement. No DOM or three.js dependency, so the
// same code runs in the browser and in the Node benchmark.

export const DEFAULTS = {
  radius: 0.25,
  height: 1.7,
  eyeHeight: 1.6,
  maxStep: 0.25,
  maxSlopeDegrees: 35,
  snapToGround: 0.3,
  gravity: 9.81,
  jumpSpeed: 3.2,
  skin: 0.02,
};

export async function inflate(base64, expectedBytes) {
  const binary = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
  if (typeof DecompressionStream === 'undefined') throw new Error('Compressed navigation data is not supported by this browser.');
  const stream = new Blob([binary]).stream().pipeThrough(new DecompressionStream('deflate'));
  const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  if (expectedBytes !== undefined && bytes.length !== expectedBytes) throw new Error('Navigation layer length mismatch.');
  return bytes;
}

export async function decodeMovementLayers(navigation) {
  const movement = navigation.movement;
  if (!movement) return null;
  const cells = navigation.width * navigation.height;
  const standable = await inflate(movement.standable_base64, cells);
  const raw = await inflate(movement.support_height_mm_base64, cells * 2);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const support = new Float32Array(cells);
  for (let index = 0; index < cells; index += 1) support[index] = view.getInt16(index * 2, true) / 1000;
  return { standable, support };
}

// Minimal GLB reader: concatenates every triangle primitive's POSITION and
// indices. The collision GLB is a single trimesh export, so this is enough.
export function readGlbTriangles(arrayBuffer) {
  const data = new DataView(arrayBuffer);
  if (data.getUint32(0, true) !== 0x46546c67) throw new Error('Collision model is not a GLB file.');
  let offset = 12;
  let json;
  let bin;
  while (offset < data.byteLength) {
    const length = data.getUint32(offset, true);
    const type = data.getUint32(offset + 4, true);
    const start = offset + 8;
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, start, length)));
    if (type === 0x004e4942) bin = { start, length };
    offset = start + length;
  }
  if (!json || !bin) throw new Error('Collision GLB is missing its JSON or binary chunk.');
  const componentArrays = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
  const sizes = { SCALAR: 1, VEC3: 3 };
  const read = (accessorIndex) => {
    const accessor = json.accessors[accessorIndex];
    const view = json.bufferViews[accessor.bufferView];
    const ArrayType = componentArrays[accessor.componentType];
    const components = sizes[accessor.type];
    const byteOffset = bin.start + (view.byteOffset || 0) + (accessor.byteOffset || 0);
    const stride = view.byteStride || components * ArrayType.BYTES_PER_ELEMENT;
    const out = new ArrayType(accessor.count * components);
    if (stride === components * ArrayType.BYTES_PER_ELEMENT && byteOffset % ArrayType.BYTES_PER_ELEMENT === 0) {
      out.set(new ArrayType(arrayBuffer, byteOffset, accessor.count * components));
    } else {
      const bytes = new Uint8Array(arrayBuffer);
      for (let item = 0; item < accessor.count; item += 1) {
        const source = new ArrayType(bytes.slice(byteOffset + item * stride, byteOffset + item * stride + components * ArrayType.BYTES_PER_ELEMENT).buffer);
        out.set(source, item * components);
      }
    }
    return out;
  };
  const positions = [];
  const indices = [];
  let base = 0;
  for (const mesh of json.meshes || []) {
    for (const primitive of mesh.primitives) {
      if (primitive.mode !== undefined && primitive.mode !== 4) continue;
      const position = read(primitive.attributes.POSITION);
      const index = primitive.indices !== undefined
        ? read(primitive.indices)
        : Uint32Array.from({ length: position.length / 3 }, (_, item) => item);
      positions.push(position);
      indices.push(Uint32Array.from(index, (value) => value + base));
      base += position.length / 3;
    }
  }
  const concat = (arrays, ArrayType) => {
    const out = new ArrayType(arrays.reduce((sum, array) => sum + array.length, 0));
    let cursor = 0;
    for (const array of arrays) { out.set(array, cursor); cursor += array.length; }
    return out;
  };
  return { positions: concat(positions, Float32Array), indices: concat(indices, Uint32Array) };
}

function transformPoint(matrix, [x, y, z]) {
  const w = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3];
  const divisor = Math.abs(w) > 1e-9 ? w : 1;
  return [0, 1, 2].map((row) => (matrix[row][0] * x + matrix[row][1] * y + matrix[row][2] * z + matrix[row][3]) / divisor);
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// Conversions between the splat viewer frame, the calibrated collision frame
// (metres, COLMAP orientation) and the nav frame.
export class NavFrame {
  constructor(navigation) {
    this.navigation = navigation;
    this.scale = navigation.source_units_to_meters;
    const { axis_x: ax, axis_y: ay, up, plane_offset: offset } = navigation.basis;
    Object.assign(this, { ax, ay, up, offset });
    const m = navigation.source_to_viewer;
    const viewerUp = [0, 1, 2].map((row) => m[row][0] * up[0] + m[row][1] * up[1] + m[row][2] * up[2]);
    const length = Math.hypot(...viewerUp) || 1;
    this.viewerUp = viewerUp.map((value) => value / length);
  }

  metersToNav(p) { return [dot(p, this.ax), dot(p, this.up) + this.offset, dot(p, this.ay)]; }

  navToMeters([x, y, z]) {
    const h = y - this.offset;
    return [0, 1, 2].map((i) => this.ax[i] * x + this.up[i] * h + this.ay[i] * z);
  }

  viewerToNav(v) {
    const source = transformPoint(this.navigation.viewer_to_source, v);
    return this.metersToNav(source.map((value) => value * this.scale));
  }

  navToViewer(n) {
    const source = this.navToMeters(n).map((value) => value / this.scale);
    return transformPoint(this.navigation.source_to_viewer, source);
  }

  // Direction vectors: drop translation, keep orientation only.
  viewerDirToNav(d) {
    const origin = this.viewerToNav([0, 0, 0]);
    const tip = this.viewerToNav(d);
    return [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]];
  }

  cellOf(x, z) {
    const n = this.navigation;
    return [
      Math.floor((x - n.origin_floor_coordinates[0]) / n.cell_size_m),
      Math.floor((z - n.origin_floor_coordinates[1]) / n.cell_size_m),
    ];
  }
}

// Collision GLB vertices are calibrated metres in COLMAP orientation.
export function collisionToNav(frame, { positions, indices }) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    out.set(frame.metersToNav([positions[i], positions[i + 1], positions[i + 2]]), i);
  }
  // The nav basis (axis_x, up, axis_y) is left-handed; flip winding so face
  // normals still point out of the surfaces.
  const flipped = new Uint32Array(indices);
  for (let i = 0; i < flipped.length; i += 3) [flipped[i + 1], flipped[i + 2]] = [indices[i + 2], indices[i + 1]];
  return { vertices: out, indices: flipped };
}

// Blocky terrain from per-cell support heights: a flat top per cell and
// vertical faces between neighbours, so a 16 cm stair is a real step for the
// autostep logic rather than a steep slope.
export function terraceMesh(navigation, support) {
  const { width, height, cell_size_m: cell, origin_floor_coordinates: [x0, z0] } = navigation;
  const vertices = [];
  const indices = [];
  const quad = (a, b, c, d) => {
    const base = vertices.length / 3;
    vertices.push(...a, ...b, ...c, ...d);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  };
  for (let gy = 0; gy < height; gy += 1) {
    for (let gx = 0; gx < width; gx += 1) {
      const h = support[gy * width + gx];
      const xa = x0 + gx * cell;
      const xb = xa + cell;
      const za = z0 + gy * cell;
      const zb = za + cell;
      quad([xa, h, za], [xa, h, zb], [xb, h, zb], [xb, h, za]);
      if (gx + 1 < width) {
        const other = support[gy * width + gx + 1];
        if (Math.abs(other - h) > 1e-3) quad([xb, Math.min(h, other), za], [xb, Math.max(h, other), za], [xb, Math.max(h, other), zb], [xb, Math.min(h, other), zb]);
      }
      if (gy + 1 < height) {
        const other = support[(gy + 1) * width + gx];
        if (Math.abs(other - h) > 1e-3) quad([xa, Math.min(h, other), zb], [xa, Math.max(h, other), zb], [xb, Math.max(h, other), zb], [xb, Math.min(h, other), zb]);
      }
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}

export class Walker {
  constructor(RAPIER, { navigation, layers, collision, options = {} }) {
    this.RAPIER = RAPIER;
    this.options = { ...DEFAULTS, ...options, radius: navigation.agent_radius_m ?? DEFAULTS.radius };
    this.navigation = navigation;
    this.frame = new NavFrame(navigation);
    this.layers = layers;
    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    const addTrimesh = ({ vertices, indices }) => {
      if (!indices.length) return;
      this.world.createCollider(RAPIER.ColliderDesc.trimesh(vertices, indices));
    };
    if (collision) addTrimesh(collisionToNav(this.frame, collision));
    if (layers) addTrimesh(terraceMesh(navigation, layers.support));

    const { radius, height, skin } = this.options;
    this.halfHeight = Math.max(0.05, height / 2 - radius);
    this.body = this.world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased());
    this.collider = this.world.createCollider(RAPIER.ColliderDesc.capsule(this.halfHeight, radius), this.body);
    this.controller = this.world.createCharacterController(skin);
    this.controller.setUp({ x: 0, y: 1, z: 0 });
    this.controller.setSlideEnabled(true);
    this.controller.enableAutostep(this.options.maxStep, radius * 0.5, false);
    this.controller.setMaxSlopeClimbAngle((this.options.maxSlopeDegrees * Math.PI) / 180);
    this.controller.setMinSlopeSlideAngle(((this.options.maxSlopeDegrees + 5) * Math.PI) / 180);
    this.controller.enableSnapToGround(this.options.snapToGround);
    this.verticalSpeed = 0;
    this.grounded = false;
    this.blockedByMask = 0;
    this.world.step();
  }

  get centerToEye() { return this.options.eyeHeight - this.options.height / 2; }

  standable(x, z) {
    if (!this.layers) return true;
    const [gx, gy] = this.frame.cellOf(x, z);
    const { width, height } = this.navigation;
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) return false;
    return this.layers.standable[gy * width + gx] === 1;
  }

  supportAt(x, z) {
    if (!this.layers) return 0;
    const [gx, gy] = this.frame.cellOf(x, z);
    const { width, height } = this.navigation;
    if (gx < 0 || gx >= width || gy < 0 || gy >= height) return 0;
    return this.layers.support[gy * width + gx];
  }

  // Place the feet on the ground at (x, z) in the nav frame.
  placeFeet(x, z, feetY = this.supportAt(x, z)) {
    const center = { x, y: feetY + this.options.height / 2 + this.options.skin, z };
    this.body.setTranslation(center, true);
    this.body.setNextKinematicTranslation(center);
    this.collider.setTranslation(center);
    this.verticalSpeed = 0;
    this.world.step();
  }

  spawn() {
    const [gx, gy] = this.navigation.spawn.grid;
    const { cell_size_m: cell, origin_floor_coordinates: [x0, z0] } = this.navigation;
    this.placeFeet(x0 + (gx + 0.5) * cell, z0 + (gy + 0.5) * cell);
  }

  get center() { return this.body.translation(); }

  get eye() {
    const c = this.center;
    return [c.x, c.y + this.centerToEye, c.z];
  }

  get feet() {
    const c = this.center;
    return [c.x, c.y - this.options.height / 2, c.z];
  }

  // velocity: desired horizontal velocity [vx, vz] in m/s (nav frame).
  step(velocity, dt, jump = false) {
    if (this.grounded && jump) this.verticalSpeed = this.options.jumpSpeed;
    this.verticalSpeed = this.grounded && this.verticalSpeed <= 0 ? -0.5 : this.verticalSpeed - this.options.gravity * dt;
    this.verticalSpeed = Math.max(this.verticalSpeed, -20);
    const desired = { x: velocity[0] * dt, y: this.verticalSpeed * dt, z: velocity[1] * dt };
    this.controller.computeColliderMovement(this.collider, desired);
    const moved = this.controller.computedMovement();
    this.grounded = this.controller.computedGrounded();
    const c = this.center;
    // Mask check on the corrected motion, then per-axis so the agent slides
    // along the edge of known space instead of sticking to it.
    // Poisson meshing can invent smooth bulges (over a glass table, between
    // furniture) that the controller would happily climb; the observed floor
    // height is the arbiter, so standing more than a step above it is refused.
    const feetY = c.y + moved.y - this.options.height / 2;
    const allowed = (px, pz) => this.standable(px, pz) && feetY <= this.supportAt(px, pz) + this.options.maxStep + 0.05;
    let { x, z } = moved;
    if (!allowed(c.x + x, c.z + z)) {
      this.blockedByMask += 1;
      if (allowed(c.x + x, c.z)) z = 0;
      else if (allowed(c.x, c.z + z)) x = 0;
      else { x = 0; z = 0; }
    }
    const next = { x: c.x + x, y: c.y + moved.y, z: c.z + z };
    if (this.grounded && this.verticalSpeed < 0) this.verticalSpeed = 0;
    this.body.setNextKinematicTranslation(next);
    this.world.step();
    return next;
  }
}
