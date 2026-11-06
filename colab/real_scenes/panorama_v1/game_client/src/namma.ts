// NAMMA additions to the vendored third-person template (MIT, see ../UPSTREAM.txt).
// Everything world-specific comes from game.json written by export_game.py:
// metric y-up splats (.rad LoD or .ply), TSDF collider, sky dome, panorama probes for lighting.
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';

const params = new URLSearchParams(location.search);
/** Folder holding game.json and its assets (default: the site root). */
export const WORLD_BASE = (params.get('world') || './world/').replace(/\/?$/, '/');

type GameJson = {
  units?: string;
  files: { splats: string; rad?: string; collider: string; ground_mesh?: string; sky?: string | null; surface?: string | null };
  spawn?: { pos: number[]; look_at: number[] };
  route?: number[][];
  ibl?: { file: string; pos: number[] }[];
  character?: { glb: string; clips: { idle: string; air: string; walk: string; trick: string } };
};

export const GAME: GameJson = await (await fetch(WORLD_BASE + 'game.json')).json();
const metric = (GAME.units || 'm') === 'm';

export const NAMMA = {
  metric,
  splatUrl: WORLD_BASE + (params.get('splats') || GAME.files.rad || GAME.files.splats),
  get splatIsRad() { return this.splatUrl.endsWith('.rad'); },
  colliderUrl: WORLD_BASE + GAME.files.collider,
  skyUrl: GAME.files.sky ? WORLD_BASE + GAME.files.sky : null,
  surfaceUrl: GAME.files.surface && !params.has('nosurface') ? WORLD_BASE + GAME.files.surface : null,
  spawn: GAME.spawn ?? { pos: [0, 1.7, 0], look_at: [0, 1.7, -1] },
  route: GAME.route ?? [],
  probes: (GAME.ibl ?? []).map((p) => ({ url: WORLD_BASE + p.file, pos: p.pos })),
  // CC0 avatar (three.js RobotExpressive by Tomás Laulhé / Quaternius) until VRM avatars land
  character: GAME.character ?? {
    glb: './avatars/RobotExpressive.glb',
    clips: { idle: 'Idle', air: 'Jump', walk: 'Walking', trick: 'Dance' },
  },
  /** Metric human scale instead of the template's 4-unit dog. */
  capsuleHeight: metric ? 1.75 : 4,
  tuning: metric
    ? {
        moveSpeed: 1.7, sprintMoveMultiplier: 2.6, jumpSpeed: 4.6, gravityY: -14,
        splatUniformScale: 1, colliderGlbUniformScale: 1,
        // the world is a real photo: no grading, only a light vignette
        ppBloomIntensity: 0.0, ppBrightness: 0, ppContrast: 0, ppVignetteDarkness: 0.35,
        shadowCameraHalfExtent: 12, shadowCameraFar: 80, sunPosX: -6, sunPosY: 30, sunPosZ: 5,
        colliderGlbShadowOpacity: 0.35,
      }
    : {},
};

// ── sky dome: equirect in the game frame, row 0 = straight up (export_game.game_dirs) ──
export async function addSkyDome(scene: THREE.Scene, camera: THREE.Camera) {
  if (!NAMMA.skyUrl) return null;
  const tex = await new THREE.TextureLoader().loadAsync(NAMMA.skyUrl);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(3000, 64, 32),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, uniforms: { map: { value: tex } },
      vertexShader: `varying vec3 vDir;
        void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D map; varying vec3 vDir;
        void main(){
          vec3 d = normalize(vDir);
          float u = 0.5 + atan(d.x, -d.z) / 6.2831853;
          float v = 0.5 + asin(clamp(d.y, -1.0, 1.0)) / 3.1415927;
          gl_FragColor = texture2D(map, vec2(u, v));
          #include <colorspace_fragment>
        }`,
    }),
  );
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  camera.add(sky);
  scene.add(camera);
  return sky;
}

// ── textured base mesh: crisp from any angle, visible only where splats are thin ──
export async function addSurface(scene: THREE.Scene) {
  if (!NAMMA.surfaceUrl) return null;
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const gltf = await new GLTFLoader().loadAsync(NAMMA.surfaceUrl);
  gltf.scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const old = mesh.material as THREE.MeshStandardMaterial;
    const map = old.map ?? null;
    if (map) { map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = 8; }
    // unlit photo texture; pushed back in depth so splats on the same surface always win
    mesh.material = new THREE.MeshBasicMaterial({
      map, toneMapped: false, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8,
    });
    mesh.receiveShadow = false;
    mesh.castShadow = false;
    old.dispose();
  });
  scene.add(gltf.scene);
  return gltf.scene;
}

// ── image-based lighting from the nearest real panorama ──
export class ProbeLighting {
  private cache = new Map<string, THREE.Texture>();
  private current: string | null = null;
  private loading = false;
  constructor(private renderer: THREE.WebGLRenderer, private onEnv: (t: THREE.Texture) => void) {}
  update(p: { x: number; z: number }) {
    if (!NAMMA.probes.length || this.loading) return;
    let best = NAMMA.probes[0];
    let bd = Infinity;
    for (const q of NAMMA.probes) {
      const d = (q.pos[0] - p.x) ** 2 + (q.pos[2] - p.z) ** 2;
      if (d < bd) { bd = d; best = q; }
    }
    if (best.url === this.current) return;
    const hit = this.cache.get(best.url);
    if (hit) { this.current = best.url; this.onEnv(hit); return; }
    this.loading = true;
    new THREE.TextureLoader().load(best.url, (tex) => {
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.colorSpace = THREE.SRGBColorSpace;
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      const env = pmrem.fromEquirectangular(tex).texture;
      tex.dispose(); pmrem.dispose();
      this.cache.set(best.url, env);
      this.current = best.url;
      this.loading = false;
      this.onEnv(env);
    }, undefined, () => { this.loading = false; });
  }
}

// ── car: Rapier's raycast vehicle controller (tuned + tested in the built-in browser) ──
export class Car {
  active = false;
  private chassis: RAPIER.RigidBody | null = null;
  private v: RAPIER.DynamicRayCastVehicleController | null = null;
  readonly group = new THREE.Group();
  private wheels: THREE.Group[] = [];
  private keys = new Set<string>();
  private acc = 0;
  constructor(private world: RAPIER.World, scene: THREE.Scene) {
    const paint = new THREE.MeshStandardMaterial({ color: 0xb01e28, roughness: 0.25, metalness: 0.6 });
    const glass = new THREE.MeshStandardMaterial({ color: 0x111820, roughness: 0.05, metalness: 0.9 });
    const tyre = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.55, 4.2), paint);
    body.position.y = 0.05; body.castShadow = true;
    const cab = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.5, 2.0), glass);
    cab.position.set(0, 0.55, 0.2); cab.castShadow = true;
    this.group.add(body, cab);
    for (let i = 0; i < 4; i++) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.25, 20), tyre);
      m.rotation.z = Math.PI / 2; m.castShadow = true;
      const g = new THREE.Group(); g.add(m); this.group.add(g); this.wheels.push(g);
    }
    this.group.visible = false;
    scene.add(this.group);
    addEventListener('keydown', (e) => this.keys.add(e.code));
    addEventListener('keyup', (e) => this.keys.delete(e.code));
  }
  get materials() {
    const out: THREE.MeshStandardMaterial[] = [];
    this.group.traverse((o) => { const m = (o as THREE.Mesh).material; if (m instanceof THREE.MeshStandardMaterial) out.push(m); });
    return out;
  }
  spawn(at: { x: number; y: number; z: number }, yaw: number) {
    const ahead = { x: at.x - Math.sin(yaw) * 3.5, z: at.z - Math.cos(yaw) * 3.5 };
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0));
    this.chassis = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setCanSleep(false).setAngularDamping(1.5).setLinearDamping(0.05)
        .setCcdEnabled(true)
        .setAdditionalMassProperties(600, { x: 0, y: -0.6, z: 0 }, { x: 300, y: 300, z: 150 }, { x: 0, y: 0, z: 0, w: 1 })
        .setTranslation(ahead.x, at.y + 0.6, ahead.z).setRotation(q),
    );
    this.world.createCollider(RAPIER.ColliderDesc.cuboid(0.9, 0.35, 2.1).setDensity(160).setFriction(0.3), this.chassis);
    this.v = this.world.createVehicleController(this.chassis);
    for (const w of [[-0.8, -0.1, -1.3], [0.8, -0.1, -1.3], [-0.8, -0.1, 1.3], [0.8, -0.1, 1.3]]) {
      this.v.addWheel({ x: w[0], y: w[1], z: w[2] }, { x: 0, y: -1, z: 0 }, { x: -1, y: 0, z: 0 }, 0.35, 0.34);
    }
    for (let i = 0; i < 4; i++) {
      this.v.setWheelSuspensionStiffness(i, 35); this.v.setWheelSuspensionCompression(i, 4.4);
      this.v.setWheelSuspensionRelaxation(i, 2.3); this.v.setWheelFrictionSlip(i, 1.6);
      this.v.setWheelMaxSuspensionForce(i, 12000); this.v.setWheelSideFrictionStiffness(i, 1.0);
    }
    this.group.visible = true;
    this.active = true;
  }
  despawn(): { x: number; y: number; z: number } | null {
    if (!this.chassis || !this.v) return null;
    const t = this.chassis.translation();
    this.world.removeVehicleController(this.v);
    this.world.removeRigidBody(this.chassis);
    this.chassis = null; this.v = null; this.active = false; this.group.visible = false;
    return { x: t.x + 2.5, y: t.y + 1.2, z: t.z };
  }
  get kmh() { return this.v ? Math.abs(this.v.currentVehicleSpeed()) * 3.6 : 0; }
  /** Fixed-timestep physics so vehicle and world always advance together (stable at walls). */
  step(dt: number) {
    if (!this.v || !this.chassis) return;
    const k = this.keys;
    const eng = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0);
    const steer = (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0) - (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0);
    const maxSteer = 0.55 / (1 + this.kmh / 35);
    for (let i = 0; i < 4; i++) {
      this.v.setWheelEngineForce(i, i >= 2 ? -eng * 2400 : 0);
      this.v.setWheelBrake(i, k.has('Space') ? 60 : 0);
    }
    this.v.setWheelSteering(0, steer * maxSteer);
    this.v.setWheelSteering(1, steer * maxSteer);
    const h = 1 / 60;
    this.world.timestep = h;
    this.acc = Math.min(this.acc + dt, 0.25);
    while (this.acc >= h) { this.v.updateVehicle(h); this.world.step(); this.acc -= h; }
    const t = this.chassis.translation();
    const r = this.chassis.rotation();
    this.group.position.set(t.x, t.y, t.z);
    this.group.quaternion.set(r.x, r.y, r.z, r.w);
    for (let i = 0; i < 4; i++) {
      const c = this.v.wheelChassisConnectionPointCs(i);
      if (c) this.wheels[i].position.set(c.x, c.y - (this.v.wheelSuspensionLength(i) ?? 0.35), c.z);
    }
  }
  followCamera(camera: THREE.PerspectiveCamera) {
    const back = new THREE.Vector3(0, 2.6, 7.0).applyQuaternion(this.group.quaternion).add(this.group.position);
    camera.position.lerp(back, 0.12);
    camera.lookAt(this.group.position.x, this.group.position.y + 0.8, this.group.position.z);
  }
}

// ── FPS HUD + benchmark (?bench=1 flies the route and records avg / 1%-low FPS) ──
export class Hud {
  private el = document.createElement('div');
  private n = 0; private t = performance.now(); fps = 0;
  private bench: { t0: number; dur: number; frames: number[] } | null = null;
  constructor() {
    this.el.style.cssText = 'position:fixed;top:12px;left:12px;padding:6px 10px;border-radius:8px;' +
      'background:rgba(10,12,16,.6);color:#f2f2f2;font:13px system-ui,sans-serif;pointer-events:none;z-index:10';
    document.body.appendChild(this.el);
    if (params.has('bench')) this.startBench();
    addEventListener('keydown', (e) => { if (e.code === 'KeyB') this.startBench(); });
  }
  startBench() { this.bench = { t0: performance.now(), dur: Number(params.get('benchSeconds') || 20) * 1000, frames: [] }; }
  /** Returns true while the benchmark is driving the camera. */
  frame(dt: number, camera: THREE.PerspectiveCamera, mode: string): boolean {
    this.n++;
    const now = performance.now();
    if (now - this.t > 500) { this.fps = (this.n * 1000) / (now - this.t); this.n = 0; this.t = now; }
    let driving = false;
    if (this.bench && NAMMA.route.length > 1) {
      const r = NAMMA.route;
      const f = Math.min(1, (now - this.bench.t0) / this.bench.dur);
      const s = f * (r.length - 1); const i = Math.min(r.length - 2, Math.floor(s)); const a = s - i;
      const p = new THREE.Vector3().fromArray(r[i]).lerp(new THREE.Vector3().fromArray(r[i + 1]), a);
      camera.position.copy(p);
      camera.lookAt(r[i + 1][0], p.y, r[i + 1][2]);
      this.bench.frames.push(dt);
      driving = true;
      if (f >= 1) {
        const ft = this.bench.frames.slice(5).sort((x, y) => x - y);
        const avg = ft.length / ft.reduce((x, y) => x + y, 0);
        (window as any).__bench = { avg_fps: +avg.toFixed(1), low1_fps: +(1 / ft[Math.floor(ft.length * 0.99)]).toFixed(1),
          frames: ft.length, width: innerWidth, height: innerHeight, dpr: devicePixelRatio };
        console.log('NAMMA_BENCH', JSON.stringify((window as any).__bench));
        this.bench = null;
      }
    }
    const b = (window as any).__bench;
    this.el.innerHTML = `<b>${this.fps.toFixed(0)}</b> fps · ${mode}` +
      (b ? `<br>bench ${b.avg_fps} avg · ${b.low1_fps} 1%-low` : '');
    return driving;
  }
}
