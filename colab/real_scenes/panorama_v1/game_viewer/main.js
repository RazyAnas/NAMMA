// NAMMA Street World: photoreal splats (Spark 2.4 LoD) + Rapier physics on the
// reconstructed collider + panorama image-based lighting for meshes.
// Assets (same folder): game.json, scene-lod.rad | scene.ply, collider.glb,
// ground.glb, sky.jpg, ibl/pNNNN.jpg  (written by export_game.py).
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import RAPIER from "rapier";

const params = new URLSearchParams(location.search);
const BASE = params.get("base") || "./";
const $ = (id) => document.getElementById(id);
const hud = $("hud");
const setStatus = (t, frac) => { $("status").textContent = t; if (frac != null) $("bar").firstElementChild.style.width = `${Math.round(frac * 100)}%`; };

const game = await (await fetch(BASE + "game.json")).json();

// ------------------------------------------------------------------ render
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, Number(params.get("dpr") || 1.5)));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping; // splats are already display-referred photos
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.05, 5000);
const spark = new SparkRenderer({ renderer, lodSplatScale: Number(params.get("lod") || 1.0) });
scene.add(spark);

// sky: equirect in the game frame, row 0 = straight up (export_game.game_dirs)
let sky = null;
if (game.files.sky && !params.has("nosky")) {
  const tex = await new THREE.TextureLoader().loadAsync(BASE + game.files.sky);
  tex.colorSpace = THREE.SRGBColorSpace;
  sky = new THREE.Mesh(
    new THREE.SphereGeometry(4000, 64, 32),
    new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, uniforms: { map: { value: tex } },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform sampler2D map; varying vec3 vDir;
        void main(){ vec3 d = normalize(vDir);
          float u = 0.5 + atan(d.x, -d.z) / 6.2831853; float v = 0.5 + asin(clamp(d.y,-1.0,1.0)) / 3.1415927;
          gl_FragColor = texture2D(map, vec2(u, v));
          #include <colorspace_fragment>
        }`,
    }));
  sky.renderOrder = -1; sky.frustumCulled = false;
  scene.add(sky);
  camera.add(sky); scene.add(camera); // the sky follows the player
}

setStatus("Loading splats…", 0.1);
const splatUrl = params.get("splats") || (game.files.rad ? game.files.rad : game.files.splats);
const splats = new SplatMesh({ url: BASE + splatUrl, paged: splatUrl.endsWith(".rad"), lod: !splatUrl.endsWith(".rad") });
scene.add(splats);
await splats.initialized;
setStatus("Building physics…", 0.6);

// ------------------------------------------------------------------ physics
await RAPIER.init();
const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
const gltf = new GLTFLoader();
async function addTrimesh(file, visible = false) {
  const g = await gltf.loadAsync(BASE + file);
  g.scene.updateMatrixWorld(true);
  g.scene.traverse((o) => {
    if (!o.isMesh) return;
    const geo = o.geometry.clone().applyMatrix4(o.matrixWorld);
    const pos = geo.attributes.position.array;
    const idx = geo.index ? geo.index.array : Uint32Array.from({ length: pos.length / 3 }, (_, i) => i);
    world.createCollider(RAPIER.ColliderDesc.trimesh(new Float32Array(pos), new Uint32Array(idx)));
    if (visible) { o.material = new THREE.MeshBasicMaterial({ wireframe: true, color: 0x44ff88 }); scene.add(g.scene); }
  });
}
await addTrimesh(game.files.collider, params.has("debug"));
if (game.files.ground_mesh) await addTrimesh(game.files.ground_mesh, false);

// ------------------------------------------------------------------ lighting for meshes (avatars, cars)
const pmrem = new THREE.PMREMGenerator(renderer);
const probeCache = new Map();
let currentProbe = null;
async function updateProbe(p) {
  if (!game.ibl?.length) return;
  let best = game.ibl[0], bd = Infinity;
  for (const q of game.ibl) { const d = (q.pos[0] - p.x) ** 2 + (q.pos[2] - p.z) ** 2; if (d < bd) { bd = d; best = q; } }
  if (best === currentProbe) return;
  currentProbe = best;
  if (!probeCache.has(best.file)) {
    const t = await new THREE.TextureLoader().loadAsync(BASE + best.file);
    t.mapping = THREE.EquirectangularReflectionMapping; t.colorSpace = THREE.SRGBColorSpace;
    probeCache.set(best.file, pmrem.fromEquirectangular(t).texture);
  }
  scene.environment = probeCache.get(best.file);
}

// ------------------------------------------------------------------ player
const spawn = game.spawn || { pos: [0, 1.7, 0], look_at: [0, 1.7, -1] };
const body = world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(spawn.pos[0], spawn.pos[1], spawn.pos[2]));
const capsule = world.createCollider(RAPIER.ColliderDesc.capsule(0.55, 0.3), body);
const cc = world.createCharacterController(0.02);
cc.enableAutostep(0.35, 0.2, true); cc.enableSnapToGround(0.4); cc.setMaxSlopeClimbAngle(50 * Math.PI / 180);
const avatar = new THREE.Mesh(new THREE.CapsuleGeometry(0.3, 1.1, 6, 12),
  new THREE.MeshStandardMaterial({ color: 0x2d6cdf, roughness: 0.55, metalness: 0.0 }));
scene.add(avatar);
let yaw = Math.atan2(-(spawn.look_at[0] - spawn.pos[0]), -(spawn.look_at[2] - spawn.pos[2])), pitch = 0, vy = 0, thirdPerson = false;

// ------------------------------------------------------------------ car
let car = null;
function makeCar(at) {
  const ahead = { x: at.x - Math.sin(yaw) * 3.5, z: at.z - Math.cos(yaw) * 3.5 };
  // arcade handling: ballast below the floor lowers the centre of mass so it corners instead of flipping
  const chassis = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setCanSleep(false).setAngularDamping(1.5)
    .setLinearDamping(0.05).setCcdEnabled(true)
    .setAdditionalMassProperties(600, { x: 0, y: -0.6, z: 0 }, { x: 300, y: 300, z: 150 }, { x: 0, y: 0, z: 0, w: 1 })
    .setTranslation(ahead.x, at.y + 0.6, ahead.z).setRotation(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0))));
  world.createCollider(RAPIER.ColliderDesc.cuboid(0.9, 0.35, 2.1).setDensity(160).setFriction(0.3), chassis);
  const v = world.createVehicleController(chassis);
  const wheels = [[-0.8, -0.1, -1.3], [0.8, -0.1, -1.3], [-0.8, -0.1, 1.3], [0.8, -0.1, 1.3]];
  for (const w of wheels) {
    v.addWheel({ x: w[0], y: w[1], z: w[2] }, { x: 0, y: -1, z: 0 }, { x: -1, y: 0, z: 0 }, 0.35, 0.34);
  }
  for (let i = 0; i < 4; i++) {
    v.setWheelSuspensionStiffness(i, 35); v.setWheelSuspensionCompression(i, 4.4); v.setWheelSuspensionRelaxation(i, 2.3);
    v.setWheelFrictionSlip(i, 1.6); v.setWheelMaxSuspensionForce(i, 12000); v.setWheelSideFrictionStiffness(i, 1.0);
  }
  const group = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({ color: 0xb01e28, roughness: 0.25, metalness: 0.6 });
  const glass = new THREE.MeshStandardMaterial({ color: 0x111820, roughness: 0.05, metalness: 0.9 });
  const tyre = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
  const b = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.55, 4.2), paint); b.position.y = 0.05; group.add(b);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.5, 2.0), glass); cab.position.set(0, 0.55, 0.2); group.add(cab);
  const wheelMeshes = wheels.map((w) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.25, 20), tyre); m.rotation.z = Math.PI / 2; const g = new THREE.Group(); g.add(m); group.add(g); return g; });
  scene.add(group);
  return { chassis, v, group, wheelMeshes };
}
function removeCar() { if (!car) return; scene.remove(car.group); world.removeVehicleController(car.v); world.removeRigidBody(car.chassis); car = null; }

// ------------------------------------------------------------------ input
const keys = new Set();
addEventListener("keydown", (e) => {
  keys.add(e.code);
  if (e.code === "KeyV") thirdPerson = !thirdPerson;
  if (e.code === "KeyC") {
    if (car) { const t = car.chassis.translation(); body.setTranslation({ x: t.x + 2.5, y: t.y + 1.2, z: t.z }, true); removeCar(); capsule.setEnabled(true); }
    else { car = makeCar(body.translation()); capsule.setEnabled(false); avatar.visible = false; }
  }
  if (e.code === "KeyB") startBench();
});
addEventListener("keyup", (e) => keys.delete(e.code));
$("start").addEventListener("click", () => { $("start").style.display = "none"; renderer.domElement.requestPointerLock?.(); });
addEventListener("mousemove", (e) => { if (document.pointerLockElement) { yaw -= e.movementX * 0.0022; pitch = Math.max(-1.45, Math.min(1.45, pitch - e.movementY * 0.0022)); } });
let touchFwd = false;
$("tFwd").addEventListener("touchstart", () => (touchFwd = true)); $("tFwd").addEventListener("touchend", () => (touchFwd = false));
$("tCar").addEventListener("click", () => dispatchEvent(new KeyboardEvent("keydown", { code: "KeyC" })));
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// ------------------------------------------------------------------ benchmark: fly the route, measure FPS
let bench = null;
function startBench() {
  const r = game.route?.length > 1 ? game.route : [spawn.pos, spawn.look_at];
  bench = { t0: performance.now(), dur: Number(params.get("benchSeconds") || 20) * 1000, route: r, frames: [] };
  $("start").style.display = "none";
}
if (params.has("bench")) startBench();

// ------------------------------------------------------------------ loop
const clock = new THREE.Timer();
let fpsT = performance.now(), fpsN = 0, fps = 0, physAcc = 0;
const fwdV = new THREE.Vector3(), rightV = new THREE.Vector3();
setStatus("Ready", 1);
window.__dbg = { THREE, scene, camera, splats, sky, renderer, spark, world };
function frame(fixedDt) {
  clock.update(); const dt = fixedDt || Math.min(clock.getDelta(), 1 / 20);
  const now = performance.now();
  if (bench) {
    const f = Math.min(1, (now - bench.t0) / bench.dur), r = bench.route, s = f * (r.length - 1), i = Math.min(r.length - 2, Math.floor(s)), a = s - i;
    const p = new THREE.Vector3().fromArray(r[i]).lerp(new THREE.Vector3().fromArray(r[i + 1]), a);
    const q = new THREE.Vector3().fromArray(r[Math.min(r.length - 1, i + 1)]);
    camera.position.copy(p); camera.lookAt(q.x, p.y, q.z);
    bench.frames.push(dt);
    if (f >= 1) {
      const ft = bench.frames.slice(5).sort((x, y) => x - y);
      const avg = ft.length / ft.reduce((x, y) => x + y, 0);
      const low1 = 1 / ft[Math.floor(ft.length * 0.99)];
      window.__bench = { avg_fps: Math.round(avg * 10) / 10, low1_fps: Math.round(low1 * 10) / 10, frames: ft.length,
        width: innerWidth, height: innerHeight, dpr: renderer.getPixelRatio() };
      console.log("NAMMA_BENCH", JSON.stringify(window.__bench));
      bench = null;
    }
  } else if (car) {
    const v = car.v, eng = (keys.has("KeyW") || touchFwd ? 1 : 0) - (keys.has("KeyS") ? 1 : 0), steer = (keys.has("KeyA") ? 1 : 0) - (keys.has("KeyD") ? 1 : 0);
    const kmh = Math.abs(v.currentVehicleSpeed()) * 3.6;
    const maxSteer = 0.55 / (1 + kmh / 35);                         // less steering at speed
    for (let k = 0; k < 4; k++) { v.setWheelEngineForce(k, k >= 2 ? -eng * 2400 : 0); v.setWheelBrake(k, keys.has("Space") ? 60 : 0); }
    v.setWheelSteering(0, steer * maxSteer); v.setWheelSteering(1, steer * maxSteer);
    const t = car.chassis.translation(), q = car.chassis.rotation();
    car.group.position.set(t.x, t.y, t.z); car.group.quaternion.set(q.x, q.y, q.z, q.w);
    for (let k = 0; k < 4; k++) { const c = v.wheelChassisConnectionPointCs(k); if (c) car.wheelMeshes[k].position.set(c.x, c.y - (v.wheelSuspensionLength?.(k) ?? 0.35), c.z); }
    const back = new THREE.Vector3(0, 2.6, 7.0).applyQuaternion(car.group.quaternion).add(car.group.position);
    camera.position.lerp(back, 0.12); camera.lookAt(t.x, t.y + 0.8, t.z);
    updateProbe(t);
  } else {
    const run = keys.has("ShiftLeft") || keys.has("ShiftRight") ? 2.4 : 1.0, speed = 1.6 * run;
    fwdV.set(-Math.sin(yaw), 0, -Math.cos(yaw)); rightV.set(-fwdV.z, 0, fwdV.x);
    const move = new THREE.Vector3();
    if (keys.has("KeyW") || touchFwd) move.add(fwdV); if (keys.has("KeyS")) move.sub(fwdV);
    if (keys.has("KeyD")) move.add(rightV); if (keys.has("KeyA")) move.sub(rightV);
    if (move.lengthSq() > 0) move.normalize().multiplyScalar(speed * dt);
    const grounded = cc.computedGrounded?.() ?? true;
    vy = grounded ? (keys.has("Space") ? 4.2 : -0.5) : vy - 9.81 * dt;
    move.y = vy * dt;
    cc.computeColliderMovement(capsule, move);
    const m = cc.computedMovement(), t = body.translation();
    const n = { x: t.x + m.x, y: t.y + m.y, z: t.z + m.z };
    body.setNextKinematicTranslation(n);
    avatar.position.set(n.x, n.y, n.z); avatar.visible = thirdPerson;
    const eye = new THREE.Vector3(n.x, n.y + 0.75, n.z);
    if (thirdPerson) { camera.position.copy(eye).addScaledVector(fwdV, -3.2).add(new THREE.Vector3(0, 0.6, 0)); camera.lookAt(eye); }
    else { camera.position.copy(eye); camera.rotation.set(pitch, yaw, 0, "YXZ"); }
    updateProbe(n);
  }
  // fixed-timestep physics: the vehicle and the world always advance by the same 1/60 s
  physAcc = Math.min(physAcc + dt, 0.25);
  while (physAcc >= world.timestep) {
    if (car) car.v.updateVehicle(world.timestep);
    world.step();
    physAcc -= world.timestep;
  }
  renderer.render(scene, camera);
  fpsN++;
  if (now - fpsT > 500) { fps = (fpsN * 1000) / (now - fpsT); fpsN = 0; fpsT = now; }
  hud.innerHTML = `<b>${fps.toFixed(0)}</b> fps · ${car ? `driving ${Math.round(Math.abs(car.v.currentVehicleSpeed()) * 3.6)} km/h` : thirdPerson ? "3rd person" : "walking"}` +
    (window.__bench ? `<br>bench: <b>${window.__bench.avg_fps}</b> avg · ${window.__bench.low1_fps} 1%-low` : "");
}
renderer.setAnimationLoop(frame);
// manual stepping for automated checks when the page is not visible (rAF paused)
window.__tick = (n = 1, keysDown = []) => { keysDown.forEach((k) => keys.add(k)); for (let i = 0; i < n; i++) frame(1 / 60); keysDown.forEach((k) => keys.delete(k)); return camera.position.toArray(); };
