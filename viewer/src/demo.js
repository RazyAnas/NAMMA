// Self-contained walking demo: loads ./data/{splat.ply,navmesh.json,collision.glb}, or with
// ./data/scenes.json one of several rooms from ./data/<dir>/ (picked on the start screen, ?scene=<id>)
// and walks it with keyboard + mouse on a laptop, or a touch stick, drag-to-look
// and optional gyro look on a phone. Same Rapier walker as the main viewer.
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import RAPIER from '@dimforge/rapier3d-compat';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import './style.css';
import './demo.css';
import { NavFrame, Walker, decodeMovementLayers, readGlbTriangles } from './walker.js';

const $ = (id) => document.getElementById(id);
const canvas = $('scene');
const touchDevice = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;
const keys = new Set();
const level = new THREE.Quaternion();
const look = new THREE.Euler(0, 0, 0, 'YXZ');
let yaw = 0;
let pitch = 0;
let flying = false;
let gyro = null; // latest device orientation, when gyro look is on
let gyroYawOffset = 0;
let stick = { x: 0, y: 0 };
let jumpQueued = false;
let viewer;
let camera;
let walker;
let frame;
// Optional data/<room>/cameras.json: the capture's camera centres in splat coordinates. A splat is only
// trustworthy near where the phone filmed from, so flying is kept inside a tube of radius_m around them.
let corridor = null;
const nearest = new THREE.Vector3();
// Hybrid view (optional data/<room>/mesh.glb): measured on a held-out stretch of the phone room, the splat is best on the
// capture path (~34 dB) but a fused mesh beats it once you leave it (18.8 vs 15.3-16.6 dB). So the mesh is drawn behind
// the splat and the splat fades out between FADE_NEAR and FADE_FAR metres from the nearest capture camera.
const FADE_NEAR = 0.15;
const FADE_FAR = 0.6;
let roomMesh = null;
let splatShare = 1;
// M cycles the view: fill = splat in front, mesh behind it only plugs the holes (default; never worse than the splat alone),
// auto = fade to the mesh away from the capture path, splat = splat only, mesh = mesh only.
const BLEND_MODES = ['fill', 'auto', 'splat', 'mesh'];
let blendMode = 'fill';

function nearestPathDistance(position) {
  let best = Infinity;
  for (const c of corridor.centers) {
    const d = c.distanceToSquared(position);
    if (d < best) { best = d; nearest.copy(c); }
  }
  return Math.sqrt(best) * frame.navigation.viewer_units_to_meters;
}

function keepNearPath(position) {
  if (!corridor) return;
  const distance = nearestPathDistance(position);
  const radius = corridor.radius;
  if (distance > radius) position.sub(nearest).setLength(radius / frame.navigation.viewer_units_to_meters).add(nearest);
}

function blendSplatAndMesh() {
  if (!roomMesh) return;
  roomMesh.visible = blendMode !== 'splat';
  if (blendMode === 'auto' && corridor) {
    splatShare = 1 - THREE.MathUtils.smoothstep(nearestPathDistance(camera.position), FADE_NEAR, FADE_FAR);
  } else splatShare = blendMode === 'mesh' ? 0 : 1;
  viewer.getSplatScene(0).opacity = splatShare;
  const share = splatShare > 0.99 ? 'splat' : splatShare < 0.01 ? 'mesh' : `splat ${Math.round(splatShare * 100)}%`;
  $('blend').textContent = blendMode === 'auto' ? `auto: ${share}` : blendMode === 'fill' ? 'splat + mesh fill' : share;
}

async function loadRoomMesh(url) {
  const buffer = await fetch(url).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null);
  if (!buffer) return null;
  // mesh.glb is packed with gltfpack (meshopt compression, ~5x smaller); plain GLBs load too.
  const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).parseAsync(buffer, '');
  const scene = new THREE.Scene();
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    o.material = new THREE.MeshBasicMaterial({ vertexColors: true });
    // The splat is drawn after this scene; clearing depth here keeps the mesh from hiding splats that sit on its surface.
    o.onAfterRender = (r) => r.clearDepth();
  });
  scene.add(gltf.scene);
  return scene;
}

const setProgress = (text) => { $('progress-text').textContent = text; };

// Room list: data/scenes.json = [{ id, dir, title, lede }]; without it, the single room in data/.
async function chooseScene() {
  const scenes = await fetch('./data/scenes.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!scenes?.length) return './data/';
  const wanted = new URLSearchParams(location.search).get('scene');
  const scene = scenes.find((s) => s.id === wanted) || scenes[0];
  $('room-title').textContent = scene.title;
  $('room-lede').textContent = scene.lede;
  const list = $('rooms');
  list.hidden = false;
  for (const s of scenes) {
    const link = document.createElement('a');
    link.href = `?scene=${encodeURIComponent(s.id)}`;
    link.textContent = s.title;
    if (s === scene) link.setAttribute('aria-current', 'true');
    list.append(link);
  }
  document.title = `NAMMA · ${scene.title}`;
  return `./data/${scene.dir}/`;
}

async function load() {
  const base = await chooseScene();
  const [navigation, collisionBuffer, cameraPath] = await Promise.all([
    fetch(`${base}navmesh.json`).then((r) => { if (!r.ok) throw new Error('navmesh.json missing'); return r.json(); }),
    fetch(`${base}collision.glb`).then((r) => (r.ok ? r.arrayBuffer() : null)),
    fetch(`${base}cameras.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    RAPIER.init({}),
  ]);
  const meshScene = await loadRoomMesh(`${base}mesh.glb`);
  // With a mesh to fall back on, flying may go further from the capture path than with the splat alone.
  const radius = (cameraPath?.radius_m ?? 0.6) * (meshScene ? 2.5 : 1);
  corridor = cameraPath?.centers?.length ? { centers: cameraPath.centers.map((c) => new THREE.Vector3(...c)), radius } : null;
  frame = new NavFrame(navigation);
  const layers = await decodeMovementLayers(navigation);
  if (!layers) throw new Error('This NavMesh has no movement layers; rebuild it with the current backend.');
  walker = new Walker(RAPIER, {
    navigation,
    layers,
    collision: collisionBuffer ? readGlbTriangles(collisionBuffer) : null,
  });
  walker.spawn();
  for (let i = 0; i < 20; i += 1) walker.step([0, 0], 1 / 60);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, touchDevice ? 1 : 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(0x07110f, 1);
  camera = new THREE.PerspectiveCamera(touchDevice ? 70 : 62, window.innerWidth / window.innerHeight, 0.03, 200);
  level.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...frame.viewerUp));
  viewer = new GaussianSplats3D.Viewer({
    selfDrivenMode: false,
    renderer,
    camera,
    useBuiltInControls: false,
    gpuAcceleratedSort: false,
    sharedMemoryForWorkers: false,
    integerBasedSort: true,
    dynamicScene: false,
    sphericalHarmonicsDegree: touchDevice ? 0 : 1,
    sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
    freeIntermediateSplatData: true,
    threeScene: meshScene ?? undefined,
    enableOptionalEffects: !!meshScene,
  });
  roomMesh = meshScene;
  $('blend-wrap').hidden = !roomMesh;
  await viewer.addSplatScene(`${base}splat.ply`, {
    format: GaussianSplats3D.SceneFormat.Ply,
    splatAlphaRemovalThreshold: 5,
    showLoadingUI: false,
    progressiveLoad: false,
    onProgress: (percent) => setProgress(`Loading room ${Math.round(percent)}%`),
  });
  faceRoom(layers);
  syncCamera();
  viewer.update();
  viewer.render();
  setProgress(touchDevice ? 'Left stick moves · drag to look' : 'Click to start, then use W A S D and the mouse');
  $('go').disabled = false;
  $('go').textContent = 'Start walking';
}

// Start out looking toward the middle of the walkable floor.
function faceRoom(layers) {
  const { width, height, cell_size_m: cell, origin_floor_coordinates: [x0, z0] } = frame.navigation;
  let sx = 0; let sz = 0; let n = 0;
  for (let gy = 0; gy < height; gy += 1) {
    for (let gx = 0; gx < width; gx += 1) {
      if (!layers.standable[gy * width + gx]) continue;
      sx += x0 + (gx + 0.5) * cell; sz += z0 + (gy + 0.5) * cell; n += 1;
    }
  }
  const eye = walker.eye;
  const target = new THREE.Vector3(...frame.navToViewer([sx / n, eye[1], sz / n]));
  const direction = target.sub(new THREE.Vector3(...frame.navToViewer(eye)));
  direction.applyQuaternion(level.clone().invert());
  if (direction.lengthSq() < 1e-6) return;
  yaw = Math.atan2(-direction.x, -direction.z);
}

function cameraQuaternion(out) {
  if (gyro) {
    // Device orientation to a camera looking out of the screen (the old
    // three.js DeviceOrientationControls formula), turned by the drag offset.
    const alpha = THREE.MathUtils.degToRad(gyro.alpha || 0);
    const beta = THREE.MathUtils.degToRad(gyro.beta || 0);
    const gamma = THREE.MathUtils.degToRad(gyro.gamma || 0);
    const orient = THREE.MathUtils.degToRad(screen.orientation?.angle ?? window.orientation ?? 0);
    out.setFromEuler(new THREE.Euler(beta, alpha, -gamma, 'YXZ'));
    out.multiply(new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2));
    out.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -orient));
    out.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), gyroYawOffset));
  } else {
    look.set(pitch, yaw, 0);
    out.setFromEuler(look);
  }
  return out.premultiply(level);
}

const scratchQuaternion = new THREE.Quaternion();
const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();

function syncCamera() {
  camera.quaternion.copy(cameraQuaternion(scratchQuaternion));
  if (!flying) camera.position.fromArray(frame.navToViewer(walker.eye));
}

function step(dt) {
  const forwardInput = Number(keys.has('KeyW') || keys.has('ArrowUp')) - Number(keys.has('KeyS') || keys.has('ArrowDown')) - stick.y;
  const strafeInput = Number(keys.has('KeyD') || keys.has('ArrowRight')) - Number(keys.has('KeyA') || keys.has('ArrowLeft')) + stick.x;
  const run = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 2.2 : 1;
  const speed = 1.4 * run;
  // Horizontal heading from wherever the camera looks (mouse or gyro).
  up.set(...frame.viewerUp);
  forward.set(0, 0, -1).applyQuaternion(camera.quaternion);
  if (!flying) forward.addScaledVector(up, -forward.dot(up));
  if (forward.lengthSq() < 1e-6) forward.set(0, 0, -1).applyQuaternion(level);
  forward.normalize();
  right.crossVectors(forward, up).normalize();
  const move = forward.clone().multiplyScalar(forwardInput).addScaledVector(right, strafeInput);
  if (move.lengthSq() > 1) move.normalize();
  if (flying) {
    const vertical = Number(keys.has('KeyE')) - Number(keys.has('KeyQ'));
    move.addScaledVector(up, vertical);
    camera.position.addScaledVector(move, (speed * dt) / frame.navigation.viewer_units_to_meters);
    keepNearPath(camera.position);
    return;
  }
  const amount = Math.min(1, move.length());
  const nav = frame.viewerDirToNav(move.toArray());
  const horizontal = Math.hypot(nav[0], nav[2]);
  const scale = horizontal > 1e-6 ? (speed * amount) / horizontal : 0;
  walker.step([nav[0] * scale, nav[2] * scale], dt, keys.has('Space') || jumpQueued);
  jumpQueued = false;
}

let previous = performance.now();
let frames = 0;
let fpsStart = previous;
function loop(now) {
  const dt = Math.min((now - previous) / 1000, 0.1);
  previous = now;
  step(dt);
  syncCamera();
  blendSplatAndMesh();
  viewer.update();
  viewer.render();
  frames += 1;
  if (now - fpsStart > 500) {
    $('fps').textContent = Math.round((frames * 1000) / (now - fpsStart));
    frames = 0;
    fpsStart = now;
  }
  requestAnimationFrame(loop);
}

function start() {
  $('start').hidden = true;
  $('hud').hidden = false;
  if (touchDevice) $('touch-ui').hidden = false;
  else {
    $('keys-help').hidden = false;
    canvas.requestPointerLock?.();
  }
  requestAnimationFrame(loop);
}

function reset() {
  walker.spawn();
  flying = false;
  $('mode').textContent = 'walking';
}

// ---- keyboard and mouse ---------------------------------------------------
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') event.preventDefault();
  if (event.code === 'KeyF' && !event.repeat && walker) {
    flying = !flying;
    if (!flying) {
      const [x, , z] = frame.viewerToNav(camera.position.toArray());
      if (walker.standable(x, z)) walker.placeFeet(x, z);
    }
    $('mode').textContent = flying ? 'flying, no collision' : 'walking';
  }
  if (event.code === 'KeyR' && walker) reset();
  if (event.code === 'KeyM' && !event.repeat && roomMesh) blendMode = BLEND_MODES[(BLEND_MODES.indexOf(blendMode) + 1) % BLEND_MODES.length];
  keys.add(event.code);
});
document.addEventListener('keyup', (event) => keys.delete(event.code));
canvas.addEventListener('click', () => { if (!touchDevice && $('start').hidden) canvas.requestPointerLock?.(); });
document.addEventListener('mousemove', (event) => {
  if (document.pointerLockElement !== canvas) return;
  yaw -= event.movementX * 0.0022;
  pitch = THREE.MathUtils.clamp(pitch - event.movementY * 0.0022, -1.5, 1.5);
});

// ---- touch: left stick moves, drag elsewhere looks -------------------------
const stickEl = $('stick');
let stickPointer = null;
let lookPointer = null;
let lookLast = null;
function updateStick(event) {
  const rect = stickEl.getBoundingClientRect();
  const radius = rect.width / 2;
  let x = (event.clientX - rect.left - radius) / radius;
  let y = (event.clientY - rect.top - radius) / radius;
  const length = Math.hypot(x, y);
  if (length > 1) { x /= length; y /= length; }
  stick = { x, y };
  $('knob').style.transform = `translate(${x * radius * 0.6}px, ${y * radius * 0.6}px)`;
}
stickEl.addEventListener('pointerdown', (event) => {
  stickPointer = event.pointerId;
  try { stickEl.setPointerCapture(event.pointerId); } catch { /* synthetic pointer */ }
  updateStick(event);
});
stickEl.addEventListener('pointermove', (event) => { if (event.pointerId === stickPointer) updateStick(event); });
const releaseStick = (event) => {
  if (event.pointerId !== stickPointer) return;
  stickPointer = null;
  stick = { x: 0, y: 0 };
  $('knob').style.transform = '';
};
stickEl.addEventListener('pointerup', releaseStick);
stickEl.addEventListener('pointercancel', releaseStick);

canvas.addEventListener('pointerdown', (event) => {
  if (event.pointerType === 'mouse' || lookPointer !== null) return;
  lookPointer = event.pointerId;
  lookLast = [event.clientX, event.clientY];
  try { canvas.setPointerCapture(event.pointerId); } catch { /* synthetic pointer */ }
});
canvas.addEventListener('pointermove', (event) => {
  if (event.pointerId !== lookPointer) return;
  const dx = event.clientX - lookLast[0];
  const dy = event.clientY - lookLast[1];
  lookLast = [event.clientX, event.clientY];
  const k = 0.005;
  if (gyro) gyroYawOffset += dx * k;
  else {
    yaw += dx * k;
    pitch = THREE.MathUtils.clamp(pitch + dy * k, -1.4, 1.4);
  }
});
const releaseLook = (event) => { if (event.pointerId === lookPointer) lookPointer = null; };
canvas.addEventListener('pointerup', releaseLook);
canvas.addEventListener('pointercancel', releaseLook);

$('jump').addEventListener('click', () => { jumpQueued = true; });
$('reset').addEventListener('click', () => reset());
$('gyro').addEventListener('click', async () => {
  if (gyro) {
    window.removeEventListener('deviceorientation', onOrientation);
    gyro = null;
    $('gyro').setAttribute('aria-pressed', 'false');
    return;
  }
  try {
    // iOS asks for permission, and only on HTTPS pages.
    if (typeof DeviceOrientationEvent?.requestPermission === 'function') {
      const answer = await DeviceOrientationEvent.requestPermission();
      if (answer !== 'granted') throw new Error('Motion permission was not granted.');
    }
    if (!window.isSecureContext) throw new Error('Gyro look needs an https:// address on this phone.');
    gyro = { alpha: 0, beta: 90, gamma: 0 };
    // Keep the current heading when switching from drag look to gyro look.
    gyroYawOffset = 0;
    window.addEventListener('deviceorientation', onOrientation);
    $('gyro').setAttribute('aria-pressed', 'true');
  } catch (error) {
    $('mode').textContent = error.message;
  }
});
function onOrientation(event) {
  if (event.alpha === null) return;
  gyro = { alpha: event.alpha, beta: event.beta, gamma: event.gamma };
}

window.addEventListener('resize', () => {
  if (!camera) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  viewer.renderer.setSize(window.innerWidth, window.innerHeight, false);
});
$('go').addEventListener('click', start);

load().catch((error) => {
  console.error(error);
  $('error').textContent = error.message || String(error);
  $('go').textContent = 'Could not load';
});
