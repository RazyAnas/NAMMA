// Self-contained walking demo: loads ./data/{splat.ply,navmesh.json,collision.glb}
// and walks it with keyboard + mouse on a laptop, or a touch stick, drag-to-look
// and optional gyro look on a phone. Same Rapier walker as the main viewer.
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import RAPIER from '@dimforge/rapier3d-compat';
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

const setProgress = (text) => { $('progress-text').textContent = text; };

async function load() {
  const [navigation, collisionBuffer] = await Promise.all([
    fetch('./data/navmesh.json').then((r) => { if (!r.ok) throw new Error('navmesh.json missing'); return r.json(); }),
    fetch('./data/collision.glb').then((r) => (r.ok ? r.arrayBuffer() : null)),
    RAPIER.init({}),
  ]);
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
  });
  await viewer.addSplatScene('./data/splat.ply', {
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
