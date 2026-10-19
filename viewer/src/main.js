import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import './style.css';
import { NavFrame, Walker, decodeMovementLayers, readGlbTriangles } from './walker.js';

const ui = Object.fromEntries(
  ['project', 'load', 'launcher', 'loading', 'progress', 'progress-bar', 'loading-detail', 'error',
    'hud', 'controls', 'crosshair', 'enter', 'status', 'status-dot', 'project-name', 'model-size', 'fps', 'nav-status', 'move-help']
    .map((id) => [id, document.getElementById(id)])
);
const canvas = document.getElementById('scene');

let viewer;
let camera;
let renderer;
let navigation;
let walker;
let navFrame;
let flying = false;
// Rotates three.js's +Y onto the reconstruction's real up direction, so mouse
// yaw turns around the floor normal instead of rolling the horizon.
const level = new THREE.Quaternion();
const look = new THREE.Euler(0, 0, 0, 'YXZ');
let previousFrame = performance.now();
let fpsFrames = 0;
let fpsStarted = previousFrame;
let yaw = 0;
let pitch = 0;
let speed = 1.5;
const keys = new Set();
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();

const setStatus = (text, state = '') => {
  ui.status.textContent = text;
  ui['status-dot'].dataset.state = state;
};

const formatBytes = (bytes) => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = Number(bytes || 0);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
};

async function loadProjects() {
  try {
    const response = await fetch('/api/v1/projects');
    if (!response.ok) throw new Error(`Project API returned ${response.status}`);
    const projects = await response.json();
    const ready = projects.filter((project) => project.status === 'ready');
    ui.project.innerHTML = ready.length
      ? ready.map((project) => `<option value="${project.id}">${escapeHtml(project.name)}</option>`).join('')
      : '<option value="">No trained reconstruction available</option>';
    ui.load.disabled = ready.length === 0;
    setStatus(ready.length ? 'Backend online' : 'Waiting for model', ready.length ? 'ready' : '');
  } catch (error) {
    ui.project.innerHTML = '<option value="">Backend unavailable</option>';
    ui.error.textContent = error.message;
    setStatus('Backend offline', 'error');
  }
}

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value;
  return node.innerHTML;
}

function updateProgress(percent, detail) {
  const bounded = Math.max(0, Math.min(100, Math.round(percent)));
  ui.progress.textContent = `${bounded}%`;
  ui['progress-bar'].style.width = `${bounded}%`;
  if (detail) ui['loading-detail'].textContent = detail;
}

async function decodeNavigationGrid(encoded, expectedLength) {
  const binary = Uint8Array.from(atob(encoded), (character) => character.charCodeAt(0));
  if (typeof DecompressionStream === 'undefined') throw new Error('Compressed navigation data is not supported by this browser.');
  const stream = new Blob([binary]).stream().pipeThrough(new DecompressionStream('deflate'));
  const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  if (bytes.length !== expectedLength) throw new Error('Navigation grid length mismatch.');
  return bytes;
}

async function loadNavigation(projectId) {
  const response = await fetch(`/api/v1/projects/${projectId}/navigation-mesh`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Navigation API returned ${response.status}`);
  const manifest = await response.json();
  if (!manifest.navmesh_ready) return manifest;
  const bytes = await decodeNavigationGrid(manifest.walkable_grid_base64, manifest.width * manifest.height);
  return { ...manifest, walkable: bytes };
}

// Physics walking needs the movement layers; the calibrated collision mesh
// sharpens walls and furniture but the height field alone is enough to walk.
async function createWalker(projectId) {
  if (!navigation?.navmesh_ready || !navigation.movement) return null;
  const [{ default: RAPIER }, layers] = await Promise.all([
    import('@dimforge/rapier3d-compat'),
    decodeMovementLayers(navigation),
  ]);
  await RAPIER.init({});
  let collision = null;
  try {
    const response = await fetch(`/api/v1/projects/${projectId}/collision-model`);
    if (response.ok) collision = readGlbTriangles(await response.arrayBuffer());
  } catch (error) {
    console.warn('Collision mesh unavailable; walking on the height field only.', error);
  }
  const created = new Walker(RAPIER, { navigation, layers, collision });
  created.spawn();
  return created;
}

function transformPoint(matrix, point) {
  if (!Array.isArray(matrix) || matrix.length !== 4) return point;
  const [x, y, z] = point;
  const w = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3];
  const divisor = Math.abs(w) > 1e-9 ? w : 1;
  return [
    (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]) / divisor,
    (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]) / divisor,
    (matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]) / divisor,
  ];
}

function applyLook() {
  look.set(pitch, yaw, 0);
  camera.quaternion.setFromEuler(look).premultiply(level);
}

function frameCamera(manifest) {
  const bounds = manifest.viewer_bounds;
  const spawn = navigation?.spawn?.source_world;
  let target = [0, 0, 0];
  let distance = 3;
  if (bounds?.min?.length === 3 && bounds?.max?.length === 3) {
    target = bounds.center?.length === 3
      ? bounds.center
      : bounds.min.map((value, index) => (value + bounds.max[index]) * 0.5);
    const extent = bounds.min.map((value, index) => bounds.max[index] - value);
    distance = Math.max(1, Math.max(...extent) * 0.9);
  }
  const up = navFrame ? new THREE.Vector3(...navFrame.viewerUp) : new THREE.Vector3(0, 1, 0);
  level.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
  if (walker) {
    camera.position.fromArray(navFrame.navToViewer(walker.eye));
  } else if (navigation?.navmesh_ready && Array.isArray(spawn) && spawn.length === 3) {
    camera.position.fromArray(transformPoint(navigation.source_to_viewer, spawn));
  } else {
    camera.position.copy(up).multiplyScalar(distance * 0.15).add(new THREE.Vector3(target[0], target[1], target[2] + distance));
  }
  // Express the view direction in the levelled frame to get yaw and pitch.
  const toTarget = new THREE.Vector3(...target).sub(camera.position);
  toTarget.applyQuaternion(level.clone().invert());
  if (toTarget.lengthSq() < 1e-9) toTarget.set(0, 0, -1);
  toTarget.normalize();
  yaw = Math.atan2(-toTarget.x, -toTarget.z);
  pitch = walker ? 0 : Math.asin(Math.max(-1, Math.min(1, toTarget.y)));
  applyLook();
}

async function loadModel() {
  const projectId = ui.project.value;
  if (!projectId) return;
  ui.load.disabled = true;
  ui.error.textContent = '';
  ui.launcher.hidden = true;
  ui.loading.hidden = false;
  setStatus('Loading model');

  try {
    const manifestResponse = await fetch(`/api/v1/projects/${projectId}/gaussian-artifact`);
    if (!manifestResponse.ok) throw new Error('The Gaussian model manifest is not available.');
    const manifest = await manifestResponse.json();
    navigation = null;
    walker = null;
    navFrame = null;
    flying = false;
    try {
      navigation = await loadNavigation(projectId);
      if (navigation?.navmesh_ready) navFrame = new NavFrame(navigation);
      walker = await createWalker(projectId);
    } catch (navigationError) {
      console.warn(navigationError);
    }
    updateProgress(2, `Downloading ${formatBytes(manifest.size_bytes)} model…`);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.setClearColor(0x07110f, 1);

    camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.03, 500);
    camera.rotation.order = 'YXZ';

    viewer = new GaussianSplats3D.Viewer({
      selfDrivenMode: false,
      renderer,
      camera,
      useBuiltInControls: false,
      gpuAcceleratedSort: false,
      enableSIMDInSort: true,
      sharedMemoryForWorkers: false,
      integerBasedSort: true,
      halfPrecisionCovariancesOnGPU: false,
      dynamicScene: false,
      renderMode: GaussianSplats3D.RenderMode.OnChange,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
      sphericalHarmonicsDegree: 2,
      inMemoryCompressionLevel: 1,
      freeIntermediateSplatData: true,
    });
    await viewer.addSplatScene(`/api/v1/projects/${projectId}/gaussian-model`, {
      format: GaussianSplats3D.SceneFormat.Ply,
      splatAlphaRemovalThreshold: 5,
      showLoadingUI: false,
      progressiveLoad: false,
      onProgress: (percentage, label) => updateProgress(5 + percentage * 0.9, label || 'Decoding Gaussian splats…'),
    });
    // First-scene setup applies the library's default pose. Frame only after
    // the scene is attached so nothing can overwrite the reconstructed bounds.
    frameCamera(manifest);

    updateProgress(100, 'Scene ready');
    const option = ui.project.selectedOptions[0];
    ui['project-name'].textContent = option?.textContent || 'Reconstruction';
    ui['model-size'].textContent = formatBytes(manifest.size_bytes);
    updateModeLabels();
    ui.loading.hidden = true;
    ui.hud.hidden = false;
    ui.controls.hidden = false;
    ui.enter.hidden = false;
    setStatus('Scene ready', 'ready');
    // Paint the first frame immediately. Background laptop tabs may throttle
    // requestAnimationFrame until they become active.
    viewer.update();
    viewer.render();
    requestAnimationFrame(frame);
  } catch (error) {
    console.error(error);
    ui.loading.hidden = true;
    ui.launcher.hidden = false;
    ui.load.disabled = false;
    ui.error.textContent = error.message || 'Could not load this reconstruction.';
    setStatus('Load failed', 'error');
  }
}

function frame(now) {
  if (!viewer) return;
  const delta = Math.min((now - previousFrame) / 1000, 0.1);
  previousFrame = now;
  updateMovement(delta);
  viewer.update();
  viewer.render();

  fpsFrames += 1;
  if (now - fpsStarted >= 500) {
    ui.fps.textContent = Math.round((fpsFrames * 1000) / (now - fpsStarted));
    fpsFrames = 0;
    fpsStarted = now;
  }
  requestAnimationFrame(frame);
}

function updateModeLabels() {
  if (walker && !flying) {
    ui['nav-status'].textContent = 'walking · collision on';
    ui['move-help'].textContent = 'W A S D · Space jump';
  } else {
    ui['nav-status'].textContent = walker ? 'flying · no collision' : (navigation?.navmesh_ready ? 'collision on' : 'review required');
    ui['move-help'].textContent = 'W A S D · Q / E down/up';
  }
}

function updateMovement(delta) {
  if (document.pointerLockElement !== canvas) return;
  const forward = Number(keys.has('KeyW')) - Number(keys.has('KeyS'));
  const strafe = Number(keys.has('KeyD')) - Number(keys.has('KeyA'));
  const vertical = Number(keys.has('KeyE')) - Number(keys.has('KeyQ'));
  const multiplier = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 3 : 1;
  if (walker && !flying) {
    // Walk on the level plane: pitch never changes height, gravity does.
    direction.set(strafe, 0, -forward);
    if (direction.lengthSq() > 0) direction.normalize();
    direction.applyEuler(new THREE.Euler(0, yaw, 0)).applyQuaternion(level);
    const nav = navFrame.viewerDirToNav(direction.toArray());
    const horizontal = Math.hypot(nav[0], nav[2]);
    const scale = horizontal > 1e-6 ? (speed * multiplier) / horizontal : 0;
    walker.step([nav[0] * scale, nav[2] * scale], delta, keys.has('Space'));
    camera.position.fromArray(navFrame.navToViewer(walker.eye));
    return;
  }
  direction.set(strafe, 0, -forward);
  if (direction.lengthSq() > 0) direction.normalize();
  direction.applyQuaternion(camera.quaternion);
  direction.addScaledVector(new THREE.Vector3(0, 1, 0).applyQuaternion(level), vertical);
  if (direction.lengthSq() > 0) direction.normalize();
  const sourceUnitsToMeters = navigation?.navmesh_ready ? navigation.viewer_units_to_meters : 1;
  velocity.copy(direction).multiplyScalar((speed * multiplier * delta) / sourceUnitsToMeters);
  if (!navigation?.navmesh_ready || flying) {
    camera.position.add(velocity);
    return;
  }
  // Substep movement to prevent a fast sprint from tunnelling through a wall.
  const distanceMeters = velocity.length() * sourceUnitsToMeters;
  const steps = Math.max(1, Math.ceil(distanceMeters / (navigation.cell_size_m * 0.5)));
  const step = velocity.clone().multiplyScalar(1 / steps);
  for (let index = 0; index < steps; index += 1) {
    const candidate = camera.position.clone().add(step);
    if (!isWalkable(candidate)) break;
    camera.position.copy(candidate);
  }
}

function toggleFlying() {
  if (!walker) return;
  flying = !flying;
  if (!flying) {
    // Land where the camera is if that spot is walkable, otherwise go back
    // to the last walking position.
    const [x, , z] = navFrame.viewerToNav(camera.position.toArray());
    if (walker.standable(x, z)) walker.placeFeet(x, z);
    camera.position.fromArray(navFrame.navToViewer(walker.eye));
    pitch = Math.max(-1.2, Math.min(1.2, pitch));
    applyLook();
  }
  updateModeLabels();
}

function isWalkable(position) {
  const basis = navigation.basis;
  const scale = navigation.source_units_to_meters;
  const source = transformPoint(navigation.viewer_to_source, position.toArray());
  const x = scale * (source[0] * basis.axis_x[0] + source[1] * basis.axis_x[1] + source[2] * basis.axis_x[2]);
  const y = scale * (source[0] * basis.axis_y[0] + source[1] * basis.axis_y[1] + source[2] * basis.axis_y[2]);
  const gx = Math.floor((x - navigation.origin_floor_coordinates[0]) / navigation.cell_size_m);
  const gy = Math.floor((y - navigation.origin_floor_coordinates[1]) / navigation.cell_size_m);
  if (gx < 0 || gx >= navigation.width || gy < 0 || gy >= navigation.height) return false;
  // Obstacles are already inflated by radius + safety margin in the backend.
  return navigation.walkable[gy * navigation.width + gx] === 1;
}

function lockPointer() {
  if (viewer && document.pointerLockElement !== canvas) canvas.requestPointerLock();
}

document.addEventListener('mousemove', (event) => {
  if (document.pointerLockElement !== canvas || !camera) return;
  yaw -= event.movementX * 0.002;
  pitch -= event.movementY * 0.002;
  pitch = Math.max(-Math.PI / 2 + 0.02, Math.min(Math.PI / 2 - 0.02, pitch));
  applyLook();
});
document.addEventListener('pointerlockchange', () => {
  const locked = document.pointerLockElement === canvas;
  ui.crosshair.hidden = !locked;
  ui.enter.textContent = locked ? 'Esc to release cursor' : 'Click to roam';
});
document.addEventListener('keydown', (event) => {
  if (event.code === 'KeyF' && !event.repeat && document.pointerLockElement === canvas) toggleFlying();
  if (event.code === 'Space' && document.pointerLockElement === canvas) event.preventDefault();
  keys.add(event.code);
});
document.addEventListener('keyup', (event) => keys.delete(event.code));
canvas.addEventListener('click', lockPointer);
ui.enter.addEventListener('click', lockPointer);
canvas.addEventListener('wheel', (event) => {
  speed = Math.max(0.25, Math.min(8, speed * (event.deltaY > 0 ? 0.9 : 1.1)));
}, { passive: true });
window.addEventListener('resize', () => {
  if (!renderer || !camera) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
});
ui.load.addEventListener('click', loadModel);

loadProjects();
