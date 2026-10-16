import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import './style.css';

const ui = Object.fromEntries(
  ['project', 'load', 'launcher', 'loading', 'progress', 'progress-bar', 'loading-detail', 'error',
    'hud', 'controls', 'crosshair', 'enter', 'status', 'status-dot', 'project-name', 'model-size', 'fps', 'nav-status']
    .map((id) => [id, document.getElementById(id)])
);
const canvas = document.getElementById('scene');

let viewer;
let camera;
let renderer;
let navigation;
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
    try {
      navigation = await loadNavigation(projectId);
    } catch (navigationError) {
      console.warn(navigationError);
    }
    updateProgress(2, `Downloading ${formatBytes(manifest.size_bytes)} model…`);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.setClearColor(0x07110f, 1);

    camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.03, 500);
    camera.position.set(0, 0, 3);
    camera.rotation.order = 'YXZ';

    viewer = new GaussianSplats3D.Viewer({
      selfDrivenMode: false,
      renderer,
      camera,
      useBuiltInControls: false,
      gpuAcceleratedSort: true,
      enableSIMDInSort: true,
      sharedMemoryForWorkers: false,
      integerBasedSort: true,
      halfPrecisionCovariancesOnGPU: true,
      dynamicScene: false,
      renderMode: GaussianSplats3D.RenderMode.Always,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Gradual,
      sphericalHarmonicsDegree: 2,
      inMemoryCompressionLevel: 1,
      freeIntermediateSplatData: true,
    });

    await viewer.addSplatScene(`/api/v1/projects/${projectId}/gaussian-model`, {
      format: GaussianSplats3D.SceneFormat.Ply,
      splatAlphaRemovalThreshold: 5,
      showLoadingUI: false,
      progressiveLoad: true,
      onProgress: (percentage, label) => updateProgress(5 + percentage * 0.9, label || 'Decoding Gaussian splats…'),
    });

    updateProgress(100, 'Scene ready');
    const option = ui.project.selectedOptions[0];
    ui['project-name'].textContent = option?.textContent || 'Reconstruction';
    ui['model-size'].textContent = formatBytes(manifest.size_bytes);
    ui['nav-status'].textContent = navigation?.navmesh_ready ? 'collision on' : 'review required';
    ui.loading.hidden = true;
    ui.hud.hidden = false;
    ui.controls.hidden = false;
    ui.enter.hidden = false;
    setStatus('Scene ready', 'ready');
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

function updateMovement(delta) {
  if (document.pointerLockElement !== canvas) return;
  const forward = Number(keys.has('KeyW')) - Number(keys.has('KeyS'));
  const strafe = Number(keys.has('KeyD')) - Number(keys.has('KeyA'));
  const vertical = Number(keys.has('KeyE')) - Number(keys.has('KeyQ'));
  const multiplier = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 3 : 1;
  direction.set(strafe, 0, -forward);
  if (direction.lengthSq() > 0) direction.normalize();
  direction.applyQuaternion(camera.quaternion);
  direction.y += vertical;
  if (direction.lengthSq() > 0) direction.normalize();
  velocity.copy(direction).multiplyScalar(speed * multiplier * delta);
  if (!navigation?.navmesh_ready) {
    camera.position.add(velocity);
    return;
  }
  // Substep movement to prevent a fast sprint from tunnelling through a wall.
  const distance = velocity.length();
  const steps = Math.max(1, Math.ceil(distance / (navigation.cell_size_m * 0.5)));
  const step = velocity.clone().multiplyScalar(1 / steps);
  for (let index = 0; index < steps; index += 1) {
    const candidate = camera.position.clone().add(step);
    if (!isWalkable(candidate)) break;
    camera.position.copy(candidate);
  }
}

function isWalkable(position) {
  const basis = navigation.basis;
  const x = position.x * basis.axis_x[0] + position.y * basis.axis_x[1] + position.z * basis.axis_x[2];
  const y = position.x * basis.axis_y[0] + position.y * basis.axis_y[1] + position.z * basis.axis_y[2];
  const gx = Math.floor((x - navigation.origin_floor_coordinates[0]) / navigation.cell_size_m);
  const gy = Math.floor((y - navigation.origin_floor_coordinates[1]) / navigation.cell_size_m);
  const radius = Math.ceil((navigation.agent_radius_m + navigation.safety_margin_m) / navigation.cell_size_m);
  for (let row = gy - radius; row <= gy + radius; row += 1) {
    for (let column = gx - radius; column <= gx + radius; column += 1) {
      if (column < 0 || column >= navigation.width || row < 0 || row >= navigation.height) return false;
      const dx = column - gx;
      const dy = row - gy;
      if (dx * dx + dy * dy > radius * radius) continue;
      if (navigation.walkable[row * navigation.width + column] === 0) return false;
    }
  }
  return true;
}

function lockPointer() {
  if (viewer && document.pointerLockElement !== canvas) canvas.requestPointerLock();
}

document.addEventListener('mousemove', (event) => {
  if (document.pointerLockElement !== canvas || !camera) return;
  yaw -= event.movementX * 0.002;
  pitch -= event.movementY * 0.002;
  pitch = Math.max(-Math.PI / 2 + 0.02, Math.min(Math.PI / 2 - 0.02, pitch));
  camera.rotation.set(pitch, yaw, 0);
});
document.addEventListener('pointerlockchange', () => {
  const locked = document.pointerLockElement === canvas;
  ui.crosshair.hidden = !locked;
  ui.enter.textContent = locked ? 'Esc to release cursor' : 'Click to roam';
});
document.addEventListener('keydown', (event) => keys.add(event.code));
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
