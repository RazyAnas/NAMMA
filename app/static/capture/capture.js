// NAMMA live capture: runs in the phone browser, picks sharp keyframes on-device and streams them.
//
// Robustness is handled at the source, before a bad frame can reach the reconstruction:
// - the camera's exposure, white balance and focus are locked once auto-exposure settles, so
//   lighting drift and flicker-chasing never start, and exposure time is capped at 1/120 s
//   (ISO raised to compensate) to cut motion blur four times compared with 1/30 s;
// - the gyroscope predicts blur in pixels (angular rate x exposure x focal length) and asks the
//   user to slow down before frames smear;
// - each 0.4 s window sends only its sharpest frame, scored like the server (Laplacian
//   variance over intensity variance, so brightness does not bias it).

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(
  ['video', 'map', 'link', 'motion-bar', 'sharp-bar', 'advice', 'sent', 'mapped', 'points', 'locks', 'start', 'finish', 'error']
    .map((id) => [id, $(id)])
);

const WINDOW_MS = 400;
const MAX_WIDTH = 1280;
const MAX_EXPOSURE_S = 1 / 120;
const params = new URLSearchParams(location.search);
let session = params.get('session');
let token = params.get('token');
let socket;
let track;
let exposureSeconds = 1 / 30;
let angularRate = 0; // rad/s
let gravity = null; // unit vector, OpenCV camera axes
let sent = 0;
let running = false;
let encoding = false;
let windowStart = 0;
let best = null;
const recentSharpness = [];

const small = document.createElement('canvas');
const smallCtx = small.getContext('2d', { willReadFrequently: true });
const full = document.createElement('canvas');
const fullCtx = full.getContext('2d');

const api = (path, options = {}) =>
  fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } }).then(async (r) => {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    return r.json();
  });

function setLevel(bar, fraction, level) {
  bar.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  bar.dataset.level = level;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[sorted.length >> 1] : 0;
}

function sharpness(width, height) {
  const { data } = smallCtx.getImageData(0, 0, width, height);
  const gray = new Float32Array(width * height);
  let sum = 0;
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    sum += gray[i];
  }
  const mean = sum / gray.length;
  let variance = 0;
  for (const g of gray) variance += (g - mean) ** 2;
  variance /= gray.length;
  let lapSum = 0;
  let lapSq = 0;
  let n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width] - 4 * gray[i];
      lapSum += lap;
      lapSq += lap * lap;
      n++;
    }
  }
  const lapVar = lapSq / n - (lapSum / n) ** 2;
  return { sharpness: lapVar / (variance + 1e-6), luma: mean / 255 };
}

async function lockCamera() {
  if (!track.getCapabilities) return 'auto (no camera controls)';
  const caps = track.getCapabilities();
  const now = track.getSettings();
  const advanced = {};
  const locked = [];
  if (caps.exposureMode?.includes('manual') && caps.exposureTime) {
    // exposureTime is in units of 100 microseconds (Media Capture Image spec).
    const current = now.exposureTime || caps.exposureTime.max;
    const target = Math.max(caps.exposureTime.min, Math.min(current, MAX_EXPOSURE_S * 1e4));
    advanced.exposureMode = 'manual';
    advanced.exposureTime = target;
    if (caps.iso && now.iso && target < current) advanced.iso = Math.min(caps.iso.max, (now.iso * current) / target);
    exposureSeconds = target / 1e4;
    locked.push(`1/${Math.round(1 / exposureSeconds)} s`);
  } else if (now.exposureTime) {
    exposureSeconds = now.exposureTime / 1e4;
  }
  if (caps.whiteBalanceMode?.includes('manual')) {
    advanced.whiteBalanceMode = 'manual';
    if (caps.colorTemperature && now.colorTemperature) advanced.colorTemperature = now.colorTemperature;
    locked.push('WB');
  }
  if (caps.focusMode?.includes('manual') && now.focusDistance != null) {
    advanced.focusMode = 'manual';
    advanced.focusDistance = now.focusDistance;
    locked.push('focus');
  }
  if (!locked.length) return 'auto (browser cannot lock)';
  try {
    await track.applyConstraints({ advanced: [advanced] });
    return `locked ${locked.join(', ')}`;
  } catch {
    return 'auto (lock refused)';
  }
}

function onMotion(event) {
  const r = event.rotationRate;
  if (r && r.alpha != null) angularRate = (Math.hypot(r.alpha, r.beta, r.gamma) * Math.PI) / 180;
  const a = event.accelerationIncludingGravity;
  if (a && a.x != null) {
    const norm = Math.hypot(a.x, a.y, a.z) || 1;
    // Device axes: x right, y up, z out of the screen; the reading points away from gravity.
    const [dx, dy, dz] = [-a.x / norm, -a.y / norm, -a.z / norm];
    const angle = (((screen.orientation && screen.orientation.angle) || 0) * Math.PI) / 180;
    const sx = dx * Math.cos(angle) + dy * Math.sin(angle);
    const sy = -dx * Math.sin(angle) + dy * Math.cos(angle);
    gravity = [sx, -sy, -dz]; // OpenCV camera axes for the rear camera: x right, y down, z forward
  }
}

function predictedBlurPixels() {
  const focalPx = 0.78 * (ui.video.videoWidth || 1280); // about 65 degrees horizontal field of view
  return angularRate * exposureSeconds * focalPx;
}

function onFrame(now) {
  if (!running) return;
  const width = 192;
  const height = Math.round((width * ui.video.videoHeight) / ui.video.videoWidth) || 108;
  small.width = width;
  small.height = height;
  smallCtx.drawImage(ui.video, 0, 0, width, height);
  const score = sharpness(width, height);

  const blur = predictedBlurPixels();
  setLevel(ui['motion-bar'], blur / 12, blur < 3 ? 'ok' : blur < 8 ? 'warn' : 'bad');
  const reference = median(recentSharpness) || score.sharpness;
  setLevel(ui['sharp-bar'], score.sharpness / (2 * reference), score.sharpness >= 0.5 * reference ? 'ok' : 'bad');

  if (!encoding && (!best || score.sharpness > best.sharpness)) {
    const scale = Math.min(1, MAX_WIDTH / ui.video.videoWidth);
    full.width = Math.round(ui.video.videoWidth * scale);
    full.height = Math.round(ui.video.videoHeight * scale);
    fullCtx.drawImage(ui.video, 0, 0, full.width, full.height);
    best = { ...score, blur, t: performance.now() / 1000, gravity };
  }
  if (now - windowStart >= WINDOW_MS && best && !encoding) {
    windowStart = now;
    const candidate = best;
    best = null;
    if (recentSharpness.length >= 5 && candidate.sharpness < 0.5 * median(recentSharpness)) {
      ui.advice.textContent = 'Too fast: slow down so frames stay sharp.';
    } else if (candidate.luma < 0.08) {
      ui.advice.textContent = 'Too dark: add light or move closer to it.';
    } else {
      ui.advice.textContent = blur > 8 ? 'Slow down a little.' : 'Good. Keep moving slowly and overlap your views.';
      send(candidate);
    }
    recentSharpness.push(candidate.sharpness);
    if (recentSharpness.length > 30) recentSharpness.shift();
  }
  requestFrame();
}

function requestFrame() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) ui.video.requestVideoFrameCallback((now) => onFrame(now));
  else requestAnimationFrame(onFrame);
}

function send(candidate) {
  if (!socket || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 2_000_000) return;
  encoding = true;
  full.toBlob(
    async (blob) => {
      try {
        const header = new TextEncoder().encode(
          JSON.stringify({ t: candidate.t, phone_sharpness: candidate.sharpness, predicted_blur_px: candidate.blur, gravity: candidate.gravity })
        );
        const packet = new Uint8Array(4 + header.length + blob.size);
        new DataView(packet.buffer).setUint32(0, header.length, true);
        packet.set(header, 4);
        packet.set(new Uint8Array(await blob.arrayBuffer()), 4 + header.length);
        socket.send(packet);
        sent++;
        ui.sent.textContent = sent;
      } finally {
        encoding = false;
      }
    },
    'image/jpeg',
    0.92
  );
}

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/api/v1/live-sessions/${session}/stream?token=${encodeURIComponent(token)}`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    ui.link.textContent = 'Streaming';
    ui.link.dataset.state = 'on';
  };
  socket.onclose = () => {
    ui.link.textContent = 'Offline';
    ui.link.dataset.state = 'off';
    if (running) setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    const ack = JSON.parse(event.data);
    if (ack.registered != null) ui.mapped.textContent = `${ack.registered}/${ack.keyframes}`;
    if (ack.points != null) ui.points.textContent = ack.points;
    if (ack.clipped > 0.15) ui.advice.textContent = 'Highlights are blown out: avoid pointing at bright windows.';
    ui.link.dataset.state = ack.lost_recent > 3 ? 'warn' : 'on';
  };
}

async function pollMap() {
  if (!session) return;
  try {
    const state = await api(`/api/v1/live-sessions/${session}?token=${encodeURIComponent(token)}`);
    drawMap(state);
  } catch {
    // The map is a preview; a missed poll is harmless.
  }
}

function drawMap(state) {
  const ctx = ui.map.getContext('2d');
  const { width, height } = ui.map;
  ctx.clearRect(0, 0, width, height);
  const all = [...state.sample_points, ...state.camera_positions];
  if (!all.length) return;
  const xs = all.map((p) => p[0]);
  const zs = all.map((p) => p[2]);
  const [minX, maxX, minZ, maxZ] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
  const scale = (0.9 * Math.min(width, height)) / Math.max(maxX - minX, maxZ - minZ, 1e-6);
  const project = (p) => [width / 2 + (p[0] - (minX + maxX) / 2) * scale, height / 2 + (p[2] - (minZ + maxZ) / 2) * scale];
  state.sample_points.forEach((p, i) => {
    const c = state.sample_colors[i] || [200, 200, 200];
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    const [x, y] = project(p);
    ctx.fillRect(x, y, 1.5, 1.5);
  });
  ctx.strokeStyle = '#b8ff66';
  ctx.beginPath();
  state.camera_positions.forEach((p, i) => {
    const [x, y] = project(p);
    if (i) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  });
  ctx.stroke();
}

async function start() {
  ui.error.textContent = '';
  try {
    if (!window.isSecureContext) throw new Error('Open this page over https:// (see README, live capture).');
    if (typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission) {
      await DeviceMotionEvent.requestPermission().catch(() => 'denied');
    }
    window.addEventListener('devicemotion', onMotion);
    if (!session) {
      const created = await api('/api/v1/live-sessions', {
        method: 'POST',
        body: JSON.stringify({ name: `Live capture ${new Date().toLocaleString()}` }),
      });
      session = created.session_id;
      token = created.token;
      history.replaceState(null, '', created.capture_path);
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
    });
    ui.video.srcObject = stream;
    track = stream.getVideoTracks()[0];
    await ui.video.play();
    ui.advice.textContent = 'Hold still while exposure settles…';
    await new Promise((resolve) => setTimeout(resolve, 1500));
    ui.locks.textContent = await lockCamera();
    connect();
    running = true;
    ui.start.disabled = true;
    ui.finish.disabled = false;
    setInterval(pollMap, 2000);
    requestFrame();
  } catch (error) {
    ui.error.textContent = error.message;
  }
}

async function finish() {
  running = false;
  ui.finish.disabled = true;
  socket?.close();
  track?.stop();
  try {
    const job = await api(`/api/v1/live-sessions/${session}/finish?token=${encodeURIComponent(token)}`, { method: 'POST' });
    ui.advice.textContent = `Building the 3D model (job ${job.id.slice(0, 8)}). Open /viewer/ on the laptop when it finishes.`;
  } catch (error) {
    ui.error.textContent = error.message;
  }
}

ui.start.addEventListener('click', start);
ui.finish.addEventListener('click', finish);
if (session) ui.advice.textContent = 'Session ready. Press Start.';
