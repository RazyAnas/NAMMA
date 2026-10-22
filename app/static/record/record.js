// NAMMA sensor recorder: records phone video, the AR camera pose of every frame and raw IMU
// samples on one clock, and saves them as one bundle for scripts/ingest_sensor_bundle.py.
//
// AR mode was removed (2026-10-22): ARCore owns the camera, so it cannot lock exposure or white
// balance, and per-frame brightness drift on plain walls was the measured cause of smeared walls.
// The recorder now uses the camera directly (getUserMedia): 4K 30 fps when the phone offers it,
// widest lens, then exposure and white balance locked (autofocus stays on). Raw gyro and accelerometer samples
// are still saved next to the video.
// Every timestamp is a DOMHighResTimeStamp in milliseconds (the performance.now() clock): XR
// frame times, Generic Sensor reading timestamps and DeviceMotion event times share it.

import { BundleFile, csvRow, ivfFrameHeader, ivfHeader, zipParts } from './bundle.js';

const $ = (id) => document.getElementById(id);
const KEYFRAME_INTERVAL = 30;
const MAX_ENCODE_QUEUE = 4;
const DEPTH_EVERY = 6;
const SNAPSHOT_WIDTH = 80; // colour for the on-phone quick model, saved next to each depth frame
const IMU_FREQUENCY_HZ = 200; // a request; Chrome clamps it (60 Hz on current Android builds, per Chromium source)
// Strict mode: a frame is kept (encoded) only if it has a tracked AR pose and fresh gyro, accelerometer and gravity
// samples (each no older than SENSOR_FRESH_MS). Rejected frames still get their pose row (video_ts empty, which the
// laptop already skips) and a line in rejects.csv saying why.
const SENSOR_FRESH_MS = 100;
// AR origin resets: when ARCore loses tracking and cannot find its old origin, the WebXR reference space fires
// 'reset' and every later pose is in a shifted frame (the same wall then appears twice in a model). Each reset, and
// any pose jump faster than JUMP_SPEED_MPS, starts a new segment; frames within RESET_SETTLE_MS of it are rejected.
const JUMP_SPEED_MPS = 3;
const RESET_SETTLE_MS = 500;

let rec = null;

const capabilities = {};

function log(message) {
  $('status').textContent = message;
}

function flag(id, ok, text) {
  const el = $(id);
  el.textContent = text;
  el.dataset.ok = ok ? 'yes' : 'no';
}

async function probe() {
  capabilities.secure = window.isSecureContext;
  flag('cap-secure', capabilities.secure, capabilities.secure ? 'yes' : 'no: open over https://');
  const codec4k = typeof VideoEncoder === 'undefined' ? null : await pickCodec(3840, 2160);
  const codec = codec4k || (typeof VideoEncoder === 'undefined' ? null : await pickCodec(1920, 1080));
  capabilities.codec = codec?.codec || null;
  flag('cap-codec', !!codec, codec ? `${codec.codec} (${codec4k ? 'up to 4K' : '1080p'})` : 'no WebCodecs encoder');
  const sensors = ['Gyroscope', 'Accelerometer', 'GravitySensor'].filter((name) => name in window);
  capabilities.sensors = sensors;
  flag('cap-imu', sensors.length || 'DeviceMotionEvent' in window, sensors.length ? sensors.join(', ') : 'DeviceMotion only');
  const cameraOk = capabilities.secure && codec && navigator.mediaDevices;
  const syncCamera = () => { $('start-camera').disabled = !cameraOk; };
  $('strict').addEventListener('change', syncCamera);
  syncCamera();
}

async function pickCodec(width, height, fps = 30) {
  // Baseline H.264 first: hardware-encoded on Android and never uses B-frames, so decode order is
  // display order and the k-th decoded frame is the k-th chunk logged in video.csv.
  // Level 4.0 covers 1080p30, 4.2 1080p60, 5.1 anything larger. Falls back to a 30 fps config.
  for (const rate of fps > 30 ? [fps, 30] : [30]) {
    const level = width * height > 1920 * 1088 ? '33' : rate > 30 ? '2A' : '28';
    const bitrate = Math.round(Math.min(50e6, (width * height * rate * 0.2)));
    const candidates = [
      { codec: `avc1.42E0${level}`, avc: { format: 'annexb' }, container: 'h264' },
      { codec: `avc1.4D40${level}`, avc: { format: 'annexb' }, container: 'h264' },
      { codec: 'vp8', container: 'ivf', fourcc: 'VP80' },
      { codec: 'vp09.00.40.08', container: 'ivf', fourcc: 'VP90' },
    ];
    for (const { container, fourcc, ...candidate } of candidates) {
      const config = { ...candidate, width, height, bitrate, framerate: rate, latencyMode: 'quality' };
      try {
        const { supported } = await VideoEncoder.isConfigSupported(config);
        if (supported) return { ...config, container, fourcc };
      } catch {
        // try the next codec
      }
    }
  }
  return null;
}

function newRecording(mode) {
  const r = {
    mode,
    startedAt: performance.now(),
    startedWall: new Date().toISOString(),
    frames: new BundleFile('frames.csv'),
    videoIndex: new BundleFile('video.csv'),
    imu: new BundleFile('imu.csv'),
    depthIndex: new BundleFile('depth.csv'),
    rejects: new BundleFile('rejects.csv'),
    strict: $('strict').checked,
    lastImu: {},
    rejected: 0,
    rejectReasons: {},
    segment: 0,
    segmentStart: -Infinity,
    resets: [],
    lastPos: null,
    depthFiles: [],
    video: null,
    encoder: null,
    codec: null,
    frameCount: 0,
    encoded: 0,
    chunks: 0,
    dropped: 0,
    tracked: 0,
    lost: 0,
    imuCounts: {},
    sensors: [],
    errors: [],
    camera: null,
    stopping: false,
  };
  r.frames.write(
    csvRow(['frame', 't_ms', 'video_ts_us', 'tracking', 'screen_angle', ...Array.from({ length: 16 }, (_, i) => `m${i}`), 'p0', 'p5', 'p8', 'p9', 'image_w', 'image_h', 'segment'])
  );
  r.videoIndex.write(csvRow(['chunk', 'ts_us', 'key', 'bytes']));
  r.imu.write(csvRow(['t_ms', 'sensor', 'x', 'y', 'z', 't_sensor_ms']));
  r.rejects.write(csvRow(['frame', 't_ms', 'reason']));
  r.depthIndex.write(csvRow(['frame', 't_ms', 'file', 'width', 'height', 'raw_to_m', ...Array.from({ length: 16 }, (_, i) => `n${i}`)]));
  return r;
}

// ---- IMU ---------------------------------------------------------------------------------------

function imuSample(sensor, t, x, y, z, tSensor) {
  if (!rec || rec.stopping) return;
  rec.imu.write(csvRow([t, sensor, x, y, z, tSensor]));
  rec.imuCounts[sensor] = (rec.imuCounts[sensor] || 0) + 1;
  // Freshness per quantity, whichever API delivered it (Generic Sensor or DeviceMotion).
  const quantity = { gyro: 'gyro', dm_gyro: 'gyro', accel: 'accel', dm_accel: 'accel', gravity: 'gravity' }[sensor];
  if (quantity) rec.lastImu[quantity] = performance.now();
  if (sensor === 'gravity') rec.hasGravitySensor = true;
  if (quantity === 'gyro') {
    // rad/s from Generic Sensor; DeviceMotion samples were already converted to rad/s in onDeviceMotion
    const degS = Math.hypot(x, y, z) * 180 / Math.PI;
    rec.turn = 0.8 * (rec.turn || 0) + 0.2 * degS;
    if (rec.turn > TURN_WARN_DEG_S) rec.lastFastTurn = performance.now();
  }
}

// Turning faster than this blurs frames and leaves too little overlap between the frames the pipeline keeps
// (it samples a few per second; past ~35 deg between two kept frames matching fails).
const TURN_WARN_DEG_S = 60;

// Why a frame fails strict mode, or null if it has everything. Gravity comes from GravitySensor once it delivers;
// where Chrome has none or blocks it, the accelerometer (which includes gravity) stands in, and meta.json says so.
function missingSensors(r, tracking) {
  const now = performance.now();
  const stale = (q) => !(now - (r.lastImu[q] ?? -Infinity) <= SENSOR_FRESH_MS);
  const missing = [];
  if (tracking !== 0) missing.push(tracking === 1 ? 'ar_tracking_lost' : 'no_ar_pose');
  if (stale('gyro')) missing.push('gyro');
  if (stale('accel')) missing.push('accel');
  if (r.hasGravitySensor ? stale('gravity') : stale('accel')) missing.push('gravity');
  return missing.length ? missing.join('+') : null;
}

function reject(r, index, t, reason) {
  r.rejected++;
  r.rejectReasons[reason] = (r.rejectReasons[reason] || 0) + 1;
  r.rejects.write(csvRow([index, t, reason]));
  r.lastReject = reason;
}

function onDeviceMotion(event) {
  // rotationRate is in deg/s. The spec puts alpha about z, but Android Chrome reports alpha, beta,
  // gamma about device x, y, z (measured on a phone against the Gyroscope sensor, 2026-10-17).
  // Stored as rad/s in device x, y, z order (x right, y up along the screen, z out of the screen).
  const r = event.rotationRate;
  const k = Math.PI / 180;
  if (r && r.alpha != null) imuSample('dm_gyro', event.timeStamp, r.alpha * k, r.beta * k, r.gamma * k);
  const a = event.accelerationIncludingGravity;
  if (a && a.x != null) imuSample('dm_accel', event.timeStamp, a.x, a.y, a.z);
  if (rec && event.interval) rec.dmIntervalMs = event.interval;
}

function startImu(r) {
  if (typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission) {
    DeviceMotionEvent.requestPermission().catch(() => 'denied');
  }
  window.addEventListener('devicemotion', onDeviceMotion);
  const kinds = { Gyroscope: 'gyro', Accelerometer: 'accel', GravitySensor: 'gravity' };
  r.hasGravitySensor = false; // set by the first GravitySensor reading; a blocked or absent sensor leaves the accelerometer
  for (const [name, tag] of Object.entries(kinds)) {
    if (!(name in window)) continue;
    try {
      const sensor = new window[name]({ frequency: IMU_FREQUENCY_HZ, referenceFrame: 'device' });
      // sensor.timestamp is the exact sample time but on Android it is not the performance.now()
      // clock, so the arrival time is logged too and the laptop maps one onto the other.
      sensor.addEventListener('reading', () => imuSample(tag, performance.now(), sensor.x, sensor.y, sensor.z, sensor.timestamp));
      sensor.addEventListener('error', (e) => r.errors.push(`${name}: ${e.error?.name || 'error'}`));
      sensor.start();
      r.sensors.push(sensor);
    } catch (error) {
      r.errors.push(`${name}: ${error.name}`);
    }
  }
}

function stopImu(r) {
  window.removeEventListener('devicemotion', onDeviceMotion);
  r.sensors.forEach((s) => s.stop());
}

// ---- Video encoding ----------------------------------------------------------------------------

async function startEncoder(r, width, height, fps = 30) {
  // H.264 needs even dimensions.
  width -= width % 2;
  height -= height % 2;
  const config = await pickCodec(width, height, fps);
  if (!config) throw new Error(`No video encoder for ${width}x${height}.`);
  const { container, fourcc, ...encoderConfig } = config;
  r.codec = { ...encoderConfig, container };
  r.video = new BundleFile(container === 'h264' ? 'video.h264' : 'video.ivf');
  if (container === 'ivf') r.video.write(ivfHeader(fourcc, width, height));
  r.encoder = new VideoEncoder({
    output: (chunk) => {
      const bytes = new Uint8Array(chunk.byteLength);
      chunk.copyTo(bytes);
      if (container === 'ivf') r.video.write(ivfFrameHeader(bytes.length, chunk.timestamp));
      r.video.write(bytes);
      r.videoIndex.write(csvRow([r.chunks++, chunk.timestamp, chunk.type === 'key' ? 1 : 0, bytes.length]));
    },
    error: (error) => r.errors.push(`encoder: ${error.message}`),
  });
  r.encoder.configure(encoderConfig);
  r.encodeWidth = width;
  r.encodeHeight = height;
}

// Returns the timestamp the frame was encoded with, or null if it was dropped. Dropping on a full
// queue keeps the capture real-time; the pose row is still written so tracking has no gap.
function encode(r, source, tMs) {
  if (!r.encoder || r.encoder.state !== 'configured' || r.encoder.encodeQueueSize > MAX_ENCODE_QUEUE) {
    r.dropped++;
    return null;
  }
  const timestamp = Math.round(tMs * 1000);
  let frame;
  try {
    frame = new VideoFrame(source, { timestamp, visibleRect: { x: 0, y: 0, width: r.encodeWidth, height: r.encodeHeight } });
  } catch (error) {
    r.dropped++;
    if (r.errors.length < 20) r.errors.push(`VideoFrame: ${error.message}`);
    return null;
  }
  r.encoder.encode(frame, { keyFrame: r.encoded % KEYFRAME_INTERVAL === 0 });
  frame.close();
  r.encoded++;
  return timestamp;
}

// ---- Camera mode (getUserMedia, no pose) -------------------------------------------------------

// Splat training assumes every frame sees the same colours: per-frame brightness drift on plain walls
// was proven to be the cause of smeared walls. But an automatic lock picked a bad (dark) exposure on a
// real phone, so the camera now starts on auto and the person filming taps "Lock exposure & colour"
// once the picture looks natural; tapping again returns to auto. Every lock/unlock is saved in
// meta.json (camera.lock_events) so the pipeline knows which frames share one exposure. Focus stays
// on continuous autofocus (a fixed focus blurred the picture). Browsers expose no HDR switch.
async function applyOne(track, constraint) {
  try {
    await track.applyConstraints({ advanced: [constraint] });
    return true;
  } catch {
    return false;
  }
}

async function prepareCamera(track) {
  const caps = track.getCapabilities?.() || {};
  const locks = { exposure: false, white_balance: false };
  if (caps.zoom && caps.zoom.min < 1) locks.wide_lens = await applyOne(track, { zoom: caps.zoom.min });
  if (caps.exposureMode?.includes('continuous')) await applyOne(track, { exposureMode: 'continuous' });
  if (caps.whiteBalanceMode?.includes('continuous')) await applyOne(track, { whiteBalanceMode: 'continuous' });
  locks.autofocus = caps.focusMode?.includes('continuous') ? await applyOne(track, { focusMode: 'continuous' }) : false;
  return locks;
}

// Freezes exposure and white balance at what auto has chosen right now, or hands them back to auto.
async function toggleLock() {
  const r = rec;
  if (!r?.track) return;
  const caps = r.track.getCapabilities?.() || {};
  const now = r.track.getSettings?.() || {};
  const lock = !r.locked;
  let exposure = false;
  let balance = false;
  if (caps.exposureMode?.includes(lock ? 'manual' : 'continuous')) {
    const c = { exposureMode: lock ? 'manual' : 'continuous' };
    if (lock && caps.exposureTime && now.exposureTime) c.exposureTime = now.exposureTime;
    if (lock && caps.iso && now.iso) c.iso = now.iso;
    exposure = await applyOne(r.track, c);
  }
  if (caps.whiteBalanceMode?.includes(lock ? 'manual' : 'continuous')) {
    const c = { whiteBalanceMode: lock ? 'manual' : 'continuous' };
    if (lock && caps.colorTemperature && now.colorTemperature) c.colorTemperature = now.colorTemperature;
    balance = await applyOne(r.track, c);
  }
  if (lock && !exposure && !balance) {
    $('lock').textContent = 'Lock not supported on this phone';
    $('lock').disabled = true;
    return;
  }
  r.locked = lock;
  r.camera.locks.exposure = lock && exposure;
  r.camera.locks.white_balance = lock && balance;
  r.camera.lock_events.push({ t_ms: performance.now() - r.startedAt, locked: lock, exposure, white_balance: balance, settings: now });
  $('lock').textContent = lock ? 'Locked · tap to go back to auto' : 'Lock exposure & colour';
  $('lock').dataset.locked = lock ? 'yes' : 'no';
}

// Camera-app style focus on the live picture: tap to focus on that spot, press and hold to lock focus
// there (hold again to go back to autofocus). Points of interest are 0..1 in the video's own frame,
// so the tap is mapped back through the object-fit: cover crop.
const HOLD_MS = 600;

function videoPoint(event) {
  const video = $('video');
  const scale = Math.max(innerWidth / video.videoWidth, innerHeight / video.videoHeight);
  const x = (event.clientX - (innerWidth - video.videoWidth * scale) / 2) / (video.videoWidth * scale);
  const y = (event.clientY - (innerHeight - video.videoHeight * scale) / 2) / (video.videoHeight * scale);
  return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
}

function showFocus(event, text) {
  const ring = $('focus-ring');
  ring.style.left = `${event.clientX}px`;
  ring.style.top = `${event.clientY}px`;
  ring.textContent = text;
  ring.hidden = false;
  clearTimeout(ring.timer);
  ring.timer = setTimeout(() => { ring.hidden = true; }, 1500);
}

async function focusAt(event, hold) {
  const r = rec;
  if (!r?.track) return;
  const caps = r.track.getCapabilities?.() || {};
  const point = videoPoint(event);
  if (hold && r.focusLocked) {
    r.focusLocked = !(await applyOne(r.track, { focusMode: 'continuous' }));
    r.camera.lock_events.push({ t_ms: performance.now() - r.startedAt, focus_locked: r.focusLocked });
    showFocus(event, r.focusLocked ? 'Unlock failed' : 'Autofocus');
    return;
  }
  // Points of interest are a Chrome extension; phones without them still focus on the centre.
  if ('pointsOfInterest' in (r.track.getSettings?.() || {}) || caps.pointsOfInterest) await applyOne(r.track, { pointsOfInterest: [point] });
  if (!hold) {
    if (caps.focusMode?.includes('single-shot')) await applyOne(r.track, { focusMode: 'single-shot' });
    if (caps.focusMode?.includes('continuous')) setTimeout(() => { if (!r.focusLocked) applyOne(r.track, { focusMode: 'continuous' }); }, 800);
    showFocus(event, '');
    return;
  }
  // Hold: let single-shot focus land on the spot, then freeze focus at that distance.
  if (caps.focusMode?.includes('single-shot')) await applyOne(r.track, { focusMode: 'single-shot' });
  await new Promise((resolve) => setTimeout(resolve, 400));
  const distance = r.track.getSettings?.().focusDistance;
  const lock = { focusMode: 'manual' };
  if (caps.focusDistance && distance) lock.focusDistance = distance;
  r.focusLocked = caps.focusMode?.includes('manual') ? await applyOne(r.track, lock) : false;
  r.camera.locks.focus = r.focusLocked;
  r.camera.lock_events.push({ t_ms: performance.now() - r.startedAt, focus_locked: r.focusLocked, point, focus_distance: distance ?? null });
  showFocus(event, r.focusLocked ? 'Focus locked' : 'Focus lock not supported');
}

function watchFocusGestures() {
  const video = $('video');
  let timer = null;
  let held = false;
  video.addEventListener('pointerdown', (event) => {
    held = false;
    timer = setTimeout(() => { held = true; focusAt(event, true); }, HOLD_MS);
  });
  const release = (event) => {
    clearTimeout(timer);
    if (!held && event.type === 'pointerup') focusAt(event, false);
  };
  video.addEventListener('pointerup', release);
  video.addEventListener('pointercancel', release);
  video.addEventListener('contextmenu', (event) => event.preventDefault());
}

// Brightness (exposure compensation) while on auto; hidden where the phone has no control for it.
function setupBrightness(track) {
  const caps = track.getCapabilities?.() || {};
  const slider = $('brightness');
  const range = caps.exposureCompensation;
  if (!range || range.max <= range.min) return;
  slider.min = range.min;
  slider.max = range.max;
  slider.step = range.step || 0.1;
  slider.value = track.getSettings?.().exposureCompensation ?? 0;
  slider.addEventListener('input', () => applyOne(track, { exposureCompensation: Number(slider.value) }));
  $('brightness-box').hidden = false;
}

// 4K at 30 fps first: resolution and sharpness matter more than 60 fps (the pipeline keeps only a few frames per
// second, the sharpest of each window). Phones without 4K, or whose encoder cannot take it, fall back to 1080p 60.
const CAMERA_MODES = {
  '4k30': { width: { min: 3200, ideal: 3840 }, height: { min: 1800, ideal: 2160 }, frameRate: { ideal: 30, max: 30 } },
  '1080p60': { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } },
};

async function openCamera() {
  for (const mode of ['4k30', '1080p60']) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, ...CAMERA_MODES[mode] } });
      const { width, height } = stream.getVideoTracks()[0].getSettings();
      const config = await pickCodec(width - (width % 2), height - (height % 2), 30);
      if (config || mode === '1080p60') return { stream, mode };
      stream.getTracks().forEach((t) => t.stop()); // camera gives 4K but nothing can encode it here
    } catch {
      // no camera mode like this on this phone; try the next
    }
  }
  throw new Error('Could not open the camera.');
}

async function startCamera() {
  const { stream, mode } = await openCamera();
  const r = newRecording('camera');
  rec = r;
  const video = $('video');
  video.srcObject = stream;
  await video.play();
  r.track = stream.getVideoTracks()[0];
  const locks = await prepareCamera(r.track);
  setupBrightness(r.track);
  r.startedAt = performance.now();
  r.startedWall = new Date().toISOString();
  const settings = r.track.getSettings?.() || {};
  r.camera = { width: video.videoWidth, height: video.videoHeight, settings, locks, mode, lock_events: [] };
  await startEncoder(r, video.videoWidth, video.videoHeight, Math.round(settings.frameRate || 30));
  startImu(r);
  showHud();
  const onFrame = (now, metadata) => {
    if (r.stopping) return;
    // captureTime is when the sensor exposed the frame; fall back to presentation time.
    const t = metadata?.captureTime || metadata?.expectedDisplayTime || now;
    const index = r.frameCount++;
    // Camera mode has no AR pose, so strict mode checks only the motion sensors here.
    const missing = r.strict ? missingSensors(r, 0) : null;
    if (missing) reject(r, index, t, missing);
    const videoTs = missing ? null : encode(r, video, t);
    r.frames.write(csvRow([index, t, videoTs, 3, screen.orientation?.angle ?? 0, ...Array(20).fill(null), video.videoWidth, video.videoHeight, 0]));
    if (index % 10 === 0) updateHud(r, t, 3);
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
}

// ---- Finish and save ---------------------------------------------------------------------------

async function finish() {
  const r = rec;
  if (!r || r.stopping) return;
  r.stopping = true;
  r.endedAt = performance.now();
  stopImu(r);
  if (r.session) r.session.end().catch(() => {});
  if (r.track) r.track.stop();
  try {
    if (r.encoder && r.encoder.state === 'configured') await r.encoder.flush();
    r.encoder?.close();
  } catch (error) {
    r.errors.push(`flush: ${error.message}`);
  }
  r.wakeLock?.release().catch(() => {});
  const durationS = (r.endedAt - r.startedAt) / 1000;
  const rates = Object.fromEntries(Object.entries(r.imuCounts).map(([k, n]) => [k, +(n / durationS).toFixed(1)]));
  const meta = {
    format: 'namma-sensor-bundle',
    version: 1,
    mode: r.mode,
    clock: 'DOMHighResTimeStamp ms (performance.now)',
    pose_convention: 'frames.csv m0..m15: column-major world_from_camera, WebXR local-floor world (y up, metres), OpenGL camera (x right, y up, looking down -z)',
    imu_convention: 'imu.csv device axes (x right, y up along screen, z out of screen); gyro rad/s, accel m/s^2 including gravity, gravity m/s^2; t_ms page clock, t_sensor_ms raw Generic Sensor timestamp (different clock)',
    started_wall: r.startedWall,
    duration_s: +durationS.toFixed(3),
    user_agent: navigator.userAgent,
    device: await navigator.userAgentData?.getHighEntropyValues?.(['model', 'platformVersion']).catch(() => null),
    screen: { width: screen.width, height: screen.height, dpr: devicePixelRatio, angle: screen.orientation?.angle ?? 0 },
    camera: r.camera,
    codec: r.codec,
    xr_features: r.features || [],
    depth_format: r.depthFormat || null,
    depth_color_size: r.snap ? [r.snap.width, r.snap.height] : null,
    frames: r.frameCount,
    frames_encoded: r.encoded,
    frames_dropped: r.dropped,
    chunks: r.chunks,
    frames_tracked: r.tracked,
    frames_tracking_lost: r.lost,
    strict: r.strict,
    strict_rule: r.strict ? `kept only with ${r.mode === 'ar' ? 'a tracked AR pose and ' : ''}gyro, accelerometer and gravity samples no older than ${SENSOR_FRESH_MS} ms` : null,
    gravity_source: r.hasGravitySensor ? 'GravitySensor' : 'accelerometer (includes gravity)',
    frames_rejected: r.rejected,
    ar_segments: r.segment + 1,
    ar_resets: r.resets,
    reject_reasons: r.rejectReasons,
    imu_rates_hz: rates,
    devicemotion_interval_ms: r.dmIntervalMs || null,
    errors: r.errors,
  };
  const metaFile = new BundleFile('meta.json');
  metaFile.write(JSON.stringify(meta, null, 2));
  const files = [metaFile, r.frames, r.videoIndex, r.imu, r.rejects, ...(r.video ? [r.video] : [])];
  if (r.depthFiles.length) files.push(r.depthIndex, ...r.depthFiles);
  const blob = new Blob(zipParts(files), { type: 'application/zip' });
  const name = `namma-${r.mode}-${r.startedWall.replace(/[:.]/g, '-').slice(0, 19)}.zip`;
  const link = $('save');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  showSummary(meta, blob.size);
  // The studio (quick model, send for render) picks the recording up from here.
  window.dispatchEvent(new CustomEvent('namma:bundle', { detail: { blob, name, meta } }));
}

// ---- UI ----------------------------------------------------------------------------------------

function showHud() {
  document.body.dataset.screen = 'recording';
  navigator.wakeLock?.request('screen').then((lock) => { if (rec) rec.wakeLock = lock; }).catch(() => {});
}

function updateHud(r, t, tracking) {
  const seconds = (t - r.startedAt) / 1000;
  $('hud-time').textContent = `${Math.max(0, seconds).toFixed(0)} s`;
  $('hud-frames').textContent = `${r.encoded} / ${r.frameCount}`;
  const gyro = r.imuCounts.gyro || r.imuCounts.dm_gyro || 0;
  $('hud-imu').textContent = seconds > 1 ? `${(gyro / seconds).toFixed(0)} Hz` : '…';
  const labels = ['tracking', 'tracking lost: slow down, aim at texture', 'no pose', 'camera only'];
  const reset = r.resets.length && t - r.resets[r.resets.length - 1].t_ms < 3000;
  const reason = reset ? 'ar_origin_reset' : r.strict ? missingSensors(r, tracking === 3 ? 0 : tracking) : null;
  const fastTurn = r.lastFastTurn && performance.now() - r.lastFastTurn < 1500;
  $('hud-track').textContent = reset ? 'AR tracking reset: walk back to an area you already scanned'
    : fastTurn ? 'Turning too fast: stop, turn slowly, then walk on'
    : reason ? `rejecting frames: ${reason.replaceAll('_', ' ').replaceAll('+', ', ')} missing` : labels[tracking];
  $('hud-track').dataset.ok = !reason && !fastTurn && (tracking === 0 || tracking === 3) ? 'yes' : 'no';
  $('hud-rejected').textContent = r.strict ? `${r.rejected}` : 'off';
  if (r.coverage) {
    const c = r.coverage.stats();
    $('hud-cover').textContent = c.seen ? `${c.percent}% good · ${c.walls} wall${c.walls === 1 ? '' : 's'}` : 'point at a wall';
  } else $('hud-cover').textContent = `${r.encodeWidth || 0} x ${r.encodeHeight || 0}`;
}

function showSummary(meta, bytes) {
  document.body.dataset.screen = 'done';
  const rows = [
    ['Mode', meta.mode],
    ['Duration', `${meta.duration_s.toFixed(1)} s`],
    ['Camera image', meta.camera ? `${meta.camera.width} x ${meta.camera.height}` : 'none'],
    ['Camera locks', meta.camera?.locks ? Object.entries(meta.camera.locks).map(([k, v]) => `${k.replace('_', ' ')} ${v ? 'on' : 'not supported'}`).join(', ') : 'n/a'],
    ['Codec', meta.codec?.codec || 'none'],
    ['Frames encoded', `${meta.frames_encoded} of ${meta.frames} (${(meta.frames_encoded / meta.duration_s).toFixed(1)} fps)`],
    ['Frames rejected', meta.strict ? `${meta.frames_rejected} (${Object.entries(meta.reject_reasons).map(([k, v]) => `${k.replaceAll('_', ' ')} ${v}`).join(', ') || 'none'})` : 'strict mode off'],
    ['IMU rates', Object.entries(meta.imu_rates_hz).map(([k, v]) => `${k} ${v} Hz`).join(', ') || 'none'],
    ['Bundle size', `${(bytes / 1e6).toFixed(1)} MB`],
    ['Problems', meta.errors.length ? meta.errors.slice(0, 5).join('; ') : 'none'],
  ];
  $('summary').replaceChildren(
    ...rows.flatMap(([k, v]) => {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = v;
      return [dt, dd];
    })
  );
}

async function begin(starter) {
  $('error').textContent = '';
  try {
    await starter();
  } catch (error) {
    rec = null;
    document.body.dataset.screen = 'setup';
    $('error').textContent = `${error.name || 'Error'}: ${error.message}`;
  }
}

$('start-camera').addEventListener('click', () => begin(startCamera));
$('stop').addEventListener('click', () => finish());
$('lock').addEventListener('click', () => toggleLock());
watchFocusGestures();
$('again').addEventListener('click', () => location.reload());
probe().catch((error) => log(error.message));

// Exposed for the automated browser test.
window.nammaRecorder = { finish, state: () => rec };
