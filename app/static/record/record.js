// NAMMA recorder: records phone video (4K 30 fps where the phone allows) and raw IMU samples on one clock, and
// saves them as one bundle for scripts/ingest_sensor_bundle.py.
//
// No AR: the reconstruction solves the camera of every frame from the images (SuperPoint + LightGlue, 173 of 180
// frames on the lobby test), and ARCore's poses drifted and reset across big spaces. Video quality matters more:
// ARCore owns the camera in AR mode, which caps it at a soft ~1080p stream with automatic exposure.
// Raw gyro and accelerometer samples are kept: they warn when the phone turns too fast and give a rotation prior.
// Every timestamp is a DOMHighResTimeStamp in milliseconds (the performance.now() clock).

import { BundleFile, csvRow, ivfFrameHeader, ivfHeader, zipParts } from './bundle.js';

const $ = (id) => document.getElementById(id);
const KEYFRAME_INTERVAL = 30;
const MAX_ENCODE_QUEUE = 4;
const IMU_FREQUENCY_HZ = 200; // a request; Chrome clamps it (60 Hz on current Android builds, per Chromium source)
// Strict mode: a frame is kept (encoded) only with fresh gyro, accelerometer and gravity samples (each no older than
// SENSOR_FRESH_MS). Rejected frames still get their row in frames.csv (video_ts empty) and a line in rejects.csv.
const SENSOR_FRESH_MS = 100;

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
  const codec = typeof VideoEncoder === 'undefined' ? null : await pickCodec(1920, 1080);
  capabilities.codec = codec?.codec || null;
  flag('cap-codec', !!codec, codec ? `${codec.codec} (1080p)` : 'no WebCodecs encoder');
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

async function applyOne(track, constraint) {
  try {
    await track.applyConstraints({ advanced: [constraint] });
    return true;
  } catch {
    return false;
  }
}

// Exposure and white balance start automatic, so the preview looks natural. The user locks them with one tap when
// the view looks right (bright part and dark part of the room both readable), so the video keeps one exposure and
// one colour for the whole walk, which splat training needs. Focus stays automatic (a locked focus blurs).
async function widenLens(track) {
  const caps = track.getCapabilities?.() || {};
  const locks = {};
  if (caps.zoom && caps.zoom.min < 1) locks.wide_lens = await applyOne(track, { zoom: caps.zoom.min });
  if (caps.focusMode?.includes('continuous')) locks.focus_continuous = await applyOne(track, { focusMode: 'continuous' });
  return locks;
}

async function setLook(track, locked) {
  const caps = track.getCapabilities?.() || {};
  const now = track.getSettings?.() || {};
  const done = {};
  if (caps.exposureMode?.includes(locked ? 'manual' : 'continuous')) {
    const exposure = { exposureMode: locked ? 'manual' : 'continuous' };
    if (locked && caps.exposureTime && now.exposureTime) exposure.exposureTime = now.exposureTime;
    if (locked && caps.iso && now.iso) exposure.iso = now.iso;
    done.exposure = await applyOne(track, exposure);
  }
  if (caps.whiteBalanceMode?.includes(locked ? 'manual' : 'continuous')) {
    const balance = { whiteBalanceMode: locked ? 'manual' : 'continuous' };
    if (locked && caps.colorTemperature && now.colorTemperature) balance.colorTemperature = now.colorTemperature;
    done.white_balance = await applyOne(track, balance);
  }
  return done;
}

// Focus, like the phone's own camera app: tap the picture to focus there, press and hold (0.6 s) to lock focus at
// that distance (tap again to go back to automatic). Works in the preview and while recording. The focus button
// toggles the same lock. Phones without these controls in the browser just keep autofocus.
function focusControls(track) {
  const caps = track.getCapabilities?.() || {};
  const video = $('video'), button = $('focus-lock');
  const canLock = !!caps.focusMode?.includes('single-shot');
  const canPoint = 'pointsOfInterest' in (navigator.mediaDevices.getSupportedConstraints?.() || {});
  button.hidden = !canLock;
  let locked = false, timer = null;
  const show = () => {
    button.setAttribute('aria-pressed', String(locked));
    button.textContent = locked ? 'Focus locked (tap to unlock)' : 'Lock focus';
    $('hud-focus').textContent = locked ? 'focus locked' : '';
  };
  // Locking focus = focus once on the chosen spot ('single-shot'), which then holds that distance. A 'manual' lock
  // without a reliable current distance made Chrome jump to the nearest focus on a real phone, so it is used only
  // when the "Focus distance" slider sets an explicit value.
  const lock = async (on, point) => {
    const target = point ? { pointsOfInterest: [point] } : {};
    const ok = on
      ? (canPoint && point ? await applyOne(track, { ...target, focusMode: 'single-shot' }) : false)
        || await applyOne(track, { focusMode: 'single-shot' })
      : await applyOne(track, { focusMode: 'continuous' });
    if (ok) locked = on;
    if (rec) (rec.focusEvents ||= []).push({ t_ms: performance.now(), locked, ok, point: point || null });
    show();
  };
  video.onpointerdown = (e) => {
    // The picture is shown with object-fit: cover, so map the tap back into the video's own frame.
    const s = Math.max(innerWidth / video.videoWidth, innerHeight / video.videoHeight);
    const x = Math.min(1, Math.max(0, (e.clientX - (innerWidth - video.videoWidth * s) / 2) / (video.videoWidth * s)));
    const y = Math.min(1, Math.max(0, (e.clientY - (innerHeight - video.videoHeight * s) / 2) / (video.videoHeight * s)));
    timer = setTimeout(() => { timer = null; if (canLock) lock(true, { x, y }); }, 600);
    if (canPoint && !locked) applyOne(track, { pointsOfInterest: [{ x, y }], focusMode: 'single-shot' })
      .then((ok) => ok || applyOne(track, { pointsOfInterest: [{ x, y }] }));
  };
  video.onpointerup = video.onpointercancel = () => {
    if (timer) { clearTimeout(timer); timer = null; if (locked) lock(false); }
  };
  button.onclick = () => lock(!locked);
  show();
  return () => locked;
}

// Manual camera controls, like a pro camera app: one slider per setting this phone's browser exposes. Moving ISO or
// shutter switches exposure to manual, moving colour temperature switches white balance to manual; the "Lock" button
// then freezes whatever is on screen. Only what the phone supports is shown.
const MANUAL = [
  ['exposureCompensation', 'Brightness (EV)', null],
  ['iso', 'ISO', { exposureMode: 'manual' }],
  ['exposureTime', 'Shutter (1/10000 s units)', { exposureMode: 'manual' }],
  ['colorTemperature', 'Colour temperature (K)', { whiteBalanceMode: 'manual' }],
  ['contrast', 'Contrast', null],
  ['saturation', 'Saturation', null],
  ['sharpness', 'Sharpness', null],
  ['zoom', 'Zoom', null],
  ['focusDistance', 'Focus distance (manual focus)', { focusMode: 'manual' }],
];
function manualControls(track) {
  const caps = track.getCapabilities?.() || {}, now = track.getSettings?.() || {}, box = $('manual');
  box.textContent = '';
  for (const [key, label, mode] of MANUAL) {
    const range = caps[key];
    if (!range || typeof range.min !== 'number' || range.max <= range.min) continue;
    const row = document.createElement('label'); row.className = 'slider';
    const name = document.createElement('span'), value = document.createElement('output'), input = document.createElement('input');
    name.textContent = label; input.type = 'range';
    Object.assign(input, { min: range.min, max: range.max, step: range.step || (range.max - range.min) / 100, value: now[key] ?? range.min });
    value.textContent = (+input.value).toFixed(key === 'exposureCompensation' ? 1 : 0);
    input.oninput = () => {
      value.textContent = (+input.value).toFixed(key === 'exposureCompensation' ? 1 : 0);
      applyOne(track, { ...(mode || {}), [key]: +input.value });
    };
    row.append(name, value, input); box.append(row);
  }
  const auto = document.createElement('button');
  auto.type = 'button'; auto.className = 'secondary'; auto.textContent = 'Back to automatic';
  auto.onclick = async () => { await setLook(track, false); manualControls(track); };
  if (box.children.length) box.append(auto);
  else box.textContent = 'This phone does not expose manual camera settings to the browser.';
}

// Preview before recording: brightness slider (exposure compensation, while automatic), a lock toggle, Start.
function previewControls(track) {
  const caps = track.getCapabilities?.() || {};
  const lock = $('lock');
  let locked = false;
  manualControls(track);
  lock.hidden = !(caps.exposureMode?.includes('manual') || caps.whiteBalanceMode?.includes('manual'));
  lock.setAttribute('aria-pressed', 'false');
  lock.textContent = 'Lock exposure and colour';
  lock.onclick = async () => {
    locked = !locked;
    const done = await setLook(track, locked);
    lock.setAttribute('aria-pressed', String(locked));
    lock.textContent = locked ? 'Locked (tap to unlock)' : 'Lock exposure and colour';
    $('manual').querySelectorAll('input, button').forEach((el) => { el.disabled = locked; });
    $('preview-note').textContent = locked && !done.exposure && !done.white_balance ? 'This phone does not let the browser lock them.' : '';
  };
  return new Promise((resolve) => { $('go').onclick = () => resolve({ locked }); });
}

// 4K at 30 fps first: resolution and sharpness matter more than 60 fps (the pipeline keeps only a few frames per
// second, the sharpest of each window). Phones without 4K, or whose encoder cannot take it, fall back to 1080p 60.
const CAMERA_MODES = {
  '4k30': { width: { min: 3200, ideal: 3840 }, height: { min: 1800, ideal: 2160 }, frameRate: { ideal: 30, max: 30 } },
  '1080p60': { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } },
};

async function openCamera() {
  const wanted = $('quality')?.value || '4k30';
  for (const mode of wanted === '4k30' ? ['4k30', '1080p60'] : ['1080p60']) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, ...CAMERA_MODES[mode] } });
      const { width, height } = stream.getVideoTracks()[0].getSettings();
      const config = await pickCodec(width - (width % 2), height - (height % 2), 30);
      if (config || mode === '1080p60') return { stream, mode };
      stream.getTracks().forEach((t) => t.stop());   // camera gives 4K but nothing can encode it here
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
  const lens = await widenLens(r.track);
  r.focusLocked = focusControls(r.track);
  document.body.dataset.screen = 'preview';
  const { locked } = await previewControls(r.track);
  const locks = { ...lens, look_locked_by_user: locked, focus_locked_at_start: r.focusLocked() };
  r.startedAt = performance.now(); // the recording starts when the user presses Start, not at getUserMedia
  r.startedWall = new Date().toISOString();
  const settings = r.track.getSettings?.() || {};
  r.camera = { width: video.videoWidth, height: video.videoHeight, settings, locks, mode };
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
    focus_events: r.focusEvents || [],
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
  const reset = false;
  const reason = r.strict ? missingSensors(r, 0) : null;
  const fastTurn = r.lastFastTurn && performance.now() - r.lastFastTurn < 1500;
  $('hud-track').textContent = reset ? 'AR tracking reset: walk back to an area you already scanned'
    : fastTurn ? 'Turning too fast: stop, turn slowly, then walk on'
    : reason ? `rejecting frames: ${reason.replaceAll('_', ' ').replaceAll('+', ', ')} missing` : labels[tracking];
  $('hud-track').dataset.ok = !reason && !fastTurn && (tracking === 0 || tracking === 3) ? 'yes' : 'no';
  $('hud-rejected').textContent = r.strict ? `${r.rejected}` : 'off';
  if (r.coverage) {   // only bundles from the old AR recorder had depth coverage
    const c = r.coverage.stats();
    $('hud-cover').textContent = c.seen ? `${c.percent}% good · ${c.walls} wall${c.walls === 1 ? '' : 's'}` : 'point at a wall';
  } else $('hud-cover').textContent = 'needs depth';
}

function showSummary(meta, bytes) {
  document.body.dataset.screen = 'done';
  const rows = [
    ['Mode', meta.mode],
    ['Duration', `${meta.duration_s.toFixed(1)} s`],
    ['Camera image', meta.camera ? `${meta.camera.width} x ${meta.camera.height}` : 'none'],
    ['Camera locks', meta.camera?.locks ? Object.entries(meta.camera.locks).map(([k, v]) => `${k.replace('_', ' ')} ${v ? 'on' : 'not supported'}`).join(', ') : 'n/a (AR mode)'],
    ['Codec', meta.codec?.codec || 'none'],
    ['Frames encoded', `${meta.frames_encoded} of ${meta.frames} (${(meta.frames_encoded / meta.duration_s).toFixed(1)} fps)`],
    ['Frames rejected', meta.strict ? `${meta.frames_rejected} (${Object.entries(meta.reject_reasons).map(([k, v]) => `${k.replaceAll('_', ' ')} ${v}`).join(', ') || 'none'})` : 'strict mode off'],
    ['Pose tracked', meta.mode === 'ar' ? `${((100 * meta.frames_tracked) / Math.max(1, meta.frames)).toFixed(0)}% of frames` : 'n/a'],
    ['IMU rates', Object.entries(meta.imu_rates_hz).map(([k, v]) => `${k} ${v} Hz`).join(', ') || 'none'],
    ['Depth', meta.xr_features.includes('depth-sensing') ? 'on' : 'off'],
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
$('again').addEventListener('click', () => location.reload());
probe().catch((error) => log(error.message));

// Exposed for the automated browser test.
window.nammaRecorder = { finish, state: () => rec };
