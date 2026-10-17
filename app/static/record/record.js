// NAMMA sensor recorder: records phone video, the AR camera pose of every frame and raw IMU
// samples on one clock, and saves them as one bundle for scripts/ingest_sensor_bundle.py.
//
// Why each stream is worth recording:
// - AR mode runs ARCore through WebXR. ARCore already fuses camera and IMU, so its per-frame pose
//   is metric and gravity-aligned; frames and poses come from the same ARCore frame, so they are
//   synchronised by construction. The camera image is read with WebXR Raw Camera Access
//   ('camera-access'), because ARCore owns the camera and getUserMedia cannot open it meanwhile.
// - Raw gyro and accelerometer samples are kept as well: they cover gaps where ARCore loses
//   tracking (pose.emulatedPosition), give a rotation prior for blurred frames, and let the
//   laptop measure the IMU-to-camera time offset instead of assuming it.
// - Camera mode (no ARCore, or Raw Camera Access unavailable) records video and IMU only.
// Every timestamp is a DOMHighResTimeStamp in milliseconds (the performance.now() clock): XR
// frame times, Generic Sensor reading timestamps and DeviceMotion event times share it.

import { BundleFile, csvRow, intrinsicsFromProjection, ivfFrameHeader, ivfHeader, zipParts } from './bundle.js';

const $ = (id) => document.getElementById(id);
const KEYFRAME_INTERVAL = 30;
const MAX_ENCODE_QUEUE = 4;
const DEPTH_EVERY = 6;
const IMU_FREQUENCY_HZ = 200; // a request; Chrome clamps it (60 Hz on current Android builds, per Chromium source)

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
  capabilities.ar = false;
  if (navigator.xr) {
    try {
      capabilities.ar = await navigator.xr.isSessionSupported('immersive-ar');
    } catch {
      capabilities.ar = false;
    }
  }
  flag('cap-ar', capabilities.ar, capabilities.ar ? 'immersive-ar supported' : navigator.xr ? 'no immersive-ar (ARCore missing?)' : 'no WebXR');
  capabilities.rawCamera = typeof XRWebGLBinding !== 'undefined' && 'getCameraImage' in XRWebGLBinding.prototype;
  flag('cap-camera', capabilities.rawCamera, capabilities.rawCamera ? 'getCameraImage present' : 'missing');
  const codec = typeof VideoEncoder === 'undefined' ? null : await pickCodec(1920, 1080);
  capabilities.codec = codec?.codec || null;
  flag('cap-codec', !!codec, codec ? `${codec.codec} (1080p)` : 'no WebCodecs encoder');
  const sensors = ['Gyroscope', 'Accelerometer', 'GravitySensor'].filter((name) => name in window);
  capabilities.sensors = sensors;
  flag('cap-imu', sensors.length || 'DeviceMotionEvent' in window, sensors.length ? sensors.join(', ') : 'DeviceMotion only');
  $('start-ar').disabled = !(capabilities.secure && capabilities.ar && capabilities.rawCamera && codec);
  $('start-camera').disabled = !(capabilities.secure && codec && navigator.mediaDevices);
}

async function pickCodec(width, height) {
  // Baseline H.264 first: hardware-encoded on Android and never uses B-frames, so decode order is
  // display order and the k-th decoded frame is the k-th chunk logged in video.csv.
  const level = width * height > 1920 * 1088 ? '33' : '28';
  const bitrate = Math.round(Math.min(40e6, (width * height * 30 * 0.17)));
  const candidates = [
    { codec: `avc1.42E0${level}`, avc: { format: 'annexb' }, container: 'h264' },
    { codec: `avc1.4D40${level}`, avc: { format: 'annexb' }, container: 'h264' },
    { codec: 'vp8', container: 'ivf', fourcc: 'VP80' },
    { codec: 'vp09.00.40.08', container: 'ivf', fourcc: 'VP90' },
  ];
  for (const { container, fourcc, ...candidate } of candidates) {
    const config = { ...candidate, width, height, bitrate, framerate: 30, latencyMode: 'quality' };
    try {
      const { supported } = await VideoEncoder.isConfigSupported(config);
      if (supported) return { ...config, container, fourcc };
    } catch {
      // try the next codec
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
    csvRow(['frame', 't_ms', 'video_ts_us', 'tracking', 'screen_angle', ...Array.from({ length: 16 }, (_, i) => `m${i}`), 'p0', 'p5', 'p8', 'p9', 'image_w', 'image_h'])
  );
  r.videoIndex.write(csvRow(['chunk', 'ts_us', 'key', 'bytes']));
  r.imu.write(csvRow(['t_ms', 'sensor', 'x', 'y', 'z']));
  r.depthIndex.write(csvRow(['frame', 't_ms', 'file', 'width', 'height', 'raw_to_m', ...Array.from({ length: 16 }, (_, i) => `n${i}`)]));
  return r;
}

// ---- IMU ---------------------------------------------------------------------------------------

function imuSample(sensor, t, x, y, z) {
  if (!rec || rec.stopping) return;
  rec.imu.write(csvRow([t, sensor, x, y, z]));
  rec.imuCounts[sensor] = (rec.imuCounts[sensor] || 0) + 1;
}

function onDeviceMotion(event) {
  // rotationRate is in deg/s with alpha about z, beta about x, gamma about y (device axes:
  // x right, y up along the screen, z out of the screen). Stored as rad/s in x, y, z order.
  const r = event.rotationRate;
  const k = Math.PI / 180;
  if (r && r.alpha != null) imuSample('dm_gyro', event.timeStamp, r.beta * k, r.gamma * k, r.alpha * k);
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
  for (const [name, tag] of Object.entries(kinds)) {
    if (!(name in window)) continue;
    try {
      const sensor = new window[name]({ frequency: IMU_FREQUENCY_HZ, referenceFrame: 'device' });
      sensor.addEventListener('reading', () => imuSample(tag, sensor.timestamp, sensor.x, sensor.y, sensor.z));
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

async function startEncoder(r, width, height) {
  // H.264 needs even dimensions.
  width -= width % 2;
  height -= height % 2;
  const config = await pickCodec(width, height);
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

// ---- AR mode (WebXR + ARCore) ------------------------------------------------------------------

const VERTEX = `attribute vec2 a; varying vec2 uv; void main() { uv = a * 0.5 + 0.5; gl_Position = vec4(a, 0.0, 1.0); }`;
const FRAGMENT = `precision mediump float; varying vec2 uv; uniform sampler2D img; void main() { gl_FragColor = texture2D(img, uv); }`;

function blitProgram(gl) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  return { program, buffer, attribute: gl.getAttribLocation(program, 'a') };
}

function drawTexture(gl, blit, texture, width, height) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  gl.useProgram(blit.program);
  gl.bindBuffer(gl.ARRAY_BUFFER, blit.buffer);
  gl.enableVertexAttribArray(blit.attribute);
  gl.vertexAttribPointer(blit.attribute, 2, gl.FLOAT, false, 0, 0);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

async function startAr() {
  const wantDepth = $('depth').checked;
  const session = await navigator.xr.requestSession('immersive-ar', {
    requiredFeatures: ['local-floor', 'camera-access'],
    optionalFeatures: ['dom-overlay', ...(wantDepth ? ['depth-sensing'] : [])],
    domOverlay: { root: $('hud') },
    depthSensing: { usagePreference: ['cpu-optimized'], dataFormatPreference: ['luminance-alpha', 'float32'] },
  });
  const r = newRecording('ar');
  rec = r;
  r.session = session;
  const canvas = document.createElement('canvas');
  // preserveDrawingBuffer so VideoFrame(canvas) sees the blit even if Chrome swaps buffers between.
  const options = { xrCompatible: true, alpha: false, antialias: false, preserveDrawingBuffer: true };
  const gl = canvas.getContext('webgl2', options) || canvas.getContext('webgl', options);
  await gl.makeXRCompatible();
  const layer = new XRWebGLLayer(session, gl);
  session.updateRenderState({ baseLayer: layer });
  const binding = new XRWebGLBinding(session, gl);
  const space = await session.requestReferenceSpace('local-floor');
  const blit = blitProgram(gl);
  r.features = session.enabledFeatures || [];
  r.depthOn = r.features.includes('depth-sensing');
  // A tap on the overlay would otherwise also fire an XR "select" (lesson from sixthsense-lab).
  $('hud').addEventListener('beforexrselect', (e) => e.preventDefault());
  session.addEventListener('end', () => finish());
  startImu(r);
  showHud();
  let encoderPending = null;

  session.requestAnimationFrame(function loop(t, frame) {
    if (r.stopping) return;
    session.requestAnimationFrame(loop);
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const pose = frame.getViewerPose(space);
    const index = r.frameCount++;
    // Tracking loss shows up as emulatedPosition (orientation keeps updating, position freezes),
    // not as a missing pose; mark it so the laptop can drop or bridge those frames.
    const tracking = !pose ? 2 : pose.emulatedPosition ? 1 : 0;
    if (tracking === 0) r.tracked++;
    else r.lost++;
    const view = pose?.views[0];
    const camera = view?.camera;
    let videoTs = null;
    if (camera) {
      if (!r.camera) {
        r.camera = { width: camera.width, height: camera.height };
        canvas.width = camera.width;
        canvas.height = camera.height;
        encoderPending = startEncoder(r, camera.width, camera.height).catch((error) => {
          r.errors.push(error.message);
          log(error.message);
        });
      }
      if (r.encoder) {
        const texture = binding.getCameraImage(camera);
        if (texture) {
          drawTexture(gl, blit, texture, camera.width, camera.height);
          videoTs = encode(r, canvas, t);
        }
      } else if (encoderPending) {
        r.dropped++;
      }
    }
    const m = view ? view.transform.matrix : null;
    const p = view ? view.projectionMatrix : null;
    r.frames.write(
      csvRow([
        index,
        t,
        videoTs,
        tracking,
        screen.orientation?.angle ?? 0,
        ...(m ? Array.from(m) : Array(16).fill(null)),
        p?.[0], p?.[5], p?.[8], p?.[9],
        camera?.width,
        camera?.height,
      ])
    );
    if (r.depthOn && view && index % DEPTH_EVERY === 0) recordDepth(r, frame, view, index, t);
    if (index % 10 === 0) updateHud(r, t, tracking);
  });
}

function recordDepth(r, frame, view, index, t) {
  let depth;
  try {
    depth = frame.getDepthInformation(view);
  } catch {
    return;
  }
  if (!depth?.data) return;
  const name = `depth/${String(index).padStart(6, '0')}.bin`;
  const file = new BundleFile(name);
  file.write(new Uint8Array(depth.data.slice(0)));
  r.depthFiles.push(file);
  const n = depth.normDepthBufferFromNormView?.matrix;
  r.depthIndex.write(csvRow([index, t, name, depth.width, depth.height, depth.rawValueToMeters, ...(n ? Array.from(n) : Array(16).fill(null))]));
}

// ---- Camera mode (getUserMedia, no pose) -------------------------------------------------------

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
  });
  const r = newRecording('camera');
  rec = r;
  const video = $('video');
  video.srcObject = stream;
  await video.play();
  r.track = stream.getVideoTracks()[0];
  r.camera = { width: video.videoWidth, height: video.videoHeight, settings: r.track.getSettings?.() };
  await startEncoder(r, video.videoWidth, video.videoHeight);
  startImu(r);
  showHud();
  const onFrame = (now, metadata) => {
    if (r.stopping) return;
    // captureTime is when the sensor exposed the frame; fall back to presentation time.
    const t = metadata?.captureTime || metadata?.expectedDisplayTime || now;
    const index = r.frameCount++;
    const videoTs = encode(r, video, t);
    r.frames.write(csvRow([index, t, videoTs, 3, screen.orientation?.angle ?? 0, ...Array(20).fill(null), video.videoWidth, video.videoHeight]));
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
    imu_convention: 'imu.csv device axes (x right, y up along screen, z out of screen); gyro rad/s, accel m/s^2 including gravity, gravity m/s^2',
    started_wall: r.startedWall,
    duration_s: +durationS.toFixed(3),
    user_agent: navigator.userAgent,
    device: await navigator.userAgentData?.getHighEntropyValues?.(['model', 'platformVersion']).catch(() => null),
    screen: { width: screen.width, height: screen.height, dpr: devicePixelRatio, angle: screen.orientation?.angle ?? 0 },
    camera: r.camera,
    codec: r.codec,
    xr_features: r.features || [],
    frames: r.frameCount,
    frames_encoded: r.encoded,
    frames_dropped: r.dropped,
    chunks: r.chunks,
    frames_tracked: r.tracked,
    frames_tracking_lost: r.lost,
    imu_rates_hz: rates,
    devicemotion_interval_ms: r.dmIntervalMs || null,
    errors: r.errors,
  };
  const metaFile = new BundleFile('meta.json');
  metaFile.write(JSON.stringify(meta, null, 2));
  const files = [metaFile, r.frames, r.videoIndex, r.imu, ...(r.video ? [r.video] : [])];
  if (r.depthFiles.length) files.push(r.depthIndex, ...r.depthFiles);
  const blob = new Blob(zipParts(files), { type: 'application/zip' });
  const name = `namma-${r.mode}-${r.startedWall.replace(/[:.]/g, '-').slice(0, 19)}.zip`;
  const link = $('save');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  showSummary(meta, blob.size);
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
  $('hud-track').textContent = labels[tracking];
  $('hud-track').dataset.ok = tracking === 0 || tracking === 3 ? 'yes' : 'no';
}

function showSummary(meta, bytes) {
  document.body.dataset.screen = 'done';
  const rows = [
    ['Mode', meta.mode],
    ['Duration', `${meta.duration_s.toFixed(1)} s`],
    ['Camera image', meta.camera ? `${meta.camera.width} x ${meta.camera.height}` : 'none'],
    ['Codec', meta.codec?.codec || 'none'],
    ['Frames encoded', `${meta.frames_encoded} of ${meta.frames} (${(meta.frames_encoded / meta.duration_s).toFixed(1)} fps)`],
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

$('start-ar').addEventListener('click', () => begin(startAr));
$('start-camera').addEventListener('click', () => begin(startCamera));
$('stop').addEventListener('click', () => finish());
$('again').addEventListener('click', () => location.reload());
probe().catch((error) => log(error.message));

// Exposed for the automated browser test.
window.nammaRecorder = { finish, state: () => rec };
