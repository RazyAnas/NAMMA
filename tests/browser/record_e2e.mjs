// End-to-end check of /record/ in headless Chromium with a scripted WebXR session and IMU.
//
// What this proves: the page's recording loop, encoder, bundle writer and pose/IMU logging work
// together and produce a bundle the Python ingest reads, and the ingest recovers a known
// IMU-to-camera offset and rotation. What it cannot prove: how real Chrome + ARCore on a phone
// behaves (camera-access image size, orientation, rates). That needs a phone.
//
//   node tests/browser/record_e2e.mjs <output-dir> [ar|camera]
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../app/static/record');
const outDir = path.resolve(process.argv[2] || '.');
const mode = process.argv[3] || 'ar';
const OFFSET_S = 0.042; // gyro reports motion 42 ms late

const server = http.createServer((req, res) => {
  const file = path.join(root, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  if (!file.startsWith(root) || !fs.existsSync(file)) return res.writeHead(404).end();
  const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html';
  res.writeHead(200, { 'Content-Type': type }).end(fs.readFileSync(file));
});
await new Promise((resolve) => server.listen(0, resolve));
const url = `http://localhost:${server.address().port}/`;

const mock = ({ offset, mode }) => {
  // Analytic trajectory: walk forward while panning and tilting.
  const rot = (t) => {
    const yaw = 0.8 * Math.sin(1.3 * t), pitch = 0.3 * Math.sin(0.9 * t + 0.5), roll = 0.1 * Math.sin(2 * t);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
    const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
    const Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
    const Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
    const mul = (a, b) => a.map((r, i) => b[0].map((_, j) => r.reduce((s, _v, k) => s + a[i][k] * b[k][j], 0)));
    return mul(mul(Ry, Rx), Rz);
  };
  const omegaBody = (t) => {
    const h = 1e-4, a = rot(t - h), b = rot(t + h);
    const d = a.map((r, i) => r.map((_, j) => (b[i][j] - a[i][j]) / (2 * h)));
    const R = rot(t);
    const W = R[0].map((_, i) => R[0].map((_, j) => R.reduce((s, _r, k) => s + R[k][i] * d[k][j], 0))); // R^T dR
    return [W[2][1], W[0][2], W[1][0]];
  };
  const t0 = performance.now() / 1000;
  window.__truth = { offset };
  class FakeSensor extends EventTarget {
    constructor(kind) { super(); this.kind = kind; }
    start() {
      this.timer = setInterval(() => {
        const now = performance.now();
        const t = now / 1000 - t0 - offset;
        const v = this.kind === 'gyro' ? omegaBody(t).map((x) => x + 0.005 * (Math.random() - 0.5)) : [0, 9.81, 0];
        [this.x, this.y, this.z] = v;
        // Android Chrome stamps Generic Sensor readings on a different clock (measured on a phone).
        this.timestamp = now + 230_000_000;
        this.dispatchEvent(new Event('reading'));
      }, 16);
    }
    stop() { clearInterval(this.timer); }
  }
  window.Gyroscope = class extends FakeSensor { constructor() { super('gyro'); } };
  window.Accelerometer = class extends FakeSensor { constructor() { super('accel'); } };
  if (mode !== 'ar') return;

  WebGLRenderingContext.prototype.makeXRCompatible = async () => {};
  if (window.WebGL2RenderingContext) WebGL2RenderingContext.prototype.makeXRCompatible = async () => {};
  window.XRWebGLLayer = class { constructor() { this.framebuffer = null; } };
  const pattern = document.createElement('canvas');
  pattern.width = 640; pattern.height = 480;
  window.XRWebGLBinding = class {
    constructor(_session, gl) { this.gl = gl; this.tex = gl.createTexture(); this.n = 0; }
    getCameraImage() {
      const g = pattern.getContext('2d');
      // A row of black/white blocks along the top carries the XR frame index in binary, so the
      // test can check that each decoded frame is paired with its own pose.
      const index = session.frameIndex;
      g.fillStyle = `hsl(${(index * 7) % 360},60%,45%)`; g.fillRect(0, 0, 640, 480);
      for (let bit = 0; bit < 10; bit++) {
        g.fillStyle = (index >> bit) & 1 ? '#fff' : '#000';
        g.fillRect(bit * 64, 0, 64, 48);
      }
      g.fillStyle = '#fff'; g.font = '80px sans-serif'; g.fillText(String(index), 40, 240);
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      // Camera textures follow the GL convention (first row at the bottom); DOM uploads do not.
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pattern);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return this.tex;
    }
  };
  // fx = fy = 500 px, cx = 330, cy = 236 for 640x480.
  const projection = new Float32Array(16);
  projection[0] = 500 / 320; projection[5] = 500 / 240; projection[8] = 1 - 330 / 320; projection[9] = 236 / 240 - 1;
  projection[10] = -1; projection[11] = -1; projection[14] = -0.02;
  const session = new EventTarget();
  session.enabledFeatures = ['local-floor', 'camera-access', 'dom-overlay'];
  session.updateRenderState = () => {};
  session.requestReferenceSpace = async () => ({});
  session.end = async () => { session.ended = true; session.dispatchEvent(new Event('end')); };
  session.requestAnimationFrame = (cb) => setTimeout(() => {
    if (session.ended) return;
    const now = performance.now();
    const t = now / 1000 - t0;
    session.frameIndex = (session.frameIndex ?? -1) + 1;
    const R = rot(t);
    const m = new Float32Array([R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, Math.sin(0.5 * t), 1.4, -0.3 * t, 1]);
    const lost = t > 2.0 && t < 2.3;
    const frame = {
      getViewerPose: () => ({ emulatedPosition: lost, views: [{ transform: { matrix: m }, projectionMatrix: projection, camera: { width: 640, height: 480 } }] }),
    };
    cb(now, frame);
  }, 33);
  Object.defineProperty(navigator, 'xr', { value: { isSessionSupported: async () => true, requestSession: async () => session } });
};

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ acceptDownloads: true });
const logs = [];
page.on('console', (m) => logs.push(m.text()));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
await page.addInitScript(mock, { offset: OFFSET_S, mode });
await page.goto(url);
await page.waitForFunction(() => document.getElementById('cap-codec').textContent !== '…');
const caps = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.caps dd')].map((d) => [d.id, d.textContent])));
await page.click(mode === 'ar' ? '#start-ar' : '#start-camera');
await page.waitForTimeout(6000);
await page.click('#stop');
await page.waitForSelector('#done', { state: 'visible', timeout: 20000 });
const summary = await page.evaluate(() => document.getElementById('summary').innerText);
const [download] = await Promise.all([page.waitForEvent('download'), page.click('#save')]);
fs.mkdirSync(outDir, { recursive: true });
const target = path.join(outDir, `bundle-${mode}.zip`);
await download.saveAs(target);
console.log(JSON.stringify({ caps, summary, bundle: target, logs: logs.slice(0, 10) }, null, 2));
await browser.close();
server.close();
