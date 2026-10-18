// End-to-end check of the phone studio (/record/ after a recording) in headless Chromium, against a
// running laptop server.
//
// Part 1, quick model: a scripted WebXR session with depth-sensing looks straight at a flat wall
// 2 m away while sliding sideways; the camera image is red on top, blue below. The quick model must
// put the wall at z = -2 and colour points above eye height red (checks the depth back-projection
// and the snapshot's row order on the phone).
// Part 2, send for render: a real phone bundle is opened from a file, sent to the server in chunks,
// rendered, and the result loaded into the viewer (a splat through Spark when a trainer is set up).
//
// What it cannot prove: real ARCore depth quality, a real phone GPU, or a tunnel's network.
//
//   NAMMA_REMOTE_KEY=k GAUSSIAN_TRAINER_COMMAND="python3 tests/fixtures/fake_splat_trainer.py --dataset {dataset} --output {output}" \
//     python3 -m uvicorn app.main:app --port 8000 &
//   node tests/browser/studio_e2e.mjs http://localhost:8000 k <bundle.zip> [screenshot.png]
import { chromium } from 'playwright';

const [server = 'http://localhost:8000', key = '', bundle, screenshot] = process.argv.slice(2);
const EYE = 1.4;
const WALL = 2.0;

const mock = ({ eye, wall }) => {
  const t0 = performance.now() / 1000;
  WebGLRenderingContext.prototype.makeXRCompatible = async () => {};
  if (window.WebGL2RenderingContext) WebGL2RenderingContext.prototype.makeXRCompatible = async () => {};
  window.XRWebGLLayer = class { constructor() { this.framebuffer = null; } };
  const pattern = document.createElement('canvas');
  pattern.width = 640; pattern.height = 480;
  const g = pattern.getContext('2d');
  g.fillStyle = '#e02020'; g.fillRect(0, 0, 640, 240);
  g.fillStyle = '#2040e0'; g.fillRect(0, 240, 640, 240);
  window.XRWebGLBinding = class {
    constructor(_session, gl) { this.gl = gl; this.tex = gl.createTexture(); }
    getCameraImage() {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); // camera textures are bottom-up, like ARCore's
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pattern);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return this.tex;
    }
  };
  const projection = new Float32Array(16);
  projection[0] = 500 / 320; projection[5] = 500 / 240; projection[8] = 0; projection[9] = 0;
  projection[10] = -1; projection[11] = -1; projection[14] = -0.02;
  // Flat wall facing the camera: every depth pixel reads the same distance (raw mm, uint16).
  const depth = new Uint16Array(160 * 90).fill(wall * 1000);
  const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const session = new EventTarget();
  session.enabledFeatures = ['local-floor', 'camera-access', 'dom-overlay', 'depth-sensing'];
  session.depthDataFormat = 'luminance-alpha';
  session.updateRenderState = () => {};
  session.requestReferenceSpace = async () => ({});
  session.end = async () => { session.ended = true; session.dispatchEvent(new Event('end')); };
  session.requestAnimationFrame = (cb) => setTimeout(() => {
    if (session.ended) return;
    const now = performance.now();
    const t = now / 1000 - t0;
    const m = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.4 * Math.sin(t), eye, 0, 1]);
    const frame = {
      getViewerPose: () => ({ emulatedPosition: false, views: [{ transform: { matrix: m }, projectionMatrix: projection, camera: { width: 640, height: 480 } }] }),
      getDepthInformation: () => ({ data: depth.buffer, width: 160, height: 90, rawValueToMeters: 0.001, normDepthBufferFromNormView: { matrix: identity } }),
    };
    cb(now, frame);
  }, 33);
  Object.defineProperty(navigator, 'xr', { value: { isSessionSupported: async () => true, requestSession: async () => session } });
};

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 1 });
const logs = [];
page.on('console', (m) => logs.push(m.text()));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
await page.addInitScript(mock, { eye: EYE, wall: WALL });
const url = `${server}/record/?server=${encodeURIComponent(server)}&key=${encodeURIComponent(key)}`;
const report = { logs };
let ok = true;
const check = (name, pass, value) => {
  report[name] = { pass, value };
  ok &&= pass;
};

// ---- Part 1: record with AR + depth, quick model on the phone ---------------------------------
await page.goto(url);
await page.waitForFunction(() => document.getElementById('cap-codec').textContent !== '…');
await page.click('#start-ar');
await page.waitForTimeout(5000);
await page.click('#stop');
await page.waitForFunction(() => window.nammaStudio?.quick, null, { timeout: 30000 });
const quick = await page.evaluate(({ eye }) => {
  const { positions, colors, stats } = window.nammaStudio.quick;
  let zSum = 0, above = 0, aboveRed = 0, below = 0, belowBlue = 0;
  for (let i = 0; i < positions.length / 3; i++) {
    zSum += positions[i * 3 + 2];
    const y = positions[i * 3 + 1], r = colors[i * 3], b = colors[i * 3 + 2];
    if (y > eye + 0.1) { above++; if (r > b) aboveRed++; }
    if (y < eye - 0.1) { below++; if (b > r) belowBlue++; }
  }
  return { stats, meanZ: zSum / (positions.length / 3), redAbove: aboveRed / above, blueBelow: belowBlue / below,
    status: document.getElementById('quick-status').textContent, pair: document.getElementById('pair-status').textContent };
}, { eye: EYE });
check('quick_points', quick.stats.points > 1000, quick.stats.points);
check('quick_wall_depth', Math.abs(quick.meanZ + WALL) < 0.05, quick.meanZ);
check('quick_colour_rows', quick.redAbove > 0.95 && quick.blueBelow > 0.95, { redAbove: quick.redAbove, blueBelow: quick.blueBelow });
check('paired', /^Connected/.test(quick.pair), quick.pair);
report.quick_status = quick.status;

// ---- Part 2: open a real bundle, send for render, load the result ------------------------------
if (bundle) {
  await page.goto(url);
  await page.setInputFiles('#open-file', bundle);
  await page.waitForFunction(() => window.nammaStudio?.quick, null, { timeout: 60000 });
  await page.waitForFunction(() => !document.getElementById('send').disabled, null, { timeout: 20000 });
  await page.selectOption('#quality', 'fast');
  const started = Date.now();
  await page.click('#send');
  await page.waitForFunction(() => window.nammaStudio?.result || /stopped|failed/.test(document.getElementById('render-status').textContent),
    null, { timeout: 15 * 60 * 1000, polling: 2000 });
  const render = await page.evaluate(() => ({
    result: window.nammaStudio.result,
    status: document.getElementById('render-status').textContent,
    steps: [...document.querySelectorAll('#render-steps li')].map((li) => li.textContent),
    splatLayer: window.nammaStudio.viewer?.layers.splat?.isPoints ? 'points' : window.nammaStudio.viewer?.layers.splat ? 'splat' : null,
    splats: window.nammaStudio.viewer?.layers.splat?.packedSplats?.numSplats ?? null,
  }));
  render.seconds = Math.round((Date.now() - started) / 1000);
  check('render_result', !!render.result, render.status);
  check('render_splat_loaded', render.result?.best !== 'splat.splat' || (render.splatLayer === 'splat' && render.splats > 0), { layer: render.splatLayer, splats: render.splats });
  report.render = render;
  await page.waitForTimeout(1500);
}
if (screenshot) await page.screenshot({ path: screenshot, fullPage: true });
report.ok = ok;
console.log(JSON.stringify(report, null, 2));
await browser.close();
process.exit(ok ? 0 : 1);
