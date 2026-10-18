// Studio: what happens after a recording (or an opened file). Hybrid design:
// 1. on the phone, seconds: quick 3D model from ARCore poses + ARCore depth (quick-model.js);
// 2. "Send for render": the bundle goes to the laptop (or a Colab GPU running the same server),
//    which runs the heavy pipeline (app/pipeline/stages/phone.py);
// 3. the photoreal splat (or dense points, if no splat trainer is available) comes back into the
//    same viewer, in the same ARCore world frame as the quick model.

import { buildQuickModel, quickModelPly } from './quick-model.js';
import { RenderClient, pairingFromLocation } from './remote.js';

const $ = (id) => document.getElementById(id);
const storage = (() => {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
})();

let current = null; // { blob, name, meta }
let viewer = null;
let pairing = pairingFromLocation(location, storage);
if (new URLSearchParams(location.search).has('server')) history.replaceState(null, '', location.pathname);

async function getViewer() {
  if (!viewer) {
    const { Viewer } = await import('./viewer3d.js');
    viewer = new Viewer($('view3d'));
    window.nammaStudio.viewer = viewer;
  }
  return viewer;
}

function hint(text) {
  $('view-hint').textContent = text || '';
  $('view-hint').hidden = !text;
}

// ---- Quick model ---------------------------------------------------------------------------------

async function showQuickModel(blob, meta) {
  if (!blob.name?.endsWith?.('.zip') && meta?.mode === undefined) {
    hint('No AR data in this file: the quick model needs a recording made with "Record with AR".');
    return null;
  }
  const v = await getViewer();
  const started = performance.now();
  hint('Building the quick 3D model…');
  const model = await buildQuickModel(blob, { onProgress: (f) => hint(`Building the quick 3D model… ${Math.round(100 * f)}%`) });
  const seconds = (performance.now() - started) / 1000;
  window.nammaStudio.quick = model;
  if (!model.stats.points) {
    hint(model.stats.mode === 'ar'
      ? 'No depth frames in this recording: tick "Record ARCore depth" next time. The camera path is shown.'
      : 'Camera-only recording: the quick model needs AR mode. Send it for render to get the 3D model.');
  } else {
    hint('');
  }
  v.setPoints('quick', model.positions, model.colors);
  v.setPath(model.path);
  v.frame(model.positions.length ? model.positions : model.path);
  $('quick-status').textContent = model.stats.points
    ? `Quick model: ${model.stats.points.toLocaleString()} points from ${model.stats.depth_frames_used} ARCore depth frames, built on this phone in ${seconds.toFixed(1)} s` +
      (model.stats.coloured_frames ? '.' : ' (coloured by height: this recording has no colour snapshots).')
    : '';
  if (model.stats.points) {
    const link = $('save-quick');
    link.href = URL.createObjectURL(quickModelPly(model));
    link.download = (current?.name || 'recording').replace(/\.zip$/, '') + '-quick.ply';
    link.hidden = false;
  }
  return model;
}

// ---- Pairing with the laptop ---------------------------------------------------------------------

async function checkPairing() {
  $('server').value = pairing.server || '';
  $('key').value = pairing.key || '';
  if (!pairing.server) {
    $('pair-status').textContent = 'Not paired: run scripts/serve_render.py on the laptop and open the link it prints on this phone.';
    $('pair-box').open = true;
    $('send').disabled = true;
    return null;
  }
  $('pair-status').textContent = `Checking ${pairing.server} …`;
  try {
    const hello = await new RenderClient(pairing.server, pairing.key).hello();
    const what = hello.photoreal ? `photoreal splats with ${hello.trainer}` : hello.dense_points ? 'dense 3D points (no splat trainer on it yet)' : 'sparse points only';
    $('pair-status').textContent = `Connected to ${hello.machine}: it can make ${what}.`;
    $('send').disabled = !current;
    return hello;
  } catch (error) {
    $('pair-status').textContent = `Can't reach the laptop (${error.message}). Is serve_render.py still running?`;
    $('pair-box').open = true;
    $('send').disabled = true;
    return null;
  }
}

function savePairing() {
  pairing = { server: $('server').value.trim(), key: $('key').value.trim() };
  try {
    storage?.setItem('namma.server', pairing.server);
    storage?.setItem('namma.key', pairing.key);
  } catch {
    // not persisted
  }
  return checkPairing();
}

// ---- Send for render -----------------------------------------------------------------------------

function showSteps(steps) {
  const icons = { done: '✓', running: '…', pending: '○', skipped: '–', failed: '!' };
  $('render-steps').replaceChildren(
    ...steps.map((step) => {
      const li = document.createElement('li');
      li.dataset.state = step.state;
      const extra = step.state === 'done' && step.seconds != null ? ` (${step.seconds} s)` : step.reason ? `: ${step.reason}` : '';
      li.textContent = `${icons[step.state] || ''} ${step.label}${extra}`;
      return li;
    })
  );
}

async function loadResult(client, renderId, status) {
  const v = await getViewer();
  const best = status.best || Object.keys(status.outputs).find((n) => n !== 'result.json');
  if (!best) {
    $('render-status').textContent = 'The laptop finished but produced no 3D output; see its log.';
    return;
  }
  $('render-status').textContent = `Downloading ${best} (${(status.outputs[best].bytes / 1e6).toFixed(1)} MB)…`;
  const bytes = await client.file(renderId, best);
  if (best === 'splat.splat') {
    await v.setSplat(new Uint8Array(bytes));
  } else {
    const { parsePointsPly } = await import('./viewer3d.js');
    const cloud = parsePointsPly(bytes);
    v.setPoints('splat', cloud.positions, cloud.colors, best === 'dense.ply' ? 0.012 : 0.03);
    if (!window.nammaStudio.quick?.positions.length) v.frame(cloud.positions);
  }
  $('show-photo').disabled = false;
  $('show-photo').checked = true;
  $('show-quick').checked = false;
  v.show('quick', false);
  let info = null;
  try {
    info = JSON.parse(new TextDecoder().decode(await client.file(renderId, 'result.json')));
  } catch {
    // older server
  }
  const psnr = info?.outputs?.['splat.splat']?.train_info?.heldout_psnr_mean;
  const label = { 'splat.splat': 'Photoreal splat', 'dense.ply': 'Dense points', 'sparse.ply': 'Sparse points' }[best];
  $('render-status').textContent = `${label} from the laptop is in the viewer` +
    (info ? `, made in ${Math.round(info.total_seconds)} s on ${info.machine}` : '') +
    (psnr ? `; held-out PSNR ${psnr} dB (measured on frames it did not train on).` : '.') +
    (best !== 'splat.splat' ? ' Install a splat trainer on the laptop for the photoreal version.' : '');
  window.nammaStudio.result = { best, bytes: bytes.byteLength, info };
}

async function follow(client, renderId) {
  $('render-progress').hidden = false;
  const status = await client.wait(renderId, (s) => {
    if (s.status === 'unreachable') {
      $('render-status').textContent = `Lost the laptop for a moment (${s.error}); still trying…`;
      return;
    }
    $('render-progress').value = s.progress || 0;
    showSteps(s.steps || []);
    $('render-status').textContent = s.status === 'queued' ? 'Waiting for the laptop…' : `Rendering on the laptop: ${s.stage || ''}`;
  });
  $('render-progress').hidden = true;
  if (status.status === 'failed') {
    $('render-status').textContent = `The laptop stopped: ${status.error || 'see its log'}`;
    return status;
  }
  await loadResult(client, renderId, status);
  return status;
}

async function send() {
  if (!current) return;
  const client = new RenderClient(pairing.server, pairing.key);
  $('send').disabled = true;
  $('render-progress').hidden = false;
  try {
    const { render_id: renderId } = await client.send(current.blob, current.name, {
      quality: $('quality').value,
      onProgress: (f) => {
        $('render-progress').value = f;
        $('render-status').textContent = `Sending to the laptop… ${Math.round(100 * f)}% of ${(current.blob.size / 1e6).toFixed(0)} MB`;
      },
    });
    try {
      storage?.setItem('namma.lastRender', JSON.stringify({ server: pairing.server, key: pairing.key, renderId, name: current.name }));
    } catch {
      // fine
    }
    window.nammaStudio.renderId = renderId;
    await follow(client, renderId);
  } catch (error) {
    $('render-status').textContent = `Sending failed: ${error.message}`;
    $('render-progress').hidden = true;
  } finally {
    $('send').disabled = false;
  }
}

// ---- Entry points --------------------------------------------------------------------------------

async function open(blob, name, meta) {
  current = { blob, name, meta };
  document.body.dataset.screen = 'done';
  $('done-title').textContent = meta ? 'Recording ready' : name;
  $('send').disabled = !pairing.server;
  checkPairing();
  if (!meta) $('save').hidden = true;
  try {
    await showQuickModel(blob, meta || (name.endsWith('.zip') ? {} : undefined));
  } catch (error) {
    hint(`Quick model failed: ${error.message}`);
  }
}

window.addEventListener('namma:bundle', (event) => {
  const { blob, name, meta } = event.detail;
  open(blob, name, meta);
});

$('open-file').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) open(file, file.name, null);
});

$('pair-test').addEventListener('click', () => savePairing());
$('send').addEventListener('click', () => send());
$('show-quick').addEventListener('change', (e) => viewer?.show('quick', e.target.checked));
$('show-photo').addEventListener('change', (e) => viewer?.show('splat', e.target.checked));
$('show-path').addEventListener('change', (e) => viewer?.show('path', e.target.checked));

const last = (() => {
  try {
    return JSON.parse(storage?.getItem('namma.lastRender') || 'null');
  } catch {
    return null;
  }
})();
if (last?.renderId) {
  $('resume').hidden = false;
  $('resume').addEventListener('click', async () => {
    document.body.dataset.screen = 'done';
    $('done-title').textContent = last.name || 'Last render';
    $('save').hidden = true;
    hint('');
    pairing = { server: last.server, key: last.key };
    checkPairing();
    await getViewer();
    follow(new RenderClient(last.server, last.key), last.renderId).catch((error) => {
      $('render-status').textContent = `Can't fetch the last render: ${error.message}`;
    });
  });
}

// Exposed for the automated browser test.
window.nammaStudio = { open, send, state: () => ({ current, pairing }) };
