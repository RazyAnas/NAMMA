// Studio: what happens after a recording (or an opened file). Hybrid design:
// 1. on the phone, seconds: quick 3D model from ARCore poses + ARCore depth (quick-model.js);
// 2. "Send for render": the bundle goes to a render server running app/pipeline/stages/phone.py: the
//    Modal cloud (modal_app.py), a Colab GPU or the laptop, all the same server;
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
      (model.stats.coloured_frames ? '.' : ' (coloured by height: this recording has no colour snapshots).') +
      (model.stats.ar_segments > 1 ? ` AR tracking reset ${model.stats.ar_segments - 1} time(s) during this recording: showing the largest consistent part, ${model.stats.depth_frames_other_segments} depth frames from the other parts left out.` : '')
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

let machine = 'the render server';

async function checkPairing() {
  $('server').value = pairing.server || '';
  $('key').value = pairing.key || '';
  if (!pairing.server) {
    $('pair-status').textContent = 'Not paired: open this page from your cloud link, or tap the link Colab or the laptop prints.';
    $('send-target').textContent = 'Not paired with a render server yet: see the first screen.';
    $('pair-box').open = true;
    $('send').disabled = true;
    return null;
  }
  $('pair-status').textContent = `Checking ${pairing.server} …`;
  try {
    let hello = await new RenderClient(pairing.server, pairing.key).hello();
    if (hello.claimable) hello = await claimServer(); // a fresh cloud server: this phone becomes its owner
    const what = hello.photoreal ? `photoreal splats with ${hello.trainer}` : hello.dense_points ? 'dense 3D points (no splat trainer on it yet)' : 'sparse points only';
    machine = hello.machine;
    $('pair-status').textContent = `Connected to ${hello.machine}: it can make ${what}.`;
    $('send-target').textContent = `Renders on ${hello.machine} (${what}).`;
    $('send').disabled = !current;
    if (hello.history) loadHistory();
    return hello;
  } catch (error) {
    const why = error.status === 401 ? 'wrong pairing key' : error.message;
    $('pair-status').textContent = `Can't reach ${pairing.server} (${why}). A Colab or laptop server stops when its tab or terminal closes.`;
    $('send-target').textContent = `Can't reach the render server (${why}).`;
    $('pair-box').open = true;
    $('send').disabled = true;
    return null;
  }
}

async function claimServer() {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  const key = btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  await new RenderClient(pairing.server, '').json('/api/v1/remote/claim', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  });
  pairing = { server: pairing.server, key };
  try {
    storage?.setItem('namma.server', pairing.server);
    storage?.setItem('namma.key', key);
  } catch {
    // not persisted: Change server shows the key to write down
  }
  $('key').value = key;
  return new RenderClient(pairing.server, key).hello();
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

function formatLeft(seconds) {
  if (seconds == null) return '';
  if (seconds < 60) return 'under a minute left';
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `about ${minutes} min left`;
  return `about ${Math.floor(minutes / 60)} h ${minutes % 60} min left`;
}

async function loadResult(client, renderId, status) {
  const v = await getViewer();
  const best = status.best || Object.keys(status.outputs).find((n) => n !== 'result.json');
  if (!best) {
    $('render-status').textContent = 'The render finished but produced no 3D output; see the server log.';
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
  $('render-status').textContent = `${label} is in the viewer` +
    (info ? `, made in ${Math.round(info.total_seconds)} s on ${info.machine}` : '') +
    (psnr ? `; held-out PSNR ${psnr} dB (measured on frames it did not train on).` : '.') +
    (best !== 'splat.splat' ? ' This server has no splat trainer: render on the cloud or Colab for the photoreal version.' : '');
  window.nammaStudio.result = { best, bytes: bytes.byteLength, info };
}

// A render the person can fix and run again (the cloud account needs a card first): what to do, a link, Try again.
function showAction(client, renderId, status) {
  const action = status.action;
  $('render-action').hidden = !action;
  if (!action) return;
  $('render-status').textContent = action.message || `The render stopped: ${status.error}`;
  $('render-action-link').hidden = !action.url;
  $('render-action-link').href = action.url || '#';
  $('render-action-link').textContent = action.label || 'Open';
  $('retry').hidden = !action.retry;
  $('retry').onclick = async () => {
    $('retry').disabled = true;
    try {
      await client.retry(renderId);
      $('render-action').hidden = true;
      await follow(client, renderId);
    } catch (error) {
      $('render-status').textContent = `Couldn't start it again: ${error.message}`;
    } finally {
      $('retry').disabled = false;
    }
  };
}

async function follow(client, renderId) {
  $('render-action').hidden = true;
  $('render-progress').hidden = false;
  const status = await client.wait(renderId, (s) => {
    if (s.status === 'unreachable') {
      $('render-status').textContent = `Lost the render server for a moment (${s.error}); still trying…`;
      return;
    }
    $('render-progress').value = s.progress || 0;
    showSteps(s.steps || []);
    const left = formatLeft(s.eta_seconds);
    $('render-status').textContent = s.status === 'queued' ? `Waiting for ${machine}…`
      : `Rendering on ${machine}: ${s.stage || ''}${left ? ` · ${left}` : ''}. You can close this page; it shows up under Your scans.`;
  });
  $('render-progress').hidden = true;
  if (status.status === 'failed') {
    $('render-status').textContent = `The render stopped: ${status.error || 'see the server log'}`;
    showAction(client, renderId, status);
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
        $('render-status').textContent = `Sending to ${machine}… ${Math.round(100 * f)}% of ${(current.blob.size / 1e6).toFixed(0)} MB`;
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

// ---- Scan history (every render this server has, with progress and time left) -------------------

let historyTimer = null;

function historyItem(render) {
  const li = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button';
  button.dataset.status = render.status;
  const title = document.createElement('strong');
  title.textContent = render.name || 'Scan';
  const meta = document.createElement('span');
  meta.className = 'meta';
  const when = render.created_at ? new Date(render.created_at * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '';
  const quality = render.quality ? ` · ${render.quality}` : '';
  const state = {
    queued: 'waiting to start',
    running: `rendering ${Math.round(100 * (render.progress || 0))}%${render.eta_seconds != null ? `, ${formatLeft(render.eta_seconds)}` : ''}`,
    succeeded: { 'splat.splat': 'photoreal splat ready', 'dense.ply': 'dense points ready', 'sparse.ply': 'sparse points ready' }[render.best] || 'finished',
    failed: render.action ? `needs you: ${render.action.label || 'tap to see'}` : `stopped: ${(render.error || '').slice(0, 80)}`,
  }[render.status] || render.status;
  meta.textContent = `${when}${quality} · ${state}`;
  button.append(title, meta);
  if (render.status === 'running' || render.status === 'queued') {
    const bar = document.createElement('progress');
    bar.max = 1;
    bar.value = render.progress || 0;
    button.append(bar);
  }
  button.addEventListener('click', () => openRender(render));
  li.append(button);
  return li;
}

async function loadHistory() {
  clearTimeout(historyTimer);
  if (!pairing.server || document.body.dataset.screen !== 'setup') return;
  try {
    const { renders } = await new RenderClient(pairing.server, pairing.key).json('/api/v1/remote/renders');
    $('history-box').hidden = false;
    $('history').replaceChildren(...renders.map(historyItem));
    $('history-status').textContent = renders.length ? '' : 'No scans sent to this server yet.';
    $('resume').hidden = true; // the list replaces "Show my last render"
    if (renders.some((r) => r.status === 'running' || r.status === 'queued')) historyTimer = setTimeout(loadHistory, 10000);
  } catch (error) {
    $('history-status').textContent = `Can't load your scans: ${error.message}`;
  }
}

async function openRender(render) {
  clearTimeout(historyTimer);
  document.body.dataset.screen = 'done';
  $('done-title').textContent = render.name || 'Scan';
  $('save').hidden = true;
  $('send').disabled = true;
  hint('');
  await getViewer();
  window.nammaStudio.renderId = render.render_id;
  follow(new RenderClient(pairing.server, pairing.key), render.render_id).catch((error) => {
    $('render-status').textContent = `Can't fetch this render: ${error.message}`;
  });
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
window.nammaStudio = { open, send, loadHistory, state: () => ({ current, pairing }) };
checkPairing();
