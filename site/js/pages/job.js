// Job (#/job/<id>, needs sign-in): live status of one training job. Asks every 20 s while the tab is visible and stops when the
// job is done or has failed.
import { h, replace, icon, extLink, notice, stateBox, spinner, toast, errText, fmtElapsed, safeUrl, session as tabStore } from '../ui.js';

const POLL_MS = 20000;
const ORDER = ['uploaded', 'training', 'publishing', 'done'];
const STEPS = [
  ['Uploaded', 'Your video is in your Hugging Face account.'],
  ['Training on your Kaggle GPU', 'The 3D model is being trained. This is the long part.'],
  ['Building the tour', 'The model is packed into a tour and saved to your account.'],
  ['Done', 'Your tour is ready.'],
];
const KAGGLE_TEXT = {
  queued: 'Waiting for a free GPU',
  running: 'Running',
  complete: 'Finished',
  error: 'Stopped with an error',
  cancelled: 'Cancelled',
};
const PRESET_TEXT = { fast: 'Fast setting, usually about 55 minutes', quality: 'Quality setting, usually about 90 minutes' };

export function render(ctx) {
  const { jobs } = ctx.api;
  const id = String(ctx.params[0] || '');
  if (!jobs) {
    return h('section', { class: 'wrap page' }, stateBox({ kind: 'error', title: 'This job could not be opened', body: 'A part of this site did not load. Reload the page and try again.',
      actions: [h('button', { type: 'button', class: 'btn btn--line', onclick: () => window.location.reload() }, 'Reload')] }));
  }

  let alive = true;
  let timer = 0;
  let clock = 0;
  let last = 0;                 // time of the last answer
  let info = null;              // { title, preset, created } from jobs.list()
  let status = null;
  let reached = 0;              // furthest step seen, so a failed job shows where it stopped
  let finished = false;
  let asking = false;

  const heading = h('span', null, 'Your job');
  const sub = h('p', { class: 'muted' }, `Job ${id}`);
  const timeline = h('ol', { class: 'timeline' });
  const elapsed = h('dd', null, '...');
  const kaggleState = h('dd', null, '...');
  const checked = h('dd', null, '...');
  const actions = h('div', { class: 'job__actions' });
  const message = h('div', { class: 'job__message' });
  const keyBox = h('div', { class: 'job__key' });

  function kernelLink() {
    const stored = safeUrl(tabStore.get(`namma.kernel.${id}`));
    if (stored) return [stored, 'Open your Kaggle notebook'];
    const user = tabStore.get('namma.kaggleUser');
    if (user && /^[a-z0-9-]+$/i.test(user)) return [`https://www.kaggle.com/${user}/code`, 'Open your Kaggle notebooks'];
    return ['https://www.kaggle.com/work', 'Open your Kaggle notebooks'];
  }

  function drawTimeline() {
    const stage = status?.stage;
    const failed = stage === 'failed';
    // "uploaded" means step 1 is finished and the Kaggle job is next. A failed job is marked at the furthest step this page
    // saw; without that, at training (or at building the tour when Kaggle itself finished without an error).
    const partial = stage === 'uploading';        // the video upload broke off
    let at = stage === 'uploaded' ? 1 : Math.max(0, ORDER.indexOf(stage));
    if (failed) at = Math.max(reached, status?.kaggle === 'complete' ? 2 : 1);
    reached = Math.max(reached, at);
    replace(timeline, STEPS.map(([label, text], i) => {
      let state = 'todo';
      if (stage === 'done' || i < at) state = 'done';
      else if (i === at) state = failed ? 'failed' : 'now';
      let detail = text;
      if (state === 'failed') detail = 'The job stopped here.';
      else if (i === 0 && partial) detail = 'The upload did not finish.';
      else if (i === 1 && state === 'now' && status?.kaggle === 'queued') detail = 'Waiting for a free GPU on Kaggle. This can take a few minutes.';
      else if (i === 1 && state === 'now' && stage === 'uploaded' && !status?.kaggle) detail = 'Not started yet.';
      const mark = state === 'done' ? icon('check', 16) : state === 'failed' ? icon('x', 16) : h('span', null, String(i + 1));
      return h('li', { class: `timeline__step timeline__step--${state}`, 'aria-current': state === 'now' ? 'step' : null },
        h('span', { class: 'timeline__mark' }, mark),
        h('div', { class: 'timeline__text' }, h('h3', null, i === 0 && partial ? 'Uploading' : label), h('p', null, detail)));
    }));
  }

  function drawClock() {
    const start = info?.created ? new Date(info.created).getTime() : NaN;
    if (!Number.isFinite(start)) {
      elapsed.textContent = 'Not known';
      return;
    }
    const endRaw = finished && status?.updated ? new Date(status.updated).getTime() : Date.now();
    const end = Number.isFinite(endRaw) ? endRaw : Date.now();
    elapsed.textContent = fmtElapsed(Math.max(0, end - start)) || 'Not known';
  }

  function draw() {
    if (info?.title) heading.textContent = String(info.title);
    sub.textContent = [`Job ${id}`, PRESET_TEXT[info?.preset] || ''].filter(Boolean).join(' · ');
    drawTimeline();
    drawClock();
    const stage = status?.stage;
    const hasKey = Boolean(ctx.kaggleKey.get());
    const live = stage === 'training' || stage === 'publishing';
    kaggleState.textContent = status?.kaggle ? (KAGGLE_TEXT[status.kaggle] || String(status.kaggle))
      : status?.kaggleError ? 'Kaggle could not be asked'
      : !live ? (stage === 'done' || stage === 'failed' ? 'Ended' : 'Not started')
      : hasKey ? 'No answer from Kaggle yet' : 'Not shown without your Kaggle key';
    checked.textContent = last ? new Date(last).toLocaleTimeString() : '...';

    const [kUrl, kLabel] = kernelLink();
    const tour = stage === 'done' ? extLink(status.link, { class: 'btn btn--primary btn--lg' }, 'Open your tour', icon('open', 20), h('span', { class: 'sr-only' }, ' (opens in a new tab)')) : null;
    replace(actions,
      tour,
      stage === 'done' ? h('a', { class: 'btn btn--line', href: '#/library' }, 'Go to your library') : null,
      extLink(kUrl, { class: 'btn btn--line' }, kLabel, icon('open', 18)),
      stage === 'failed' ? h('a', { class: 'btn btn--primary', href: '#/create' }, 'Try again with a new upload') : null,
      !finished ? h('button', { type: 'button', class: 'btn btn--ghost', onclick: () => ask(true) }, 'Check now') : null);

    if (stage === 'failed') {
      replace(message, notice('error', h('b', null, 'The job stopped. '), status.reason ? String(status.reason) : 'No reason was reported. The log in your Kaggle notebook shows what happened.'));
    } else if (stage === 'done' && !tour) {
      replace(message, notice('info', 'The tour is ready. You find it in your library.'));
    } else if (stage === 'done') {
      replace(message, notice('ok', 'Your tour is ready. It opens in a new tab.'));
    } else if (stage === 'uploading') {
      replace(message, notice('info', 'The video upload did not finish. Go to Create and choose the same file again: the upload continues where it stopped.'),
        h('a', { class: 'btn btn--primary', href: '#/create' }, 'Go to Create'));
    } else if (stage === 'uploaded' && hasKey) {
      replace(message, notice('info', 'The video is uploaded, but the training has not been started.'),
        h('button', { type: 'button', class: 'btn btn--primary', onclick: startNow }, 'Start on Kaggle'));
    } else if (stage === 'uploaded') {
      replace(message, notice('info', 'The video is uploaded, but the training has not been started. Add your Kaggle key to start it.'));
    } else {
      replace(message, notice('info', 'You can leave this page open or come back later. The job runs on Kaggle, not in your browser.'));
    }
    drawKeyBox(hasKey);
  }

  function drawKeyBox(hasKey) {
    if (hasKey || finished) {
      replace(keyBox);
      return;
    }
    if (keyBox.firstChild) return;         // keep what the user is typing
    const input = h('input', { type: 'password', class: 'input', autocomplete: 'off', spellcheck: 'false', placeholder: 'Paste your Kaggle key', 'aria-label': 'Your Kaggle key' });
    replace(keyBox, h('form', { class: 'field', onsubmit: (e) => {
      e.preventDefault();
      if (!input.value.trim()) return;
      ctx.kaggleKey.set(input.value, false);
      replace(keyBox);
      ask(true);
    } },
      h('p', { class: 'field__help' }, 'Add your Kaggle key to see the live state of the Kaggle job, or to start it. It is kept for this browser tab only.'),
      h('div', { class: 'field__row' }, input, h('button', { type: 'submit', class: 'btn btn--line' }, 'Use this key'))));
  }

  async function startNow(e) {
    const button = e.currentTarget;
    button.disabled = true;
    try {
      const res = await jobs.start(id, ctx.kaggleKey.get());
      if (res?.kernelUrl) tabStore.set(`namma.kernel.${id}`, String(res.kernelUrl));
      toast('The job was started on Kaggle.', 'ok');
      ask(true);
    } catch (err) {
      button.disabled = false;
      toast('The job did not start. ' + errText(err, ''), 'error');
    }
  }

  function schedule() {
    clearTimeout(timer);
    if (!alive || finished || document.hidden) return;
    const wait = Math.max(1000, POLL_MS - (Date.now() - last));
    timer = setTimeout(() => ask(false), wait);
  }

  async function ask(byHand) {
    if (!alive || asking) return;
    asking = true;
    clearTimeout(timer);
    try {
      const res = await jobs.status(id, ctx.kaggleKey.get() || null);
      if (!alive) return;
      if (!res || typeof res !== 'object') throw new Error('The status came back empty.');
      status = res;
      last = Date.now();
      finished = res.stage === 'done' || res.stage === 'failed';
      root.classList.remove('job--stale');
      draw();
    } catch (err) {
      if (!alive) return;
      last = Date.now();
      root.classList.add('job--stale');
      if (!status) {
        replace(message, notice('error', 'The status could not be read. ', errText(err, ''), ' The page tries again in 20 seconds.'));
      } else if (byHand) {
        toast('The status could not be read. ' + errText(err, ''), 'error');
      }
      checked.textContent = 'Last try failed';
    } finally {
      asking = false;
      schedule();
    }
  }

  async function loadInfo() {
    try {
      const list = (await jobs.list()) || [];
      if (!alive) return;
      info = list.find((j) => j && String(j.job) === id) || null;
      if (info) {
        heading.textContent = String(info.title || 'Your job');
        sub.textContent = [`Job ${id}`, PRESET_TEXT[info.preset] || ''].filter(Boolean).join(' · ');
        drawClock();
      } else {
        elapsed.textContent = 'Not known';
      }
    } catch (err) {
      console.warn('[namma] job list', err);
      if (alive) elapsed.textContent = 'Not known';
    }
  }

  const onVisible = () => {
    if (document.hidden) clearTimeout(timer);
    else if (!finished) {
      if (Date.now() - last >= POLL_MS) ask(false);
      else schedule();
    }
  };
  document.addEventListener('visibilitychange', onVisible);
  clock = setInterval(() => { if (!document.hidden && !finished && info) drawClock(); }, 1000);
  ctx.onLeave(() => {
    alive = false;
    clearTimeout(timer);
    clearInterval(clock);
    document.removeEventListener('visibilitychange', onVisible);
  });

  replace(timeline, h('li', { class: 'timeline__wait' }, spinner('Reading the status')));
  const root = h('section', { class: 'wrap page page--narrow job' },
    h('a', { class: 'back', href: '#/library' }, 'Your library'),
    h('h1', { class: 'title title--page' }, heading),
    sub,
    h('div', { class: 'job__grid' },
      h('div', { class: 'panel' }, timeline),
      h('div', { class: 'panel job__side' },
        h('dl', { class: 'kv' },
          h('dt', null, 'Time so far'), elapsed,
          h('dt', null, 'Kaggle job'), kaggleState,
          h('dt', null, 'Last checked'), checked),
        keyBox)),
    message,
    actions);

  loadInfo();
  ask(false);
  return root;
}
