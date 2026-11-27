// Create (#/create, needs sign-in): "From a video" (upload, then start the user's Kaggle job) and "I already have a splat file".
import { h, replace, icon, extLink, notice, progressBar, toast, errText, fmtBytes, fmtDuration, session as tabStore } from '../ui.js';

const VIDEO_EXT = /\.(mp4|mov)$/i;
const SPLAT_EXT = /\.(compressed\.ply|ply|sog|splat)$/i;
const PRESETS = [
  ['fast', 'Fast', 'About 55 minutes. Good for most rooms.'],
  ['quality', 'Quality', 'About 90 minutes. Sharper details.'],
];
const TIPS = [
  'Walk slowly. No fast turns.',
  'One loop around the room with the phone at eye level.',
  'One loop with the phone tilted down 30 to 40 degrees, looking about 2 m ahead.',
  'One loop with the phone tilted up.',
  'Turn the lights on.',
  '2 to 4 minutes per room.',
];

let uid = 0;
const nextId = (p) => `${p}-${++uid}`;

function field(label, control, help) {
  const id = control.id || (control.id = nextId('f'));
  return h('div', { class: 'field' }, h('label', { class: 'field__label', for: id }, label), control, help ? h('p', { class: 'field__help' }, help) : null);
}

function titleFromFile(name) {
  return name.replace(/\.[a-z0-9.]+$/i, '').replace(/[_-]+/g, ' ').trim().slice(0, 80);
}

/** A file picker that is also a drop target. onFile(file | null). */
function dropZone({ accept, test, hint, wrong, onFile }) {
  const input = h('input', { type: 'file', class: 'sr-only', accept, id: nextId('file') });
  const info = h('p', { class: 'drop__info' }, hint);
  const zone = h('label', { class: 'drop', for: input.id }, input,
    h('span', { class: 'drop__icon' }, icon('upload', 26)),
    h('span', { class: 'drop__title' }, 'Choose a file or drop it here'), info);
  const take = (file) => {
    if (!file) return;
    if (!test.test(file.name)) {
      toast(wrong, 'error');
      return;
    }
    zone.classList.add('drop--has');
    onFile(file, info);
  };
  input.addEventListener('change', () => take(input.files && input.files[0]));
  for (const ev of ['dragenter', 'dragover']) zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('drop--over'); });
  for (const ev of ['dragleave', 'drop']) zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('drop--over'); });
  zone.addEventListener('drop', (e) => take(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]));
  return { root: zone, input };
}

/** The length of a video, read by a hidden <video> element from the local file. Resolves 0 when the browser cannot read it. */
function videoSeconds(file) {
  return new Promise((resolve) => {
    let url = '';
    const video = document.createElement('video');
    const finish = (value) => {
      clearTimeout(timer);
      video.removeAttribute('src');
      try { video.load(); } catch { /* nothing to unload */ }
      if (url) URL.revokeObjectURL(url);
      resolve(Number.isFinite(value) && value > 0 ? value : 0);
    };
    const timer = setTimeout(() => finish(0), 8000);
    video.preload = 'metadata';
    video.muted = true;
    video.addEventListener('loadedmetadata', () => finish(video.duration));
    video.addEventListener('error', () => finish(0));
    try {
      url = URL.createObjectURL(file);
      video.src = url;
    } catch {
      finish(0);
    }
  });
}

function consentBox() {
  const input = h('input', { type: 'checkbox', id: nextId('ok') });
  return { input, root: h('label', { class: 'check', for: input.id }, input, h('span', null, 'I have the right to share this place.')) };
}

// --------------------------------------------------------------------------------------------------------- from a video
function videoForm(ctx) {
  const { jobs, kaggle } = ctx.api;
  let file = null;
  let jobId = null;        // set once the upload is finished, so a failed start can be tried again without a new upload
  let keyUser = '';

  const title = h('input', { type: 'text', class: 'input', maxlength: 80, autocomplete: 'off', placeholder: 'Living room' });
  const key = h('input', { type: 'password', class: 'input', autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off', placeholder: 'KGAT_... or the text of kaggle.json', value: ctx.kaggleKey.get() });
  const remember = h('input', { type: 'checkbox', id: nextId('rem'), checked: ctx.kaggleKey.remembered() });
  const keyState = h('p', { class: 'field__state', role: 'status' }, '');
  const check = h('button', { type: 'button', class: 'btn btn--line' }, 'Check');
  const consent = consentBox();
  const go = h('button', { type: 'submit', class: 'btn btn--primary btn--lg' }, icon('upload', 20), 'Upload and start');
  const out = h('div', { class: 'form__out' });
  const hint = h('p', { class: 'field__help', role: 'status' }, '');

  const drop = dropZone({ accept: '.mp4,.mov,video/mp4,video/quicktime', test: VIDEO_EXT, hint: 'MP4 or MOV from your phone', wrong: 'Please choose an MP4 or MOV video.',
    onFile: async (f, info) => {
      file = f;
      jobId = null;
      info.textContent = `${f.name} · ${fmtBytes(f.size)} · reading the length...`;
      if (!title.value.trim()) title.value = titleFromFile(f.name);
      const seconds = await videoSeconds(f);
      if (file !== f) return;
      info.textContent = `${f.name} · ${fmtBytes(f.size)} · ${seconds ? fmtDuration(seconds) : 'length not readable in this browser'}`;
      hint.textContent = seconds && seconds < 60 ? 'This video is short. 2 to 4 minutes per room gives a better tour.'
        : seconds > 600 ? 'This video is long. 2 to 4 minutes per room is enough.' : '';
    } });

  const presetName = nextId('preset');
  const presets = h('fieldset', { class: 'choice' }, h('legend', { class: 'field__label' }, 'Setting'),
    PRESETS.map(([value, label, text], i) => h('label', { class: 'choice__opt' },
      h('input', { type: 'radio', name: presetName, value, checked: i === 0 }),
      h('span', { class: 'choice__face' }, h('b', null, label), h('span', null, text)))));
  const preset = () => presets.querySelector('input:checked')?.value || 'fast';

  const saveKey = () => ctx.kaggleKey.set(key.value, remember.checked);
  key.addEventListener('input', () => { keyUser = ''; keyState.textContent = ''; keyState.className = 'field__state'; });
  key.addEventListener('change', saveKey);
  remember.addEventListener('change', saveKey);

  async function checkKey() {
    const value = key.value.trim();
    if (!value) {
      keyState.className = 'field__state field__state--bad';
      keyState.textContent = 'Paste your Kaggle key first.';
      return false;
    }
    if (!kaggle) {
      keyState.className = 'field__state';
      keyState.textContent = 'The key cannot be checked in this preview. It is checked when the job starts.';
      return true;
    }
    check.disabled = true;
    keyState.className = 'field__state';
    keyState.textContent = 'Checking...';
    try {
      const res = await kaggle.check(value);
      keyUser = String(res?.username || '');
      saveKey();
      keyState.className = 'field__state field__state--ok';
      keyState.textContent = keyUser ? `The key works. Kaggle account: ${keyUser}.` : 'The key works.';
      return true;
    } catch (err) {
      keyUser = '';
      keyState.className = 'field__state field__state--bad';
      keyState.textContent = errText(err, 'Kaggle did not accept this key.');
      return false;
    } finally {
      check.disabled = false;
    }
  }
  check.addEventListener('click', checkKey);

  function lock(on) {
    for (const el of form.querySelectorAll('input, button')) el.disabled = on;
    ctx.setBusy(on ? 'An upload is running. Stay on this page until it finishes.' : null);
  }

  async function start() {
    const bar = progressBar('Starting the job');
    replace(out, bar.root);
    bar.busy('Starting the job on your Kaggle account...');
    const res = await jobs.start(jobId, key.value.trim());
    bar.done();
    if (res?.kernelUrl) tabStore.set(`namma.kernel.${jobId}`, String(res.kernelUrl));
    if (keyUser) tabStore.set('namma.kaggleUser', keyUser);
    return res;
  }

  async function submit(e) {
    e.preventDefault();
    const problems = [];
    if (!file) problems.push('Choose a video.');
    if (!title.value.trim()) problems.push('Give the tour a name.');
    if (!key.value.trim()) problems.push('Paste your Kaggle key.');
    if (!consent.input.checked) problems.push('Tick the box that you have the right to share this place.');
    if (!jobs) problems.push('A part of this site did not load. Reload the page.');
    if (problems.length) {
      replace(out, notice('error', problems.join(' ')));
      return;
    }
    replace(out);
    if (!keyUser && !(await checkKey())) {
      replace(out, notice('error', 'The Kaggle key did not pass the check. Nothing was uploaded.'));
      return;
    }
    saveKey();
    lock(true);
    let step = 'prepare';
    try {
      if (!jobId) {
        const bar = progressBar('Uploading the video');
        replace(out, bar.root);
        bar.busy('Preparing the storage in your Hugging Face account...');
        await jobs.ensureRepos();
        bar.done();
        step = 'upload';
        bar.set(0, 'Uploading the video');
        jobId = await jobs.create(file, { title: title.value.trim(), preset: preset() }, (fraction, text) => bar.set(fraction, text || 'Uploading the video'));
        if (jobId == null || jobId === '') throw new Error('The upload finished without a job number.');
        jobId = String(jobId);
        bar.set(1, 'Uploaded');
      }
      step = 'start';
      await start();
      ctx.setBusy(null);
      ctx.navigate(`#/job/${encodeURIComponent(jobId)}`);
    } catch (err) {
      lock(false);
      const text = errText(err);
      if (step === 'start') {
        replace(out, notice('error', 'The video is uploaded, but the Kaggle job did not start. ', text),
          h('div', { class: 'form__row' },
            h('button', { type: 'submit', class: 'btn btn--primary' }, 'Try to start again'),
            h('a', { class: 'btn btn--line', href: `#/job/${encodeURIComponent(jobId)}` }, 'Open the job page')));
      } else if (step === 'upload') {
        replace(out, notice('error', 'The upload did not finish. ', text, ' Press "Upload and start" again with the same file: it continues where it stopped.'));
      } else {
        replace(out, notice('error', 'The storage in your Hugging Face account could not be prepared. ', text));
      }
    }
  }

  const form = h('form', { class: 'form', novalidate: true, onsubmit: submit },
    drop.root, hint,
    field('Name of the tour', title),
    presets,
    h('div', { class: 'field' },
      h('label', { class: 'field__label', for: key.id || (key.id = nextId('key')) }, 'Your Kaggle key'),
      h('div', { class: 'field__row' }, key, check),
      keyState,
      h('label', { class: 'check check--small', for: remember.id }, remember, h('span', null, 'Remember on this device. Leave this off on a shared computer.')),
      h('details', { class: 'help' },
        h('summary', null, 'Where do I get this?'),
        h('ol', null,
          h('li', null, 'Open ', extLink('https://www.kaggle.com/settings', null, 'kaggle.com'), ' and go to Settings.'),
          h('li', null, 'In the API section press "Create new token".'),
          h('li', null, 'Copy the token that starts with KGAT_ and paste it here. The whole text of a kaggle.json file works too.')),
        h('p', null, 'Your Kaggle account must be phone-verified, otherwise Kaggle gives it no GPU. The key stays in this browser and is sent only to Kaggle. The site uses it to start one private notebook on your account, which also receives your Hugging Face sign-in pass so it can read the video and save the tour.'))),
    consent.root,
    h('div', { class: 'form__row' }, go),
    out);
  return form;
}

// ------------------------------------------------------------------------------------------------------- a splat file
function splatForm(ctx) {
  const { jobs } = ctx.api;
  let file = null;
  const title = h('input', { type: 'text', class: 'input', maxlength: 80, autocomplete: 'off', placeholder: 'Living room' });
  const consent = consentBox();
  const out = h('div', { class: 'form__out' });
  const go = h('button', { type: 'submit', class: 'btn btn--primary btn--lg' }, icon('upload', 20), 'Upload');
  const drop = dropZone({ accept: '.ply,.sog,.splat', test: SPLAT_EXT, hint: '.ply, .compressed.ply, .sog or .splat', wrong: 'Please choose a .ply, .compressed.ply, .sog or .splat file.',
    onFile: (f, info) => {
      file = f;
      info.textContent = `${f.name} · ${fmtBytes(f.size)}`;
      if (!title.value.trim()) title.value = titleFromFile(f.name);
    } });

  async function submit(e) {
    e.preventDefault();
    const problems = [];
    if (!file) problems.push('Choose a splat file.');
    if (!title.value.trim()) problems.push('Give it a name.');
    if (!consent.input.checked) problems.push('Tick the box that you have the right to share this place.');
    if (!jobs) problems.push('A part of this site did not load. Reload the page.');
    if (problems.length) {
      replace(out, notice('error', problems.join(' ')));
      return;
    }
    const bar = progressBar('Uploading the splat file');
    replace(out, bar.root);
    for (const el of form.querySelectorAll('input, button')) el.disabled = true;
    ctx.setBusy('An upload is running. Stay on this page until it finishes.');
    try {
      bar.busy('Preparing the storage in your Hugging Face account...');
      await jobs.ensureRepos();
      bar.done();
      bar.set(0, 'Uploading');
      const entry = await jobs.uploadSplat(file, { title: title.value.trim() }, (fraction, text) => bar.set(fraction, text || 'Uploading'));
      bar.set(1, 'Uploaded');
      replace(out, notice('ok', 'Saved in your library as link-only. You can change that there.'),
        h('div', { class: 'form__row' },
          h('a', { class: 'btn btn--primary', href: '#/library' }, 'Open your library'),
          extLink(entry?.url, { class: 'btn btn--line' }, 'Open it', icon('open', 18))));
      form.reset();
      file = null;
    } catch (err) {
      replace(out, notice('error', 'The upload did not finish. ', errText(err)));
    } finally {
      ctx.setBusy(null);
      for (const el of form.querySelectorAll('input, button')) el.disabled = false;
    }
  }

  const form = h('form', { class: 'form', novalidate: true, onsubmit: submit },
    drop.root,
    field('Name', title),
    h('p', { class: 'field__help' }, 'No training is needed and no Kaggle key. The file goes into the public tour Space of your Hugging Face account and starts as link-only: anyone who has the link can open it.'),
    consent.root,
    h('div', { class: 'form__row' }, go),
    out);
  return form;
}

// ----------------------------------------------------------------------------------------------------------------- page
export function render(ctx) {
  const tabs = [
    { id: 'video', label: 'From a video', icon: 'video', build: () => videoForm(ctx) },
    { id: 'splat', label: 'I already have a splat file', icon: 'cube', build: () => splatForm(ctx) },
  ];
  const panel = h('div', { class: 'panel', role: 'tabpanel', id: 'create-panel', tabindex: 0 });
  const tips = h('aside', { class: 'tips', 'aria-labelledby': 'tips-title' },
    h('h2', { id: 'tips-title', class: 'tips__title' }, icon('video', 20), 'How to film'),
    h('ul', { class: 'tips__list' }, TIPS.map((t) => h('li', null, t))));
  const buttons = tabs.map((t) => h('button', { type: 'button', class: 'tab', role: 'tab', id: `tab-${t.id}`, 'aria-controls': 'create-panel' }, icon(t.icon, 18), t.label));
  const built = {};

  function show(i, focus) {
    buttons.forEach((b, k) => {
      b.setAttribute('aria-selected', String(k === i));
      b.tabIndex = k === i ? 0 : -1;
    });
    const t = tabs[i];
    built[t.id] = built[t.id] || t.build();
    panel.setAttribute('aria-labelledby', `tab-${t.id}`);
    replace(panel, built[t.id]);
    tips.hidden = t.id !== 'video';
    if (focus) buttons[i].focus();
  }
  buttons.forEach((b, i) => {
    b.addEventListener('click', () => show(i, false));
    b.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        show((i + (e.key === 'ArrowRight' ? 1 : buttons.length - 1)) % buttons.length, true);
      }
    });
  });
  show(0, false);
  ctx.onLeave(() => ctx.setBusy(null));

  return h('section', { class: 'wrap page' },
    h('h1', { class: 'title title--page' }, h('span', null, 'Make a tour'), h('span', { class: 'title__dim' }, 'from your own video')),
    h('div', { class: 'tabs', role: 'tablist', 'aria-label': 'How do you want to start?' }, buttons),
    h('div', { class: 'create' }, panel, tips));
}
