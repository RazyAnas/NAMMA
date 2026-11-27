// NAMMA site: hash router and shared state. Static page, no server of its own: it talks to Hugging Face and Kaggle straight
// from the browser through the modules in ./api (loaded on demand, so the page still opens when one of them is missing or broken).
import { h, replace, icon, toast, errText, stateBox, spinner, safeUrl, session as tabStore, device as deviceStore } from './ui.js';
import * as home from './pages/home.js';
import * as explore from './pages/explore.js';
import * as library from './pages/library.js';
import * as create from './pages/create.js';
import * as job from './pages/job.js';

const ROUTES = [
  { re: /^\/?$/, page: home, title: 'NAMMA: walk through a place filmed on a phone' },
  { re: /^\/explore\/?$/, page: explore, title: 'Public works', nav: 'explore' },
  { re: /^\/library\/?$/, page: library, title: 'Your library', nav: 'library', auth: true },
  { re: /^\/create\/?$/, page: create, title: 'Make a tour', nav: 'create', auth: true },
  { re: /^\/job\/([^/]+)\/?$/, page: job, title: 'Your job', nav: 'library', auth: true },
  { re: /^\/privacy\/?$/, page: { render: home.renderPrivacy }, title: 'Privacy' },
];

const KAGGLE_KEY = 'namma.kaggleKey';
const RETURN_TO = 'namma.returnTo';

const state = {
  booting: true,
  api: { hf: null, jobs: null, kaggle: null },
  apiErrors: {},
  session: null,
  busy: null,          // text shown when the user tries to leave during an upload
  gallery: null,       // cached promise of the public tour list
  leave: [],           // clean-up functions of the page on screen
  lastHash: '',
};

const view = () => document.getElementById('view');
const embedded = (() => { try { return window.self !== window.top; } catch { return true; } })();

// ------------------------------------------------------------------------------------------------------------------- api
async function loadApi() {
  const wanted = [['hf', './api/hf.js'], ['jobs', './api/jobs.js'], ['kaggle', './api/kaggle.js']];
  await Promise.all(wanted.map(async ([name, path]) => {
    try {
      const mod = await import(path);
      if (!mod || !mod[name]) throw new Error(`${path} has no export "${name}"`);
      state.api[name] = mod[name];
    } catch (err) {
      state.api[name] = null;
      state.apiErrors[name] = errText(err, 'could not be loaded');
      console.warn('[namma] api module not available:', name, err);
    }
  }));
}

function readSession() {
  const { hf } = state.api;
  if (!hf) return null;
  try {
    const s = hf.session();
    return s && s.user ? s : null;
  } catch {
    return null;
  }
}

function signInAvailable() {
  const { hf } = state.api;
  if (!hf) return false;
  try { return Boolean(hf.available()); } catch { return false; }
}

async function signIn() {
  const { hf } = state.api;
  if (!hf || !signInAvailable()) {
    toast('Sign-in works on the published site.', 'info');
    return;
  }
  if (embedded) {
    // Inside the Hugging Face page frame the sign-in redirect is blocked: open the site in its own tab instead.
    window.open(window.location.href, '_blank', 'noopener');
    toast('NAMMA opened in its own tab. Sign in there.', 'info');
    return;
  }
  try {
    tabStore.set(RETURN_TO, window.location.hash || '#/');
    await hf.login();
  } catch (err) {
    toast('Sign-in could not start. ' + errText(err, ''), 'error');
  }
}

function signOut() {
  try { state.api.hf?.logout(); } catch (err) { console.warn('[namma] logout', err); }
  kaggleKey.clear();
  state.session = null;
  renderAuth();
  route();
  toast('Signed out. The saved Kaggle key was removed from this browser.', 'ok');
}

// The Kaggle key: this tab only (sessionStorage), or this browser (localStorage) when the user ticks "remember".
const kaggleKey = {
  get() { return tabStore.get(KAGGLE_KEY) || deviceStore.get(KAGGLE_KEY) || ''; },
  remembered() { return Boolean(deviceStore.get(KAGGLE_KEY)); },
  set(value, remember) {
    const v = String(value || '').trim();
    if (!v) return this.clear();
    tabStore.set(KAGGLE_KEY, v);
    if (remember) deviceStore.set(KAGGLE_KEY, v);
    else deviceStore.del(KAGGLE_KEY);
  },
  clear() { tabStore.del(KAGGLE_KEY); deviceStore.del(KAGGLE_KEY); },
};

function loadGallery(force = false) {
  if (!state.gallery || force) {
    state.gallery = (async () => {
      const res = await fetch('gallery.json', { cache: 'no-cache' });
      if (!res.ok) throw new Error(`The tour list answered with status ${res.status}.`);
      const data = await res.json();
      const list = Array.isArray(data) ? data : Array.isArray(data?.entries) ? data.entries : [];
      return list.filter((e) => e && typeof e === 'object' && safeUrl(e.url)).map((e) => ({
        id: String(e.id ?? e.run ?? e.url),
        run: e.run ? String(e.run) : '',
        title: String(e.title || 'Untitled tour'),
        kind: e.kind ? String(e.kind) : '',
        tags: Array.isArray(e.tags) ? e.tags.map(String) : [],
        by: e.by ? String(e.by) : '',
        created: e.created || '',
        url: safeUrl(e.url),
        cover: safeUrl(e.cover),
        splats: Number(e.splats) || 0,
        area: Number(e.area_m2) || 0,
        sizeMb: Number(e.size_mb) || 0,
      }));
    })();
    state.gallery.catch(() => { state.gallery = null; });
  }
  return state.gallery;
}

// ---------------------------------------------------------------------------------------------------------------- header
function renderAuth() {
  const slot = document.getElementById('auth');
  if (!slot) return;
  const s = state.session;
  if (state.booting) {
    replace(slot, h('span', { class: 'auth__wait' }, ''));
    return;
  }
  if (!s) {
    const can = signInAvailable();
    replace(slot, h('button', { type: 'button', class: 'btn btn--line btn--sm', onclick: signIn, title: can ? null : 'Sign-in works on the published site' },
      icon('user', 18), h('span', { class: 'auth__long' }, 'Sign in with Hugging Face'), h('span', { class: 'auth__short' }, 'Sign in')));
    return;
  }
  const name = String(s.user.fullname || s.user.name || 'You');
  const avatar = safeUrl(s.user.avatarUrl);
  const menu = h('div', { class: 'menu', hidden: true, role: 'menu' },
    h('p', { class: 'menu__who' }, 'Signed in as ', h('b', null, String(s.user.name || name))),
    h('a', { class: 'menu__item', href: '#/library', role: 'menuitem' }, icon('cube', 18), 'Your library'),
    h('button', { type: 'button', class: 'menu__item', role: 'menuitem', onclick: signOut }, icon('out', 18), 'Sign out'));
  const button = h('button', { type: 'button', class: 'avatar', 'aria-haspopup': 'menu', 'aria-expanded': 'false', 'aria-label': `Account menu for ${name}` },
    avatar ? h('img', { src: avatar, alt: '', width: 32, height: 32, referrerpolicy: 'no-referrer' }) : h('span', { class: 'avatar__ph' }, icon('user', 18)),
    h('span', { class: 'avatar__name' }, name));
  const close = () => { menu.hidden = true; button.setAttribute('aria-expanded', 'false'); };
  button.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
    button.setAttribute('aria-expanded', String(!menu.hidden));
  });
  menu.addEventListener('click', close);
  const wrap = h('div', { class: 'auth__menu' }, button, menu);
  wrap.addEventListener('keydown', (e) => { if (e.key === 'Escape') { close(); button.focus(); } });
  replace(slot, wrap);
}

document.addEventListener('click', (e) => {
  const open = document.querySelector('.menu:not([hidden])');
  if (open && !open.parentElement.contains(e.target)) {
    open.hidden = true;
    open.parentElement.querySelector('.avatar')?.setAttribute('aria-expanded', 'false');
  }
});

function markNav(name) {
  for (const a of document.querySelectorAll('[data-nav]')) {
    if (a.dataset.nav === name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
}

// ---------------------------------------------------------------------------------------------------------------- router
function signInPanel(what) {
  const can = signInAvailable();
  if (!state.api.hf) {
    return stateBox({ kind: 'empty', iconName: 'lock', title: 'Sign-in is not available here',
      body: 'Sign-in works on the published site. You can still look at the public tours.',
      actions: [h('a', { class: 'btn btn--primary', href: '#/explore' }, 'Explore tours')] });
  }
  return stateBox({ kind: 'empty', iconName: 'lock', title: `Sign in to ${what}`,
    body: can
      ? 'NAMMA keeps your videos and tours in your own Hugging Face account, so it needs you to sign in there. A free account is enough.'
      : 'Sign-in works on the published site. This copy of the page is a preview without it.',
    actions: [
      can ? h('button', { type: 'button', class: 'btn btn--primary', onclick: signIn }, 'Sign in with Hugging Face') : null,
      h('a', { class: 'btn btn--ghost', href: '#/explore' }, 'Explore tours'),
    ] });
}

function currentPath() {
  const raw = window.location.hash.replace(/^#/, '');
  return raw.startsWith('/') ? raw : '/';
}

function route() {
  const path = currentPath();
  if (state.busy && window.location.hash !== state.lastHash) {
    // An upload is running in this page: stay here, leaving would lose it.
    toast(state.busy, 'info');
    window.history.replaceState(null, '', state.lastHash || '#/');
    return;
  }
  state.lastHash = window.location.hash;
  for (const fn of state.leave.splice(0)) { try { fn(); } catch (err) { console.warn('[namma] leave', err); } }

  let match = null;
  let found = null;
  for (const r of ROUTES) {
    match = r.re.exec(path);
    if (match) { found = r; break; }
  }
  const root = view();
  const before = state.session?.user?.name || null;
  state.session = readSession();
  if (!state.booting && before !== (state.session?.user?.name || null)) renderAuth();   // the 8-hour sign-in ran out
  if (!found) {
    document.title = 'Not found | NAMMA';
    markNav('');
    replace(root, h('section', { class: 'wrap page' }, stateBox({ kind: 'empty', title: 'This page does not exist',
      actions: [h('a', { class: 'btn btn--primary', href: '#/' }, 'Go to the start page')] })));
    return;
  }
  document.title = found.page === home ? found.title : `${found.title} | NAMMA`;
  markNav(found.nav || '');

  let node;
  if (found.auth && state.booting) {
    node = h('section', { class: 'wrap page page--center' }, spinner('Checking sign-in'));
  } else if (found.auth && !state.session) {
    const what = found.page === create ? 'make a tour' : found.page === job ? 'see this job' : 'see your library';
    node = h('section', { class: 'wrap page' }, signInPanel(what));
  } else {
    const ctx = {
      params: match.slice(1).map((p) => { try { return decodeURIComponent(p); } catch { return p; } }),
      session: state.session,
      api: state.api,
      apiErrors: state.apiErrors,
      signIn,
      signInAvailable,
      kaggleKey,
      gallery: loadGallery,
      navigate(hash) { window.location.hash = hash; },
      onLeave(fn) { state.leave.push(fn); },
      setBusy(text) { state.busy = text || null; },
    };
    try {
      node = found.page.render(ctx);
    } catch (err) {
      console.error('[namma] page failed', err);
      node = h('section', { class: 'wrap page' }, stateBox({ kind: 'error', title: 'This page could not be shown', body: errText(err),
        actions: [h('button', { type: 'button', class: 'btn btn--line', onclick: () => route() }, 'Try again')] }));
    }
  }
  replace(root, node);
  window.scrollTo(0, 0);
  if (state.focusNext) root.focus({ preventScroll: true });
  state.focusNext = true;
}

window.addEventListener('hashchange', route);
window.addEventListener('beforeunload', (e) => {
  if (state.busy) { e.preventDefault(); e.returnValue = ''; }
});

// ------------------------------------------------------------------------------------------------------------------ boot
async function boot() {
  const year = document.getElementById('year');
  if (year) year.textContent = String(new Date().getFullYear());
  if (embedded) {
    const bar = document.getElementById('embed-note');
    if (bar) {
      replace(bar, h('span', null, 'For sign-in and uploads, open NAMMA in its own tab. '),
        h('a', { href: window.location.href.startsWith('https:') ? window.location.href : './', target: '_blank', rel: 'noopener' }, 'Open in a new tab'));
      bar.hidden = false;
    }
  }
  renderAuth();
  route();                               // Home and Explore show at once, without waiting for sign-in

  await loadApi();
  const hadCode = /[?&]code=/.test(window.location.search);
  if (state.api.hf) {
    try {
      await state.api.hf.init();
    } catch (err) {
      toast('Sign-in did not finish. ' + errText(err, 'Please try again.'), 'error');
    }
  }
  state.session = readSession();
  state.booting = false;
  if (hadCode && /[?&]code=/.test(window.location.search)) {
    // the sign-in answer is used up: take it out of the address bar
    window.history.replaceState(null, '', window.location.pathname + window.location.hash);
  }
  const back = tabStore.get(RETURN_TO);
  tabStore.del(RETURN_TO);
  renderAuth();
  if (hadCode && state.session && back && back.startsWith('#/') && back !== window.location.hash) {
    window.location.hash = back;         // the hashchange event draws the page
    return;
  }
  const path = currentPath();
  if (ROUTES.some((r) => r.auth && r.re.test(path))) route();
}

boot().catch((err) => {
  console.error('[namma] start failed', err);
  state.booting = false;
  renderAuth();
  route();
});
