// Small DOM helpers for the NAMMA site. No framework, no HTML strings: every node is built with createElement and every text
// goes in through textContent, because sign-in and Kaggle keys live in this page's memory and remote strings must never be
// parsed as HTML.

const SVG_NS = 'http://www.w3.org/2000/svg';
const DOM_PROPS = new Set(['value', 'checked', 'disabled', 'selected', 'hidden', 'multiple', 'required', 'readOnly', 'open', 'tabIndex']);
const URL_ATTRS = new Set(['href', 'src', 'action', 'poster']);
const BLOCKED = new Set(['innerHTML', 'outerHTML', 'srcdoc']);

/** A safe absolute https URL (string) or null. Use for every link or image address that comes from a file or from the network. */
export function safeUrl(value) {
  if (typeof value !== 'string' || !value) return null;
  try {
    const u = new URL(value);
    return u.protocol === 'https:' ? u.href : null;
  } catch {
    return null;
  }
}

// Addresses the page may put into href/src: https, in-page routes, relative local files, blob: previews and data: images.
function attrUrl(value) {
  const s = String(value).trim();
  if (s.startsWith('#') || s.startsWith('./') || s.startsWith('../')) return s;
  if (/^blob:/i.test(s) || /^data:image\//i.test(s)) return s;
  if (/^[a-z][a-z0-9+.-]*:/i.test(s) || s.startsWith('//')) return safeUrl(s.startsWith('//') ? 'https:' + s : s);
  return s;
}

function append(parent, kids) {
  for (const kid of kids) {
    if (kid == null || kid === false || kid === true) continue;
    if (Array.isArray(kid)) append(parent, kid);
    else if (kid instanceof Node) parent.appendChild(kid);
    else parent.appendChild(document.createTextNode(String(kid)));
  }
}

/**
 * h('a', { class: 'btn', href: url, onclick: fn }, 'Open', icon('open'))
 * Strings become text nodes. There is deliberately no way to pass markup.
 */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value == null || BLOCKED.has(key)) continue;
      if (key.startsWith('on')) {
        if (typeof value === 'function') el.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === 'class') {
        el.className = Array.isArray(value) ? value.filter(Boolean).join(' ') : String(value);
      } else if (key === 'text') {
        el.textContent = String(value);
      } else if (key === 'dataset') {
        for (const [k, v] of Object.entries(value)) if (v != null) el.dataset[k] = String(v);
      } else if (key === 'style' && typeof value === 'object') {
        for (const [k, v] of Object.entries(value)) el.style.setProperty(k, String(v));
      } else if (DOM_PROPS.has(key)) {
        el[key] = value;
      } else if (URL_ATTRS.has(key)) {
        const u = attrUrl(value);
        if (u) el.setAttribute(key, u);
      } else if (value === true) {
        el.setAttribute(key, '');
      } else if (value !== false) {
        el.setAttribute(key, String(value));
      }
    }
  }
  append(el, children);
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...children) {
  clear(el);
  append(el, children);
  return el;
}

// ---------------------------------------------------------------------------------------------------------------- icons
// 24 x 24 line icons drawn here (no icon font, no image files). Each entry is a list of [tag, attributes].
const ICONS = {
  open: [['path', { d: 'M7 17 17 7' }], ['path', { d: 'M8 7h9v9' }]],
  arrow: [['path', { d: 'M5 12h14' }], ['path', { d: 'm13 6 6 6-6 6' }]],
  copy: [['rect', { x: 9, y: 9, width: 11, height: 11, rx: 2 }], ['path', { d: 'M5 15V6a2 2 0 0 1 2-2h9' }]],
  trash: [['path', { d: 'M4 7h16' }], ['path', { d: 'M9 7V4h6v3' }], ['path', { d: 'M6 7l1 13h10l1-13' }], ['path', { d: 'M10 11v6M14 11v6' }]],
  check: [['path', { d: 'm5 12.5 4.5 4.5L19 7.5' }]],
  x: [['path', { d: 'M6 6l12 12M18 6 6 18' }]],
  upload: [['path', { d: 'M12 16V4' }], ['path', { d: 'm7 9 5-5 5 5' }], ['path', { d: 'M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3' }]],
  video: [['rect', { x: 3, y: 6, width: 13, height: 12, rx: 2.5 }], ['path', { d: 'm16 10.5 5-3v9l-5-3' }]],
  cube: [['path', { d: 'M12 3 4 7.5v9L12 21l8-4.5v-9L12 3Z' }], ['path', { d: 'M4 7.5 12 12l8-4.5M12 12v9' }]],
  lock: [['rect', { x: 5, y: 11, width: 14, height: 9, rx: 2 }], ['path', { d: 'M8 11V8a4 4 0 0 1 8 0v3' }]],
  link: [['path', { d: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1' }], ['path', { d: 'M14 10a4 4 0 0 0-5.7 0l-3 3A4 4 0 0 0 11 18.700l1-1' }]],
  globe: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M3 12h18' }], ['path', { d: 'M12 3c2.800 3 2.800 15 0 18M12 3c-2.800 3-2.800 15 0 18' }]],
  user: [['circle', { cx: 12, cy: 8, r: 4 }], ['path', { d: 'M4 21c1.500-4 4.500-6 8-6s6.500 2 8 6' }]],
  out: [['path', { d: 'M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3' }], ['path', { d: 'M10 8 6 12l4 4M6 12h10' }]],
  info: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M12 11v6' }], ['path', { d: 'M12 7.500v.500' }]],
  alert: [['path', { d: 'M12 4 2.500 20h19L12 4Z' }], ['path', { d: 'M12 10v5' }], ['path', { d: 'M12 17.500v.500' }]],
  clock: [['circle', { cx: 12, cy: 12, r: 9 }], ['path', { d: 'M12 7v5l3 2' }]],
  phone: [['rect', { x: 7, y: 2.500, width: 10, height: 19, rx: 2.500 }], ['path', { d: 'M11 18h2' }]],
  chip: [['rect', { x: 6, y: 6, width: 12, height: 12, rx: 2 }], ['path', { d: 'M9 2.500V6M15 2.500V6M9 18v3.500M15 18v3.500M2.500 9H6M2.500 15H6M18 9h3.500M18 15h3.500' }]],
  walk: [['circle', { cx: 13, cy: 4.500, r: 1.800 }], ['path', { d: 'm9 21 2.500-7-2-2.500 1-4.500 3 1.500 2 3h3' }], ['path', { d: 'M12 14.500 15 17l1 4' }]],
  key: [['circle', { cx: 8, cy: 15, r: 4 }], ['path', { d: 'm11 12 8-8M16 7l3 3M14 9l2 2' }]],
  search: [['circle', { cx: 11, cy: 11, r: 6.500 }], ['path', { d: 'm16 16 4.500 4.500' }]],
  plus: [['path', { d: 'M12 5v14M5 12h14' }]],
  box: [['path', { d: 'M4 8 12 4l8 4v8l-8 4-8-4V8Z' }]],
};

export function icon(name, size = 20) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  const attrs = { viewBox: '0 0 24 24', width: size, height: size, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.600,
    'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true', focusable: 'false', class: 'ico' };
  for (const [k, v] of Object.entries(attrs)) svg.setAttribute(k, String(v));
  for (const [tag, a] of ICONS[name] || []) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(a)) node.setAttribute(k, String(v));
    svg.appendChild(node);
  }
  return svg;
}

// ---------------------------------------------------------------------------------------------------------------- text
export function errText(err, fallback = 'Something went wrong. Please try again.') {
  let msg = '';
  if (typeof err === 'string') msg = err;
  else if (err && typeof err.message === 'string') msg = err.message;
  msg = msg.replace(/\s+/g, ' ').trim();
  if (!msg || msg === '[object Object]') return fallback;
  if (/failed to fetch|networkerror|load failed/i.test(msg)) return 'The network request did not go through. Check your connection and try again.';
  return msg.length > 280 ? msg.slice(0, 277) + '...' : msg;
}

export function fmtDate(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (!value || Number.isNaN(d.getTime())) return '';
  try {
    return new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short', year: 'numeric' }).format(d);
  } catch {
    return d.toDateString();
  }
}

export function fmtBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes < 1e6) return Math.max(1, Math.round(bytes / 1e3)) + ' KB';
  if (bytes < 1e9) return (bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0) + ' MB';
  return (bytes / 1e9).toFixed(2) + ' GB';
}

export function fmtMb(mb) {
  return Number.isFinite(mb) && mb > 0 ? fmtBytes(mb * 1e6) : '';
}

/** Seconds of video: "3 min 20 s". */
export function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  return m ? `${m} min ${s % 60} s` : `${s} s`;
}

/** Milliseconds of waiting: "1 h 04 min" or "12 min 05 s". */
export function fmtElapsed(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '';
  const s = Math.floor(ms / 1000);
  const hrs = Math.floor(s / 3600);
  const min = Math.floor((s % 3600) / 60);
  const two = (n) => String(n).padStart(2, '0');
  if (hrs) return `${hrs} h ${two(min)} min`;
  return `${min} min ${two(s % 60)} s`;
}

export function fmtCount(n) {
  if (!Number.isFinite(n) || n <= 0) return '';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + ' million';
  if (n >= 1e3) return Math.round(n / 1e3) + ' thousand';
  return String(Math.round(n));
}

// ------------------------------------------------------------------------------------------------------------ building blocks
/** A link that opens in a new tab on its own origin. Returns null when the address is not a plain https URL. */
export function extLink(url, props, ...children) {
  const safe = safeUrl(url);
  if (!safe) return null;
  return h('a', { ...props, href: safe, target: '_blank', rel: 'noopener noreferrer' }, ...children);
}

export function spinner(label = 'Loading') {
  return h('span', { class: 'spinner', role: 'status' }, h('span', { class: 'spinner__dot', 'aria-hidden': 'true' }), h('span', { class: 'spinner__text' }, label));
}

/** Empty, error and "sign in first" panels. */
export function stateBox({ kind = 'empty', iconName, title, body, actions = [] }) {
  return h('div', { class: `state state--${kind}`, role: kind === 'error' ? 'alert' : null },
    h('div', { class: 'state__icon' }, icon(iconName || (kind === 'error' ? 'alert' : 'info'), 26)),
    h('h2', { class: 'state__title' }, title),
    body ? h('p', { class: 'state__body' }, body) : null,
    actions.length ? h('div', { class: 'state__actions' }, actions) : null);
}

export function notice(kind, ...children) {
  return h('p', { class: `notice notice--${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    icon(kind === 'error' ? 'alert' : kind === 'ok' ? 'check' : 'info', 18), h('span', null, ...children));
}

export function progressBar(label) {
  const fill = h('div', { class: 'bar__fill' });
  const text = h('p', { class: 'bar__text' }, '');
  const bar = h('div', { class: 'bar', role: 'progressbar', 'aria-label': label, 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': 0 }, fill);
  const root = h('div', { class: 'progress' }, bar, text);
  return {
    root,
    set(fraction, message) {
      const pct = Math.max(0, Math.min(100, Math.round((Number(fraction) || 0) * 100)));
      fill.style.width = pct + '%';
      bar.setAttribute('aria-valuenow', String(pct));
      text.textContent = message ? `${pct}%  ${message}` : `${pct}%`;
    },
    busy(message) {
      bar.removeAttribute('aria-valuenow');
      bar.classList.add('bar--busy');
      text.textContent = message || '';
    },
    done() {
      bar.classList.remove('bar--busy');
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------- toasts
export function toast(message, kind = 'info') {
  const region = document.getElementById('toasts');
  if (!region) return;
  const item = h('div', { class: `toast toast--${kind}` }, icon(kind === 'error' ? 'alert' : kind === 'ok' ? 'check' : 'info', 18), h('span', null, message));
  region.appendChild(item);
  while (region.children.length > 3) region.removeChild(region.firstChild);
  setTimeout(() => item.remove(), kind === 'error' ? 9000 : 5000);
}

// --------------------------------------------------------------------------------------------------------------- dialogs
/** A yes/no question in a native <dialog> (focus is kept inside it, Escape cancels). Resolves true or false. */
export function confirmDialog({ title, body, confirmLabel = 'Yes', cancelLabel = 'Cancel', danger = false }) {
  return new Promise((resolve) => {
    let settled = false;
    // Answer straight from the button: the dialog's own "close" event can arrive late (browsers send it with the next frame).
    const finish = (answer) => {
      if (settled) return;
      settled = true;
      try { if (dlg.open && typeof dlg.close === 'function') dlg.close(); } catch { /* already closed */ }
      dlg.remove();
      resolve(answer);
    };
    const cancel = h('button', { type: 'button', class: 'btn btn--ghost', onclick: () => finish(false) }, cancelLabel);
    const ok = h('button', { type: 'button', class: danger ? 'btn btn--danger' : 'btn btn--primary', onclick: () => finish(true) }, confirmLabel);
    const dlg = h('dialog', { class: 'dialog', 'aria-labelledby': 'dlg-title' },
      h('h2', { id: 'dlg-title', class: 'dialog__title' }, title),
      body ? h('p', { class: 'dialog__body' }, body) : null,
      h('div', { class: 'dialog__actions' }, cancel, ok));
    dlg.addEventListener('cancel', (e) => { e.preventDefault(); finish(false); });      // Escape
    dlg.addEventListener('close', () => finish(false));
    dlg.addEventListener('click', (e) => { if (e.target === dlg) finish(false); });     // a click on the dark area around it
    document.body.appendChild(dlg);
    if (typeof dlg.showModal === 'function') dlg.showModal();
    else dlg.setAttribute('open', '');
    cancel.focus();
  });
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = h('textarea', { class: 'sr-only', readonly: true });
    area.value = text;
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    area.remove();
    return ok;
  }
}

/** Storage that never throws (private windows, blocked storage). */
export function store(kind) {
  const mem = new Map();
  const real = () => { try { return window[kind]; } catch { return null; } };
  return {
    get(key) { try { const v = real()?.getItem(key); return v == null ? (mem.get(key) ?? null) : v; } catch { return mem.get(key) ?? null; } },
    set(key, value) { try { real().setItem(key, value); } catch { mem.set(key, value); } },
    del(key) { mem.delete(key); try { real()?.removeItem(key); } catch { /* nothing to remove */ } },
  };
}

export const session = store('sessionStorage');
export const device = store('localStorage');
