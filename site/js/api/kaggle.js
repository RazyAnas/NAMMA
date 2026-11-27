// Kaggle from the browser: check a key, see whether a notebook is running, push the private GPU notebook, read its status,
// delete it. Same calls as the official client (Kaggle/kagglesdk kaggle_http_client.py): every call is
// POST https://api.kaggle.com/v1/<service>/<method> with a JSON body. The key is passed in by the caller on every call,
// goes only into the Authorization header, and is never stored or logged here. Notes and sources: site/api.md.

const API = 'https://api.kaggle.com/v1';
const KERNELS = 'kernels.KernelsApiService';
const WEB = 'https://www.kaggle.com';

let customFetch = null;
/** Use this fetch instead of the browser's (node tests pass a fake one). */
export function setFetch(fn) { customFetch = fn || null; }

function fail(message, status = 0) {
  const e = new Error(message);
  e.status = status;
  return e;
}

const BAD_KEY = 'Kaggle did not accept this key. Make a new one at kaggle.com/settings and paste it again.';

/** What the user pasted -> { bearer } (a KGAT_ token) or { username, key } (kaggle.json). Accepts the kaggle.json text,
 *  "username:key", or a token starting with KGAT_. */
export function parseToken(token) {
  let t = String(token == null ? '' : token).trim();
  if (!t) throw fail('Paste your Kaggle key first.');
  if (t.startsWith('{')) {
    let j;
    try { j = JSON.parse(t); } catch { throw fail('This kaggle.json text is not complete. Paste the whole file.'); }
    const key = String(j.key || j.api_token || j.apiToken || j.token || '').trim();
    if (/^KGAT_/i.test(key)) return { bearer: key };
    if (!j.username || !key) throw fail('This kaggle.json has no username and key. Paste the whole file.');
    return { username: String(j.username).trim(), key };
  }
  t = t.replace(/^KAGGLE_API_TOKEN\s*=\s*/i, '').replace(/^["']|["']$/g, '').trim();
  if (/^KGAT_/i.test(t)) {
    if (/\s/.test(t)) throw fail('The Kaggle token must be one word that starts with KGAT_.');
    return { bearer: t };
  }
  const colon = t.indexOf(':');
  if (colon > 0 && colon < t.length - 1 && !/\s/.test(t)) {
    const key = t.slice(colon + 1);
    return /^KGAT_/i.test(key) ? { bearer: key } : { username: t.slice(0, colon), key };
  }
  throw fail('Paste the whole kaggle.json, or a token that starts with KGAT_.');
}

function authHeader(cred) {
  if (cred.bearer) return `Bearer ${cred.bearer}`;
  const raw = new TextEncoder().encode(`${cred.username}:${cred.key}`);
  return `Basic ${btoa(String.fromCharCode(...raw))}`;
}

async function call(token, service, method, body) {
  const cred = typeof token === 'object' && token ? token : parseToken(token);
  let res;
  try {
    res = await (customFetch || globalThis.fetch)(`${API}/${service}/${method}`, {
      method: 'POST', headers: { Authorization: authHeader(cred), 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}), credentials: 'omit', cache: 'no-store',
    });
  } catch {
    throw fail('This page could not reach Kaggle. Check your connection. If it keeps failing, the browser is blocking the request.');
  }
  let data = null;
  try { data = await res.json(); } catch { /* an empty or non-JSON answer */ }
  const err = data && (data.error && typeof data.error === 'object' ? data.error : data.code >= 400 ? data : null);
  const status = res.ok && err ? Number(err.code) || 400 : res.status;
  if (res.ok && !err) return data || {};
  const said = String((err && err.message) || '').slice(0, 160);
  throw fail(status === 401 ? BAD_KEY
    : status === 403 ? 'Kaggle refused this. Check that the key is yours and that your Kaggle account is phone-verified.'
    : status === 404 ? 'Kaggle could not find this.'
    : status === 429 ? 'Kaggle is getting too many requests. Wait a minute and try again.'
    : status >= 500 ? 'Kaggle has a problem right now. Try again in a minute.'
    : `Kaggle answered with an error: ${said || status}`, status);
}

const STATUS = {
  queued: 'queued', new_script: 'queued', running: 'running', complete: 'complete', error: 'error',
  cancel_requested: 'cancelled', cancel_acknowledged: 'cancelled',
};
const STATUS_BY_NUMBER = ['queued', 'running', 'complete', 'error', 'cancelled', 'cancelled', 'queued'];

/** Kaggle's KernelWorkerStatus (a name such as "RUNNING", in any spelling, or its number) -> our five words. */
export function normalStatus(value) {
  if (typeof value === 'number') return STATUS_BY_NUMBER[value] || 'queued';
  const s = String(value || '').replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase().replace(/^kernel_worker_status[._]/, '');
  return STATUS[s] || (s.includes('cancel') ? 'cancelled' : s.includes('error') ? 'error' : s.includes('run') ? 'running' : s.includes('complete') ? 'complete' : 'queued');
}

export const kaggle = {
  /** -> { username }. Throws a readable error when the key is wrong. */
  async check(token) {
    const cred = parseToken(token);
    if (cred.bearer) {
      let r;
      try { r = await call(cred, 'security.OAuthService', 'IntrospectToken', { token: cred.bearer }); }
      catch (e) { throw [400, 401, 403, 404].includes(e.status) ? fail(BAD_KEY, e.status) : e; }
      if (!r.active || !r.username) throw fail(BAD_KEY, 401);
      return { username: r.username };
    }
    await call(cred, KERNELS, 'ListKernels', { group: 'PROFILE', pageSize: 1 });
    return { username: cred.username };
  },

  /** True when any of the user's own notebooks is queued or running (the same check as v3/kaggle_io.py busy). */
  async busy(token, username) {
    const cred = parseToken(token);
    const r = await call(cred, KERNELS, 'ListKernels', {
      group: 'PROFILE', sortBy: 'DATE_RUN', pageSize: 40, user: '', language: 'all', kernelType: 'all', outputType: 'all',
      dataset: '', competition: '', parentKernel: '', search: '',
    });
    const mine = String(username || '').toLowerCase();
    const slugs = [];
    for (const k of r.kernels || []) {
      const parts = String(k.ref || '').split('/').filter(Boolean).slice(-2);
      if (parts.length === 2 && (!mine || parts[0].toLowerCase() === mine)) slugs.push(parts);
    }
    for (let i = 0; i < slugs.length; i += 6) {
      const states = await Promise.all(slugs.slice(i, i + 6).map(([owner, slug]) => this.status(cred, owner, slug).catch(() => null)));
      if (states.some((s) => s === 'queued' || s === 'running')) return true;
    }
    return false;
  },

  /** Save the notebook (a Python script) and run it. -> { url, ref, version } */
  async push(token, { username, slug, title, code, gpu = true, internet = true, isPrivate = true }) {
    if (!username || !slug || !code) throw fail('The notebook is not complete.');
    if (String(title || slug).length < 5) throw fail('The notebook title needs at least five letters.');
    const body = {
      slug: `${username}/${slug}`, newTitle: title || slug, text: code, language: 'python', kernelType: 'script',
      isPrivate: Boolean(isPrivate), enableGpu: Boolean(gpu), enableTpu: false, enableInternet: Boolean(internet),
      datasetDataSources: [], competitionDataSources: [], kernelDataSources: [], modelDataSources: [], categoryIds: [],
    };
    if (gpu) body.machineShape = 'NvidiaTeslaT4';
    const r = await call(token, KERNELS, 'SaveKernel', body);
    if (r.error) throw fail(`Kaggle did not take the notebook: ${String(r.error).slice(0, 200)}`);
    const url = !r.url ? `${WEB}/code/${username}/${slug}` : String(r.url).startsWith('/') ? WEB + r.url : String(r.url);
    return { url, ref: r.ref || `${username}/${slug}`, version: r.versionNumber == null ? null : r.versionNumber };
  },

  /** -> 'queued' | 'running' | 'complete' | 'error' | 'cancelled' */
  async status(token, username, slug) {
    const r = await call(token, KERNELS, 'GetKernelSessionStatus', { userName: username, kernelSlug: slug });
    return normalStatus(r.status);
  },

  async remove(token, username, slug) {
    await call(token, KERNELS, 'DeleteKernel', { userName: username, kernelSlug: slug });
  },
};
