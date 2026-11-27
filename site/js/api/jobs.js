// The site's jobs and library, kept in the signed-in user's own Hugging Face account:
//   <user>/namma-inbox   private dataset: one folder per video job, the format of the Android app (Upload.kt) and v3/inbox.py
//   <user>/namma-v3-out  private dataset: results, and library.json (the list of the user's tours and splat files)
//   <user>/namma-tour    public static Space: the published tours and uploaded splat files
// A job runs as a private notebook on the user's own Kaggle account (kaggle.js, boot.js). Protocol notes: site/api.md.
import { hf, Sha256, sha256Hex, setFetch as setHubFetch } from './hf.js';
import { kaggle, setFetch as setKaggleFetch } from './kaggle.js';
import { bootScript, PRESETS, DEFAULT_REF } from './boot.js';

export const INBOX = 'namma-inbox';
export const RESULTS = 'namma-v3-out';
export const TOUR = 'namma-tour';
export const KERNEL = 'namma-job';
export const GALLERY = 'namma-gallery';
const HUB = 'https://huggingface.co';
const JOB_ID = /^\d{8}-\d{6}-[0-9a-f]{4}$/;
const STOPPED = 'The Kaggle job stopped. Open the notebook log on Kaggle.';

/** Settings tests may change. partSize: bytes per video part (32 MB, as the Android app). commitEvery: parts per commit while
 *  uploading. needsHours: how long the Hugging Face sign-in must still last to start a job of that preset. */
export const config = { partSize: 32 * 1024 * 1024, commitEvery: 4, needsHours: { fast: 2, quality: 4 }, errorGraceMs: 120000 };

/** Use this fetch for Hugging Face and Kaggle (node tests pass a fake one). */
export function setFetch(fn) { setHubFetch(fn); setKaggleFetch(fn); }

const isoNow = () => new Date().toISOString().replace(/\.\d+Z$/, 'Z');
const pretty = (o) => JSON.stringify(o, null, 1);
const pad = (n, w) => String(n).padStart(w, '0');

/** A new id: YYYYMMDD-HHMMSS-xxxx (UTC time + 4 hex), the same as Jobs.kt newId. */
export function newId(now = new Date()) {
  const d = `${now.getUTCFullYear()}${pad(now.getUTCMonth() + 1, 2)}${pad(now.getUTCDate(), 2)}-${pad(now.getUTCHours(), 2)}${pad(now.getUTCMinutes(), 2)}${pad(now.getUTCSeconds(), 2)}`;
  const r = globalThis.crypto.getRandomValues(new Uint16Array(1))[0];
  return `${d}-${pad(r.toString(16), 4)}`;
}

function idTime(id) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-/.exec(id);
  return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z` : null;
}

function me() {
  const s = hf.session();
  if (!s) throw new Error('Sign in with Hugging Face first.');
  return s.user.name;
}

function checkJob(jobId) {
  if (!JOB_ID.test(String(jobId))) throw new Error('This job number is not valid.');
  return String(jobId);
}

// The folder the site is served from, to turn the library's site-relative links into full addresses.
function siteBase() {
  const loc = globalThis.window?.location;
  return loc && loc.origin ? loc.origin + loc.pathname.replace(/[^/]*$/, '') : '';
}
// The splat viewer lives in a Space of its own, <site Space>-viewer (scripts/site_deploy.py), never on the site's origin: a
// page here can read the signed-in visitor's tokens, and a viewer opens files and settings other people made.
function viewerBase() {
  const loc = globalThis.window?.location;
  const o = String((loc && loc.origin) || '');
  return /^https:\/\/[a-z0-9-]+\.static\.hf\.space$/.test(o) ? o.replace(/\.static\.hf\.space$/, '-viewer.static.hf.space') + '/' : '';
}
function fullUrl(url) {
  if (!url) return url || null;
  try {
    if (/^viewer\//.test(url) && viewerBase()) return new URL(url.slice(7), viewerBase()).href;
    return new URL(url, siteBase() || undefined).href;
  } catch { return url; }
}

async function pool(items, size, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(size, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
}

/** Update <job>/status.json as v3/inbox.py set_status does: the stage history is kept, an old failure reason goes with a new stage. */
async function setStatus(repo, job, stage, extra = {}) {
  const st = (await hf.readJson('dataset', repo, `${job}/status.json`)) || {};
  const now = isoNow();
  if (stage !== 'failed') delete st.reason;
  Object.assign(st, extra, { stage, updated: now });
  (st.history = Array.isArray(st.history) ? st.history : []).push([now, stage]);
  await hf.commit('dataset', repo, [{ path: `${job}/status.json`, content: pretty(st) }], `${job}: ${stage}`);
  return st;
}

// Every job folder of the inbox with its status.json and meta.json (null when missing), newest first.
async function scan(repo) {
  const tree = await hf.listTree('dataset', repo, '');
  const ids = tree.filter((e) => e.type === 'directory' && JOB_ID.test(e.path)).map((e) => e.path).sort().reverse();
  return pool(ids, 6, async (job) => {
    const [status, meta] = await Promise.all([hf.readJson('dataset', repo, `${job}/status.json`), hf.readJson('dataset', repo, `${job}/meta.json`)]);
    return { job, status, meta };
  });
}

async function readLibrary(user) {
  const j = await hf.readJson('dataset', `${user}/${RESULTS}`, 'library.json');
  const list = Array.isArray(j) ? j : Array.isArray(j?.items) ? j.items : [];
  return list.filter((e) => e && e.id != null);
}

async function writeLibrary(user, list, message) {
  await hf.commit('dataset', `${user}/${RESULTS}`, [{ path: 'library.json', content: pretty(list) }], message);
}

async function findEntry(user, id) {
  const list = await readLibrary(user);
  const entry = list.find((e) => String(e.id) === String(id));
  if (!entry) throw new Error('This item is not in your library any more.');
  return { list, entry };
}

// Where a library item's files are: { space, folder } in one of the user's own tour Spaces, or null. Splat entries say it
// (space, path). A tour entry (v3/site_tail.py) has only its link, https://<user>-namma-tour[-N].static.hf.space/<folder>/index.html.
function spaceFolder(user, entry) {
  let space = `${user}/${TOUR}`, folder = String(entry.path || '').replace(/^\/+|\/+$/g, '');
  if (!folder) {
    try {
      const u = new URL(entry.url);
      const sub = (/^([a-z0-9-]+)(\.static)?\.hf\.space$/i.exec(u.hostname) || [])[1] || '';
      const mine = `${user.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-`;
      const parts = u.pathname.split('/').filter(Boolean);
      if (!sub.toLowerCase().startsWith(mine) || parts.length < 2) return null;
      space = `${user}/${sub.slice(mine.length)}`;
      folder = decodeURIComponent(parts[0]);
    } catch { return null; }
  } else if (entry.space) space = String(entry.space);
  const safe = folder.split('/').every((s) => /^[A-Za-z0-9._-]+$/.test(s) && !/^\.+$/.test(s));
  const name = space.startsWith(`${user}/`) ? space.slice(user.length + 1) : '';
  return safe && /^namma-tour(-\d+)?$/.test(name) ? { space, folder } : null;
}

/** The Hugging Face account that owns this site (the gallery dataset lives there). Static Spaces are not told their own
 *  name, so: the Space variable SITE_OWNER, else SPACE_ID / SPACE_AUTHOR_NAME if present, else the name behind
 *  SPACE_CREATOR_USER_ID. */
export async function siteOwner() {
  const v = hf.variables();
  const direct = v.SITE_OWNER || String(v.SPACE_ID || v.HF_SPACE_ID || '').split('/')[0] || v.SPACE_AUTHOR_NAME || v.HF_SPACE_OWNER;
  if (direct) return String(direct);
  if (/^[0-9a-f]{8,40}$/i.test(v.SPACE_CREATOR_USER_ID || '')) {
    const name = await hf.userName(v.SPACE_CREATOR_USER_ID).catch(() => null);
    if (name) return String(name);
  }
  throw new Error('This site does not know who owns its public list. The owner must set the Space variable SITE_OWNER.');
}

const ensured = new Set();

export const jobs = {
  /** Create what is missing: the private inbox and results datasets and the public static tour Space. */
  async ensureRepos() {
    const user = me();
    const ids = { inbox: `${user}/${INBOX}`, results: `${user}/${RESULTS}`, tour: `${user}/${TOUR}` };
    if (ensured.has(user)) return ids;
    const wanted = [['dataset', INBOX, { private: true }], ['dataset', RESULTS, { private: true }], ['space', TOUR, { private: false, sdk: 'static' }]];
    for (const [type, name, opts] of wanted) {
      if (!(await hf.repoExists(type, `${user}/${name}`))) await hf.createRepo(type, name, opts);
    }
    ensured.add(user);
    return ids;
  },

  /** The user's tours and splat files, newest first ([] when there is no library.json yet). Links are full addresses. */
  async library() {
    const list = await readLibrary(me());
    return list.map((e) => ({
      ...e, id: String(e.id), run: e.run || null, title: e.title || 'Untitled', created: e.created || null,
      kind: e.kind === 'splat' ? 'splat' : 'tour', visibility: ['private', 'link', 'public'].includes(e.visibility) ? e.visibility : 'private',
      url: fullUrl(e.url), cover: fullUrl(e.cover), sizeMb: Number.isFinite(Number(e.sizeMb)) ? Number(e.sizeMb) : null,
    })).sort((a, b) => String(b.created || '').localeCompare(String(a.created || '')));
  },

  /** The inbox jobs, newest first. A video whose upload did not finish has stage 'uploading' (choose the same file again to go on). */
  async list() {
    const found = await scan(`${me()}/${INBOX}`);
    return found.filter((j) => j.status || j.meta).map(({ job, status, meta }) => ({
      job, title: meta?.title || meta?.name || job, preset: meta?.preset || null, created: meta?.created || idTime(job),
      stage: status?.stage || 'uploading', updated: status?.updated || null, link: status?.link || null, reason: status?.reason || null,
    }));
  },

  /** Upload a video into the inbox in parts and commit the job as 'uploaded'. The file is read one part at a time.
   *  Choosing the same file again after a break continues the same job: parts already in the repo are skipped.
   *  onProgress(fraction, text). -> job id */
  async create(file, { title, preset = 'fast', job: wantedJob } = {}, onProgress) {
    const user = me();
    if (!file || !(file.size > 0)) throw new Error('This video file is empty.');
    if (!Object.prototype.hasOwnProperty.call(PRESETS, preset)) throw new Error(`There is no setting called "${preset}".`);
    const say = (f, text) => { if (onProgress) onProgress(Math.max(0, Math.min(1, f)), text); };
    await this.ensureRepos();
    const repo = `${user}/${INBOX}`;
    const size = file.size, partSize = config.partSize;
    const name = String(file.name || 'video.mp4').split(/[\\/]/).pop();

    let job = wantedJob ? checkJob(wantedJob) : null;
    let meta = null;
    if (job) {
      if (await hf.readJson('dataset', repo, `${job}/status.json`)) return job;             // committed before the page closed
      meta = await hf.readJson('dataset', repo, `${job}/meta.json`);
    } else {
      const open = (await scan(repo)).find((j) => !j.status && j.meta && j.meta.file === name && j.meta.size === size && j.meta.part_size === partSize);
      if (open) { job = open.job; meta = open.meta; }
    }
    if (!job) job = newId();
    if (!meta) {
      meta = { name: title || name, title: title || name, preset, source: 'site', file: name, size, part_size: partSize, created: isoNow() };
      await hf.commit('dataset', repo, [{ path: `${job}/meta.json`, content: pretty(meta) }], `${job}: upload started`);
    } else {
      meta = { ...meta, name: title || meta.name, title: title || meta.title, preset };
    }

    const have = new Map((await hf.listTree('dataset', repo, `${job}/parts`)).map((e) => [e.path, e]));
    const whole = new Sha256();
    const parts = [];
    let waiting = [];
    const flush = async () => {
      if (waiting.length) await hf.commit('dataset', repo, waiting, `${job}: parts up to ${waiting[waiting.length - 1].path.split('/').pop()}`);
      waiting = [];
    };
    say(0, 'Uploading the video');
    for (let i = 0, offset = 0; offset < size; i++, offset += partSize) {
      const n = Math.min(partSize, size - offset);
      const slice = file.slice(offset, offset + n);
      const bytes = new Uint8Array(await slice.arrayBuffer());
      if (bytes.length !== n) throw new Error('The video file changed while it was being read. Choose it again.');
      whole.update(bytes);
      const oid = await sha256Hex(bytes);
      const path = `${job}/parts/${pad(i, 3)}`;
      const old = have.get(path);
      if (!(old && old.size === n && (!old.lfs || old.lfs.oid === oid))) {
        await hf.lfsUpload('dataset', repo, bytes, oid);
        waiting.push({ path, lfs: { oid, size: n } });
        if (waiting.length >= config.commitEvery) await flush();
      }
      parts.push(path);
      say((offset + n) / size * 0.99, 'Uploading the video');
    }
    await flush();
    const now = isoNow();
    const manifest = { name, size, sha256: whole.hex(), parts };                             // Upload.kt's video.json, read by inbox.py assemble
    const status = { stage: 'uploaded', updated: now, history: [[now, 'uploaded']] };
    await hf.commit('dataset', repo, [
      { path: `${job}/video.json`, content: pretty(manifest) },
      { path: `${job}/meta.json`, content: pretty(meta) },
      { path: `${job}/status.json`, content: pretty(status) },
    ], `${job}: uploaded`);
    say(1, 'Uploaded');
    return job;
  },

  /** Start the job's notebook on the user's Kaggle account (one at a time). -> { kernelUrl } */
  async start(jobId, kaggleToken) {
    const user = me();
    const job = checkJob(jobId);
    const repo = `${user}/${INBOX}`;
    const [status, meta] = await Promise.all([hf.readJson('dataset', repo, `${job}/status.json`), hf.readJson('dataset', repo, `${job}/meta.json`)]);
    if (!status) throw new Error('The video upload is not finished. Upload the video first.');
    const preset = Object.prototype.hasOwnProperty.call(PRESETS, meta?.preset) ? meta.preset : 'fast';
    const left = (Date.parse(hf.session().expires) - Date.now()) / 3600000;
    if (left < (config.needsHours[preset] || 2)) throw new Error('Your Hugging Face sign-in ends too soon for this job. Sign out, sign in again, then start it.');
    const { username } = await kaggle.check(kaggleToken);
    if (await kaggle.busy(kaggleToken, username)) {
      throw new Error('A notebook is already running on your Kaggle account. Kaggle gives each account a limited number of GPU sessions, so wait until it finishes, then start this job.');
    }
    const ref = hf.variables().NAMMA_REF || DEFAULT_REF;
    const code = bootScript({ ref, hfUser: user, job, preset, hfToken: hf.token() });
    const pushed = await kaggle.push(kaggleToken, { username, slug: KERNEL, title: KERNEL, code, gpu: true, internet: true, isPrivate: true });
    await setStatus(repo, job, 'training', { note: `${username}/${KERNEL}` });
    return { kernelUrl: pushed.url };
  },

  /** Where the job is. With a Kaggle key the notebook's own state is read too: a notebook that stopped with an error while
   *  the job still says 'training' is reported as failed. -> { stage, updated, link, reason, kaggle } */
  async status(jobId, kaggleToken) {
    const user = me();
    const job = checkJob(jobId);
    const st = await hf.readJson('dataset', `${user}/${INBOX}`, `${job}/status.json`);
    const out = { stage: st?.stage || 'uploading', updated: st?.updated || null, link: st?.link || null, reason: st?.reason || null, kaggle: null };
    const [owner, slug] = String(st?.note || '').split('/');
    if (!kaggleToken || !slug || !['training', 'publishing'].includes(out.stage)) return out;
    try {
      out.kaggle = await kaggle.status(kaggleToken, owner, slug);
    } catch (e) {
      out.kaggleError = e.message;
      return out;
    }
    // just after a push Kaggle can still show the previous version's end state, so a fresh 'training' is not judged yet
    const settled = !out.updated || Date.now() - Date.parse(out.updated) > config.errorGraceMs;
    if (out.stage === 'training' && settled && (out.kaggle === 'error' || out.kaggle === 'cancelled')) {
      out.stage = 'failed';
      out.reason = STOPPED;
    }
    return out;
  },

  /** Put a splat file into the user's public tour Space and list it in the library, open to people who have the link.
   *  -> the library entry (url is a full address of the site's viewer). */
  async uploadSplat(file, { title } = {}, onProgress) {
    const user = me();
    if (!file || !(file.size > 0)) throw new Error('This file is empty.');
    await this.ensureRepos();
    const id = newId();
    const name = String(file.name || 'scene.ply').split(/[\\/]/).pop().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^\.+/, '') || 'scene.ply';
    const path = `splats/${id}/${name}`;
    await hf.uploadLarge('space', `${user}/${TOUR}`, path, file, (f) => { if (onProgress) onProgress(f * 0.98, 'Uploading'); });
    const content = `${HUB}/spaces/${user}/${TOUR}/resolve/main/${path}`;
    const entry = {
      id, run: null, title: title || name, created: isoNow(), kind: 'splat', visibility: 'link',
      url: `viewer/index.html?content=${encodeURIComponent(content)}`, cover: null, sizeMb: Math.round(file.size / 1048576 * 10) / 10,
      space: `${user}/${TOUR}`, path: `splats/${id}`,
    };
    const list = await readLibrary(user);
    await writeLibrary(user, [entry, ...list], `library: add ${id}`);
    if (onProgress) onProgress(1, 'Uploaded');
    return { ...entry, url: fullUrl(entry.url) };
  },

  /** v: 'private' | 'link' | 'public' (who the site shows the item to; see site/api.md for what it does not hide). */
  async setVisibility(id, v) {
    if (!['private', 'link', 'public'].includes(v)) throw new Error('Choose private, link or public.');
    const user = me();
    const { list, entry } = await findEntry(user, id);
    if (entry.visibility === v) return;
    entry.visibility = v;
    await writeLibrary(user, list, `library: ${id} is ${v}`);
  },

  /** Take the item out of the library and delete its folder in the tour Space. */
  async remove(id) {
    const user = me();
    const { list, entry } = await findEntry(user, id);
    const place = spaceFolder(user, entry);
    if (place) {
      const there = await hf.listTree('space', place.space, place.folder);
      if (there.length) await hf.commit('space', place.space, [], `remove ${place.folder}`, { deletes: [`${place.folder}/`] });
    }
    await writeLibrary(user, list.filter((e) => e !== entry), `library: remove ${id}`);
  },

  /** Ask the site owner to show a public item in the gallery: a pull request on <owner>/namma-gallery. -> { prUrl } */
  async submitToGallery(id) {
    const user = me();
    const { entry } = await findEntry(user, id);
    if (entry.visibility !== 'public') throw new Error('Set this to Public first.');
    const owner = await siteOwner();
    const repo = `${owner}/${GALLERY}`;
    const item = {
      user, id: String(entry.id), title: entry.title || 'Untitled', kind: entry.kind === 'splat' ? 'splat' : 'tour',
      url: fullUrl(entry.url), cover: fullUrl(entry.cover), created: entry.created || null, submitted: isoNow(),
    };
    const pr = await hf.openPullRequest('dataset', repo, [{ path: `entries/${user}--${entry.id}.json`, content: pretty(item) }],
      `Gallery: ${item.title} by ${user}`.slice(0, 120), 'Sent from the NAMMA site. Merging this shows the item in the public list.');
    return { prUrl: pr.url || `${HUB}/datasets/${repo}/discussions` };
  },
};
