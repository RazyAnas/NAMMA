// Checks for the site's browser-side API modules (site/js/api), with a fake network: node --test tests/browser/site_api.test.mjs
// No request leaves the machine. The Hugging Face Hub, its file storage and the Kaggle API are small fakes below that keep
// repos in memory and check every upload's sha256, so a wrong call order or a wrong hash fails here.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { hf, Sha256, sha256Hex, hashBlob, tuning } from '../../site/js/api/hf.js';
import { kaggle, parseToken, normalStatus } from '../../site/js/api/kaggle.js';
import { jobs, config, setFetch, newId, siteOwner } from '../../site/js/api/jobs.js';
import { bootScript, PRESETS, PROVISIONAL_PRESETS, DEFAULT_REF } from '../../site/js/api/boot.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const HUB = 'https://huggingface.co';
const TOKEN = 'hf_oauth_FAKE-token.for_tests_only';            // not a real token
const KGAT = 'KGAT_' + '0123456789abcdef'.repeat(2);            // not a real token
const LEGACY = { username: 'kagada', key: 'f'.repeat(32) };     // not a real key
const USER = 'ada';
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const reply = (obj, status = 200, headers = {}) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...headers } });
const hasPython = spawnSync('python3', ['--version']).status === 0;

// ------------------------------------------------------------------------------------------------------- fake browser
function browser({ search = '', variables = {} } = {}) {
  const kept = new Map();
  Object.defineProperty(globalThis, 'sessionStorage', { configurable: true, writable: true, value: {
    getItem: (k) => (kept.has(k) ? kept.get(k) : null), setItem: (k, v) => kept.set(k, String(v)), removeItem: (k) => kept.delete(k), kept,
  } });
  globalThis.window = {
    huggingface: { variables: { OAUTH_CLIENT_ID: 'client-123', OAUTH_SCOPES: 'openid profile read-repos write-repos manage-repos write-discussions', ...variables } },
    location: { origin: 'https://owner-namma.static.hf.space', pathname: '/index.html', search, hash: '#/create', assigned: null, assign(u) { this.assigned = u; } },
    history: { replaced: null, replaceState(a, b, u) { this.replaced = u; } },
  };
  return globalThis.window;
}

function signIn(hours = 8) {
  globalThis.sessionStorage.setItem('namma.hf.session', JSON.stringify({
    token: TOKEN, expires: new Date(Date.now() + hours * 3600000).toISOString(), user: { name: USER, fullname: 'Ada L', avatarUrl: 'https://example.org/a.png' },
  }));
}

// ------------------------------------------------------------------------------------------------------- fake network
class Net {
  constructor() {
    this.calls = []; this.repos = new Map(); this.objects = new Map(); this.commits = []; this.pieces = new Map();
    this.multipart = 0;                                            // > 0: the Hub answers the batch call with a multipart upload of this chunk size
    this.kaggle = { calls: [], list: [], status: {}, pushes: [], deleted: [] };
    this.fetch = this.fetch.bind(this);
  }

  files(type, id, make = false) {
    const key = `${type}:${id}`;
    if (make && !this.repos.has(key)) this.repos.set(key, new Map());
    return this.repos.get(key);
  }
  text(type, id, p) { const f = this.files(type, id)?.get(p); return f && f.bytes ? new TextDecoder().decode(f.bytes) : null; }
  json(type, id, p) { const t = this.text(type, id, p); return t == null ? null : JSON.parse(t); }
  hubCalls(part) { return this.calls.filter((c) => c.url.includes(part)); }

  async fetch(url, init = {}) {
    url = String(url);
    const method = (init.method || 'GET').toUpperCase();
    const headers = init.headers || {};
    const call = { method, url, headers, body: init.body };
    this.calls.push(call);
    assert.ok(!url.includes(TOKEN) && !url.includes(KGAT) && !url.includes(LEGACY.key), 'a token must never be in a URL');
    const u = new URL(url);
    if (u.host === 'api.kaggle.com') return this.kaggleApi(u, headers, JSON.parse(init.body));
    if (u.host === 's3.test') {
      assert.equal(headers.Authorization, undefined, 'the Hugging Face token must not go to the storage host');
      return this.storage(u, method, init.body);
    }
    assert.equal(u.origin, HUB, `unexpected host ${u.host}`);
    if (u.pathname === '/oauth/token') return this.oauthToken(call);
    assert.equal(headers.Authorization, `Bearer ${TOKEN}`, `no token on ${u.pathname}`);
    return this.hub(u, method, headers, init.body);
  }

  oauthToken(call) {
    const form = new URLSearchParams(call.body);
    call.form = Object.fromEntries(form);
    return reply({ access_token: TOKEN, expires_in: 28800, scope: 'openid profile', token_type: 'bearer', id_token: 'x' });
  }

  async hub(u, method, headers, body) {
    const p = decodeURIComponent(u.pathname);
    let m;
    if (p === '/oauth/userinfo') return reply({ sub: '1', preferred_username: USER, name: 'Ada L', picture: 'https://example.org/a.png' });
    if (p === '/api/whoami-v2') return reply({ name: USER, fullname: 'Ada L', avatarUrl: 'https://example.org/a.png' });
    if ((m = /^\/api\/users\/([^/]+)\/overview$/.exec(p))) return reply({ user: 'siteowner', fullname: 'Site Owner' });
    if (p === '/api/repos/create') {
      const j = JSON.parse(body); const id = `${USER}/${j.name}`;
      if (this.files(j.type, id)) return reply({ error: 'You already created this repo' }, 409);
      this.files(j.type, id, true); this.created = [...(this.created || []), j];
      return reply({ url: `${HUB}/${id}` });
    }
    if ((m = /^\/api\/(dataset|space)s\/([^/]+\/[^/]+)$/.exec(p))) return this.files(m[1], m[2]) ? reply({ id: m[2] }) : reply({ error: 'Repository not found' }, 404);
    if ((m = /^\/api\/(dataset|space)s\/([^/]+\/[^/]+)\/tree\/main\/?(.*)$/.exec(p))) return this.tree(m[1], m[2], m[3]);
    if ((m = /^\/api\/(dataset|space)s\/([^/]+\/[^/]+)\/preupload\/main$/.exec(p))) {
      return reply({ files: JSON.parse(body).files.map((f) => ({ path: f.path, uploadMode: Buffer.from(f.sample, 'base64').includes(0) ? 'lfs' : 'regular', shouldIgnore: false })) });
    }
    if ((m = /^\/api\/(dataset|space)s\/([^/]+\/[^/]+)\/commit\/main$/.exec(p))) return this.commit(m[1], m[2], body, headers, u.searchParams.get('create_pr') === '1');
    if ((m = /^\/(dataset|space)s\/([^/]+\/[^/]+)\.git\/info\/lfs\/objects\/batch$/.exec(p))) return this.batch(JSON.parse(body), headers);
    if ((m = /^\/(dataset|space)s\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/.exec(p))) {
      const f = this.files(m[1], m[2])?.get(m[3]);
      return f && f.bytes ? new Response(f.bytes) : reply({ error: 'Entry not found' }, 404);
    }
    if (p === '/api/complete_multipart') {
      const j = JSON.parse(body); const got = this.pieces.get(j.oid) || [];
      assert.deepEqual(j.parts, got.map((_, i) => ({ partNumber: i + 1, etag: `"etag-${i + 1}"` })));
      const all = Buffer.concat(got); assert.equal(sha(all), j.oid, 'multipart pieces do not add up to the object');
      this.objects.set(j.oid, all);
      return reply({});
    }
    if (p === '/api/lfs_verify') { const j = JSON.parse(body); assert.equal(this.objects.get(j.oid)?.length, j.size); assert.equal(headers['x-verify'], 'v'); return reply({}); }
    return reply({ error: `fake hub: no route ${method} ${p}` }, 500);
  }

  tree(type, id, dir) {
    const files = this.files(type, id);
    if (!files) return reply({ error: 'Repository not found' }, 404);
    const out = new Map(); const prefix = dir ? `${dir}/` : '';
    for (const [p, f] of files) {
      if (!p.startsWith(prefix)) continue;
      const rest = p.slice(prefix.length); const first = rest.split('/')[0];
      if (rest.includes('/')) out.set(prefix + first, { type: 'directory', path: prefix + first, oid: 'tree', size: 0 });
      else out.set(p, f.lfs ? { type: 'file', path: p, oid: 'ptr', size: f.lfs.size, lfs: { oid: f.lfs.oid, size: f.lfs.size, pointerSize: 130 } } : { type: 'file', path: p, oid: 'blob', size: f.bytes.length });
    }
    return dir && !out.size ? reply({ error: 'not found' }, 404) : reply([...out.values()]);
  }

  batch(j, headers) {
    assert.equal(headers.Accept, 'application/vnd.git-lfs+json'); assert.equal(headers['Content-Type'], 'application/vnd.git-lfs+json');
    assert.deepEqual({ operation: j.operation, transfers: j.transfers, hash_algo: j.hash_algo, ref: j.ref }, { operation: 'upload', transfers: ['basic', 'multipart'], hash_algo: 'sha256', ref: { name: 'main' } });
    return reply({ transfer: this.multipart ? 'multipart' : 'basic', objects: j.objects.map(({ oid, size }) => {
      if (this.objects.has(oid)) return { oid, size };
      if (this.multipart) {
        const header = { chunk_size: String(this.multipart) };
        for (let n = 1; n <= Math.ceil(size / this.multipart); n++) header[String(n)] = `https://s3.test/part/${oid}/${n}`;
        return { oid, size, actions: { upload: { href: `${HUB}/api/complete_multipart?oid=${oid}`, header } } };
      }
      return { oid, size, actions: { upload: { href: `https://s3.test/put/${oid}`, header: { 'x-test': '1' } }, verify: { href: `${HUB}/api/lfs_verify`, header: { 'x-verify': 'v' } } } };
    }) });
  }

  async storage(u, method, body) {
    assert.equal(method, 'PUT');
    const bytes = Buffer.from(body instanceof Blob ? await body.arrayBuffer() : body);
    const [, how, oid, n] = u.pathname.split('/');
    if (how === 'put') { assert.equal(sha(bytes), oid, 'uploaded bytes do not match their sha256'); this.objects.set(oid, bytes); return new Response('', { status: 200 }); }
    const got = this.pieces.get(oid) || []; got[Number(n) - 1] = bytes; this.pieces.set(oid, got);
    return new Response('', { status: 200, headers: { ETag: `"etag-${n}"` } });
  }

  commit(type, id, body, headers, pr) {
    assert.equal(headers['Content-Type'], 'application/x-ndjson');
    const files = pr ? new Map() : this.files(type, id);
    if (!files) return reply({ error: 'Repository not found' }, 404);
    const lines = body.trim().split('\n').map((l) => JSON.parse(l));
    assert.equal(lines[0].key, 'header');
    const ops = [];
    for (const { key, value } of lines.slice(1)) {
      ops.push({ key, path: value.path });
      if (key === 'file') { assert.equal(value.encoding, 'base64'); files.set(value.path, { bytes: Buffer.from(value.content, 'base64') }); }
      else if (key === 'lfsFile') {
        assert.equal(value.algo, 'sha256');
        assert.equal(this.objects.get(value.oid)?.length, value.size, `lfsFile ${value.path}: the object was not uploaded first`);
        files.set(value.path, { lfs: { oid: value.oid, size: value.size } });
      } else if (key === 'deletedFolder') { for (const k of [...files.keys()]) if (k.startsWith(`${value.path}/`)) files.delete(k); }
      else if (key === 'deletedFile') files.delete(value.path);
      else assert.fail(`unknown commit line ${key}`);
    }
    this.commits.push({ type, id, pr, summary: lines[0].value.summary, description: lines[0].value.description, ops, files });
    return reply({ success: true, commitOid: 'c'.repeat(40), ...(pr ? { pullRequestUrl: `${HUB}/datasets/${id}/discussions/7` } : {}) });
  }

  kaggleApi(u, headers, body) {
    const [, , service, method] = u.pathname.split('/');
    this.kaggle.calls.push({ service, method, auth: headers.Authorization, body });
    const basic = `Basic ${Buffer.from(`${LEGACY.username}:${LEGACY.key}`).toString('base64')}`;
    if (method === 'IntrospectToken') return body.token === KGAT ? reply({ active: true, username: LEGACY.username, scope: '' }) : reply({ error: { code: 404, message: 'Not found', status: 'NOT_FOUND' } }, 404);
    if (headers.Authorization !== `Bearer ${KGAT}` && headers.Authorization !== basic) return reply({ error: { code: 401, message: 'Unauthenticated', status: 'UNAUTHENTICATED' } }, 401);
    if (method === 'ListKernels') return reply({ kernels: this.kaggle.list.map((ref) => ({ ref, title: ref })), nextPageToken: '' });
    if (method === 'GetKernelSessionStatus') {
      const s = this.kaggle.status[`${body.userName}/${body.kernelSlug}`];
      return s === undefined ? reply({ error: { code: 404, message: 'Not found' } }, 404) : reply(s === 'QUEUED' ? {} : { status: s });
    }
    if (method === 'SaveKernel') { this.kaggle.pushes.push(body); return reply({ ref: `/${body.slug}`, url: `https://www.kaggle.com/code/${body.slug}`, versionNumber: 3 }); }
    if (method === 'DeleteKernel') { this.kaggle.deleted.push(body); return reply({}); }
    return reply({ error: { code: 404, message: 'Not found' } }, 404);
  }
}

function setup({ hours = 8, variables } = {}) {
  browser({ variables });
  signIn(hours);
  const net = new Net();
  setFetch(net.fetch);
  for (const [type, name] of [['dataset', 'namma-inbox'], ['dataset', 'namma-v3-out'], ['space', 'namma-tour']]) net.files(type, `${USER}/${name}`, true);
  tuning.retryMs = 0;
  config.partSize = 1000; config.commitEvery = 4; config.errorGraceMs = 120000;
  return net;
}

const video = (size, name = 'walk.mp4') => { const bytes = randomBytes(size); return { bytes, file: new File([bytes], name, { type: 'video/mp4' }) }; };

// What v3/inbox.py assemble does with the manifest: join the parts in order, then check size and sha256.
function assemble(net, repo, manifestPath) {
  const m = net.json('dataset', repo, manifestPath);
  const joined = Buffer.concat(m.parts.map((p) => net.objects.get(net.files('dataset', repo).get(p).lfs.oid)));
  assert.ok(joined.length === m.size && sha(joined) === m.sha256, 'video parts do not add up');
  return { m, joined };
}

// ---------------------------------------------------------------------------------------------------------- sign-in
test('login builds the PKCE authorize URL and keeps the verifier in sessionStorage', async () => {
  const w = browser({ search: '?x=1' });
  assert.equal(hf.available(), true);
  assert.equal(hf.session(), null);
  await hf.login();
  const url = new URL(w.location.assigned);
  assert.equal(url.origin + url.pathname, 'https://huggingface.co/oauth/authorize');
  const q = Object.fromEntries(url.searchParams);
  const kept = JSON.parse(sessionStorage.getItem('namma.hf.pkce'));
  assert.equal(q.client_id, 'client-123');
  assert.equal(q.response_type, 'code');
  assert.equal(q.scope, 'openid profile read-repos write-repos manage-repos write-discussions');
  assert.equal(q.redirect_uri, 'https://owner-namma.static.hf.space/index.html');     // no query, no hash
  assert.equal(q.code_challenge_method, 'S256');
  assert.equal(q.state, kept.state);
  assert.match(kept.verifier, /^[A-Za-z0-9_-]{43,128}$/);
  assert.equal(q.code_challenge, createHash('sha256').update(kept.verifier).digest('base64url'));
  assert.ok(!w.location.assigned.includes(kept.verifier));
  assert.deepEqual(Object.keys(q).sort(), ['client_id', 'code_challenge', 'code_challenge_method', 'redirect_uri', 'response_type', 'scope', 'state']);
});

test('available is false without the Space OAuth variables', () => {
  browser();
  delete window.huggingface.variables.OAUTH_CLIENT_ID;
  assert.equal(hf.available(), false);
});

test('init finishes the redirect: token exchange, user, clean address, session without the token', async () => {
  const w = browser();
  const net = new Net(); setFetch(net.fetch);
  await hf.login();
  const { state, verifier } = JSON.parse(sessionStorage.getItem('namma.hf.pkce'));
  w.location.search = `?code=the-code&state=${encodeURIComponent(state)}&keep=1`;
  const s = await hf.init();
  const tokenCall = net.calls.find((c) => c.url === `${HUB}/oauth/token`);
  assert.equal(tokenCall.method, 'POST');
  assert.equal(tokenCall.headers['Content-Type'], 'application/x-www-form-urlencoded');
  assert.deepEqual(tokenCall.form, { grant_type: 'authorization_code', code: 'the-code', redirect_uri: 'https://owner-namma.static.hf.space/index.html', code_verifier: verifier, client_id: 'client-123' });
  assert.deepEqual(s.user, { name: USER, fullname: 'Ada L', avatarUrl: 'https://example.org/a.png' });
  assert.ok(Math.abs(Date.parse(s.expires) - Date.now() - 28800000) < 5000);
  assert.ok(!JSON.stringify(s).includes(TOKEN), 'session() must not hand out the token');
  assert.equal(hf.token(), TOKEN);
  assert.equal(w.history.replaced, '/index.html?keep=1#/create');
  assert.equal(sessionStorage.getItem('namma.hf.pkce'), null);
  assert.deepEqual([...sessionStorage.kept.keys()], ['namma.hf.session']);
  hf.logout();
  assert.equal(hf.session(), null); assert.equal(hf.token(), null);
});

test('init refuses an answer with the wrong state, and a refused sign-in', async () => {
  let w = browser();
  const net = new Net(); setFetch(net.fetch);
  await hf.login();
  w.location.search = '?code=the-code&state=someone-elses';
  await assert.rejects(() => hf.init(), /did not match/);
  assert.equal(net.calls.length, 0);
  assert.equal(hf.session(), null);
  assert.equal(w.history.replaced, '/index.html#/create');
  w = browser({ search: '?error=access_denied&error_description=no' });
  await assert.rejects(() => hf.init(), /did not allow/);
  w = browser({ search: '?code=old-code&state=x' });               // a reload after the code was used: nothing stored, no request
  assert.equal(await hf.init(), null);
  assert.equal(net.calls.length, 0);
});

test('an expired session is gone', () => {
  browser(); signIn(-1);
  assert.equal(hf.session(), null);
  assert.equal(hf.token(), null);
});

// ---------------------------------------------------------------------------------------------------------- hashing
test('the piecewise SHA-256 equals node:crypto for every split', async () => {
  for (const size of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 70001]) {
    const bytes = randomBytes(size);
    const s = new Sha256();
    for (let o = 0; o < size;) { const n = 1 + Math.floor(Math.random() * 300); s.update(bytes.subarray(o, o + n)); o += n; }
    assert.equal(s.hex(), sha(bytes), `size ${size}`);
    assert.equal(await sha256Hex(bytes), sha(bytes));
  }
  tuning.hashChunk = 4096;
  const big = randomBytes(50000);
  const seen = [];
  assert.equal(await hashBlob(new Blob([big]), (f) => seen.push(f)), sha(big));
  assert.equal(seen.at(-1), 1);
  tuning.hashChunk = 8 << 20;
});

// ------------------------------------------------------------------------------------------------------- video upload
test('create uploads the video in parts and writes the manifest inbox.py assemble expects', async () => {
  const net = setup();
  const { bytes, file } = video(2500);
  const seen = [];
  const job = await jobs.create(file, { title: 'My flat', preset: 'fast' }, (f, text) => seen.push([f, text]));
  const repo = `${USER}/namma-inbox`;
  assert.match(job, /^\d{8}-\d{6}-[0-9a-f]{4}$/);

  const raw = net.text('dataset', repo, `${job}/video.json`);
  const m = JSON.parse(raw);
  assert.deepEqual(Object.keys(m), ['name', 'size', 'sha256', 'parts']);          // Upload.kt's fields, in its order
  assert.equal(raw, JSON.stringify(m, null, 1));
  assert.deepEqual(m, { name: 'walk.mp4', size: 2500, sha256: sha(bytes), parts: [`${job}/parts/000`, `${job}/parts/001`, `${job}/parts/002`] });
  const { joined } = assemble(net, repo, `${job}/video.json`);
  assert.ok(joined.equals(bytes));
  m.parts.forEach((p, i) => {                                                     // each part is its own LFS object, named by its own sha256
    const part = bytes.subarray(i * 1000, (i + 1) * 1000);
    assert.deepEqual(net.files('dataset', repo).get(p).lfs, { oid: sha(part), size: part.length });
  });

  const st = net.json('dataset', repo, `${job}/status.json`);
  assert.deepEqual(Object.keys(st), ['stage', 'updated', 'history']);
  assert.equal(st.stage, 'uploaded');
  assert.match(st.updated, /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$/);
  assert.deepEqual(st.history, [[st.updated, 'uploaded']]);
  const meta = net.json('dataset', repo, `${job}/meta.json`);
  assert.deepEqual({ name: meta.name, preset: meta.preset, file: meta.file, size: meta.size, part_size: meta.part_size }, { name: 'My flat', preset: 'fast', file: 'walk.mp4', size: 2500, part_size: 1000 });

  const last = net.commits.at(-1);                                                // the job only exists for the pipeline with this last commit
  assert.equal(last.summary, `${job}: uploaded`);
  assert.deepEqual(last.ops.map((o) => o.path), [`${job}/video.json`, `${job}/meta.json`, `${job}/status.json`]);
  assert.deepEqual(seen.at(-1), [1, 'Uploaded']);
  assert.ok(seen.every(([f], i) => f >= 0 && f <= 1 && (i === 0 || f >= seen[i - 1][0])), 'progress is a rising fraction');

  const listed = await jobs.list();
  assert.deepEqual(listed.map((j) => [j.job, j.title, j.preset, j.stage]), [[job, 'My flat', 'fast', 'uploaded']]);
});

test('the real v3/inbox.py assemble accepts the upload', { skip: !hasPython && 'python3 is not installed' }, async () => {
  const net = setup();
  const { bytes, file } = video(3210);
  const job = await jobs.create(file, { title: 't', preset: 'fast' });
  const repo = `${USER}/namma-inbox`;
  const store = {};
  for (const [p, f] of net.files('dataset', repo)) store[p] = (f.bytes || net.objects.get(f.lfs.oid)).toString('base64');
  const py = `
import base64, hashlib, json, os, sys, tempfile, types
store = json.load(sys.stdin)
def hf_hub_download(repo, path, repo_type=None, token=None, local_dir='.', **kw):
    assert repo == '${repo}' and repo_type == 'dataset'
    dst = os.path.join(local_dir, path); os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, 'wb').write(base64.b64decode(store[path])); return dst
sys.modules['huggingface_hub'] = types.SimpleNamespace(hf_hub_download=hf_hub_download)
sys.path.insert(0, ${JSON.stringify(path.join(ROOT, 'colab/real_scenes/v3'))})
import inbox
with tempfile.TemporaryDirectory() as d:
    os.chdir(d)
    m = inbox.assemble('${repo}', '${job}/video.json', 'video.bin', 'no-token')
    print(json.dumps(dict(name=m['name'], size=os.path.getsize('video.bin'), sha256=hashlib.sha256(open('video.bin', 'rb').read()).hexdigest())))
`;
  const r = spawnSync('python3', ['-B', '-c', py], { input: JSON.stringify(store), encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(JSON.parse(r.stdout.trim().split('\n').pop()), { name: 'walk.mp4', size: 3210, sha256: sha(bytes) });
});

test('create resumes the same job and skips the parts already in the repo', async () => {
  const net = setup();
  config.commitEvery = 1;
  const { bytes, file } = video(3500);
  const realFetch = net.fetch;
  let puts = 0;
  setFetch(async (url, init) => {                                                 // the connection drops while the third part is sent
    if (String(url).startsWith('https://s3.test/') && ++puts > 2) throw new TypeError('Failed to fetch');
    return realFetch(url, init);
  });
  await assert.rejects(() => jobs.create(file, { title: 'Resume me', preset: 'fast' }), /Could not reach Hugging Face/);
  const repo = `${USER}/namma-inbox`;
  const [open] = await jobs.list();
  assert.equal(open.stage, 'uploading');
  assert.equal(open.title, 'Resume me');
  assert.deepEqual([...net.files('dataset', repo).keys()].filter((p) => p.includes('/parts/')).map((p) => p.slice(-3)), ['000', '001']);

  setFetch(net.fetch);
  const before = net.calls.length;
  const job = await jobs.create(file, { title: 'Resume me', preset: 'fast' });
  assert.equal(job, open.job);
  const after = net.calls.slice(before);
  const sent = after.filter((c) => c.url.startsWith('https://s3.test/put/')).map((c) => c.url.split('/').pop());
  assert.deepEqual(sent, [sha(bytes.subarray(2000, 3000)), sha(bytes.subarray(3000, 3500))], 'only the missing parts are sent');
  assert.equal(after.filter((c) => c.url.endsWith('/info/lfs/objects/batch')).length, 2, 'no upload call at all for parts 000 and 001');
  const { m, joined } = assemble(net, repo, `${job}/video.json`);
  assert.equal(m.parts.length, 4);
  assert.ok(joined.equals(bytes));
  assert.equal((await jobs.list())[0].stage, 'uploaded');

  const other = await jobs.create(video(3500).file, { title: 'Another video', preset: 'fast' });   // same name and size, new content: a finished job is never reused
  assert.notEqual(other, job);
});

test('a part that changed on disk is sent again', async () => {
  const net = setup();
  config.commitEvery = 1;
  const a = video(2000);
  const job = await jobs.create(a.file, { title: 'v', preset: 'fast' });
  const repo = `${USER}/namma-inbox`;
  for (const f of ['video.json', 'status.json']) net.files('dataset', repo).delete(`${job}/${f}`);   // as if the last commit never happened
  const changed = Buffer.from(a.bytes); changed[1500] ^= 0xff;
  const before = net.calls.length;
  assert.equal(await jobs.create(new File([changed], 'walk.mp4'), { title: 'v', preset: 'fast' }), job);
  assert.equal(net.calls.slice(before).filter((c) => c.url.startsWith('https://s3.test/put/')).length, 1);
  assert.ok(assemble(net, repo, `${job}/video.json`).joined.equals(changed));
});

// --------------------------------------------------------------------------------------------------- large-file calls
test('uploadLarge: batch, one PUT, verify, then a commit with an lfsFile entry', async () => {
  const net = setup();
  const bytes = randomBytes(5000);
  const seen = [];
  const r = await hf.uploadLarge('space', `${USER}/namma-tour`, 'splats/x/room.ply', new Blob([bytes], { type: 'application/octet-stream' }), (f) => seen.push(f));
  assert.deepEqual(r, { path: 'splats/x/room.ply', oid: sha(bytes), size: 5000 });
  assert.deepEqual(net.calls.map((c) => `${c.method} ${c.url.replace(HUB, '')}`), [
    `POST /spaces/${USER}/namma-tour.git/info/lfs/objects/batch`,
    `PUT https://s3.test/put/${sha(bytes)}`,
    'POST /api/lfs_verify',
    `POST /api/spaces/${USER}/namma-tour/commit/main`,
  ]);
  assert.deepEqual(JSON.parse(net.calls[0].body).objects, [{ oid: sha(bytes), size: 5000 }]);
  assert.equal(net.calls[1].headers['x-test'], '1');
  const lines = net.calls[3].body.trim().split('\n').map((l) => JSON.parse(l));
  assert.deepEqual(lines[1], { key: 'lfsFile', value: { path: 'splats/x/room.ply', algo: 'sha256', oid: sha(bytes), size: 5000 } });
  assert.equal(seen.at(-1), 1);

  const before = net.calls.length;                                                // the Hub already has the object: no PUT
  await hf.uploadLarge('space', `${USER}/namma-tour`, 'splats/y/room.ply', new Blob([bytes]));
  assert.deepEqual(net.calls.slice(before).map((c) => c.method + ' ' + c.url.split('/').pop()), ['POST batch', 'POST main']);
});

test('uploadLarge: a multipart answer is sent chunk by chunk and completed with the ETags', async () => {
  const net = setup();
  net.multipart = 2048;
  const bytes = randomBytes(5000);
  const seen = [];
  await hf.uploadLarge('dataset', `${USER}/namma-v3-out`, 'big.bin', new Blob([bytes]), (f) => seen.push(f));
  const o = sha(bytes);
  assert.deepEqual(net.calls.map((c) => `${c.method} ${c.url.replace(HUB, '')}`), [
    `POST /datasets/${USER}/namma-v3-out.git/info/lfs/objects/batch`,
    `PUT https://s3.test/part/${o}/1`, `PUT https://s3.test/part/${o}/2`, `PUT https://s3.test/part/${o}/3`,
    `POST /api/complete_multipart?oid=${o}`,
    `POST /api/datasets/${USER}/namma-v3-out/commit/main`,
  ]);
  assert.deepEqual(net.pieces.get(o).map((p) => p.length), [2048, 2048, 904]);
  assert.ok(net.objects.get(o).equals(bytes));
  assert.deepEqual(net.files('dataset', `${USER}/namma-v3-out`).get('big.bin').lfs, { oid: o, size: 5000 });
  assert.ok(seen.length >= 4 && seen.every((f, i) => i === 0 || f >= seen[i - 1]));
});

test('commit: text goes in as a file, binary content is asked about first (preupload), deletes are folder or file', async () => {
  const net = setup();
  const repo = `${USER}/namma-v3-out`;
  const bin = new Uint8Array([1, 0, 2, 0, 3]);
  await hf.commit('dataset', repo, [{ path: 'a.json', content: '{"é": 1}' }, { path: 'cover.bin', content: bin }, { path: 'note.txt', content: new TextEncoder().encode('plain') }], 'three files');
  assert.deepEqual(net.calls.map((c) => c.url.split('/').slice(-2).join('/')), ['preupload/main', 'objects/batch', `put/${sha(bin)}`, 'api/lfs_verify', 'commit/main']);
  assert.deepEqual(JSON.parse(net.calls[0].body).files.map((f) => [f.path, f.size]), [['cover.bin', 5], ['note.txt', 5]]);
  assert.equal(net.text('dataset', repo, 'a.json'), '{"é": 1}');
  assert.equal(net.text('dataset', repo, 'note.txt'), 'plain');
  assert.deepEqual(net.files('dataset', repo).get('cover.bin').lfs, { oid: sha(bin), size: 5 });
  await hf.commit('dataset', repo, [], 'tidy', { deletes: ['a.json', 'old/'] });
  assert.deepEqual(net.commits.at(-1).ops, [{ key: 'deletedFile', path: 'a.json' }, { key: 'deletedFolder', path: 'old' }]);
  assert.deepEqual((await hf.listTree('dataset', repo, '')).map((e) => e.path).sort(), ['cover.bin', 'note.txt']);
  assert.deepEqual(await hf.listTree('dataset', repo, 'nothing/here'), []);
  assert.equal(await hf.readJson('dataset', repo, 'missing.json'), null);
});

test('Hub errors become short messages', async () => {
  setup();
  setFetch(async () => reply({ error: 'Invalid credentials' }, 401));
  await assert.rejects(() => hf.whoami(), /sign-in has ended/);
  setFetch(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(() => hf.readText('dataset', 'a/b', 'x'), /Could not reach Hugging Face/);
  hf.logout();
  await assert.rejects(() => hf.whoami(), /Sign in with Hugging Face first/);
});

// ------------------------------------------------------------------------------------------------------------ kaggle
test('a Kaggle key is accepted as kaggle.json text, username:key or a KGAT_ token', () => {
  assert.deepEqual(parseToken(JSON.stringify(LEGACY)), LEGACY);
  assert.deepEqual(parseToken(` ${LEGACY.username}:${LEGACY.key}\n`), LEGACY);
  assert.deepEqual(parseToken(KGAT), { bearer: KGAT });
  assert.deepEqual(parseToken(`"${KGAT}"`), { bearer: KGAT });
  assert.deepEqual(parseToken(JSON.stringify({ username: 'x', key: KGAT })), { bearer: KGAT });
  for (const bad of ['', LEGACY.key, '{"username": "x"}', '{not json', 'two words']) assert.throws(() => parseToken(bad), Error);
});

test('push sends the official client\'s SaveKernel body, with Basic or Bearer auth', async () => {
  const net = setup();
  const basic = `Basic ${Buffer.from(`${LEGACY.username}:${LEGACY.key}`).toString('base64')}`;
  for (const [token, auth] of [[JSON.stringify(LEGACY), basic], [`${LEGACY.username}:${LEGACY.key}`, basic], [KGAT, `Bearer ${KGAT}`]]) {
    net.calls.length = 0; net.kaggle.calls.length = 0; net.kaggle.pushes.length = 0;
    const r = await kaggle.push(token, { username: LEGACY.username, slug: 'namma-job', title: 'namma-job', code: 'print(1)\n' });
    assert.deepEqual(r, { url: 'https://www.kaggle.com/code/kagada/namma-job', ref: '/kagada/namma-job', version: 3 });
    const [c] = net.calls;
    assert.equal(c.method, 'POST');
    assert.equal(c.url, 'https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel');
    assert.deepEqual(c.headers, { Authorization: auth, 'Content-Type': 'application/json' });
    assert.deepEqual(JSON.parse(c.body), {
      slug: 'kagada/namma-job', newTitle: 'namma-job', text: 'print(1)\n', language: 'python', kernelType: 'script',
      isPrivate: true, enableGpu: true, enableTpu: false, enableInternet: true,
      datasetDataSources: [], competitionDataSources: [], kernelDataSources: [], modelDataSources: [], categoryIds: [],
      machineShape: 'NvidiaTeslaT4',
    });
  }
});

test('check names the account, and says so plainly when the key is wrong', async () => {
  const net = setup();
  assert.deepEqual(await kaggle.check(KGAT), { username: 'kagada' });
  assert.deepEqual(net.kaggle.calls.at(-1), { service: 'security.OAuthService', method: 'IntrospectToken', auth: `Bearer ${KGAT}`, body: { token: KGAT } });
  assert.deepEqual(await kaggle.check(JSON.stringify(LEGACY)), { username: 'kagada' });
  assert.deepEqual(net.kaggle.calls.at(-1).body, { group: 'PROFILE', pageSize: 1 });
  await assert.rejects(() => kaggle.check('KGAT_' + 'a'.repeat(32)), /Kaggle did not accept this key/);
  await assert.rejects(() => kaggle.check('kagada:' + 'a'.repeat(32)), /Kaggle did not accept this key/);
  setFetch(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(() => kaggle.check(KGAT), /could not reach Kaggle/);
});

test('status words, busy and remove', async () => {
  const net = setup();
  for (const [raw, word] of [['QUEUED', 'queued'], ['RUNNING', 'running'], ['COMPLETE', 'complete'], ['ERROR', 'error'], ['CANCEL_REQUESTED', 'cancelled'],
    ['CANCEL_ACKNOWLEDGED', 'cancelled'], ['NEW_SCRIPT', 'queued'], ['KernelWorkerStatus.RUNNING', 'running'], ['cancelAcknowledged', 'cancelled'], [3, 'error'], [undefined, 'queued']]) {
    assert.equal(normalStatus(raw), word, String(raw));
  }
  net.kaggle.list = ['kagada/old-one', 'kagada/namma-job', 'someone/else'];
  net.kaggle.status = { 'kagada/old-one': 'COMPLETE', 'kagada/namma-job': 'ERROR', 'someone/else': 'RUNNING' };
  assert.equal(await kaggle.busy(KGAT, 'kagada'), false);
  const list = net.kaggle.calls.find((c) => c.method === 'ListKernels');
  assert.equal(list.body.group, 'PROFILE');
  assert.deepEqual(net.kaggle.calls.filter((c) => c.method === 'GetKernelSessionStatus').map((c) => c.body), [{ userName: 'kagada', kernelSlug: 'old-one' }, { userName: 'kagada', kernelSlug: 'namma-job' }]);
  net.kaggle.status['kagada/old-one'] = 'QUEUED';
  assert.equal(await kaggle.busy(KGAT, 'kagada'), true);
  assert.equal(await kaggle.status(KGAT, 'kagada', 'namma-job'), 'error');
  await kaggle.remove(KGAT, 'kagada', 'namma-job');
  assert.deepEqual(net.kaggle.deleted, [{ userName: 'kagada', kernelSlug: 'namma-job' }]);
  assert.equal(net.kaggle.calls.at(-1).method, 'DeleteKernel');
});

// -------------------------------------------------------------------------------------------------------- start, status
async function uploaded(net, preset = 'fast') {
  return jobs.create(video(1500).file, { title: 'Start me', preset });
}

test('start refuses while a Kaggle notebook is running, and changes nothing', async () => {
  const net = setup();
  const job = await uploaded(net);
  net.kaggle.list = ['kagada/something']; net.kaggle.status = { 'kagada/something': 'RUNNING' };
  const commits = net.commits.length;
  await assert.rejects(() => jobs.start(job, KGAT), /already running on your Kaggle account/);
  assert.equal(net.kaggle.pushes.length, 0);
  assert.equal(net.commits.length, commits);
  assert.equal(net.json('dataset', `${USER}/namma-inbox`, `${job}/status.json`).stage, 'uploaded');
});

test('start pushes the private GPU notebook and marks the job as training', async () => {
  const net = setup();
  const job = await uploaded(net);
  net.kaggle.list = ['kagada/namma-job']; net.kaggle.status = { 'kagada/namma-job': 'COMPLETE' };
  const r = await jobs.start(job, JSON.stringify(LEGACY));
  assert.deepEqual(r, { kernelUrl: 'https://www.kaggle.com/code/kagada/namma-job' });
  const [push] = net.kaggle.pushes;
  assert.deepEqual([push.slug, push.newTitle, push.isPrivate, push.enableGpu, push.enableInternet, push.kernelType], ['kagada/namma-job', 'namma-job', true, true, true, 'script']);
  assert.equal(push.text, bootScript({ ref: DEFAULT_REF, hfUser: USER, job, preset: 'fast', hfToken: TOKEN }));
  const order = net.kaggle.calls.map((c) => c.method);
  assert.ok(order.indexOf('ListKernels', 1) < order.indexOf('SaveKernel'), 'the busy check comes before the push');
  const st = net.json('dataset', `${USER}/namma-inbox`, `${job}/status.json`);
  assert.equal(st.stage, 'training');
  assert.equal(st.note, 'kagada/namma-job');
  assert.deepEqual(st.history.map((h) => h[1]), ['uploaded', 'training']);
  assert.equal(net.commits.at(-1).summary, `${job}: training`);
  assert.ok(net.calls.every((c) => !String(c.url).includes(TOKEN)));
});

test('start refuses a sign-in that ends too soon, an unfinished upload and a bad job number', async () => {
  const net = setup({ hours: 1 });
  const job = await uploaded(net);
  await assert.rejects(() => jobs.start(job, KGAT), /sign-in ends too soon/);
  signIn(8);
  await assert.rejects(() => jobs.start('20260101-000000-abcd', KGAT), /upload is not finished/);
  await assert.rejects(() => jobs.start('../x', KGAT), /not valid/);
  assert.equal(net.kaggle.pushes.length, 0);
});

test('status: what the job file says, plus the notebook state when a Kaggle key is given', async () => {
  const net = setup();
  const job = await uploaded(net);
  assert.deepEqual(await jobs.status(job, null), { stage: 'uploaded', updated: net.json('dataset', `${USER}/namma-inbox`, `${job}/status.json`).updated, link: null, reason: null, kaggle: null });
  net.kaggle.status = { 'kagada/namma-job': 'ERROR' };
  await jobs.start(job, KGAT);
  const calls = net.kaggle.calls.length;
  assert.equal((await jobs.status(job, null)).stage, 'training');
  assert.equal(net.kaggle.calls.length, calls, 'no Kaggle call without a key');

  // right after the push Kaggle may still show the last version's end state: not judged yet
  assert.deepEqual(pick(await jobs.status(job, KGAT)), { stage: 'training', reason: null, kaggle: 'error' });
  config.errorGraceMs = -1;
  assert.deepEqual(pick(await jobs.status(job, KGAT)), { stage: 'failed', reason: 'The Kaggle job stopped. Open the notebook log on Kaggle.', kaggle: 'error' });
  net.kaggle.status['kagada/namma-job'] = 'CANCEL_ACKNOWLEDGED';
  assert.equal((await jobs.status(job, KGAT)).stage, 'failed');
  net.kaggle.status['kagada/namma-job'] = 'RUNNING';
  assert.deepEqual(pick(await jobs.status(job, KGAT)), { stage: 'training', reason: null, kaggle: 'running' });
  net.kaggle.status['kagada/namma-job'] = 'COMPLETE';
  assert.deepEqual(pick(await jobs.status(job, KGAT)), { stage: 'training', reason: null, kaggle: 'complete' });

  const files = net.files('dataset', `${USER}/namma-inbox`);                      // the kernel finished and wrote the link
  const done = { ...net.json('dataset', `${USER}/namma-inbox`, `${job}/status.json`), stage: 'done', link: 'https://ada-namma-tour.static.hf.space/abc/index.html', run: '20261127-1010' };
  files.set(`${job}/status.json`, { bytes: Buffer.from(JSON.stringify(done)) });
  const before = net.kaggle.calls.length;
  assert.deepEqual(pick(await jobs.status(job, KGAT), ['stage', 'link', 'kaggle']), { stage: 'done', link: done.link, kaggle: null });
  assert.equal(net.kaggle.calls.length, before, 'a finished job needs no Kaggle call');

  net.kaggle.status = {};                                                         // Kaggle cannot be asked: the job file still answers
  files.set(`${job}/status.json`, { bytes: Buffer.from(JSON.stringify({ ...done, stage: 'training', link: undefined })) });
  const s = await jobs.status(job, KGAT);
  assert.equal(s.stage, 'training'); assert.equal(s.kaggle, null); assert.match(s.kaggleError, /Kaggle/);
});
const pick = (o, keys = ['stage', 'reason', 'kaggle']) => Object.fromEntries(keys.map((k) => [k, o[k]]));

// ------------------------------------------------------------------------------------------------------------ library
test('ensureRepos creates only what is missing: two private datasets and the public static Space', async () => {
  const net = setup();
  net.repos.clear();
  net.files('dataset', `${USER}/namma-inbox`, true);
  browser(); signIn();
  globalThis.sessionStorage.setItem('namma.hf.session', JSON.stringify({ token: TOKEN, expires: new Date(Date.now() + 3600000).toISOString(), user: { name: 'ada', fullname: '', avatarUrl: '' } }));
  const fresh = await import('../../site/js/api/jobs.js?fresh=1');                 // its own "already checked" memory; same hf module
  const ids = await fresh.jobs.ensureRepos();
  assert.deepEqual(ids, { inbox: 'ada/namma-inbox', results: 'ada/namma-v3-out', tour: 'ada/namma-tour' });
  assert.deepEqual(net.created, [{ type: 'dataset', name: 'namma-v3-out', private: true }, { type: 'space', name: 'namma-tour', private: false, sdk: 'static' }]);
  const n = net.calls.length;
  await fresh.jobs.ensureRepos();
  assert.equal(net.calls.length, n, 'checked once per page');
});

test('library, uploadSplat, setVisibility, remove', async () => {
  const net = setup();
  assert.deepEqual(await jobs.library(), []);
  const bytes = randomBytes(3000);
  const entry = await jobs.uploadSplat(new File([bytes], 'my room (1).ply'), { title: 'Room' });
  const content = `${HUB}/spaces/${USER}/namma-tour/resolve/main/splats/${entry.id}/my_room_1_.ply`;
  assert.equal(entry.url, `https://owner-namma-viewer.static.hf.space/index.html?content=${encodeURIComponent(content)}`);   // its own Space, not the site's origin
  assert.deepEqual(net.files('space', `${USER}/namma-tour`).get(`splats/${entry.id}/my_room_1_.ply`).lfs, { oid: sha(bytes), size: 3000 });
  const [saved] = net.json('dataset', `${USER}/namma-v3-out`, 'library.json');
  assert.deepEqual(saved, {
    id: entry.id, run: null, title: 'Room', created: saved.created, kind: 'splat', visibility: 'link',
    url: `viewer/index.html?content=${encodeURIComponent(content)}`, cover: null, sizeMb: 0, space: `${USER}/namma-tour`, path: `splats/${entry.id}`,
  });

  // an entry the Kaggle job appended (v3/site_tail.py library_entry): the folder is only in its link
  const FOLDER = '20261127-1010-0123456789abcdef0123';
  const lib = net.json('dataset', `${USER}/namma-v3-out`, 'library.json');
  lib.push({ id: '20261127-101500-0a0a', run: '20261127-1010', title: 'Flat', created: '2026-11-27T10:15:00Z', kind: 'tour', visibility: 'private', url: `https://ada-namma-tour-2.static.hf.space/${FOLDER}/index.html`, cover: `https://ada-namma-tour-2.static.hf.space/${FOLDER}/thumbs/spot1.jpg`, sizeMb: 41.5, job: '20261127-090000-0b0b' });
  net.files('dataset', `${USER}/namma-v3-out`).set('library.json', { bytes: Buffer.from(JSON.stringify(lib)) });
  net.files('space', `${USER}/namma-tour-2`, true).set(`${FOLDER}/index.html`, { bytes: Buffer.from('<html>') });   // a tour that spilled over
  net.files('space', `${USER}/namma-tour-2`).set('20261101-0101-ffffffffffffffffffff/index.html', { bytes: Buffer.from('<html>') });
  const shown = await jobs.library();
  assert.deepEqual(shown.map((e) => [e.id, e.kind, e.visibility, e.sizeMb]), [[entry.id, 'splat', 'link', 0], ['20261127-101500-0a0a', 'tour', 'private', 41.5]]);
  assert.ok(shown.every((e) => e.url.startsWith('https://')));

  await jobs.setVisibility('20261127-101500-0a0a', 'public');
  assert.equal(net.json('dataset', `${USER}/namma-v3-out`, 'library.json')[1].visibility, 'public');
  await assert.rejects(() => jobs.setVisibility(entry.id, 'everyone'), /private, link or public/);
  await assert.rejects(() => jobs.setVisibility('nope', 'link'), /not in your library/);

  net.files('space', `${USER}/namma-tour`).set('index.html', { bytes: Buffer.from('<html>') });
  await jobs.remove(entry.id);
  assert.deepEqual([...net.files('space', `${USER}/namma-tour`).keys()], ['index.html']);
  await jobs.remove('20261127-101500-0a0a');
  assert.deepEqual([...net.files('space', `${USER}/namma-tour-2`).keys()], ['20261101-0101-ffffffffffffffffffff/index.html'], 'only this tour\'s folder goes');
  assert.deepEqual(net.json('dataset', `${USER}/namma-v3-out`, 'library.json'), []);

  // a link that is not in the user's own tour Space is never turned into a delete
  const odd = [{ id: 'x1', url: 'https://someone-namma-tour.static.hf.space/abc/index.html' }, { id: 'x2', url: 'https://ada-namma-tour.static.hf.space/index.html' }, { id: 'x3', url: 'https://ada-other.static.hf.space/abc/index.html' }, { id: 'x4', path: '../x' }];
  net.files('dataset', `${USER}/namma-v3-out`).set('library.json', { bytes: Buffer.from(JSON.stringify(odd)) });
  const commits = net.commits.filter((c) => c.type === 'space').length;
  for (const e of odd) await jobs.remove(e.id);
  assert.equal(net.commits.filter((c) => c.type === 'space').length, commits);
});

test('submitToGallery opens a pull request on the site owner\'s gallery dataset', async () => {
  const net = setup({ variables: { SPACE_CREATOR_USER_ID: '61d2f90c3c2083e1c08af22d' } });
  const lib = [{ id: 'a1', title: 'Flat', kind: 'tour', visibility: 'link', url: 'https://ada-namma-tour.static.hf.space/k3j2/index.html', cover: null, created: '2026-11-27T10:15:00Z' }];
  net.files('dataset', `${USER}/namma-v3-out`).set('library.json', { bytes: Buffer.from(JSON.stringify(lib)) });
  await assert.rejects(() => jobs.submitToGallery('a1'), /Public first/);
  lib[0].visibility = 'public';
  net.files('dataset', `${USER}/namma-v3-out`).set('library.json', { bytes: Buffer.from(JSON.stringify(lib)) });
  assert.equal(await siteOwner(), 'siteowner');                                     // looked up from the Space creator's id
  const r = await jobs.submitToGallery('a1');
  assert.deepEqual(r, { prUrl: `${HUB}/datasets/siteowner/namma-gallery/discussions/7` });
  const c = net.commits.at(-1);
  assert.deepEqual([c.type, c.id, c.pr, c.ops], ['dataset', 'siteowner/namma-gallery', true, [{ key: 'file', path: `entries/${USER}--a1.json` }]]);
  const item = JSON.parse(c.files.get(`entries/${USER}--a1.json`).bytes.toString());
  assert.deepEqual({ ...item, submitted: null }, { user: USER, id: 'a1', title: 'Flat', kind: 'tour', url: lib[0].url, cover: null, created: lib[0].created, submitted: null });
  window.huggingface.variables.SITE_OWNER = 'chosen';
  assert.equal(await siteOwner(), 'chosen');
});

// --------------------------------------------------------------------------------------------------------- the notebook
const JOB = '20261127-101500-0a0a';
const boot = (over = {}) => bootScript({ hfUser: USER, job: JOB, preset: 'fast', hfToken: TOKEN, ...over });

test('bootScript sets the job\'s settings and never prints them', () => {
  const src = boot();
  const settings = JSON.parse(JSON.parse(/os\.environ\.update\(json\.loads\((".*")\)\)/.exec(src)[1]));
  assert.deepEqual(settings, {
    NAMMA_REF: 'claude/fast-train', SOURCE: `hf:${USER}/namma-inbox:${JOB}/video.json`, INBOX_JOB: JOB, INBOX_REPO: `${USER}/namma-inbox`,
    RESULTS_REPO: `${USER}/namma-v3-out`, TOUR_SPACE: `${USER}/namma-tour`, SITE_JOB: '1', HF_TOKEN: TOKEN, ...PRESETS.fast,
  });
  assert.ok(Object.values(settings).every((v) => typeof v === 'string'));
  assert.equal(src.split(TOKEN).length, 2, 'the token is in the script once: in the settings line');
  const prints = src.split('\n').filter((l) => /\bprint\s*\(/.test(l));
  assert.ok(prints.length >= 1);
  for (const l of prints) assert.ok(!l.includes(TOKEN) && !/environ|HF_TOKEN|head\b|_SETTINGS/.test(l), `a print could show a secret: ${l}`);
  assert.ok(!/print\s*\(\s*os\.environ|pprint|env\b\s*\)/.test(src));
  assert.match(src, /git', 'clone', '-q', '--depth', '1', '-b', os\.environ\['NAMMA_REF'\], 'https:\/\/github\.com\/RazyAnas\/NAMMA', _REPO/);
  assert.match(src, /_REPO = '\/kaggle\/working\/NAMMA'/);
  assert.match(src, /colab\/real_scenes\/v3\/run\.py/);
  assert.equal(JSON.parse(JSON.parse(/json\.loads\((".*")\)\)/.exec(boot({ ref: 'main', preset: 'quality' }))[1])).FAST_STEPS, '28000');
});

test('bootScript refuses values that could change the script', () => {
  for (const over of [{ ref: '-x' }, { ref: 'a b' }, { ref: 'a"b' }, { ref: '../x' }, { hfUser: 'a/b' }, { hfUser: '' }, { job: '1' }, { job: `${JOB}'` }, { preset: 'best' },
    { hfToken: '' }, { hfToken: 'a"b\\c' }, { hfToken: 'tok en-with-space' }, { hfToken: undefined }]) {
    assert.throws(() => boot(over), /could not be prepared/, JSON.stringify(over));
  }
});

test('the preset table matches v3/PIPELINE.json', () => {
  const file = JSON.parse(readFileSync(path.join(ROOT, 'colab/real_scenes/v3/PIPELINE.json'), 'utf8')).presets;
  for (const name of Object.keys(PRESETS)) {
    if (file[name]) assert.deepEqual(PRESETS[name], file[name].env, `preset ${name} differs from PIPELINE.json: copy it into site/js/api/boot.js`);
    else assert.ok(PROVISIONAL_PRESETS.includes(name), `preset ${name} is not in PIPELINE.json and is not marked provisional`);
  }
});

// Runs the notebook source in python3 with git, the network and run.py replaced: what it does when the clone or the run fails.
const HARNESS = `
import builtins, contextlib, io, json, os, shutil, subprocess, sys, urllib.request
mode, src = sys.argv[1], sys.stdin.read()
code = compile(src, 'namma-job.py', 'exec')
calls = []
class Answer:
    def __init__(self, body=b'{}'): self.body = body
    def read(self, *a): return self.body
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
def fake_run(cmd, **kw):
    calls.append(['git', cmd, kw])
    if mode == 'clone-fails': raise subprocess.CalledProcessError(128, cmd)
def fake_urlopen(req, timeout=None):
    calls.append(['http', req.get_method(), req.full_url, {k.lower(): v for k, v in req.header_items()}, (req.data or b'').decode()])
    if req.get_method() == 'GET':
        stage = 'failed' if mode == 'already-failed' else 'training'
        return Answer(json.dumps(dict(stage=stage, updated='t0', history=[['t0', stage]], note='k/namma-job')).encode())
    return Answer()
RUNPY = {'run-exits': "raise SystemExit('NO GPU in this Kaggle session: stopping')", 'already-failed': "raise RuntimeError('late')",
         'run-ok': "import os\\nprint('ran as', __name__, os.environ['SOURCE'])", 'run-leaks': "raise RuntimeError('bad token ' + __import__('os').environ['HF_TOKEN'])"}
real_open = builtins.open
def fake_open(p, *a, **kw):
    if str(p).endswith('v3/run.py'): return io.StringIO(RUNPY[mode])
    return real_open(p, *a, **kw)
subprocess.run = fake_run; urllib.request.urlopen = fake_urlopen; builtins.open = fake_open
shutil.rmtree = lambda p, **kw: calls.append(['rmtree', p])
out, err = io.StringIO(), None
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
    try: exec(code, {'__name__': '__main__'})
    except BaseException as e: err = type(e).__name__
builtins.open = real_open
sys.stdout.write(json.dumps(dict(err=err, out=out.getvalue(), calls=calls)))
`;
function runBoot(mode) {
  const r = spawnSync('python3', ['-B', '-c', HARNESS, mode], { input: boot(), encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  const j = JSON.parse(r.stdout);
  const posts = j.calls.filter((c) => c[0] === 'http' && c[1] === 'POST');
  const written = posts.map((c) => { const f = JSON.parse(c[4].trim().split('\n')[1]).value; return { url: c[2], auth: c[3].authorization, path: f.path, status: JSON.parse(Buffer.from(f.content, 'base64').toString()) }; });
  return { ...j, written };
}

test('the notebook compiles as Python', { skip: !hasPython && 'python3 is not installed' }, () => {
  for (const preset of Object.keys(PRESETS)) {
    const r = spawnSync('python3', ['-B', '-c', 'import sys; compile(sys.stdin.read(), "namma-job.py", "exec")'], { input: boot({ preset }), encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
  }
});

test('the notebook writes status failed when the clone fails', { skip: !hasPython && 'python3 is not installed' }, () => {
  const r = runBoot('clone-fails');
  assert.equal(r.err, 'CalledProcessError');
  assert.deepEqual(r.calls[0].slice(0, 2), ['git', ['git', 'clone', '-q', '--depth', '1', '-b', 'claude/fast-train', 'https://github.com/RazyAnas/NAMMA', '/kaggle/working/NAMMA']]);
  assert.equal(r.calls[0][2].check, true);
  assert.equal(r.calls[0][2].shell, undefined);
  assert.equal(r.written.length, 1);
  const [w] = r.written;
  assert.equal(w.url, `${HUB}/api/datasets/${USER}/namma-inbox/commit/main`);
  assert.equal(w.auth, `Bearer ${TOKEN}`);
  assert.equal(w.path, `${JOB}/status.json`);
  assert.deepEqual({ stage: w.status.stage, reason: w.status.reason, note: w.status.note }, { stage: 'failed', reason: 'could not fetch the NAMMA code', note: 'k/namma-job' });
  assert.deepEqual(w.status.history.map((h) => h[1]), ['training', 'failed']);
  assert.ok(!r.out.includes(TOKEN) && !r.out.includes('HF_TOKEN'), 'the log must not show the token');
});

test('the notebook runs run.py as the main script, and reports a run that stops', { skip: !hasPython && 'python3 is not installed' }, () => {
  const ok = runBoot('run-ok');
  assert.equal(ok.err, null);
  assert.equal(ok.out.trim(), `ran as __main__ hf:${USER}/namma-inbox:${JOB}/video.json`);
  assert.equal(ok.written.length, 0);
  assert.deepEqual(ok.calls.map((c) => c[0]), ['git', 'rmtree'], 'the fetched copy is removed before run.py runs: /kaggle/working is uploaded as the result');
  assert.equal(ok.calls[1][1], '/kaggle/working/NAMMA');

  const stop = runBoot('run-exits');
  assert.equal(stop.err, 'SystemExit');
  assert.deepEqual([stop.written[0].status.stage, stop.written[0].status.reason], ['failed', 'NO GPU in this Kaggle session: stopping']);

  const kept = runBoot('already-failed');                                           // the pipeline's own reason is not overwritten
  assert.equal(kept.err, 'RuntimeError');
  assert.equal(kept.written.length, 0);

  const leak = runBoot('run-leaks');                                                // an error text that holds the token is scrubbed
  assert.equal(leak.written[0].status.reason, 'bad token ***');
  assert.ok(!leak.out.includes(TOKEN));
});

test('ids look like the Android app\'s', () => {
  assert.match(newId(new Date(Date.UTC(2026, 10, 27, 9, 5, 7))), /^20261127-090507-[0-9a-f]{4}$/);
});
