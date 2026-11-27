# Site API modules (`site/js/api`)

Plain ES modules, no dependencies, no build step. The browser talks to Hugging Face and Kaggle directly; there is no server.
Tests: `node --test tests/browser/site_api.test.mjs` (fake network; the Python checks need `python3`).

| File | Exports | Job |
|---|---|---|
| `hf.js` | `hf`, `setFetch`, `Sha256`, `sha256Hex`, `hashBlob`, `tuning` | sign-in, Hub calls, large-file upload |
| `kaggle.js` | `kaggle`, `setFetch`, `parseToken`, `normalStatus` | Kaggle key check, notebook push, status |
| `jobs.js` | `jobs`, `setFetch`, `config`, `newId`, `siteOwner` | the user's jobs and library |
| `boot.js` | `bootScript`, `PRESETS`, `DEFAULT_REF` | Python source of the Kaggle notebook |

Every function returns data only and throws `Error` with a short message a person can read (`err.status` holds the HTTP code).
`setFetch(fn)` replaces `fetch` (tests). `jobs.js`'s `setFetch` sets both `hf.js` and `kaggle.js`.

## Contract

```js
hf.available()      // true when window.huggingface.variables.OAUTH_CLIENT_ID exists and sessionStorage works
await hf.init()     // once at startup; finishes ?code=&state=, cleans the address, returns session|null. THROWS when the
                    // sign-in was refused or the state does not match (the address is already clean then)
hf.session()        // null | { user: { name, fullname, avatarUrl }, expires }   (ISO time)
await hf.login()    // redirects this tab to huggingface.co/oauth/authorize
hf.logout(); hf.token(); hf.variables()
await hf.whoami(); hf.createRepo(type, name, { private, sdk }); hf.repoExists(type, id); hf.userName(userId)
await hf.readText(type, repo, path)   // null when missing        hf.readJson(...)   // null when missing or not JSON
await hf.listTree(type, repo, path, { recursive })    // [] when missing; [{ type: 'file'|'directory', path, size, oid, lfs? }]
await hf.commit(type, repo, files, message, { deletes, description, createPr })
        // files: { path, content: string|Blob|Uint8Array } or { path, lfs: { oid, size } }; deletes: 'a/file', 'a/folder/'
await hf.lfsUpload(type, repo, blobOrBytes, sha256hex, onProgress)   // false: the Hub already had it
await hf.uploadLarge(type, repo, path, blob, onProgress)             // -> { path, oid, size }
await hf.openPullRequest(type, repo, files, title, description)      // -> { url, num }

await kaggle.check(token)                 // -> { username }
await kaggle.busy(token, username)        // true when one of the user's notebooks is queued or running
await kaggle.push(token, { username, slug, title, code, gpu, internet, isPrivate })   // -> { url, ref, version }
await kaggle.status(token, username, slug)   // 'queued'|'running'|'complete'|'error'|'cancelled'
await kaggle.remove(token, username, slug)

await jobs.ensureRepos()                  // -> { inbox, results, tour } (repo ids)
await jobs.library()                      // [{ id, run, title, created, kind, visibility, url, cover, sizeMb, ... }], url and cover absolute
await jobs.list()                         // [{ job, title, preset, created, stage, updated, link, reason }], newest first
await jobs.create(file, { title, preset }, onProgress)   // -> job id; onProgress(fraction, text)
await jobs.start(jobId, kaggleToken)      // -> { kernelUrl }
await jobs.status(jobId, kaggleTokenOrNull)   // -> { stage, updated, link, reason, kaggle, kaggleError? }
await jobs.uploadSplat(file, { title }, onProgress)      // -> the library entry
await jobs.setVisibility(id, 'private'|'link'|'public'); jobs.remove(id)
await jobs.submitToGallery(id)            // -> { prUrl }
```

Stages: `uploading` (site only: a video whose upload did not finish) -> `uploaded` -> `training` -> `publishing` -> `done` | `failed`.

## Tokens

- Hugging Face token: `sessionStorage` key `namma.hf.session` only. Sent only to `https://huggingface.co` in the
  `Authorization` header, never to the storage hosts that take the file bytes, never in a URL, never logged.
- The token is also written into the notebook source (`HF_TOKEN`), so the job can read the inbox and publish. The notebook is
  private, in the user's own Kaggle account. Anyone who can open that Kaggle account can read it until it expires.
- Kaggle key: passed in on every call, used for the `Authorization` header, not kept. Calls use `credentials: 'omit'`.
- `jobs.start` refuses when the sign-in ends in less than 2 h (`fast`) or 4 h (`quality`): the job would finish with a dead token.

## Hugging Face sign-in (static Space)

Source: https://huggingface.co/docs/hub/spaces-oauth and https://huggingface.co/docs/hub/spaces-sdks-static

- Space `README.md`: `hf_oauth: true`, `hf_oauth_scopes` (`openid profile` are always added), `hf_oauth_expiration_minutes`
  (default 480, at most 43200). The page then finds `window.huggingface.variables.OAUTH_CLIENT_ID` and `OAUTH_SCOPES`.
- A live static Space (`huggingfacejs-static-variables.static.hf.space`, read on 27 Nov 2026) has exactly these variables:
  `OAUTH_CLIENT_ID`, `OAUTH_SCOPES`, `OPENID_PROVIDER_URL`, `SPACE_CREATOR_USER_ID`, plus the Space's own variables.
  `SPACE_HOST` and `SPACE_ID` are NOT there. Space secrets are also given to the page, so a static Space must have none.
- Authorize: `https://huggingface.co/oauth/authorize?client_id&scope&response_type=code&redirect_uri&state&code_challenge&code_challenge_method=S256`.
  `redirect_uri` = the page's own address without query and hash (any address inside the Space is allowed).
- Token: `POST https://huggingface.co/oauth/token`, form body `grant_type=authorization_code, code, redirect_uri,
  code_verifier, client_id`, no secret. Answer: `access_token, expires_in, id_token, scope`. No refresh token is used.
- User: `GET https://huggingface.co/oauth/userinfo` -> `preferred_username` (user name), `name`, `picture`.
- The verifier and state wait in `sessionStorage` (`namma.hf.pkce`) during the redirect.
- Inside the huggingface.co page frame sign-in can fail on cookies: open the site at its own `*.static.hf.space` address.

Scopes this code needs: `read-repos` and `write-repos` (private datasets, commits), `manage-repos` (create the three repos),
`write-discussions` (gallery pull request). This is what `site/README.md` asks for.

## Hub calls (the ones `Hub.kt` makes)

- `GET /api/whoami-v2`; `POST /api/repos/create` `{ type, name, private, sdk }` (409 = exists); `GET /api/<type>s/<repo>` (404 = missing)
- `GET /<type>s/<repo>/resolve/main/<path>`; `GET /api/<type>s/<repo>/tree/main/<path>` (next page in the `Link` header)
- Large file: `POST /<type>s/<repo>.git/info/lfs/objects/batch` `{ operation: 'upload', transfers: ['basic','multipart'],
  hash_algo: 'sha256', ref: { name: 'main' }, objects: [{ oid, size }] }`. No `actions.upload` = already stored. With
  `header.chunk_size`: PUT each chunk to `header['1'], header['2'], ...`, read each `ETag`, then POST `{ oid, parts:
  [{ partNumber, etag }] }` to `upload.href`. Without: one PUT to `upload.href` with `upload.header`. Then POST `{ oid, size }`
  to `actions.verify.href` when it is given.
- `POST /api/<type>s/<repo>/preupload/main` `{ files: [{ path, size, sample }] }` -> `uploadMode: 'lfs'|'regular'` (only asked
  for binary content given to `hf.commit`; text is always a regular file, video parts and splats are always large files).
- `POST /api/<type>s/<repo>/commit/main` (`?create_pr=1` for a pull request), NDJSON lines: `header { summary, description }`,
  `file { path, content, encoding: 'base64' }`, `lfsFile { path, algo: 'sha256', oid, size }`, `deletedFile { path }`,
  `deletedFolder { path }`.

## Inbox format (same as the Android app; read by `v3/inbox.py`)

`<user>/namma-inbox`, one folder per job, id `YYYYMMDD-HHMMSS-xxxx` (UTC + 4 hex):

- `parts/000`, `parts/001`, ...: the video cut every 32 MB (`config.partSize`), each its own large-file object (oid = its sha256).
- `video.json`: `{ name, size, sha256, parts: ["<job>/parts/000", ...] }` (these four keys, this order, indent 1).
- `meta.json`: `{ name, title, preset, source: 'site', file, size, part_size, created }`.
- `status.json`: `{ stage, updated, history: [[time, stage], ...], run, link, reason, note }`. A new stage keeps the history
  and drops an old `reason`.

Upload order: `meta.json` first, then the parts (one commit per 4 parts), last one commit with `video.json`, `meta.json` and
`status.json` (`uploaded`). The pipeline only sees a job that has `status.json`. Resume: choosing the same file again (same
name, size and part size) finds the folder that has `meta.json` but no `status.json`, lists `parts/` and skips every part
whose size and sha256 match. The file is read one part at a time; the hash of the whole file is built piece by piece
(`Sha256`), because WebCrypto can only hash one buffer. Progress steps once per part.

## Library and tour Space

- `<user>/namma-v3-out/library.json`: a JSON list, newest first. Entry: `{ id, run, title, created, kind: 'tour'|'splat',
  visibility: 'private'|'link'|'public', url, cover, sizeMb }` plus anything else (the kernel adds `job, preset, splats, psnr`).
- `uploadSplat`: file at `<user>/namma-tour/splats/<id>/<filename>`, entry `{ kind: 'splat', visibility: 'link',
  url: 'viewer/index.html?content=<encoded https://huggingface.co/spaces/<user>/namma-tour/resolve/main/splats/<id>/<filename>>',
  space, path: 'splats/<id>' }`. The stored `url` is relative to the site; `jobs.library()` and `uploadSplat` return it as a
  full address (the deploy step puts the stock viewer under `site/viewer/`).
- `setVisibility` only changes the entry. It is a label for the site: the tour Space is public, so a `private` item is still
  reachable by anyone who has its link.
- `remove` deletes the entry and the item's folder in the user's own tour Space (`path`, or for a tour the first folder of its
  link `https://<user>-namma-tour[-N].static.hf.space/<folder>/index.html`). The run's files in `namma-v3-out/<run>` stay.
- `library.json` is read, changed and written back; the site and a finishing job writing in the same second can lose one change.
- `submitToGallery`: needs `visibility: 'public'`. Pull request on the dataset `<SITE_OWNER>/namma-gallery` adding
  `entries/<user>--<id>.json` = `{ user, id, title, kind, url, cover, created, submitted }`.
  `SITE_OWNER`: the Space variable `SITE_OWNER` if set; else the owner part of `SPACE_ID` if the page ever gets it; else the
  user name behind `SPACE_CREATOR_USER_ID` (`GET /api/users/<id>/overview`). The host name is not used: `a-b-c.hf.space`
  does not say where the owner's name ends. Set `SITE_OWNER` when the Space belongs to an organisation.

## Kaggle

Source: `Kaggle/kagglesdk` (`kaggle_http_client.py`, `kernels/types/kernels_api_service.py`, `kernels/types/kernels_enums.py`,
`security/types/oauth_service.py`) and `Kaggle/kaggle-cli` (`src/kaggle/api/kaggle_api_extended.py`) on GitHub, main, 27 Nov 2026.

- Every call: `POST https://api.kaggle.com/v1/<service>/<method>`, JSON body, `Content-Type: application/json`.
- Auth: `Authorization: Bearer KGAT_...` for an API token, `Authorization: Basic base64(username:key)` for `kaggle.json`.
- `security.OAuthService/IntrospectToken` `{ token }` -> `{ active, username }` (a wrong token answers 404).
- `kernels.KernelsApiService/ListKernels` `{ group: 'PROFILE', sortBy: 'DATE_RUN', pageSize, ... }` -> `{ kernels: [{ ref }] }`.
- `.../GetKernelSessionStatus` `{ userName, kernelSlug }` -> `{ status, failureMessage }`, status one of `QUEUED, RUNNING,
  COMPLETE, ERROR, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, NEW_SCRIPT`.
- `.../SaveKernel` `{ slug: 'user/slug', newTitle, text, language: 'python', kernelType: 'script', isPrivate, enableGpu,
  enableTpu, enableInternet, datasetDataSources, competitionDataSources, kernelDataSources, modelDataSources, categoryIds,
  machineShape: 'NvidiaTeslaT4' }` -> `{ ref, url, versionNumber, error }`.
- `.../DeleteKernel` `{ userName, kernelSlug }`.
- `busy` = `v3/kaggle_io.py busy`: list the user's notebooks, ask each one's status, true on the first queued or running one.

`jobs.start`: check the key -> refuse when `busy` -> push the private notebook `namma-job` (GPU, internet) -> `status.json`
`training` with `note: "<kaggle user>/namma-job"`. `jobs.status` reads `status.json`; with a key and stage `training` or
`publishing` it also asks Kaggle (using `note`). A notebook in `error` or `cancelled` while the stage is `training` is reported
as `failed`, "The Kaggle job stopped. Open the notebook log on Kaggle." (not written to `status.json`; not judged in the
first 2 minutes after the push, when Kaggle can still show the previous version's end state).

## The notebook (`boot.js`) and what the kernel side must do

`bootScript({ ref, hfUser, job, preset, hfToken })` returns a Python script that:

1. sets `os.environ`: `NAMMA_REF` (default `claude/fast-train`; the Space variable `NAMMA_REF` overrides it),
   `SOURCE=hf:<user>/namma-inbox:<job>/video.json`, `INBOX_JOB`, `INBOX_REPO`, `RESULTS_REPO=<user>/namma-v3-out`,
   `TOUR_SPACE=<user>/namma-tour`, `SITE_JOB=1`, `HF_TOKEN`, and the preset's variables (`PRESETS`, a copy of
   `v3/PIPELINE.json` presets `fast` and `quality`; the test fails when they differ);
2. `git clone --depth 1 -b <ref> https://github.com/RazyAnas/NAMMA /kaggle/working/NAMMA`, reads
   `colab/real_scenes/v3/run.py` from it, deletes the clone (`/kaggle/working` is uploaded as the job's result) and runs the
   source as `__main__`, as if `run.py` had been pushed with the settings in front (`gh_run.py`);
3. never prints its settings; when the clone or the run raises (also `SystemExit` with a message), writes `status.json`
   `failed` with a short reason, unless the stage is already `done` or `failed`. Only the standard library is used for that.

The kernel side (`v3/run.py`, `v3/site_tail.py`) is expected to:

- read the token from `HF_TOKEN` in the environment (no Kaggle secret exists in a visitor's account);
- accept `SOURCE=hf:<repo>:<job>/video.json`: join the parts with `inbox.assemble` and check size and sha256;
- with `SITE_JOB=1`, after training: `status.json` `publishing`, build the tour in the kernel, publish it to the user's
  public Space (`TOUR_SPACE`, deleting nothing else there: `splats/` holds uploaded files), then prepend one entry to
  `RESULTS_REPO/library.json` (`id, run, title` from `<job>/meta.json`, `created, kind: 'tour', visibility, url, cover,
  sizeMb`) and write `status.json` `done` with `run` and `link`;
- on any failure write `status.json` `failed` with a `reason` a person can read, and keep `note` as it is;
- never print the environment or the token.

## Not proven yet

- A real signed-in run in a browser: sign-in on the deployed Space, repo creation with the OAuth token, a commit, a large upload.
- Which host takes the file bytes. The Hub hands out the upload address; if it is not covered by the page's `connect-src`
  (`*.hf.co`, `*.huggingface.co`, `storage.googleapis.com` today), the browser blocks the PUT. Old addresses were on
  `*.amazonaws.com`.
- Multipart uploads need the storage host to let the page read the `ETag` header.
- Any authenticated Kaggle call from a page. Preflight and error answers from `api.kaggle.com` were checked (CORS headers are
  sent, a wrong key gives 401/404 JSON); a call with a real key was not.
- Whether Kaggle answers `QUEUED` (the default) for a notebook that never ran; `busy` would then say true for it.
