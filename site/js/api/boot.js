// The Kaggle notebook the site pushes for one job (a Python script; kaggle.js push). It only sets the job's settings, fetches
// the code and runs colab/real_scenes/v3/run.py as the main script, the way v3/gh_run.py makes the kernel run it (settings
// in os.environ before run.py's first line). What run.py must do with SITE_JOB=1 is listed in site/api.md.
// The script never prints its settings, and it writes status.json 'failed' with a short reason when the clone or the run raises.

export const DEFAULT_REF = 'claude/fast-train';
export const GITHUB_REPO = 'https://github.com/RazyAnas/NAMMA';

/** Settings per preset: a copy of colab/real_scenes/v3/PIPELINE.json presets.<name>.env (tests/browser/site_api.test.mjs
 *  fails when a preset that exists there differs from this table). */
export const PRESETS = {
  fast: {
    FAST: '1', FAST_DEPS: '1', NFRAMES: 'auto', LONG: '1440', OBJECTS: '1',
    FAST_STEPS: '15000', FAST_CAP: '1000000', FAST_KIND: 'fbg+res0.5_0.6+dense',
  },
  quality: {
    FAST: '1', FAST_DEPS: '1', NFRAMES: 'auto', LONG: '1440', OBJECTS: '1', FAST_FRAMES: '420',
    FAST_STEPS: '28000', FAST_CAP: '1500000', FAST_KIND: 'fbg+dense+prior',
  },
};
/** Presets listed here may be missing from PIPELINE.json (placeholder values, not measured yet). None at the moment. */
export const PROVISIONAL_PRESETS = [];

const JOB = /^\d{8}-\d{6}-[0-9a-f]{4}$/;
const REF = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$/;
const USER = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const TOKEN = /^[A-Za-z0-9._~+/=-]{8,4096}$/;

const PYTHON = String.raw`# NAMMA site job. Written by the website (site/js/api/boot.js) for one video; do not edit, start jobs from the site.
import base64, json, linecache, os, shutil, subprocess, sys, time, urllib.request
os.environ.update(json.loads(__SETTINGS__))  # set by the site


def _status(stage, reason):
    """<inbox>/<job>/status.json like v3/inbox.py set_status. An end state the pipeline already wrote is kept."""
    repo, job = os.environ['INBOX_REPO'], os.environ['INBOX_JOB']
    head = {'Authorization': 'Bearer ' + os.environ['HF_TOKEN']}
    try:
        with urllib.request.urlopen(urllib.request.Request(f'https://huggingface.co/datasets/{repo}/resolve/main/{job}/status.json', headers=head), timeout=60) as r:
            st = json.load(r)
    except Exception:
        st = {}
    if st.get('stage') in ('done', 'failed'):
        return
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    st.update(stage=stage, updated=now, reason=reason)
    st.setdefault('history', []).append([now, stage])
    lines = [dict(key='header', value=dict(summary=f'{job}: {stage}', description='')),
             dict(key='file', value=dict(path=f'{job}/status.json', encoding='base64',
                                         content=base64.b64encode(json.dumps(st, indent=1).encode()).decode()))]
    req = urllib.request.Request(f'https://huggingface.co/api/datasets/{repo}/commit/main', method='POST',
                                 data=('\n'.join(json.dumps(x) for x in lines) + '\n').encode(),
                                 headers=dict(head, **{'Content-Type': 'application/x-ndjson'}))
    urllib.request.urlopen(req, timeout=60).close()


def _failed(e):
    why = str(e.code if isinstance(e, SystemExit) else e) or type(e).__name__
    if isinstance(e, subprocess.CalledProcessError):
        why = 'could not fetch the NAMMA code'
    elif isinstance(e, KeyboardInterrupt):
        why = 'the job was stopped'
    why = why.replace(os.environ.get('HF_TOKEN') or '\0', '***').strip()[:300]
    try:
        _status('failed', why)
        print('job failed:', why, flush=True)
    except Exception as e2:
        print('could not write the failed status:', type(e2).__name__, flush=True)


_REPO = '/kaggle/working/NAMMA'
_RUN = _REPO + '/colab/real_scenes/v3/run.py'
try:
    if not os.path.isdir(_REPO + '/.git'):
        subprocess.run(['git', 'clone', '-q', '--depth', '1', '-b', os.environ['NAMMA_REF'], '__GITHUB__', _REPO], check=True)
    _src = open(_RUN).read()
    # run.py fetches its own copy of the code; this one must not stay in /kaggle/working, the folder the job uploads as its result
    shutil.rmtree(_REPO, ignore_errors=True)
    linecache.cache[_RUN] = (len(_src), None, _src.splitlines(True), _RUN)      # error messages still show run.py's lines
    exec(compile(_src, _RUN, 'exec'), {'__name__': '__main__', '__file__': _RUN})
except SystemExit as e:
    if e.code not in (None, 0):
        _failed(e)
    raise
except BaseException as e:
    _failed(e)
    raise
`;

/** Python source of the notebook for one job. hfToken goes into the script (the user's own private notebook on Kaggle) so the
 *  job can read the inbox and write the results; nothing else of the user's is in it. */
export function bootScript({ ref = DEFAULT_REF, hfUser, job, preset = 'fast', hfToken } = {}) {
  const bad = (what) => new Error(`The job could not be prepared: ${what}.`);
  if (!REF.test(String(ref)) || String(ref).includes('..')) throw bad('the code branch name is not valid');
  if (!USER.test(String(hfUser || ''))) throw bad('the Hugging Face user name is not valid');
  if (!JOB.test(String(job || ''))) throw bad('the job number is not valid');
  if (!Object.prototype.hasOwnProperty.call(PRESETS, preset)) throw bad(`there is no setting called "${preset}"`);
  if (!TOKEN.test(String(hfToken || ''))) throw bad('you are not signed in to Hugging Face');
  const env = {
    NAMMA_REF: ref,
    SOURCE: `hf:${hfUser}/namma-inbox:${job}/video.json`,
    INBOX_JOB: job,
    INBOX_REPO: `${hfUser}/namma-inbox`,
    RESULTS_REPO: `${hfUser}/namma-v3-out`,
    TOUR_SPACE: `${hfUser}/namma-tour`,
    SITE_JOB: '1',
    HF_TOKEN: hfToken,
    ...PRESETS[preset],
  };
  // JSON inside a JSON string: a valid Python string literal whatever the values are (v3/kaggle_io.py push_kernel does the same).
  return PYTHON.replace('__SETTINGS__', () => JSON.stringify(JSON.stringify(env))).replace('__GITHUB__', GITHUB_REPO);
}
