"""The end of a Kaggle job started from the website (SITE_JOB=1; site/js/api/boot.js): build the tour here in the kernel and
publish it to the visitor's OWN Hugging Face account, with nothing of ours in between (27 Nov; owner: "a full website where
people can upload their work use their own api keys"). Until now the tour was built on a GitHub runner with the owner's
secrets (gh_run.tour_step), which cannot serve anybody else. Same steps, same code: make_tour.plan -> solidify clean ->
make_tour.build -> tour_publish.publish. Then <job>/status.json = done + link (the site polls it) and one more entry in
<results>/library.json (the visitor's library).
Environment (set by the notebook the site pushes): INBOX_JOB, INBOX_REPO, RESULTS_REPO, HF_TOKEN (the visitor's sign-in token,
8 hours, only in their own private notebook; never printed here)."""
import json, os, re, subprocess, sys, time

NODE = 'v22.12.0'                                   # splat-transform needs Node 22; Kaggle's image has none
PKGS = 'playcanvas@2.23.2 @playcanvas/supersplat-viewer@1.42.0 @playcanvas/splat-transform@3.10.1'   # as v3/gh_run.py tour_step
SKIP = ['images/*', 'sfm_work/*']


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read_json(api, repo, path, default=None):
    from huggingface_hub import hf_hub_download
    try: return json.load(open(hf_hub_download(repo, path, repo_type='dataset', token=api.token, force_download=True, local_dir='site_dl')))
    except Exception: return default


def status(api, repo, job, stage, **kw):
    """<job>/status.json in the inbox, the schema of v3/inbox.py set_status (stage, updated, history, run, link, reason, note)."""
    if not job: return None
    st = read_json(api, repo, f'{job}/status.json', {}) or {}
    if stage != 'failed': st.pop('reason', None)
    st.update(kw, stage=stage, updated=now()); st.setdefault('history', []).append([st['updated'], stage])
    api.upload_file(path_or_fileobj=json.dumps(st, indent=1).encode(), path_in_repo=f'{job}/status.json', repo_id=repo, repo_type='dataset',
                    commit_message=f'{job}: {stage}')
    return st


def library_entry(run, job, link, meta, tour_dir, metrics=None):
    """One library row as the site reads it (site/api.md): id, run, title, created, kind, visibility, url, cover, sizeMb."""
    mb = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(tour_dir) for f in fs) / 1e6 if os.path.isdir(tour_dir) else 0
    cover = link.rsplit('/', 1)[0] + '/thumbs/spot1.jpg' if os.path.exists(os.path.join(tour_dir, 'thumbs', 'spot1.jpg')) else None
    title = str((meta or {}).get('title') or '').strip()[:80] or f'Tour {run[:4]}-{run[4:6]}-{run[6:8]}'
    e = dict(id=run, run=run, job=job, title=title, created=now(), kind='tour', visibility='link', url=link, cover=cover, sizeMb=round(mb, 1),
             preset=(meta or {}).get('preset'))
    if metrics: e.update(splats=metrics.get('splats'), psnr=metrics.get('heldout_psnr_exposure_fair'))
    return e


def add_to_library(api, repo, entry):
    """library.json = a list, newest first; the same run twice replaces its row (a retry)."""
    lib = read_json(api, repo, 'library.json', []) or []
    lib = [entry] + [x for x in lib if isinstance(x, dict) and x.get('id') != entry['id']]
    api.upload_file(path_or_fileobj=json.dumps(lib, indent=1).encode(), path_in_repo='library.json', repo_id=repo, repo_type='dataset',
                    commit_message=f"library: {entry['id']}")
    return lib


def node_setup(T, sh):
    """Node 22 + the viewer and converter packages under T. Returns the PATH prefix that makes `node` visible."""
    os.makedirs(f'{T}/node', exist_ok=True)
    sh(f'curl -fsSL https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz | tar -xJ -C {T}/node --strip-components=1', check=True)
    sh(f'cd {T} && export PATH={T}/node/bin:$PATH && npm init -y >/dev/null && npm i -s --no-audit --no-fund {PKGS} 2>&1 | tail -2', check=True)
    assert os.path.exists(f'{T}/node_modules/@playcanvas/supersplat-viewer/public/index.js'), 'the viewer package did not install'
    return f'{T}/node/bin'


def build(W, O, D, EX, F, sh, say):
    """The tour folder <O>/tour from the trained model, in the kernel (the steps of gh_run.tour_step)."""
    T = f'{W}/tourwork'; os.makedirs(T, exist_ok=True); nb = node_setup(T, sh)
    open(f'{EX}/datasets/__init__.py', 'a').close()        # else Kaggle's installed Hugging Face `datasets` package is imported instead
    sh('pip install -q plyfile trimesh fast-simplification scipy pillow 2>&1 | tail -1')
    sh(f'cd {T} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/make_tour.py plan {EX} {D} {O} {O}/tour', check=True)
    env = ''
    if sh(f'cd {EX} && SOLID_ONLY_CLEAN=1 PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/solidify.py {D} {O}/full/model.ply {O}/solid 480 3') == 0 \
            and os.path.exists(f'{O}/solid/clean.ply'): env = 'TOUR_MODEL=solid/clean.ply '
    else: say('clean-up failed, the tour uses the trained model as is')
    sh(f'cd {T} && PATH={nb}:$PATH {env}PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/make_tour.py build {O} {O}/tour '
       f'{T}/node_modules/@playcanvas/supersplat-viewer/public {T}/node_modules/.bin/splat-transform', check=True)
    assert os.path.exists(f'{O}/tour/index.html') and os.path.getsize(f'{O}/tour/index.js') > 1e6, 'the tour page was not written'
    return T


def run(W, O, D, EX, F, RUN, token, say, sh, S=None):
    """Called at the end of v3/run.py. Never raises: the job's own result (the model) is already on Kaggle; what failed is said
    in status.json, in words a visitor can read."""
    job = os.environ.get('INBOX_JOB')
    if not token: say('site job: no Hugging Face token in the notebook, nothing published'); return None
    from huggingface_hub import HfApi
    api = HfApi(token=token); user = api.whoami()['name']
    inbox = os.environ.get('INBOX_REPO') or f'{user}/namma-inbox'; results = os.environ.get('RESULTS_REPO') or f'{user}/namma-v3-out'
    fail = lambda why: (say('site job failed:', why), status(api, inbox, job, 'failed', reason=why, run=RUN))
    try:
        if not os.path.exists(f'{O}/full/model.ply'):
            fail('Training did not produce a model. The video may be too short, too dark or too shaky: the camera positions could not be worked out.'); return None
        status(api, inbox, job, 'publishing', run=RUN)
        api.create_repo(results, repo_type='dataset', private=True, exist_ok=True)
        try: T = build(W, O, D, EX, F, sh, say)
        except Exception as e:
            fail(f'The model was trained, but building the tour page failed ({str(e)[:160]}). The model is kept in your Kaggle notebook output.'); return None
        keep_out = [f'{n}/model.ply' for n in os.listdir(O) if os.path.isdir(f'{O}/{n}') and n not in ('full', 'prior')] + SKIP
        api.upload_folder(folder_path=O, path_in_repo=RUN, repo_id=results, repo_type='dataset', ignore_patterns=keep_out)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tour_publish
        os.environ['TOUR_KEEP_OTHER'] = '1'                  # the visitor's Space also holds their uploaded splat files: delete nothing
        cwd = os.getcwd(); os.chdir(T)                       # publish() writes its small files where it runs
        try: tour_publish.publish(api, user, results, RUN, f'{O}/tour', say); link = open('tour_url.txt').read().strip()
        finally: os.chdir(cwd)
        meta = read_json(api, inbox, f'{job}/meta.json', {}) if job else {}
        add_to_library(api, results, library_entry(RUN, job, link, meta, f'{O}/tour', (S or {}).get('full')))
        status(api, inbox, job, 'done', run=RUN, link=link); say('site job: tour published to the visitor\'s Space and library')
        return link
    except Exception as e:
        try: fail(f'Publishing failed ({type(e).__name__}: {re.sub(r"hf_[A-Za-z0-9]+", "hf_***", str(e))[:160]}). The model is kept in your Kaggle notebook output.')
        except Exception: pass
        return None
