"""A website job's last step in the Kaggle kernel (v3/site_tail.py): status, library and failure words. No network, no Node."""
import json, os, sys, types
import pytest

V3 = os.path.join(os.path.dirname(__file__), '..', '..', 'colab', 'real_scenes', 'v3')
sys.path.insert(0, os.path.abspath(V3))
import site_tail                                                                          # noqa: E402


class FakeApi:
    token = 'hf_secretsecret'

    def __init__(self): self.files = {}; self.repos = []; self.folders = []
    def whoami(self): return {'name': 'visitor'}
    def create_repo(self, repo, **kw): self.repos.append((repo, kw.get('repo_type'), kw.get('private')))
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message=None):
        self.files[(repo_id, path_in_repo)] = path_or_fileobj if isinstance(path_or_fileobj, bytes) else open(path_or_fileobj, 'rb').read()
    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, ignore_patterns=None): self.folders.append((repo_id, path_in_repo, tuple(ignore_patterns or ())))


@pytest.fixture
def api(monkeypatch):
    a = FakeApi()
    monkeypatch.setattr(site_tail, 'read_json', lambda api_, repo, path, default=None: json.loads(a.files[(repo, path)]) if (repo, path) in a.files else default)
    return a


def test_status_keeps_the_history_and_drops_an_old_reason(api):
    site_tail.status(api, 'visitor/namma-inbox', 'j1', 'failed', reason='x')
    st = site_tail.status(api, 'visitor/namma-inbox', 'j1', 'publishing', run='20261127-1700')
    assert st['stage'] == 'publishing' and 'reason' not in st and [h[1] for h in st['history']] == ['failed', 'publishing'] and st['run'] == '20261127-1700'
    assert site_tail.status(api, 'visitor/namma-inbox', None, 'done') is None               # not an inbox job: nothing written


def test_library_row_and_newest_first(api, tmp_path):
    t = tmp_path / 'tour'; (t / 'thumbs').mkdir(parents=True); (t / 'thumbs' / 'spot1.jpg').write_bytes(b'x' * 1000); (t / 'scene.compressed.ply').write_bytes(b'x' * 2_000_000)
    link = 'https://visitor-namma-tour.static.hf.space/20261127-1700-0123456789abcdef0123/index.html'
    e = site_tail.library_entry('20261127-1700', 'j1', link, {'title': '  My kitchen ', 'preset': 'quality'}, str(t), {'splats': 1500000, 'heldout_psnr_exposure_fair': 29.9})
    assert e['title'] == 'My kitchen' and e['kind'] == 'tour' and e['visibility'] == 'link' and e['url'] == link and e['sizeMb'] == 2.0
    assert e['cover'].endswith('/20261127-1700-0123456789abcdef0123/thumbs/spot1.jpg') and e['splats'] == 1500000
    assert site_tail.library_entry('20261127-1700', None, link, {}, str(tmp_path / 'none'))['title'] == 'Tour 2026-11-27'
    site_tail.add_to_library(api, 'visitor/namma-v3-out', dict(e, id='old')); lib = site_tail.add_to_library(api, 'visitor/namma-v3-out', e)
    lib = site_tail.add_to_library(api, 'visitor/namma-v3-out', dict(e, title='again'))     # a retry replaces its own row
    assert [x['id'] for x in lib] == ['20261127-1700', 'old'] and lib[0]['title'] == 'again'


def test_no_model_is_said_in_plain_words_and_nothing_is_published(api, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(HfApi=lambda token=None: api))
    monkeypatch.setenv('INBOX_JOB', 'j1'); out = []
    r = site_tail.run(str(tmp_path), str(tmp_path / 'out'), '', '', '', '20261127-1700', 'hf_secretsecret', lambda *a: out.append(' '.join(map(str, a))), None)
    st = json.loads(api.files[('visitor/namma-inbox', 'j1/status.json')])
    assert r is None and st['stage'] == 'failed' and 'did not produce a model' in st['reason'] and not api.folders
    assert 'hf_secretsecret' not in json.dumps(st) + ' '.join(out)                          # the token is never written or said


def test_a_failed_build_keeps_the_model_and_says_so(api, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(HfApi=lambda token=None: api))
    monkeypatch.setenv('INBOX_JOB', 'j2'); (tmp_path / 'out' / 'full').mkdir(parents=True); (tmp_path / 'out' / 'full' / 'model.ply').write_bytes(b'ply')
    monkeypatch.setattr(site_tail, 'build', lambda *a: (_ for _ in ()).throw(RuntimeError('npm is down')))
    assert site_tail.run(str(tmp_path), str(tmp_path / 'out'), '', '', '', 'R', 'tok', lambda *a: None, None) is None
    st = json.loads(api.files[('visitor/namma-inbox', 'j2/status.json')])
    assert st['stage'] == 'failed' and 'building the tour page failed' in st['reason'] and [h[1] for h in st['history']] == ['publishing', 'failed']


def test_the_job_script_calls_it_only_for_site_jobs():
    src = open(os.path.join(V3, 'run.py')).read()
    assert "if os.environ.get('SITE_JOB') == '1':" in src and "SOURCE.startswith('hf:')" in src
    pub = open(os.path.join(V3, 'tour_publish.py')).read(); assert "TOUR_KEEP_OTHER" in pub      # never deletes in a visitor's Space
