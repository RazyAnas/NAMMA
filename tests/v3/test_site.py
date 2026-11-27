"""The website (site/) and its deploy script (scripts/site_deploy.py). Cheap checks, no network, no browser:
every JS file parses (node --check, when node is installed), the page loads only its own local files, no HTML is ever built
from a string that is not a literal (sign-in and Kaggle keys live in the page), and the gallery / deploy logic against a fake Hub."""
import importlib.util, json, re, shutil, subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'site'
JS = sorted((SITE / 'js').rglob('*.js'))
OWN_JS = [SITE / 'js' / n for n in ('app.js', 'ui.js', 'pages/home.js', 'pages/explore.js', 'pages/library.js', 'pages/create.js', 'pages/job.js')]

spec = importlib.util.spec_from_file_location('site_deploy', ROOT / 'scripts' / 'site_deploy.py')
site_deploy = importlib.util.module_from_spec(spec); spec.loader.exec_module(site_deploy)


# ------------------------------------------------------------------------------------------------------------- the page
def test_the_files_are_there():
    for f in OWN_JS + [SITE / 'index.html', SITE / 'css' / 'site.css', SITE / 'gallery.src.json', SITE / 'README.md']:
        assert f.is_file() and f.stat().st_size > 0, f


@pytest.mark.parametrize('js', JS, ids=lambda p: str(p.relative_to(SITE)))
def test_js_parses(js, tmp_path):
    node = shutil.which('node')
    if not node: pytest.skip('node is not installed')
    f = tmp_path / (js.stem + '.mjs'); f.write_text(js.read_text(encoding='utf-8'), encoding='utf-8')      # .mjs: checked as an ES module
    r = subprocess.run([node, '--check', str(f)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]


class Refs(HTMLParser):
    """Every address index.html loads something from (links a person clicks, <a href>, are kept apart)."""
    def __init__(self):
        super().__init__(); self.loads, self.links, self.inline, self.csp, self.tag = [], [], [], '', None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs); self.tag = tag
        if tag == 'a': self.links.append(a.get('href') or ''); return
        if tag == 'meta' and (a.get('http-equiv') or '').lower() == 'content-security-policy': self.csp = a.get('content') or ''
        if tag == 'script' and 'src' not in a: self.inline.append('script')
        if tag == 'style': self.inline.append('style')
        self.inline += [k for k in a if k.startswith('on')]
        for k in ('src', 'href', 'poster', 'data', 'action', 'srcset'):
            if a.get(k) and not (tag == 'use' and a[k].startswith('#')): self.loads.append((tag, a[k]))


def page():
    p = Refs(); p.feed((SITE / 'index.html').read_text(encoding='utf-8')); return p


def test_index_loads_only_local_files():
    p = page()
    assert p.loads, 'index.html loads its stylesheet and script'
    for tag, url in p.loads:
        assert not re.match(r'^([a-z][a-z0-9+.-]*:|//|/)', url, re.I), f'<{tag}> {url}: not a relative local address'
        assert (SITE / url.split('#')[0].split('?')[0]).is_file(), f'<{tag}> {url}: file missing'
    assert not p.inline, f'inline script, style or on* attribute in index.html: {p.inline} (blocked by the page policy)'
    for href in p.links:                                         # links for people: in-page routes, or the project's own repository
        assert href.startswith('#') or href.startswith('https://github.com/RazyAnas/NAMMA'), href


def test_page_policy():
    csp = dict((part.split()[0], part.split()[1:]) for part in page().csp.split(';') if part.strip())
    assert csp['default-src'] == ["'self'"] and csp['script-src'] == ["'self'"], 'only the site runs code'
    assert set(csp['connect-src']) == {"'self'", 'https://huggingface.co', 'https://*.huggingface.co', 'https://*.hf.co', 'https://*.hf.space', 'https://*.amazonaws.com',
                                       'https://www.kaggle.com', 'https://api.kaggle.com', 'https://storage.googleapis.com'}
    assert csp['frame-src'] == ['https://*.hf.space'] and "'unsafe-eval'" not in page().csp
    html = (SITE / 'index.html').read_text(encoding='utf-8')
    assert html.index('Content-Security-Policy') < html.index('<script'), 'the policy comes before the first script'


def test_css_and_js_reach_nothing_outside():
    css = (SITE / 'css' / 'site.css').read_text(encoding='utf-8')
    assert '@import' not in css and not re.search(r'url\(\s*["\']?(https?:|//)', css, re.I), 'no external fonts or images in the CSS'
    for js in JS:
        src = js.read_text(encoding='utf-8')
        for m in re.finditer(r'''(?:\bfrom\s*|\bimport\s*\(\s*|^\s*import\s+)(['"])([^'"]+)\1''', src, re.M):
            target = m.group(2)
            assert target.startswith('.'), f'{js.name}: import of {target} (only local modules)'
            if js in OWN_JS and '/api/' in target: continue      # written by the API side; loaded on demand, see below
            assert (js.parent / target).resolve().is_file(), f'{js.name}: {target} does not exist'


def test_api_modules_are_loaded_on_demand():
    """The pages must open while site/js/api/*.js is missing or broken: no static import of them anywhere in the front end."""
    for js in OWN_JS:
        src = js.read_text(encoding='utf-8')
        assert not re.search(r'''^\s*import\s[^;]*?from\s*['"][^'"]*/api/''', src, re.M), f'{js.name}: static import of an api module'
    app = (SITE / 'js' / 'app.js').read_text(encoding='utf-8')
    assert 'await import(path)' in app and all(f"'./api/{n}.js'" in app for n in ('hf', 'jobs', 'kaggle'))


HTML_SINKS = re.compile(r'\.(innerHTML|outerHTML)\s*\+?=\s*(?P<a>[^;\n]*)|\.insertAdjacentHTML\s*\(\s*[^,]+,\s*(?P<b>[^;\n]*)|document\.write(ln)?\s*\((?P<c>[^;\n]*)')
LITERAL = re.compile(r'''^\s*(''|""|'[^'<>]*'|"[^"<>]*")\s*\)?\s*$''')


def html_sinks(src):
    """[(line, code)] where HTML is built from something that is not a plain, tag-free string literal."""
    src = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), src, flags=re.S)
    bad = []
    for m in HTML_SINKS.finditer(src):
        value = m.group('a') if m.group('a') is not None else m.group('b') if m.group('b') is not None else m.group('c')
        line = src.count('\n', 0, m.start()) + 1; text = src.splitlines()[line - 1].strip()
        if text.startswith('//'): continue
        if not LITERAL.match(value or ''): bad.append((line, text[:160]))
    return bad


def test_html_sink_scanner():
    assert html_sinks('el.innerHTML = `<b>${name}</b>`;') and html_sinks('el.innerHTML = a + b;') and html_sinks('x.insertAdjacentHTML("beforeend", s)')
    assert html_sinks('el.outerHTML = html') and html_sinks("document.write(x)") and html_sinks("el.innerHTML += '<i>' + t")
    assert not html_sinks("el.innerHTML = '';") and not html_sinks('el.textContent = name;') and not html_sinks('// el.innerHTML = x')


@pytest.mark.parametrize('js', JS, ids=lambda p: str(p.relative_to(SITE)))
def test_no_html_from_strings(js):
    src = js.read_text(encoding='utf-8')
    assert not html_sinks(src), f'{js.name}: HTML built from a string (use textContent / createElement): {html_sinks(src)}'
    assert not re.search(r'\beval\s*\(|new\s+Function\s*\(|\bsrcdoc\s*=', src), f'{js.name}: eval / new Function / srcdoc'


def test_tours_open_in_a_new_tab_never_inside_the_page():
    ui = (SITE / 'js' / 'ui.js').read_text(encoding='utf-8')
    assert "target: '_blank', rel: 'noopener noreferrer'" in ui, 'extLink opens a new tab without an opener'
    for js in OWN_JS:
        src = js.read_text(encoding='utf-8')
        assert not re.search(r'''h\(\s*['"](iframe|embed|object)['"]|createElement\(\s*['"](iframe|embed|object)['"]''', src), f'{js.name}: a tour inside the page'
        for m in re.finditer(r"target:\s*'_blank'(?P<rest>[^}]*)", src): assert 'noopener' in m.group('rest'), f'{js.name}: new tab without noopener'


def test_kaggle_key_field():
    src = (SITE / 'js' / 'pages' / 'create.js').read_text(encoding='utf-8')
    field = re.search(r"const key = h\('input', \{([^}]*)\}", src).group(1)
    assert "type: 'password'" in field and "autocomplete: 'off'" in field
    app = (SITE / 'js' / 'app.js').read_text(encoding='utf-8')
    assert re.search(r'if \(remember\) deviceStore\.set\(KAGGLE_KEY, v\);\s*else deviceStore\.del\(KAGGLE_KEY\)', app), 'kept on the device only when asked'


def test_space_card_and_curated_list():
    readme = (SITE / 'README.md').read_text(encoding='utf-8'); assert readme.startswith('---\n')
    head = readme.split('---\n')[1]
    for line in ('title: NAMMA', 'colorFrom: red', 'colorTo: yellow', 'sdk: static', 'pinned: true', 'hf_oauth: true', 'hf_oauth_expiration_minutes: 480'):
        assert re.search(rf'^{re.escape(line)}$', head, re.M), line
    assert re.search(r'^emoji: \S+$', head, re.M)
    scopes = re.search(r'^hf_oauth_scopes:\n((?:\s+- \S+\n)+)', head, re.M).group(1).split()
    assert [s for s in scopes if s != '-'] == ['read-repos', 'write-repos', 'manage-repos', 'write-discussions']
    src = json.loads((SITE / 'gallery.src.json').read_text(encoding='utf-8'))
    assert src['spaces'] == ['namma-tour', 'namma-tour-2'] and len(src['titles']) == 12
    assert src['titles']['20261031-1528'] == dict(title='Whole house', kind='house', public=True)
    assert src['titles']['20261027-1849-world']['kind'] == 'experiment'
    assert all(site_deploy.created(run) for run in src['titles']), 'every run id starts with its date'


def test_plain_text_for_people():
    """Owner rule: plain language, no em-dash marketing text."""
    for f in OWN_JS + [SITE / 'index.html', SITE / 'README.md']:
        assert '—' not in f.read_text(encoding='utf-8'), f'{f.name}: em dash'
    note = 'Link-only tours sit in a public Space with a random folder name. Anyone who knows how can list the folder names, so do not treat link-only as private.'
    assert note in (SITE / 'js' / 'pages' / 'home.js').read_text(encoding='utf-8')


# ---------------------------------------------------------------------------------------------------- gallery and deploy
HEX = 'a1b2c3d4e5f60718293a'                                              # 20 hex characters, as tour_publish adds them
PLY = b'ply\nformat binary_little_endian 1.0\nelement chunk 12\nproperty float min_x\nelement vertex 345678\nproperty uint packed_position\nend_header\n' + b'\0' * 64


class FakeHub:
    """list_repo_tree / list_repo_files / create_repo / upload_folder of huggingface_hub.HfApi over a dict {space: {path: bytes}}."""
    def __init__(self, spaces, dates=None):
        self.spaces, self.dates, self.calls, self.reads, self.uploaded = spaces, dates or {}, [], [], None

    def whoami(self): return dict(name='Some.User')

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, expand=False, repo_type=None):
        assert repo_type == ('dataset' if repo_id.endswith('/namma-gallery') else 'space') and not recursive
        if repo_id not in self.spaces: raise FileNotFoundError(repo_id)
        base = (path_in_repo.rstrip('/') + '/') if path_in_repo else ''; seen = {}
        for path, data in self.spaces[repo_id].items():
            if not path.startswith(base): continue
            head, _, rest = path[len(base):].partition('/')
            if rest: seen[base + head] = SimpleNamespace(path=base + head, tree_id='t', last_commit=SimpleNamespace(date=self.dates.get(head)) if expand else None)
            else: seen[base + head] = SimpleNamespace(path=base + head, size=len(data), last_commit=None)
        return iter(seen.values())

    def read(self, repo_id, path, limit=None, repo_type='space'):
        assert repo_type == ('dataset' if repo_id.endswith('/namma-gallery') else 'space')
        self.reads.append((path, limit)); data = self.spaces[repo_id][path]
        return data[:limit] if limit else data

    def list_repo_files(self, repo_id, repo_type=None): return list(self.spaces.get(repo_id, {}))

    def create_repo(self, repo_id, **kw): self.calls.append(('create_repo', repo_id, kw)); self.spaces.setdefault(repo_id, {'.gitattributes': b'', 'README.md': b''})

    def update_repo_settings(self, repo_id, **kw): self.calls.append(('settings', repo_id, kw))

    def add_space_variable(self, repo_id, key, value): self.calls.append(('variable', repo_id, key, value))

    def upload_folder(self, folder_path, repo_id, **kw):
        self.calls.append(('upload_folder', repo_id, kw))
        self.uploaded = {str(p.relative_to(folder_path)): p.read_bytes() for p in Path(folder_path).rglob('*') if p.is_file()}


def tour(folder, thumbs=3, overlay=True, extra=None):
    f = {f'{folder}/index.html': b'<!doctype html>', f'{folder}/settings.json': b'{}', f'{folder}/floorplan.png': b'png',
         f'{folder}/tour.json': json.dumps(dict(room_box=[[-1.24, -0.1, 0.0], [4.96, 2.6, 3.72]], spots=thumbs, scene_mb=41.26, format='cply')).encode()}
    for n in range(thumbs): f[f'{folder}/thumbs/spot{n + 1}.jpg'] = b'jpg'
    if overlay: f[f'{folder}/overlay.json'] = json.dumps(dict(spots=[dict(title=f'Spot {n + 1}', thumb=f'thumbs/spot{n + 1}.jpg') for n in range(thumbs)])).encode()
    f.update({f'{folder}/{k}': v for k, v in (extra or {}).items()}); return f


@pytest.fixture
def hub():
    a, b, old = f'20261025-2336-{HEX}', f'20261031-1528-{HEX}', '20261031-1528-' + '0' * 20      # the house was published twice
    world, hidden, new = f'20261027-1849-world-{HEX}', f'20261026-0309-{HEX}', f'20261201-0900-{HEX}'
    one = {'index.html': b'x', 'README.md': b'x', 'editor/index.html': b'x', 'not-a-tour/index.html': b'x', '20261025-2336-short/index.html': b'x'}
    one.update(tour(a, extra={'scene.compressed.ply': PLY + b'\1' * 5000}))
    one.update(tour(old)); one.update(tour(b, thumbs=12, overlay=False, extra={'scene.sog': b's' * 2_500_000,
               'walls.json': json.dumps(dict(source='walls', outline=[[0, 0], [5.6, 0], [5.6, 3.5], [0, 3.5]])).encode()}))
    one.update({f'{world}/index.html': b'x', f'{world}/input.jpg': b'jpg', f'{world}/world_sheet.jpg': b'jpg'})
    one.update(tour(hidden)); one.update({f'20261101-0000-{HEX}/tour.json': b'{}'})
    two = tour(new, thumbs=0, overlay=False); two[f'{new}/tour.json'] = b'not json'
    dates = {b: datetime(2026, 11, 2, tzinfo=timezone.utc), old: datetime(2026, 10, 31, tzinfo=timezone.utc)}
    return FakeHub({'Some.User/namma-tour': one, 'Some.User/namma-tour-2': two}, dates)


SRC = dict(spaces=['namma-tour', 'namma-tour-2', 'namma-tour-3'],
           titles={'20261025-2336': dict(title='Hotel room, first tour', kind='room', public=True), '20261031-1528': dict(title='Whole house', kind='house', public=True),
                   '20261027-1849-world': dict(title='Generated world', kind='experiment', public=True), '20261026-0309': dict(title='Hidden', kind='room', public=False)})


def test_gallery_from_the_tour_spaces(hub):
    said = []
    g = site_deploy.build_gallery(hub, hub.read, 'Some.User', SRC, say=lambda *a: said.append(' '.join(map(str, a))), now=datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc))
    assert g['generated'] == '2026-11-27T12:00:00Z'
    e = {x['run']: x for x in g['entries']}
    assert [x['run'] for x in g['entries']] == ['20261201-0900', '20261031-1528', '20261027-1849-world', '20261025-2336'], 'newest first, one per run, hidden and non-tours left out'
    host = 'https://some-user-namma-tour.static.hf.space'
    first = e['20261025-2336']
    assert first == dict(id=f'20261025-2336-{HEX}', run='20261025-2336', title='Hotel room, first tour', kind='room', created='2026-10-25T23:36:00',
                         url=f'{host}/20261025-2336-{HEX}/index.html', cover=f'{host}/20261025-2336-{HEX}/thumbs/spot1.jpg',
                         splats=345678, area_m2=15.0, size_mb=41.3)
    house = e['20261031-1528']
    assert house['id'] == f'20261031-1528-{HEX}', 'of two folders of one run, the one committed last'
    assert house['cover'].endswith('/thumbs/spot1.jpg'), 'without overlay.json: the first thumbnail in natural order (spot1, not spot10)'
    assert house['area_m2'] == 19.6 and 'splats' not in house, 'area from the measured outline; no count for a .sog'
    world = e['20261027-1849-world']
    assert world['kind'] == 'experiment' and world['cover'].endswith('/input.jpg') and 'area_m2' not in world
    new = e['20261201-0900']
    assert new['title'] == 'Tour 2026-12-01' and new['kind'] == 'tour' and new['cover'] is None and new['url'].startswith('https://some-user-namma-tour-2.static.hf.space/')
    assert any('namma-tour-3' in s and 'skipped' in s for s in said), 'a tour Space that does not exist is skipped, not fatal'
    big = [p for p, limit in hub.reads if p.endswith(('.ply', '.sog')) and (limit is None or limit > 4096)]
    assert not big, f'the scene file is never read whole: {big}'


def test_submitted_tours_join_the_gallery(hub):
    """entries/*.json of <user>/namma-gallery (merged pull requests from jobs.submitToGallery): listed after a check of every field."""
    good = dict(user='ana', id='20261120-101500-ab12', title='Our kitchen\n::error::x', kind='tour', created='2026-11-20T10:15:00.000Z',
                url='https://ana-namma-tour.static.hf.space/20261120-1015-' + HEX + '/index.html', cover='javascript:alert(1)')
    hub.spaces['Some.User/namma-gallery'] = {'README.md': b'x', 'entries/ana--1.json': json.dumps(good).encode(), 'entries/bad--2.json': json.dumps(dict(good, url='http://x/')).encode(),
                                             'entries/bad--3.json': b'[1, 2', 'entries/note.txt': b'x'}
    g = site_deploy.build_gallery(hub, hub.read, 'Some.User', SRC, say=lambda *a: None)
    mine = [e for e in g['entries'] if e.get('by')]
    assert mine == [dict(id='ana--20261120-101500-ab12', run=None, title='Our kitchen ::error::x', kind='tour', created='2026-11-20T10:15:00.000Z', url=good['url'], cover=None, by='ana')]
    assert [e['run'] for e in g['entries']] == ['20261201-0900', None, '20261031-1528', '20261027-1849-world', '20261025-2336'], 'sorted in with the rest by date'


def test_dry_run_writes_only_the_gallery(hub, tmp_path, capsys):
    g = site_deploy.main(['--dry-run', '--out', str(tmp_path / 'out'), '--user', 'Some.User'], api=hub, read=hub.read)
    written = json.loads((tmp_path / 'out' / 'gallery.json').read_text(encoding='utf-8'))
    assert written == g and len(written['entries']) == 5, 'gallery.json holds the entries (site/gallery.src.json shows all five runs of the fake Hub)'
    assert [e['title'] for e in written['entries']][1:] == ['Whole house', 'Generated world', 'Hotel room, second build', 'Hotel room, first tour']
    assert not hub.calls and hub.uploaded is None, 'nothing created, nothing uploaded'
    assert 'dry run' in capsys.readouterr().out


def test_deploy_uploads_the_site_with_its_gallery(hub):
    g = dict(generated='now', entries=[dict(id='x', url='https://a.static.hf.space/x/index.html')])
    repo = site_deploy.deploy(hub, 'Some.User', str(SITE), g, say=lambda *a: None)
    assert repo == 'Some.User/namma'
    create = next(c for c in hub.calls if c[0] == 'create_repo')
    assert create[2] == dict(repo_type='space', space_sdk='static', private=False, exist_ok=True)
    up = next(c for c in hub.calls if c[0] == 'upload_folder')
    assert up[1] == 'Some.User/namma' and up[2]['repo_type'] == 'space' and up[2]['delete_patterns'] == '*'
    files = hub.uploaded
    assert json.loads(files['gallery.json']) == g
    for name in ('index.html', 'README.md', 'gallery.src.json', 'css/site.css', 'js/app.js', 'js/ui.js', 'js/pages/job.js'): assert name in files, name
    assert not [n for n in files if n.endswith('.DS_Store') or n == 'api.md'], 'only the page itself is published'
    assert ('variable', 'Some.User/namma', 'SITE_OWNER', 'Some.User') in hub.calls, 'the page is told whose gallery dataset takes submissions'
    assert not [c for c in hub.calls if c[0] == 'create_repo' and 'gallery' in c[1]], 'the gallery dataset is created only when asked (--make-gallery-repo)'


def test_deploy_can_add_a_built_viewer(hub, tmp_path):
    v = tmp_path / 'viewer'; v.mkdir(); (v / 'index.html').write_text('<!doctype html>'); (v / 'index.js').write_text('1')
    site_deploy.deploy(hub, 'Some.User', str(SITE), dict(generated='now', entries=[]), viewer=str(v), say=lambda *a: None)
    assert 'viewer/index.html' in hub.uploaded and 'viewer/index.js' in hub.uploaded and 'index.html' in hub.uploaded
    site_deploy.deploy(hub, 'Some.User', str(SITE), dict(generated='now', entries=[]), say=lambda *a: None)
    assert not [n for n in hub.uploaded if n.startswith('viewer/')], 'no viewer unless one is given'


def test_deploy_never_wipes_a_space_that_is_something_else(hub):
    hub.spaces['Some.User/namma'] = {'README.md': b'x', 'app.py': b'print(1)'}
    with pytest.raises(SystemExit): site_deploy.deploy(hub, 'Some.User', str(SITE), dict(generated='now', entries=[]), say=lambda *a: None)
    assert not [c for c in hub.calls if c[0] == 'upload_folder']
    site_deploy.deploy(hub, 'Some.User', str(SITE), dict(generated='now', entries=[]), force=True, say=lambda *a: None)
    assert [c for c in hub.calls if c[0] == 'upload_folder']


def test_names():
    assert site_deploy.subdomain('Some.User/namma_tour') == 'some-user-namma-tour'
    assert site_deploy.run_id(f'20261027-1849-world-{HEX}') == '20261027-1849-world'
    assert site_deploy.FOLDER_RE.match(f'20261027-1849-world-{HEX}') and not site_deploy.FOLDER_RE.match('20261027-1849-world')
    assert site_deploy.created('20261126-1119') == '2026-11-26T11:19:00' and site_deploy.created('world1') is None
