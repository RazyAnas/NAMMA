"""The tour page's view and quality settings (27 Nov): horizontal field of view, spot markers inside the room, the spot
repair on long walks, quality by device instead of the lowest setting for everyone. No GPU, no viewer, no downloads."""
import os, re, subprocess, sys, shutil
import numpy as np
import pytest

FIN = os.path.join(os.path.dirname(__file__), '..', '..', 'colab', 'real_scenes', 'final')
sys.path.insert(0, os.path.abspath(FIN))
import make_tour                                                                          # noqa: E402

HTML = '<html><head><title>SuperSplat Viewer</title></head><body></body></html>'


def test_fov_is_the_horizontal_angle_of_the_frames():
    # the switch room's lens: 1440 x 808 px, f = 1001.9 px -> 71.4 deg across, 43.9 deg up and down (the old, wrong number)
    assert abs(make_tour.lens_fov(1440, 1001.9343) - 71.4) < 0.1
    src = open(os.path.join(FIN, 'make_tour.py')).read()
    assert 'arctan(Hd / 2 / K[1, 1])' not in src and "fov_kind='horizontal'" in src       # plan() and tour.json say which


def test_look_target_stays_inside_the_room():
    lo, hi = np.array([-2.0, 0, -1.5]), np.array([2.0, 2.6, 1.5])
    assert make_tour.look_inside([0, 1.35, 0], [1, 0, 0], lo, hi, 1.25) == 1.25            # room for the whole distance
    assert abs(make_tour.look_inside([1.5, 1.35, 0], [1, 0, 0], lo, hi, 1.25) - 0.6) < 1e-9   # 0.5 m from the wall: the least
    assert abs(make_tour.look_inside([0.9, 1.35, 0], [1, 0, 0], lo, hi, 1.25) - 0.8) < 1e-9   # 0.3 m short of the wall
    d = np.array([0.6, -0.2, 0.77]); d /= np.linalg.norm(d); far = make_tour.look_inside([1.0, 1.35, 1.0], d, lo, hi, 1.25)
    p = np.array([1.0, 1.35, 1.0]) + far * d; assert lo[0] < p[0] < hi[0] and lo[2] < p[2] < hi[2]


def test_respot_accepts_both_smoothings_ever_used():
    src = open(os.path.join(FIN, 'make_tour.py')).read()
    assert "pl['smooth']" in src and 'max(2, min(6, n // 60))' in src and 'smooth=r' in src   # 900 frames: 15 before 1 Nov, 6 since


def head_script(levels):
    h = make_tour.head(HTML, 'cply', 2, levels); m = re.search(r'<script>(\(function\(\)\{var q=new URLSearchParams.*?\}\)\(\);)</script>', h, re.S)
    assert m and h.count('nammaQuality') >= 1 and make_tour.head(h, 'cply', 2, levels).count('var q=new URLSearchParams') == 1   # put in once
    return m.group(1)


RUN = r'''
const [script, ua, mem, cores, stored, query, session] = process.argv.slice(2);
const mk = (o) => ({ _: o, getItem(k) { return k in this._ ? this._[k] : null; }, setItem(k, v) { this._[k] = String(v); }, removeItem(k) { delete this._[k]; } });
globalThis.localStorage = mk(JSON.parse(stored)); globalThis.sessionStorage = mk(JSON.parse(session));
Object.defineProperty(globalThis, 'navigator', { value: { userAgent: ua, deviceMemory: +mem || undefined, hardwareConcurrency: +cores, maxTouchPoints: /Android|iPhone/.test(ua) ? 5 : 0 }, configurable: true });
globalThis.location = { search: query, pathname: '/t/index.html', hash: '' }; globalThis.window = globalThis; let url = '';
globalThis.history = { replaceState(a, b, u) { url = u; } };
(0, eval)(require('fs').readFileSync(script, 'utf8'));
console.log(JSON.stringify({ q: window.nammaQuality, ls: localStorage._, url }));
'''
PHONE = 'Mozilla/5.0 (Linux; Android 14; Pixel 6a) AppleWebKit/537.36 Chrome/130 Mobile Safari/537.36'
MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130 Safari/537.36'


def run_head(tmp_path, levels, ua, mem, cores, stored='{}', query='', session='{}'):
    import json
    js = tmp_path / 'head.js'; js.write_text(head_script(levels)); r = tmp_path / 'run.js'; r.write_text(RUN)
    out = subprocess.run(['node', str(r), str(js), ua, str(mem), str(cores), stored, query, session], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-600:]
    return json.loads(out.stdout)


@pytest.mark.skipif(not shutil.which('node'), reason='node not installed')
def test_quality_follows_the_device(tmp_path):
    a = run_head(tmp_path, False, PHONE, 4, 8)                                             # a 4 GB phone: low = half resolution
    assert a['q']['level'] == 'low' and a['ls']['performanceMode'] == 'true' and 'hpr=0' in a['url'] and a['ls']['gamingControls'] == 'true'
    b = run_head(tmp_path, False, PHONE, 8, 8)                                             # a strong phone: full resolution
    assert b['q']['level'] == 'balanced' and b['ls']['performanceMode'] == 'false' and 'hpr=1' in b['url']
    c = run_head(tmp_path, False, MAC, 8, 10)                                              # a laptop: never the lowest setting again
    assert c['q']['level'] == 'high' and c['ls']['performanceMode'] == 'false' and 'budget=2' in c['url'] and 'showcollision=0' in c['url']
    d = run_head(tmp_path, False, MAC, 0, 4); assert d['q']['level'] == 'balanced'          # small laptop, memory not reported
    # a viewer with quality levels (1.42): the level goes into 'quality'; performance mode (its softer renderer) stays off
    e = run_head(tmp_path, True, PHONE, 4, 4); assert e['ls']['quality'] == 'performance' and e['ls']['performanceMode'] == 'false' and e['q']['levels'] is True
    f = run_head(tmp_path, True, MAC, 8, 10); assert f['ls']['quality'] == 'high'


@pytest.mark.skipif(not shutil.which('node'), reason='node not installed')
def test_the_visitors_choice_wins_and_auto_remembers_a_slow_device(tmp_path):
    a = run_head(tmp_path, False, MAC, 8, 10, stored='{"nammaLevel":"low"}'); assert a['q'] == dict(pick='low', level='low', device='high', mobile=False, levels=False)
    b = run_head(tmp_path, False, PHONE, 4, 4, query='?q=high'); assert b['q']['level'] == 'high' and b['ls']['nammaLevel'] == 'high' and 'q=' not in b['url']
    c = run_head(tmp_path, False, MAC, 8, 10, session='{"nammaAuto":"low"}'); assert c['q']['pick'] == 'auto' and c['q']['level'] == 'low'   # the governor's step down
    d = run_head(tmp_path, False, MAC, 8, 10, stored='{"nammaLevel":"nonsense"}'); assert d['q']['pick'] == 'auto' and d['q']['level'] == 'high'
    e = run_head(tmp_path, False, MAC, 8, 10, stored='{"quality":"performance","performanceMode":"true","nammaQ":"4"}')   # a returning visitor
    assert e['ls']['performanceMode'] == 'false' and e['ls']['quality'] == 'high' and e['ls']['nammaQ'] == '5'


def test_viewer_levels_is_read_from_the_viewer_script():
    assert make_tour.viewer_levels("const k = ['quality', 'performanceMode']; const q = { performance: 1, 'default': 2, 'high': 3 };") is True
    assert make_tour.viewer_levels("state.performanceMode ? 0.5 : 1; // performance") is False


@pytest.mark.skipif(not shutil.which('node'), reason='node not installed')
def test_overlay_script_still_parses(tmp_path):
    h = make_tour.overlay(HTML); m = re.search(r'<script type="module">(.*?)</script>', h, re.S); assert m and 'Picture quality' in m.group(1) and 'nammaEye2' in m.group(1)
    f = tmp_path / 'o.mjs'; f.write_text(m.group(1)); r = subprocess.run(['node', '--check', str(f)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]
