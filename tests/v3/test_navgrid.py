"""Walkable grid of a tour (colab/real_scenes/final/navgrid.py) and the places plumbing around it (make_tour, tour_publish).
A synthetic flat in room metres: 8 x 5 m, a wall at x = 4 with a 0.9 m doorway, a sofa in room A, a walled-in cupboard in
room B, and (second scene) a room 0.4 m higher reached by two steps. No GPU, no network."""
import json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'colab' / 'real_scenes' / 'final'
sys.path.insert(0, str(FINAL)); sys.path.insert(0, str(ROOT / 'colab' / 'real_scenes' / 'v3'))
import navgrid  # noqa: E402

trimesh = pytest.importorskip('trimesh')
BOX = [0.0, -0.1, 0.0, 8.0, 2.6, 5.0]


def box(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float); T = np.eye(4); T[:3, 3] = (lo + hi) / 2
    return trimesh.creation.box(extents=hi - lo, transform=T)


def flat(extra=()):
    parts = [box([-0.5, -0.1, -0.5], [8.5, 0.0, 5.5]),                                     # floor slab, top at y = 0
             box([-0.1, 0, -0.1], [0.0, 2.6, 5.1]), box([8.0, 0, -0.1], [8.1, 2.6, 5.1]),   # outer walls
             box([-0.1, 0, -0.1], [8.1, 2.6, 0.0]), box([-0.1, 0, 5.0], [8.1, 2.6, 5.1]),
             box([3.95, 0, 0.0], [4.05, 2.6, 2.0]), box([3.95, 0, 2.9], [4.05, 2.6, 5.0]),   # inner wall, doorway z = 2.0 .. 2.9
             box([1.0, 0, 3.6], [2.6, 0.8, 4.4]),                                           # sofa in room A
             box([6.0, 0, 0.5], [7.2, 2.6, 0.55]), box([6.0, 0, 1.45], [7.2, 2.6, 1.5]),     # cupboard: four walls, floor inside
             box([6.0, 0, 0.5], [6.05, 2.6, 1.5]), box([7.15, 0, 0.5], [7.2, 2.6, 1.5])] + list(extra)
    return np.asarray(trimesh.util.concatenate(parts).triangles, float)


WALK = np.array([[1.0, 1.35, 1.0], [2.5, 1.35, 2.4], [4.0, 1.35, 2.45], [5.5, 1.35, 2.5], [7.0, 1.35, 4.0]])   # A -> doorway -> B


@pytest.fixture(scope='module')
def nav():
    return navgrid.build(flat(), BOX, WALK)


def at(nav, x, z):
    ix, iz = navgrid.cell_of(nav, x, z); return bool(nav['grid'][iz, ix])


def test_doorway_open_walls_closed(nav):
    assert at(nav, 4.0, 2.45), 'the doorway is walkable'
    assert not at(nav, 4.0, 1.0) and not at(nav, 4.0, 4.0), 'the wall is not'
    assert not at(nav, 4.0, 2.1) and not at(nav, 4.0, 2.85), 'a body does not fit against the door frame'
    assert not at(nav, 0.15, 2.5) and at(nav, 0.45, 2.5), 'the 0.25 m body radius keeps off the outer wall'
    assert not at(nav, -0.3, 2.5), 'outside the room box'


def test_both_rooms_reachable_from_the_walk(nav):
    a, b = navgrid.cell_of(nav, 1.0, 1.0), navgrid.cell_of(nav, 7.0, 4.0)
    assert navgrid.reachable(nav, a, b)
    assert navgrid.reachable(nav, a, navgrid.cell_of(nav, 0.5, 4.6)), 'a corner of room A nobody filmed'
    from scipy import ndimage
    assert ndimage.label(nav['grid'])[1] == 1, 'one connected piece (4-connected, as the browser A*)'


def test_sealed_cupboard_is_left_out(nav):
    assert not at(nav, 6.6, 1.0), 'floor inside four walls: never walkable'
    assert not navgrid.reachable(nav, navgrid.cell_of(nav, 1.0, 1.0), navgrid.cell_of(nav, 6.6, 1.0))


def test_furniture_blocks_and_its_top_is_not_floor(nav):
    assert not at(nav, 1.8, 4.0), 'the sofa'
    assert not at(nav, 0.85, 4.0), 'nor within the body radius of it'
    assert at(nav, 1.8, 3.2), 'the floor in front of it'
    assert np.nanmax(nav['floor']) < 0.1, 'nothing 0.8 m up counts as floor'


def test_filmed_path_is_always_walkable():
    walk = np.array([[0.6, 1.35, 3.0], [1.35, 1.35, 3.9], [0.6, 1.35, 4.7]])       # the smoothed track clips the sofa's corner
    nav = navgrid.build(flat(), BOX, walk)
    for x, _, z in navgrid.dense(walk, 0.02): assert at(nav, x, z), (x, z)
    from scipy import ndimage
    assert ndimage.label(nav['grid'])[1] == 1, 'the forced cells join the rest in one piece'
    assert not at(nav, 2.2, 3.75) and not at(nav, 1.7, 4.0), 'the rest of the sofa still blocks'
    assert np.nanmax(nav['floor']) < 0.1, 'and its top did not become floor'


def test_place_inside_furniture_snaps_beside_it(nav):
    x, z = navgrid.snap(nav, 1.8, 4.0); assert at(nav, x, z)
    assert 0.3 < np.hypot(x - 1.8, z - 4.0) < 0.9, 'beside the sofa, not across the room'
    assert navgrid.snap(nav, 1.8, 4.0, max_m=0.2) is None
    assert navgrid.snap(nav, 1.02, 1.02) == pytest.approx((1.05, 1.05), abs=1e-6), 'already on the floor: its own cell centre'


def test_raised_room_by_a_low_step_is_floor_a_table_is_not():
    # room B's floor is 0.3 m up; one 0.15 m step in the doorway leads to it. A 0.45 m table stands in room A.
    extra = [box([4.05, 0, 0.0], [8.0, 0.3, 5.0]), box([3.4, 0, 2.0], [4.05, 0.15, 2.9]), box([2.2, 0, 0.6], [3.2, 0.45, 1.4])]
    walk = np.array([[1.0, 1.35, 2.45], [3.2, 1.35, 2.45], [3.7, 1.5, 2.45], [4.5, 1.65, 2.45], [7.0, 1.65, 3.5]])
    nav = navgrid.build(flat(extra), BOX, walk); f = nav['floor']
    ix, iz = navgrid.cell_of(nav, 7.0, 3.5); assert nav['grid'][iz, ix] and f[iz, ix] == pytest.approx(0.3, abs=0.06)
    ix, iz = navgrid.cell_of(nav, 5.0, 4.5); assert nav['grid'][iz, ix] and f[iz, ix] == pytest.approx(0.3, abs=0.06), 'unfilmed part of the raised room'
    ix, iz = navgrid.cell_of(nav, 1.0, 3.2); assert nav['grid'][iz, ix] and f[iz, ix] == pytest.approx(0.0, abs=0.06)
    ix, iz = navgrid.cell_of(nav, 3.7, 2.45); assert nav['grid'][iz, ix] and f[iz, ix] == pytest.approx(0.15, abs=0.06), 'the step'
    assert not at(nav, 2.7, 1.0), 'a 0.45 m table top is not a step'
    assert navgrid.reachable(nav, navgrid.cell_of(nav, 1.0, 2.45), navgrid.cell_of(nav, 5.0, 4.5))


def test_stairs_are_walkable_where_filmed_and_levels_follow_the_camera():
    # an upper floor 2.4 m up (a slab over half the flat) and a flight of 0.2 m risers the grid cannot climb by itself
    extra = [box([4.05, 2.3, 0.0], [8.0, 2.4, 5.0])] + [box([1.0 + 0.25 * i, 0, 2.2], [1.25 + 0.25 * i, 0.2 * (i + 1), 2.7]) for i in range(11)]
    walk = np.array([[0.6, 1.35, 2.45]] + [[1.125 + 0.25 * i, 1.35 + 0.2 * (i + 1), 2.45] for i in range(11)] + [[4.6, 3.75, 2.45], [7.0, 3.75, 3.5]])
    b2 = BOX[:4] + [4.2] + BOX[5:]; nav = navgrid.build(flat(extra), b2, walk); f = nav['floor']
    for x, _, z in navgrid.dense(walk, 0.02): assert at(nav, x, z), (x, z)
    ix, iz = navgrid.cell_of(nav, 7.0, 3.5); assert f[iz, ix] == pytest.approx(2.4, abs=0.08), 'upstairs is at the upper level'
    ix, iz = navgrid.cell_of(nav, 6.0, 4.5); assert nav['grid'][iz, ix] and f[iz, ix] == pytest.approx(2.4, abs=0.08), 'and its unfilmed floor too'
    ix, iz = navgrid.cell_of(nav, 0.6, 2.45); assert f[iz, ix] == pytest.approx(0.0, abs=0.06)
    assert navgrid.reachable(nav, navgrid.cell_of(nav, 0.6, 2.45), navgrid.cell_of(nav, 6.0, 4.5)), 'joined through the filmed stairs'


def test_nav_json_round_trip_and_browser_reads_it(nav, tmp_path):
    d = navgrid.encode(nav); back = navgrid.decode(json.loads(json.dumps(d)))
    assert np.array_equal(back['grid'], nav['grid']) and back['w'] == nav['w'] and back['cell'] == 0.1
    assert np.allclose(np.nan_to_num(back['floor']), np.nan_to_num(nav['floor']), atol=0.03)
    assert d['walkable'] == int(nav['grid'].sum()) and len(d['grid']) < 4000, 'a few kB at most'
    node = shutil.which('node')
    if not node: pytest.skip('node is not installed: the browser side is covered by tests/browser/poi_e2e.mjs')
    (tmp_path / 'nav.json').write_text(json.dumps(d)); js = str(ROOT / 'tests' / 'browser' / 'poi_e2e.mjs')
    run = lambda a, b: json.loads(subprocess.run([node, js, '--nav', str(tmp_path / 'nav.json'), '--from', a, '--to', b], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    r = run('1.0,1.0', '7.0,4.0')
    assert r['ok'] and r['onFloor'] and r['walkable'] == d['walkable'] and [r['w'], r['h']] == [nav['w'], nav['h']]
    assert np.hypot(6, 3) <= r['length'] < 1.4 * np.hypot(6, 3), 'through the doorway, not around the world'
    assert run('1.0,1.0', '6.6,1.0')['ok'] is True, 'a place in the cupboard is reached from outside it (approach point)'
    assert run('1.0,1.0', '30,30')['reason'] == 'target-off-floor'


def test_morton_keys_match_the_browser():
    want = {(0, 0, 0): 939524096, (1, 0, 0): 939524097, (0, 1, 0): 939524098, (0, 0, 1): 939524100, (-1, -1, -1): 134217727,
            (3, 5, 7): 939524527, (-12, 2, 40): 824612944, (511, 511, 511): 1073741823, (-512, -512, -512): 0}     # also in poi_e2e.mjs
    for c, k in want.items(): assert navgrid.morton3(*c) == k
    assert navgrid.poi_cell([1.26, 0.9, -3.4]) == 479348750 and navgrid.poi_cell([-0.01, 0.0, 0.01]) == 824480329
    assert len({navgrid.morton3(x, y, z) for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)}) == 343, 'no two cells share a key'


def tour_dir(tmp_path, collision=True):
    t = tmp_path / 'tour'; t.mkdir()
    (t / 'plan.json').write_text(json.dumps(dict(rotate=[0, 0, 0], translate=[0, 0, 0], scale=1.0, box=BOX)))
    pos = [float(v) for p in navgrid.dense(WALK, 0.3) for v in p]
    (t / 'settings.json').write_text(json.dumps(dict(animTracks=[dict(keyframes=dict(values=dict(position=pos)))], cameras=[dict(initial=dict(position=[1.0, 1.35, 1.0]))])))
    if collision: trimesh.Trimesh(*_mesh()).export(t / 'collision.glb')
    return t


def _mesh():
    m = trimesh.util.concatenate([trimesh.Trimesh(vertices=tr, faces=[[0, 1, 2]]) for tr in flat()]); return m.vertices, m.faces


def test_make_tour_navgrid_writes_and_degrades(tmp_path, capsys):
    import make_tour
    t = tour_dir(tmp_path); info = make_tour.navgrid(str(t))
    assert info and (t / 'nav.json').exists() and 12 < info['walkable_m2'] < 40
    before = (t / 'collision.glb').read_bytes(); make_tour.navgrid(str(t)); assert (t / 'collision.glb').read_bytes() == before, 'collision is never changed'
    nav = navgrid.decode(json.loads((t / 'nav.json').read_text())); assert at(nav, 4.0, 2.45) and not at(nav, 6.6, 1.0)
    # no collision file: no nav.json (a stale one goes), no exception
    (t / 'collision.glb').unlink(); assert make_tour.navgrid(str(t)) is None and not (t / 'nav.json').exists()
    # a broken collision file: the tour still builds
    (t / 'collision.glb').write_bytes(b'not a glb'); assert make_tour.navgrid(str(t)) is None and not (t / 'nav.json').exists()
    assert 'no routes' in capsys.readouterr().out


def test_overlay_carries_the_places_code_and_is_valid_js(tmp_path):
    import make_tour
    o = make_tour.OVERLAY; a = o.index('<script type="module">') + len('<script type="module">'); b = o.rindex('</script>'); js = o[a:b]
    assert '/*NM_POI*/' not in o and 'class SpatialIndex' in js and 'window.nammaPoi' in js and 'You have reached your destination' in js
    assert not [ln for ln in js.splitlines() if ln.startswith('export ')], 'export words must be dropped inside the page script'
    assert o.count('</script>') == 1 and o.count('id="nm-plan"') == 1
    html = '<html><head></head><body><div id="app"></div></body></html>'; once = make_tour.overlay(html); assert make_tour.overlay(once) == once, 'putting it in twice changes nothing'
    node = shutil.which('node')
    if not node: pytest.skip('node is not installed')
    f = tmp_path / 'overlay.mjs'; f.write_text(js); r = subprocess.run([node, '--check', str(f)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]


def test_publish_keeps_live_places(tmp_path):
    import tour_publish
    live = dict(version=1, updated='2026-11-10T00:00:00.000Z', pois=[dict(id='a', name='Kitchen (renamed by the admin)', pos=[1, 1, 1], source='admin'),
                                                                     dict(id='ai7', name='Socket', pos=[2, 0.3, 2], source='ai')])
    built = dict(version=1, updated='2026-11-01T00:00:00.000Z', pois=[dict(id='a', name='Kitchen', pos=[1, 1, 1], source='ai'), dict(id='ai9', name='Fridge', pos=[3, 1, 3], source='ai')])
    m = tour_publish.merge_pois(live, built)
    assert [p['id'] for p in m['pois']] == ['a', 'ai7', 'ai9'] and m['pois'][0]['name'].startswith('Kitchen (renamed'), 'the live entry wins, new built ones are added'
    gone = dict(live, removed=['ai9']); assert [p['id'] for p in tour_publish.merge_pois(gone, built)['pois']] == ['a', 'ai7'], 'a place the admin deleted does not come back'
    assert tour_publish.merge_pois(live, None) == live and tour_publish.merge_pois(None, built) == built and tour_publish.merge_pois(None, None) is None
    assert tour_publish.merge_pois(dict(pois='broken'), built)['pois'] == built['pois'], 'a broken live file does not lose the built places'

    class Api:                                              # the Space already has this tour folder with places in it
        token = None
        def __init__(self, files): self.files = files
        def file_exists(self, repo, path, repo_type=None): return path in self.files
    t = tmp_path / 'tour'; t.mkdir(); (t / 'pois.json').write_text(json.dumps(built))
    got = tour_publish.keep_pois(Api({'f1/pois.json': live}), 'me/namma-tour', 'f1', str(t), fetch=lambda api, sp, path: api.files[path])
    assert got == 3 and [p['id'] for p in json.loads((t / 'pois.json').read_text())['pois']] == ['a', 'ai7', 'ai9']
    t2 = tmp_path / 'tour2'; t2.mkdir()                     # nothing built, nothing live: no file appears
    assert tour_publish.keep_pois(Api({}), 'me/namma-tour', 'f2', str(t2), fetch=lambda *a: None) == 0 and not (t2 / 'pois.json').exists()
    t3 = tmp_path / 'tour3'; t3.mkdir()                     # live places, a rebuilt tour without any: they come along
    assert tour_publish.keep_pois(Api({'f1/pois.json': live}), 'me/namma-tour', 'f1', str(t3), fetch=lambda api, sp, path: api.files[path]) == 2
    assert json.loads((t3 / 'pois.json').read_text())['pois'][0]['id'] == 'a'
