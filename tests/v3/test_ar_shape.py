"""Shape lock (colab/real_scenes/final/ar_shape.py and its browser twin ar_geo.js): finding the phone in a scanned room from
depth points alone. A made-up room (the switch room of test_walls.py with a ceiling), a made-up visit with simulated depth
(scripts/ar/exp_geo.py), and the same visit given to both implementations."""
import json, os, shutil, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (('..', '..', 'colab', 'real_scenes', 'final'), ('..', '..', 'scripts', 'ar'), ()): sys.path.insert(0, os.path.join(HERE, *d))
import ar_shape, exp_geo, test_walls


@pytest.fixture(scope='module')
def room():
    P = np.concatenate(test_walls.switch_room() + [test_walls.floor(0, 0, 6, 4, y=test_walls.H)])       # 6 x 4 m, trays, panels, a cabinet, a ceiling
    t = np.linspace(0, 2 * np.pi, 80); pos = np.c_[3 + 1.6 * np.cos(t), np.full(80, 1.35), 2 + 0.9 * np.sin(t)]
    tgt = pos + np.c_[-np.sin(t), np.zeros(80), np.cos(t)] * 2.0                                    # walking a loop, looking ahead
    return P, (pos, tgt, 60.0), ar_shape.ShapeMap(P)


def off(J, L, truth, corners):
    """How far the room's corners end up from where they are (m), for a join p_tour = Ry(yaw)(p - c)/s + t."""
    yaw, s, t = truth; cp = (corners @ ar_shape.ry(yaw).T) * s + t - L['c']
    return float(np.linalg.norm((cp @ ar_shape.ry(J['yaw']).T / J['s'])[:, [0, 2]] + J['t'] - corners[:, [0, 2]], axis=1).max())


CORNERS = np.array([[0, 0, 0], [6, 0, 0], [6, 0, 4], [0, 0, 4.0]])


def test_looking_round_finds_the_room(room):
    P, walk, M = room; rng = np.random.default_rng(1); errs = []
    assert abs(M.H - test_walls.H) < 0.05
    for _ in range(4):
        pts, path, truth = exp_geo.session(P, walk, 8, 'look', 0.03, rng); L = ar_shape.prepare(pts, path); J = ar_shape.locate(M, L)
        for h in L['heights']: assert min(abs(h / truth[1] - v) for v in (test_walls.H, 2.3)) < 0.1, 'a level above head height measured wrongly'
        errs.append(off(J, L, truth, CORNERS)); assert abs(J['s'] / truth[1] - 1) < 0.04
    assert max(errs) < 0.2, errs                                                               # every corner within 20 cm, every time


def test_map_file_round_trip(room, tmp_path):
    _, _, M = room; f = tmp_path / 'ar_shape.bin'; M.write(str(f)); raw = f.read_bytes()
    x0, z0, cell, H, N, nb = __import__('struct').unpack('<4f2I', raw[:24]); assert (N, nb) == (ar_shape.N, len(ar_shape.BANDS)) and abs(cell - ar_shape.CELL) < 1e-6
    assert len(raw) == 28 + 8 * nb + (nb + 1) * N * N and abs(H - M.H) < 1e-4 and len(M.Hs) == 2          # ceiling 2.8 and the trays 2.3
    band0 = np.frombuffer(raw, np.uint8, N * N, 28 + 8 * nb).reshape(N, N) / 255; assert np.abs(band0 - M.L[0]).max() < 0.003
    free = np.frombuffer(raw, np.uint8, N * N, 28 + 8 * nb + nb * N * N).reshape(N, N); assert ((free > 0) == (M.free > 0)).all() and 0.02 < free.mean() < 0.5


def test_the_browser_gives_the_same_answer(room, tmp_path):
    """ar_geo.js on the same visit: the same join as ar_shape.py (2 cm, 0.3 deg, 0.3 % of scale), and its join in the AR page's
    form puts the room's corners where they are."""
    if not shutil.which('node'): pytest.skip('node not installed')
    P, walk, M = room; rng = np.random.default_rng(7); pts, path, truth = exp_geo.session(P, walk, 8, 'look', 0.03, rng)
    L = ar_shape.prepare(pts, path); J = ar_shape.locate(M, L); M.write(str(tmp_path / 'ar_shape.bin'))
    json.dump(dict(pts=np.round(pts, 5).reshape(-1).tolist(), path=np.round(path, 5).reshape(-1).tolist()), open(tmp_path / 'live.json', 'w'))
    r = subprocess.run(['node', os.path.join(HERE, '..', 'browser', 'ar_geo_run.mjs'), str(tmp_path / 'ar_shape.bin'), str(tmp_path / 'live.json')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    o = json.loads(r.stdout.strip().splitlines()[-1]); B = o['J']
    assert abs(o['points'] - len(L['p'])) <= 5 and np.allclose(o['c'], L['c'], atol=1e-3) and np.allclose(o['heights'], L['heights'], atol=1e-3)
    assert np.hypot(B['tx'] - J['t'][0], B['tz'] - J['t'][1]) < 0.02 and abs(np.degrees(np.angle(np.exp(1j * (B['yaw'] - J['yaw']))))) < 0.3 and abs(B['s'] - J['s']) < 0.003, (B, J)
    assert off(dict(yaw=B['yaw'], s=B['s'], t=np.array([B['tx'], B['tz']])), L, truth, CORNERS) < 0.15
    # the page's form: p_phone = s Ry(yaw) p_tour + t  (ar_core.js apply)
    j = o['join']; yaw, s, t = truth; ry = lambda a, p: np.array([np.cos(a) * p[0] + np.sin(a) * p[2], p[1], -np.sin(a) * p[0] + np.cos(a) * p[2]])
    for cn in CORNERS:
        want = ar_shape.ry(yaw) @ cn * s + t; got = j['s'] * ry(j['yaw'], cn) + np.array(j['t'])
        assert np.linalg.norm((got - want)[[0, 2]]) < 0.15 and abs(got[1] - want[1]) < 0.08, (got, want)
    print('browser: %d points, prepare %.0f ms, match %.0f ms' % (o['points'], o['prepare_ms'], o['locate_ms']))
