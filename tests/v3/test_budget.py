"""The session time budget (colab/real_scenes/final/budget.py), the motion-aware frame choice (frames.py) and the trainer's
deadline stop (budget_patch.py), without a GPU. Recorded runs are replayed against the planner (numbers from final/NOTES.md and
docs/HANDOFF_V3.md): run 4 (the reference), run 6, the whole house.
  pytest -q --confcutdir=tests/v3 tests/v3
  GSPLAT_TRAINER=<gsplat v1.5.3 examples/simple_trainer.py> also patches and compiles the real trainer."""
import json, os, shutil, sys, time
import numpy as np
import pytest

FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'colab', 'real_scenes', 'final')
sys.path.insert(0, os.path.abspath(FINAL))
import budget, budget_patch, frames


# ---- cameras
def test_sfm_minutes_matches_the_measured_solves():
    for n, minutes, tol in ((200, 5.3, 0.4), (411, 9.0, 0.4), (600, 15.0, 0.25), (3470, 456.0, 0.1)):
        assert abs(budget.sfm_minutes(n) - minutes) <= tol * minutes, (n, budget.sfm_minutes(n))


def test_frames_never_exceed_what_the_solve_can_finish():
    assert budget.sfm_max_frames(budget.sfm_budget_min(11.0)) == budget.SFM_MAX          # a fresh session: the proven range
    late = budget.sfm_max_frames(budget.sfm_budget_min(1.5))                                # 1.5 h left: 22 min for cameras
    assert 50 <= late < budget.SFM_MAX and budget.sfm_minutes(late) <= budget.sfm_budget_min(1.5) + 0.5
    assert budget.sfm_max_frames(0) == 50
    # the 31 Oct house video (17 min): 3,470 frames took 7.6 h; the plan now stops at the cap, and its time limit is hours shorter
    n = budget.frame_count(17 * 60, budget.sfm_max_frames(budget.sfm_budget_min(11.0)))
    assert n == 1000 and budget.sfm_timeout_min(n) < 120


def test_frame_count_is_run4_density_by_walking_time():
    assert budget.frame_count(123.3, 1000) == 411
    assert budget.frame_count(5, 1000) == 50 and budget.frame_count(3600, 1000) == 1000


# ---- splats: recorded runs
def test_run4_replay_is_the_reference_recipe():
    p = budget.fit(411, 37.0, 1440, 0.1815, train_left_h=9.0)
    assert p['rung'] == 0 and p['kind'] == 'full' and p['long'] == 1440 and p['thin'] == 1
    assert abs(p['steps'] - 30000) < 50 and p['cap'] == 1_500_000
    assert abs(p['steps'] * p['s_per_step'] - 5445) < 0.02 * 5445                         # run 4 trained 5445 s


def test_house_replay_predicts_within_20_percent_of_the_real_time():
    p = budget.fit(982, 46.6, 1440, 0.221, train_left_h=9.0)                               # dry run 0.221 s/step; real 4.7 h
    assert p['rung'] == 0 and p['steps'] == 71812 and abs(p['cap'] - 1_890_000) < 5000
    assert abs(p['hours'] - 4.7) <= 0.2 * 4.7
    assert budget.refine_stop(p['steps']) == 59843 and budget.refine_stop(30000) == 25000


def test_run6_inputs_get_run4_density_not_more():
    p = budget.fit(900, 37.0, 1920, 0.4, train_left_h=10.0)                                # run 6: 51 steps/view, 3M splats, worse
    assert p['spv'] == budget.R4['steps_per_view'] and p['cap'] == 1_500_000 and p['long'] <= 1920


# ---- splats: the ladder
def test_slow_session_steps_down_instead_of_refusing():
    full = budget.fit(982, 46.6, 1440, 0.6, train_left_h=99)                               # the recipe itself: 13 h
    assert full['rung'] == 0 and full['hours'] > 12
    p = budget.fit(982, 46.6, 1440, 0.6, train_left_h=9.0)
    assert p['kind'] == 'full' and p['rung'] > 0 and p['hours'] <= 9.0 and p['steps'] >= budget.PREVIEW['steps']
    assert p['spv'] >= 60 and p['long'] in (1440, 1280, 1080, 960)
    assert p['tried_hours'] == sorted(p['tried_hours'], reverse=True)                      # every rung is cheaper than the one before


def test_ladder_order_is_steps_then_pixels_then_frames():
    r = budget.rungs(1440)
    assert [x['spv'] for x in r[:3]] == [83.6, 70.0, 60.0] and all(x['long'] == 1440 and x['thin'] == 1 for x in r[:3])
    assert [x['long'] for x in r[3:6]] == [1280, 1080, 960] and [x['thin'] for x in r[6:]] == [2, 3, 4]
    assert all(x['long'] <= 960 for x in budget.rungs(960)) and len(budget.rungs(960)) == 6   # a small source is never enlarged


def test_tiny_deadline_gets_the_preview_recipe():
    p = budget.fit(411, 37.0, 1440, 0.1815, train_left_h=0.2)
    assert p['kind'] == 'preview' and p['steps'] <= budget.PREVIEW['steps'] and p['long'] == 960 and p['hours'] <= 0.21
    assert budget.fit(411, 37.0, 1440, 0.1815, train_left_h=-1.0)['steps'] == 500           # past the deadline: the least there is


def test_fit_always_returns_a_recipe():
    rng = np.random.default_rng(0)
    for _ in range(300):
        reg = int(rng.integers(20, 1200)); p = budget.fit(reg, float(rng.uniform(5, 300)), int(rng.choice([720, 960, 1280, 1440, 1920])),
                                                           float(rng.uniform(0.05, 3.0)), float(rng.uniform(-1, 11)))
        assert p['steps'] >= 500 and 1_000_000 <= p['cap'] <= 4_000_000 and p['kind'] in ('full', 'preview')
        if p['kind'] == 'full': assert p['spv'] <= budget.R4['steps_per_view']


def test_checkpoints_every_20_minutes_and_at_the_end():
    s = budget.save_steps(71812, 0.221)
    assert s[-1] == 71812 and s == sorted(set(s)) and len(s) <= 25
    assert all((b - a) * 0.221 >= 19 * 60 for a, b in zip(s, s[1:-1]))                      # spacing between checkpoints
    assert budget.save_steps(3000, 0.2) == [3000] and budget.save_steps(7000, 0.17) == [7000]   # short runs: only the end
    long = budget.save_steps(200000, 0.05); assert len(long) <= 25 and long[-1] == 200000


# ---- clock and tail
def test_clock_and_tail(tmp_path):
    t0 = time.time() - 3 * 3600; c = budget.Clock(t0, deadline_h=11.25, path=str(tmp_path / 'budget.json'))
    assert abs(c.left_h() - 8.25) < 0.01 and abs(c.train_left_h(mesh=True) - (8.25 - 35 / 60)) < 0.01
    assert abs(c.train_deadline(mesh=False) - (t0 + (11.25 - 0.25) * 3600)) < 1
    assert budget.Clock(t0, deadline_h=13).deadline_h == 11.75                              # never past Kaggle's own limit
    c.mark('frames', time.time() - 120, frames=411); c.note(mesh_skipped=True)
    j = json.load(open(tmp_path / 'budget.json'))
    assert j['stages'][0]['stage'] == 'frames' and j['stages'][0]['minutes'] == 2.0 and j['mesh_skipped'] is True
    assert budget.tail(60, mesh=True) == dict(mesh=True, objects=False, mesh_timeout_min=20)
    assert budget.tail(30, mesh=True)['mesh'] is False and budget.tail(60, mesh=True, objects=True)['objects'] is True
    assert budget.tail(50, mesh=True, objects=True) == dict(mesh=True, objects=False, mesh_timeout_min=20)


# ---- frames by motion
def test_pick_by_motion_follows_the_camera():
    m = 400; M = np.r_[np.full(100, 0.01), np.full(100, 1.0), np.full(100, 4.0), np.full(100, 1.0)]   # pause, walk, fast turn, walk
    S = np.full(m, 100.0); kept, info = frames.pick_by_motion(S, M, 100)
    seg = [sum(a <= k < a + 100 for k in kept) for a in (0, 100, 200, 300)]
    assert kept == sorted(set(kept)) and len(kept) <= 100 and sum(seg) == len(kept)
    assert seg[2] > 1.5 * seg[1] and seg[1] > 1.5 * seg[0] and seg[0] >= 5                   # turn > walk > pause, the pause not empty
    assert abs(info['active'] - 0.75) < 0.01 and info['fast'] == [[200, 299]]                # the pause is a quarter of the video;
                                                                                             # the turn (4x the usual motion) is reported
    # equal motion = the old equal time windows
    kept, _ = frames.pick_by_motion(S, np.ones(m), 100); assert np.all(np.diff(kept) == 4)


def test_pick_by_motion_takes_the_sharp_frame_and_drops_a_blurred_stretch():
    m = 400; S = np.where(np.arange(m) % 4 == 1, 100.0, 10.0); kept, info = frames.pick_by_motion(S, np.ones(m), 100)
    assert all(k % 4 == 1 for k in kept) and info['dropped_blurred'] == 0
    S[200:204] = 1.0; kept, info = frames.pick_by_motion(S, np.ones(m), 100)                 # one group has only blurred frames
    assert info['dropped_blurred'] == 1 and not any(200 <= k < 204 for k in kept) and len(kept) == 99
    assert frames.pick_by_motion([], [], 10)[0] == []
    S[:] = 1.0; kept, info = frames.pick_by_motion(S, np.zeros(m), 50); assert len(kept) == 50   # nothing to compare: by time


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg')
def test_from_video_smart_on_a_real_file(tmp_path):
    """A camera panning over a textured wall: 4 s slow, 4 s paused, 4 s fast. The fast part must get the most frames, the pause
    the fewest, names sort in walk order, and the unused candidates stay for a later camera-solve retry."""
    import cv2
    rng = np.random.default_rng(1); wall = cv2.GaussianBlur((rng.random((480, 6000, 3)) * 255).astype(np.uint8), (0, 0), 2)
    v = str(tmp_path / 'pan.mp4'); w = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480)); x = 0.0
    for i in range(360):
        x += 2.0 if i < 120 else 0.0 if i < 240 else 24.0; w.write(wall[:, int(x):int(x) + 640])
    w.release()
    n, rep = frames.from_video_smart(v, str(tmp_path / 'img'), cap=60, long_side=320, total=360, fps=30.0, rate=5.0, cand=str(tmp_path / 'cand'))
    names = sorted(os.listdir(tmp_path / 'img')); idx = [int(f[:5]) * rep['window'] for f in names]
    assert n == len(names) == rep['frames'] and all(len(f) == 9 for f in names)
    seg = [sum(a <= k < a + 120 for k in idx) for a in (0, 120, 240)]
    assert seg[2] > seg[0] > seg[1] and seg[1] >= 1, seg
    assert 7.0 < rep['active_seconds'] < 9.0 and rep['seconds'] == 12.0                       # the 4 s pause is not walking time
    assert cv2.imread(str(tmp_path / 'img' / names[0])).shape[:2] == (240, 320)
    assert os.path.exists(tmp_path / 'cand' / 'pick.json') and len(os.listdir(tmp_path / 'cand')) == rep['candidates'] - n + 1
    lost = names[len(names) // 2]; added = frames.add_candidates(str(tmp_path / 'cand'), str(tmp_path / 'img'), [lost], around=2)
    assert added and all(abs(int(a[:5]) - int(lost[:5])) <= 2 for a in added) and sorted(os.listdir(tmp_path / 'img')) == sorted(names + added)
    assert frames.thin_dir(str(tmp_path / 'img'), 10) == 10 and len(os.listdir(tmp_path / 'img')) == 10
    assert frames.resize_dir(str(tmp_path / 'img'), 160) == 10
    assert max(cv2.imread(str(tmp_path / 'img' / sorted(os.listdir(tmp_path / 'img'))[0])).shape[:2]) == 160


# ---- the trainer's deadline stop
TRAINER = '''import json
import tqdm
class Runner:
    def train(self):
        cfg = self.cfg; max_steps = cfg.max_steps; init_step = 0; done = []
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            self.work(step)
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                done.append(('save', step))
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                done.append(('ply', step))
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)
        return done
'''


def _run(src, env, work=None):
    """The (patched) loop above on a fake runner: which steps saved / wrote a .ply / evaluated."""
    ns = {}; sys.modules.setdefault('tqdm', type(sys)('tqdm')); sys.modules['tqdm'].tqdm = getattr(sys.modules['tqdm'], 'tqdm', lambda x: x)
    exec(compile(src, 'trainer', 'exec'), ns); r = ns['Runner'](); ev = []
    r.cfg = type('C', (), dict(max_steps=400, save_steps=[200, 400], ply_steps=[400], eval_steps=[400], save_ply=True))
    r.work = work or (lambda step: None); r.eval = lambda s: ev.append(('eval', s)); r.render_traj = lambda s: ev.append(('traj', s))
    old = os.environ.get('NM_DEADLINE'); os.environ.pop('NM_DEADLINE', None)
    if env is not None: os.environ['NM_DEADLINE'] = str(env)
    try: return r.train() + ev
    finally:
        os.environ.pop('NM_DEADLINE', None)
        if old is not None: os.environ['NM_DEADLINE'] = old


def test_deadline_patch_stops_and_keeps_the_model(tmp_path):
    f = tmp_path / 'simple_trainer.py'; f.write_text(TRAINER)
    plain = _run(TRAINER, None)
    assert budget_patch.patch(str(f)) is True and budget_patch.patch(str(f)) is True         # idempotent
    src = f.read_text(); assert src.count('_nm_stop = True') == 1 and src.startswith('import time\nimport os\n')
    assert _run(src, None) == plain == [('save', 199), ('save', 399), ('ply', 399), ('eval', 399), ('traj', 399)]   # no deadline: as before
    assert _run(src, time.time() + 3600) == plain                                              # deadline far away: as before
    # the deadline passes while step 130 trains: the next check (every 20 steps) saves, writes the .ply, evaluates and stops
    cut = _run(src, time.time() + 0.2, work=lambda step: time.sleep(0.4) if step == 130 else None)
    assert cut == [('save', 140), ('ply', 140), ('eval', 140), ('traj', 140)]


def test_deadline_patch_refuses_an_unknown_trainer(tmp_path):
    f = tmp_path / 't.py'; other = TRAINER.replace('step == max_steps - 1:', 'step + 1 == max_steps:'); f.write_text(other)
    assert budget_patch.patch(str(f)) is False and f.read_text() == other                     # left alone


@pytest.mark.skipif(not os.environ.get('GSPLAT_TRAINER'), reason='GSPLAT_TRAINER = path of gsplat v1.5.3 examples/simple_trainer.py')
def test_deadline_patch_fits_the_real_trainer(tmp_path):
    f = tmp_path / 'simple_trainer.py'; shutil.copy(os.environ['GSPLAT_TRAINER'], f)
    assert budget_patch.patch(str(f)) is True
    import py_compile; py_compile.compile(str(f), doraise=True); s = f.read_text()
    assert s.count('or _nm_stop') == 3 and s.count('if _nm_stop:') == 1


# ---- checkpoints
def test_checkpoint_to_ply_and_splat(tmp_path):
    torch = pytest.importorskip('torch'); pytest.importorskip('plyfile')
    from plyfile import PlyData
    n = 50; g = torch.Generator().manual_seed(0); r = lambda *s: torch.randn(*s, generator=g)
    sp = dict(means=r(n, 3), scales=r(n, 3), quats=r(n, 4), opacities=r(n), sh0=r(n, 1, 3), shN=r(n, 15, 3))
    ck = tmp_path / 'ckpt_139_rank0.pt'; torch.save(dict(step=139, splats=sp), ck)
    assert budget.ckpt_to_ply(str(ck), str(tmp_path / 'm.ply')) == n
    v = PlyData.read(str(tmp_path / 'm.ply'))['vertex'].data
    assert len(v) == n and len([x for x in v.dtype.names if x.startswith('f_rest_')]) == 45
    assert np.allclose(v['x'], sp['means'][:, 0]) and np.allclose(v['opacity'], sp['opacities']) and np.allclose(v['f_dc_2'], sp['sh0'][:, 0, 2])
    # f_rest is channel-major, as gsplat writes it and objects3d.load_ply reads it back: f_rest_k = shN[:, k % 15, k // 15]
    assert np.allclose(v['f_rest_0'], sp['shN'][:, 0, 0]) and np.allclose(v['f_rest_15'], sp['shN'][:, 0, 1]) and np.allclose(v['f_rest_44'], sp['shN'][:, 14, 2])
    import to_splat
    assert budget.ckpt_to_splat(str(ck), str(tmp_path / 'm.splat')) == n
    a = np.fromfile(str(tmp_path / 'm.splat'), dtype=to_splat.REC); to_splat.convert(str(tmp_path / 'm.ply'), str(tmp_path / 'ref.splat'))
    b = np.fromfile(str(tmp_path / 'ref.splat'), dtype=to_splat.REC)                           # the same bytes as the .ply route
    assert len(a) == n and np.array_equal(a['c'], b['c']) and np.allclose(a['p'], b['p']) and np.allclose(a['s'], b['s'], rtol=1e-5) and np.array_equal(a['r'], b['r'])
