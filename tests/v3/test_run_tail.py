"""The training part of v3/run.py (start -> watch loop -> snapshots -> finish -> tail) run for real, with a stand-in trainer,
QA and mesh script instead of the GPU ones: what happens when a trainer finishes, reaches its deadline, hangs or crashes.
The code under test is run.py's own text from `def start(` to the end; only its waits are shortened (30 s -> 0.2 s etc.).
Needs torch (checkpoints are real torch files): pytest -q --confcutdir=tests/v3 tests/v3/test_run_tail.py"""
import glob, json, os, shutil, subprocess, sys, time
import pytest

torch = pytest.importorskip('torch'); pytest.importorskip('plyfile')
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FINAL = f'{ROOT}/colab/real_scenes/final'; sys.path.insert(0, FINAL)
import budget

# the stand-in for gsplat's patched simple_trainer.py: same flags, same files, same NM_DEADLINE rule (budget_patch.py).
# FAKE_MODE: ok | hang (ignores the deadline, never ends) | crash (exit 1 after its first checkpoint)
TRAINER = '''
import json, os, sys, time, torch
a = sys.argv[1:]; assert a[0] == 'mcmc', a
def val(k): return a[a.index(k) + 1]
def many(k):
    i = a.index(k) + 1; out = []
    while i < len(a) and not a[i].startswith('--'): out.append(int(a[i])); i += 1
    return out
res, steps, saves, plys = val('--result-dir'), int(val('--max-steps')), many('--save-steps'), many('--ply-steps')
assert '--save-ply' in a and '--disable-viewer' in a and int(val('--strategy.cap-max')) > 0 and val('--data-factor') == '1'
if steps > 30000: assert int(val('--strategy.refine-stop-iter')) == round(steps * 25 / 30)
for d in ('ckpts', 'ply', 'stats', 'videos'): os.makedirs(f'{res}/{d}', exist_ok=True)
mode = os.environ.get('FAKE_MODE', 'ok'); dl = float(os.environ.get('NM_DEADLINE') or 0); dt = float(os.environ.get('FAKE_DT', 0.01))
sys.path.insert(0, os.environ['REAL_FINAL']); import budget
n = 40; g = torch.Generator().manual_seed(0); r = lambda *s: torch.randn(*s, generator=g)
sp = dict(means=r(n, 3), scales=r(n, 3), quats=r(n, 4), opacities=r(n), sh0=r(n, 1, 3), shN=r(n, 15, 3))
for step in range(steps):
    stop = mode != 'hang' and dl and step % 20 == 0 and step > 0 and time.time() > dl
    if stop: print(f'NM_DEADLINE reached at step {step} of {steps}: saving and stopping', flush=True)
    time.sleep(dt)
    if step in [i - 1 for i in saves] or step == steps - 1 or stop:
        torch.save(dict(step=step, splats=sp), f'{res}/ckpts/ckpt_{step}_rank0.pt')
        if mode == 'crash': sys.exit(1)
    if step in [i - 1 for i in plys] or step == steps - 1 or stop:
        budget.ckpt_to_ply(f'{res}/ckpts/ckpt_{step}_rank0.pt', f'{res}/ply/point_cloud_{step}.ply')
        json.dump(dict(psnr=25.5, cc_psnr=29.2, lpips=0.2), open(f'{res}/stats/val_step{step:04d}.json', 'w')); open(f'{res}/videos/traj_{step}.mp4', 'w').write('v')
    if stop: break
    while mode == 'hang' and step == steps - 2: time.sleep(0.05)
'''
# v3_qa.py's part: metrics from the trainer's eval (None if it never ran), the model from the newest .ply
QA = '''
import glob, json, os, shutil, sys
ex, data, res, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
ck = sorted(glob.glob(f'{res}/ckpts/ckpt_*_rank0.pt'), key=lambda f: int(f.split('_')[-2]))[-1]; step = int(ck.split('_')[-2])
st = sorted(glob.glob(f'{res}/stats/val_step*.json')); m = json.load(open(st[-1])) if st else {}
json.dump(dict(step=step + 1, heldout_psnr=m.get('psnr'), heldout_psnr_exposure_fair=m.get('cc_psnr'), heldout_lpips=m.get('lpips')), open(f'{out}/metrics.json', 'w'))
json.dump(dict(matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), open(f'{out}/level.json', 'w'))
ply = sorted(glob.glob(f'{res}/ply/point_cloud_*.ply'), key=lambda f: int(f.split('_')[-1][:-4]))
if ply: shutil.copy(ply[-1], f'{out}/model.ply')
'''
# objects3d.py's part: objects.json + crops + the region index, and (like the real one) a tinted full/model.ply in ITS out dir
OBJECTS = '''
import json, os, sys, time
data, ply, level, mpu, out = sys.argv[1:6]; assert float(mpu) == 0 and os.path.exists(ply) and os.path.exists(level)
time.sleep(float(os.environ.get('FAKE_OBJ_S', 0))); os.makedirs(out + '/crops', exist_ok=True); os.makedirs(out + '/full', exist_ok=True)
if os.environ.get('FAKE_OBJ_FAIL'): sys.exit(1)
json.dump(dict(objects=[dict(id=1, label='sofa')]), open(out + '/objects.json', 'w')); open(out + '/crops/1.jpg', 'w').write('c')
json.dump(dict(count=3), open(out + '/regions.json', 'w')); open(out + '/regions.f16', 'wb').write(b'0' * 6); open(out + '/full/model.ply', 'w').write('TINTED')
'''
MESH = '''
import json, os, sys, time
time.sleep(float(os.environ.get('FAKE_MESH_S', 0))); os.makedirs(sys.argv[4], exist_ok=True); json.dump(dict(faces=10), open(sys.argv[4] + '/mesh.json', 'w'))
'''


def tail_source():
    s = open(f'{ROOT}/colab/real_scenes/v3/run.py').read(); s = s[s.index("def start(name, steps, cap, gpu, kind='gs'):"):]
    for a, b in (('time.sleep(30)', 'time.sleep(0.2)'), ('time.time() - os.path.getmtime(f) > 20', 'time.time() - os.path.getmtime(f) > 0.3'),
                 ('TRAIN_END + 600', 'TRAIN_END + 2'), ('TRAIN_END - 900', 'TRAIN_END - 1'), ('time.time() - last > 300', 'time.time() - last > 1')):
        assert s.count(a) == 1, a; s = s.replace(a, b)
    return s


def run_tail(tmp, variants, train_s, mode='ok', dt=0.01, deadline_h=None, mesh_s=0.0, ngpu=2, env=None):
    """run.py's training part in tmp. train_s = seconds from now until the trainers' deadline (TRAIN_END)."""
    W, O, fake, ex, bins = (f'{tmp}/{x}' for x in ('w', 'o', 'fake', 'ex', 'bin'))
    for d in (W, O, fake, ex, bins, f'{W}/data/images', f'{W}/data/sparse/0'): os.makedirs(d, exist_ok=True)
    open(f'{ex}/simple_trainer.py', 'w').write(TRAINER); open(f'{fake}/v3_qa.py', 'w').write(QA); open(f'{fake}/mesh_tsdf.py', 'w').write(MESH); open(f'{fake}/objects3d.py', 'w').write(OBJECTS)
    open(f'{W}/data/images/00000.jpg', 'w').write('x'); open(f'{W}/data/sparse/0/images.bin', 'w').write('x')
    sh_ = lambda name, body: (open(f'{bins}/{name}', 'w').write('#!/bin/sh\n' + body + '\n'), os.chmod(f'{bins}/{name}', 0o755))
    sh_('python', f'exec "{sys.executable}" "$@"'); sh_('pip', 'exit 0')                      # no installs in a test
    if not shutil.which('timeout'):                                                       # macOS: `timeout -k 30 <s> cmd ...`
        open(f'{bins}/timeout', 'w').write(f'#!{sys.executable}\nimport subprocess, sys\na = sys.argv[1:]\nif a[0] == "-k": a = a[2:]\n'
                                           'try: sys.exit(subprocess.run(a[1:], timeout=float(a[0])).returncode)\nexcept subprocess.TimeoutExpired: sys.exit(124)\n')
        os.chmod(f'{bins}/timeout', 0o755)
    old = dict(os.environ); os.environ.update(PATH=f"{bins}:{old['PATH']}", FAKE_MODE=mode, FAKE_DT=str(dt), REAL_FINAL=FINAL, FAKE_MESH_S=str(mesh_s), **(env or {}))
    pushed = []; T0 = time.time()
    clk = budget.Clock(T0, deadline_h=deadline_h or 11.25, path=f'{O}/budget.json')
    ns = dict(glob=glob, json=json, os=os, shutil=shutil, subprocess=subprocess, sys=sys, time=time, budget=budget, T0=T0, W=W, O=O, D=f'{W}/data',
              EX=ex, F=fake, PP='', DEPTH='', SPS={}, NGPU=ngpu, VARIANTS=variants, CLK=clk, MESH=True, SOURCE='test', rep=dict(registered=8, frames=8),
              TRAIN_END=time.time() + train_s, push=lambda p, name: pushed.append(name),
              say=lambda *a: print('[run]', *a, flush=True),
              sh=lambda c, check=False: subprocess.call(c, shell=True, executable='/bin/bash'))
    err = None
    try: exec(compile(tail_source(), 'run_tail', 'exec'), ns)
    except SystemExit as e: err = str(e)
    finally:
        os.environ.clear(); os.environ.update(old)
    summ = json.load(open(f'{O}/v3_summary.json')) if os.path.exists(f'{O}/v3_summary.json') else None
    bud = json.load(open(f'{O}/budget.json')) if os.path.exists(f'{O}/budget.json') else None
    return dict(O=O, W=W, pushed=pushed, summary=summ, budget=bud, err=err, seconds=time.time() - T0)


def stage(r, name): return next(s for s in r['budget']['stages'] if s['stage'] == name)


def test_normal_run_two_models_snapshots_and_mesh(tmp_path):
    r = run_tail(str(tmp_path), [['full', 300, 1500000], ['preview', 60, 1000000]], train_s=600, dt=0.01)
    assert r['err'] is None and set(r['summary']['models']) == {'full', 'preview', 'mesh'}
    assert r['summary']['models']['full']['step'] == 300 and r['summary']['models']['full']['heldout_psnr_exposure_fair'] == 29.2
    for f in ('full/model.ply', 'full/train.log', 'preview/model.ply', 'mesh/mesh.json', 'sparse/0/images.bin', 'images/00000.jpg', 'budget.json'):
        assert os.path.exists(f"{r['O']}/{f}"), f
    assert stage(r, 'train full')['stopped_early'] is None and stage(r, 'mesh')['ok'] is True and r['budget']['done'] is True
    assert {'full', 'preview', 'v3_summary.json', 'budget.json'} <= set(r['pushed'])


def test_checkpoints_become_snapshots_and_old_ones_are_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'save_steps', lambda steps, sps, **k: [100, 200, 300, steps])   # as if 20 min apart
    r = run_tail(str(tmp_path), [['full', 400, 1500000]], train_s=600, dt=0.02, ngpu=1)
    assert r['err'] is None and r['summary']['models']['full']['step'] == 400
    snap = json.load(open(f"{r['O']}/snapshot/full.json")); assert snap['step'] in (100, 200, 300) and snap['of'] == 400 and snap['splats'] == 40
    assert os.path.getsize(f"{r['O']}/snapshot/full.splat") == 40 * 32 and 'snapshot/full.splat' in r['pushed']
    left = sorted(int(f.split('_')[-2]) for f in glob.glob(f"{r['W']}/full/ckpts/*.pt"))
    assert left[-1] == 399 and len(left) <= 2, left                                          # the final one + at most the newest snapshot


def test_trainer_reaching_its_deadline_keeps_the_model(tmp_path):
    r = run_tail(str(tmp_path), [['full', 100000, 1500000]], train_s=3, dt=0.01, ngpu=1)       # 100k steps would take 17 min
    m = r['summary']['models']['full']
    assert r['err'] is None and 20 < m['step'] < 2000 and m['heldout_psnr_exposure_fair'] == 29.2    # its own eval ran
    assert 'NM_DEADLINE reached' in stage(r, 'train full')['stopped_early'] and os.path.exists(f"{r['O']}/full/model.ply")
    assert r['seconds'] < 30


def test_hung_trainer_is_stopped_and_its_checkpoint_becomes_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'save_steps', lambda steps, sps, **k: [50, steps])
    r = run_tail(str(tmp_path), [['full', 200, 1500000]], train_s=4, mode='hang', dt=0.005, ngpu=1)
    m = r['summary']['models']['full']; st = stage(r, 'train full')
    assert r['err'] is None and m['step'] == 50 and m['heldout_psnr'] is None                  # no eval: the numbers are missing, the model is not
    assert st['stopped_early'] == 'still training 10 min after its deadline' and st['salvaged_step'] == 50
    assert os.path.exists(f"{r['O']}/full/model.ply") and r['seconds'] < 40
    assert not subprocess.run(['pgrep', '-f', 'simple_trainer.py mcmc.*' + str(tmp_path)], capture_output=True).stdout   # really gone


def test_crashed_trainer_is_salvaged_from_its_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'save_steps', lambda steps, sps, **k: [80, steps])
    r = run_tail(str(tmp_path), [['full', 300, 1500000]], train_s=600, mode='crash', ngpu=1)
    assert r['err'] is None and r['summary']['models']['full']['step'] == 80 and stage(r, 'train full')['exit'] == 1
    assert stage(r, 'train full')['salvaged_step'] == 80 and os.path.exists(f"{r['O']}/full/model.ply")


def test_no_time_left_skips_the_mesh_and_a_late_queue(tmp_path, monkeypatch):
    # 24 min to the deadline and a scene whose coarsest mesh needs 40: no mesh; the second model is not started 15 min before the deadline
    monkeypatch.setattr(budget, 'mesh_minutes', lambda frames, voxels=512: 40.0)
    r = run_tail(str(tmp_path), [['full', 60, 1500000], ['extra', 60, 1500000]], train_s=0.5, deadline_h=0.4, ngpu=1)
    assert r['err'] is None and set(r['summary']['models']) == {'full'} and r['budget']['mesh_skipped'] is True
    assert not os.path.exists(f"{r['O']}/mesh") and not os.path.exists(f"{r['O']}/extra")


def test_slow_mesh_is_cut_off(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'mesh_minutes', lambda frames, voxels=512: 0.008); monkeypatch.setattr(budget, 'MESH_FLOOR_MIN', 0.02)   # 1.2 s limit, the mesh takes 20 s
    r = run_tail(str(tmp_path), [['full', 60, 1500000]], train_s=600, mesh_s=20, ngpu=1)
    assert r['err'] is None and 'mesh' not in r['summary']['models'] and stage(r, 'mesh')['ok'] is False and r['seconds'] < 15


def test_challenger_replaces_full_only_if_better(tmp_path):
    r = run_tail(str(tmp_path), [['full', 60, 1500000], ['challenger', 60, 1500000, 'pose']], train_s=600, env=dict(CHALLENGER='pose'))
    assert r['summary']['models']['challenger_won'] is False and 'challenger' in r['summary']['models']   # same score: full stays
    assert os.path.exists(f"{r['O']}/full/model.ply") and not os.path.exists(f"{r['O']}/full_base")


def test_no_model_at_all_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), 'TRAINER', 'import sys; sys.exit(3)\n')                    # dies before any checkpoint
    r = run_tail(str(tmp_path), [['full', 300, 1500000]], train_s=600, ngpu=1)
    assert r['err'] and 'V3 FAILED' in r['err'] and stage(r, 'train full')['exit'] == 3


def test_object_layer_runs_in_its_time_and_never_touches_the_model(tmp_path):
    r = run_tail(str(tmp_path), [['full', 60, 1500000]], train_s=600, ngpu=1, env=dict(OBJECTS='1'))
    O = r['O']; assert r['err'] is None and r['summary']['models']['objects'] == 1 and stage(r, 'objects')['ok'] is True
    for f in ('objects.json', 'regions.json', 'regions.f16', 'crops/1.jpg'): assert os.path.exists(f'{O}/{f}'), f
    assert open(f'{O}/full/model.ply', 'rb').read()[:3] == b'ply' and 'regions.f16' in r['pushed']      # not the tinted copy


def test_object_layer_failure_or_no_time_does_not_fail_the_run(tmp_path):
    r = run_tail(str(tmp_path / 'a'), [['full', 60, 1500000]], train_s=600, ngpu=1, env=dict(OBJECTS='1', FAKE_OBJ_FAIL='1'))
    assert r['err'] is None and 'objects' not in r['summary']['models'] and stage(r, 'objects')['ok'] is False and 'full' in r['summary']['models']
    r = run_tail(str(tmp_path / 'b'), [['full', 60, 1500000]], train_s=1.0, deadline_h=0.5, ngpu=1, env=dict(OBJECTS='1'))   # 30 min: mesh yes, objects no
    assert r['err'] is None and r['budget']['objects_skipped'] is True and not os.path.exists(f"{r['O']}/objects.json") and 'mesh' in r['summary']['models']
    r = run_tail(str(tmp_path / 'c'), [['full', 60, 1500000]], train_s=600, ngpu=1)                       # OBJECTS not set: not run at all
    assert not os.path.exists(f"{r['O']}/objects.json") and not any(s['stage'] == 'objects' for s in r['budget']['stages'])
