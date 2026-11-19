"""Deadline stop for gsplat's examples/simple_trainer.py (v1.5.3), patched in like room_prior / geo_prior. The stock trainer
saves and evaluates only at the steps given on the command line and has no time limit: a run that is slower than the dry run
predicted trains until Kaggle kills the session at 12 h and nothing is kept.
With NM_DEADLINE=<epoch seconds> in the environment the patched trainer, once that time has passed, treats the step it is on
as its last one: checkpoint, .ply and held-out eval exactly as at the planned end, then it stops. Without NM_DEADLINE it
behaves as before. The learning-rate schedule is not re-planned (it ends higher than planned), so this is the safety net; the
plan itself comes from budget.fit, which sizes the run to finish on time.
  python budget_patch.py <simple_trainer.py>   ->  prints True / False (False: this trainer version has different lines)"""
import sys

MARK = '_nm_deadline'
# (text that must occur exactly once, its replacement)
EDITS = [
    ('        pbar = tqdm.tqdm(range(init_step, max_steps))\n',
     "        _nm_deadline = float(os.environ.get('NM_DEADLINE') or 0); _nm_stop = False   # budget_patch: stop at this time\n"
     '        pbar = tqdm.tqdm(range(init_step, max_steps))\n'),
    ('        for step in pbar:\n',
     '        for step in pbar:\n'
     '            if _nm_deadline and step % 20 == 0 and step > 0 and time.time() > _nm_deadline:\n'
     '                _nm_stop = True; print(f"NM_DEADLINE reached at step {step} of {max_steps}: saving and stopping", flush=True)\n'),
    ('            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:\n',
     '            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1 or _nm_stop:\n'),
    ('                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1\n',
     '                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1 or _nm_stop\n'),
    ('            if step in [i - 1 for i in cfg.eval_steps]:\n                self.eval(step)\n                self.render_traj(step)\n',
     '            if step in [i - 1 for i in cfg.eval_steps] or _nm_stop:\n                self.eval(step)\n                self.render_traj(step)\n'
     '            if _nm_stop:\n                break                                           # budget_patch (the viewer is off in our runs)\n'),
]


def patch(trainer_py):
    """Insert the deadline stop (idempotent). Returns False and leaves the file alone unless every edit point is found once."""
    s = open(trainer_py).read()
    if MARK in s: return True
    if any(s.count(a) != 1 for a, _ in EDITS): return False
    for a, b in EDITS: s = s.replace(a, b)
    for mod in ('os', 'time'):
        if f'import {mod}\n' not in s: s = f'import {mod}\n' + s
    open(trainer_py, 'w').write(s); return True


if __name__ == '__main__':
    print(patch(sys.argv[1]))
