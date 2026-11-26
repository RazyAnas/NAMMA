"""Time budget for one Kaggle session (plan of 16 Nov; used by v3/run.py). Kaggle stops a session at 12 h, so every stage gets a
share of that time, the run is sized to what fits, and a video that does not fit the best recipe steps down instead of ending
with nothing trained (the 31 Oct house pre-check spent 7.6 h on cameras alone; a NO-GO after decode + cameras + dry run threw
that work away).
  frames   run 4's density by walking time, never more than the camera solve can finish in its share (sfm_minutes: measured)
  cameras  a time limit per solve; a retry with fewer frames if it runs out
  splats   run 4's recipe if it fits; else the first rung of the ladder that does (fit): fewer steps per view, then fewer
           pixels, then every k-th frame, last the 7k preview recipe. More than run 4 is never tried: run 6 (900 frames,
           1920 px, 3M splats) took 3x as long and looked worse, 3M splats scored the same as 1.5M (run 3)
  trainer  stops itself at the deadline and keeps what it has (budget_patch.py); checkpoints on the way (save_steps)
  tail     QA, mesh, object layer only while there is time (tail)
Everything decided here is written to budget.json so each run shows what it chose and why.
Pure functions, no GPU and no Kaggle: tests/v3/test_budget.py replays runs 4, 6 and the house against them."""
import json, math, os, time

HARD_H = 12.0                                  # Kaggle's limit per session
DEADLINE_H = 11.25                             # ours: 45 min spare for a slow upload or a stuck copy
MARGIN = 1.1                                   # predicted training time x this (the house: predicted 5.34 h, real 4.7 h)
MEM_GB = 13.0                                  # of a T4's 15 GB
R4 = dict(steps_per_view=83.6, splats=1_500_000, area_m2=37.0, long=1440, s_per_step=0.1815, per_second=411 / 123.3)
PREVIEW = dict(steps=7000, long=960, cap=1_000_000)       # run 2's preview: 10 min on a T4, 26.8 dB
TAIL_MIN = dict(base=15, mesh=20, objects=25, after=8)    # QA + copies + uploads; the time held back for a mesh (enough for a
                                                          # coarse one: mesh_minutes; a finer one runs when more is left);
                                                          # objects3d (14.5 min, 27 Oct); after = copies + uploads alone
MESH_VOX = (512, 384, 256)                     # voxels on the mesh grid's long side, best first
MESH_FLOOR_MIN = 30                            # a mesh's time limit is never shorter than this (if that much is left)
SFM_MAX = 1000                                 # frames: runs 4 / 6 / house solved 411 / 900 / 982 in one piece


# ---- frames and cameras
def sfm_minutes(n):
    """Camera solve (hloc ALIKED + LightGlue, incremental mapper) for n frames on a T4, minutes. Measured: 200 -> 5.3, 411 -> ~9,
    600 -> 15, 3,470 -> 456 (30-31 Oct): a fixed part + one that grows with the square of the frames (pairs and bundle adjustment)."""
    return 4.0 + 3.7e-5 * n * n


def sfm_max_frames(budget_min, hard=SFM_MAX):
    """Most frames whose solve is predicted to finish in budget_min (never above the proven range `hard`, never below 50)."""
    return int(max(50, min(hard, math.sqrt(max(budget_min - 4.0, 0.0) / 3.7e-5))))


def sfm_budget_min(left_h):
    """The camera solve's share: a quarter of the time left, at most 2 h (more frames than that buys has never helped)."""
    return max(10.0, min(120.0, left_h * 60 * 0.25))


def sfm_timeout_min(n):
    """When to give up on a solve: twice the prediction + the hloc install (~8 min) + model downloads."""
    return max(30.0, 2.2 * sfm_minutes(n) + 15.0)


def frame_count(active_s, cap, lo=50):
    """Run 4's density (411 frames in 123.3 s of walking) for the seconds the camera actually moved."""
    return int(max(lo, min(cap, round(active_s * R4['per_second']))))


def fast_frame_cap(active_s, base=300, base_s=300.0):
    """Preset 'fast': at most `base` frames for a walk of up to `base_s` seconds (one room: Postshot and Polycam work with
    100-300 photos; the switch room's 934 frames for 19 m2 were 3x what it needed), more in proportion for a longer walk.
    Frames drive the training steps, the camera solve (their square) and the mesh, so this is the biggest single saving."""
    return int(round(base * max(1.0, active_s / base_s)))


# ---- splats
def train_views(registered): return registered - math.ceil(registered / 8)        # every 8th frame is held out


def cap_for(area_m2): return int(min(4e6, max(1.5e6, 1.5e6 * area_m2 / R4['area_m2'])))   # run 4: 1.5M splats for 37 m2


def refine_stop(steps):
    """MCMC refreshes splats until this step. gsplat's default is 25k whatever the length; run 4 = 25k of 30k, so keep that
    share (the house's 71.8k steps ran 47k steps without a refresh)."""
    return int(round(steps * 25000 / 30000))


def speed_at(sps0, long0, long_side):
    """s/step at another image size, from the dry run's speed at long0. One part does not depend on the pixels (splat sorting,
    densification, optimiser), one does: run 2 (960 px) 0.136 s/step vs run 4 (1440 px) 0.1815 with the same 1.5M splats."""
    return sps0 * (0.55 + 0.45 * (long_side / long0) ** 2)


def rungs(long0):
    """The ladder, best first. Steps per view go first (60 of 83.6 still trains every view 60 times; run 6 at 51 looked worse,
    so not below 60), then pixels, then frames."""
    longs = [l for l in (1280, 1080, 960) if l < long0]; low = longs[-1] if longs else long0
    return ([dict(spv=s, long=long0, thin=1) for s in (R4['steps_per_view'], 70.0, 60.0)]
            + [dict(spv=60.0, long=l, thin=1) for l in longs] + [dict(spv=60.0, long=low, thin=k) for k in (2, 3, 4)])


def fit(registered, area_m2, long0, sps0, train_left_h, cap=None):
    """The best rung whose predicted training time fits train_left_h (hours until the trainer's deadline). Always returns a
    recipe: after the ladder comes the preview recipe, cut to the steps that fit. `why` says what was given up."""
    cap = cap or cap_for(area_m2); tried = []
    for i, r in enumerate(rungs(long0)):
        tv = train_views(math.ceil(registered / r['thin'])); steps = max(PREVIEW['steps'], int(round(r['spv'] * tv)))
        sps = speed_at(sps0, long0, r['long']); h = steps * sps * MARGIN / 3600; tried.append(round(h, 2))
        if h <= train_left_h:
            why = 'run 4 recipe' if i == 0 else f"stepped down {i} rung(s): {r['spv']:.0f} steps/view, {r['long']} px, every {r['thin']}. frame"
            return dict(rung=i, kind='full', steps=steps, cap=cap, long=r['long'], thin=r['thin'], views=tv, spv=r['spv'],
                        s_per_step=round(sps, 4), hours=round(h, 2), why=why, tried_hours=tried)
    lp = min(long0, PREVIEW['long']); sps = speed_at(sps0, long0, lp)
    steps = int(min(PREVIEW['steps'], max(0.0, train_left_h) * 3600 / MARGIN / sps)); steps = max(500, steps)
    return dict(rung='preview', kind='preview', steps=steps, cap=min(cap, PREVIEW['cap']), long=lp, thin=1, views=train_views(registered),
                spv=None, s_per_step=round(sps, 4), hours=round(steps * sps * MARGIN / 3600, 2),
                why='nothing on the ladder fits: preview recipe', tried_hours=tried)


def save_steps(steps, sps, every_min=20.0, most=24):
    """Checkpoint steps about every `every_min` of training (at most `most`), ending with the last step."""
    k = max(200, int(round(max(every_min * 60 / max(sps, 1e-6), steps / most))))
    return [s for s in range(k, steps - k // 2, k)] + [steps]


# ---- the clock
class Clock:
    """Session time. t0 = when the Kaggle job started; deadline_h = when everything must be done (hours after t0)."""

    def __init__(self, t0, deadline_h=None, path=None):
        self.t0 = t0; self.deadline_h = float(deadline_h or os.environ.get('DEADLINE_H') or DEADLINE_H)
        self.deadline_h = min(self.deadline_h, HARD_H - 0.25); self.path = path; self.log = []; self.notes = {}

    def elapsed_h(self, now=None): return ((now or time.time()) - self.t0) / 3600

    def left_h(self, now=None): return self.deadline_h - self.elapsed_h(now)

    def train_left_h(self, mesh=True, now=None):
        """Hours the trainer may use from now: the time left minus what must still run after it."""
        return self.left_h(now) - (TAIL_MIN['base'] + (TAIL_MIN['mesh'] if mesh else 0)) / 60

    def train_deadline(self, mesh=True):
        """Epoch seconds at which the trainer stops itself (NM_DEADLINE)."""
        return self.t0 + (self.deadline_h - (TAIL_MIN['base'] + (TAIL_MIN['mesh'] if mesh else 0)) / 60) * 3600

    def mark(self, stage, t_start, **info):
        """One finished stage: its minutes and what it decided. Rewrites budget.json each time (a killed run keeps the trail)."""
        self.log.append(dict(stage=stage, minutes=round((time.time() - t_start) / 60, 1), at_h=round(self.elapsed_h(), 2), **info)); self.write()

    def note(self, **kv): self.notes.update(kv); self.write()

    def write(self):
        if not self.path: return
        try: json.dump(dict(deadline_h=self.deadline_h, elapsed_h=round(self.elapsed_h(), 2), stages=self.log, **self.notes),
                       open(self.path, 'w'), indent=1, default=str)
        except Exception as e: print('budget.json not written:', str(e)[:120], flush=True)


def mesh_minutes(frames, voxels=512):
    """mesh_tsdf.py on a T4, minutes: every voxel is projected into every camera. Measured at 512 voxels and 411 frames:
    35 min (run 4, from a 2DGS model, 59 % of the voxels seen) and 70 min (run 5, from the 3DGS model, 74 %); the 3DGS figure
    is used. (A fixed 20 min limit killed the mesh of Kaggle test 2 with 9 h to spare, 16 Nov.)"""
    return 2.0 + 68.0 * frames / 411 * (voxels / 512) ** 3


def tail(left_min, mesh=True, objects=False, frames=411, voxels=512):
    """What still runs after training and QA with left_min minutes to the deadline. (QA is done by now: asking for its minutes
    again skipped the mesh of a run whose trainer had stopped exactly on its deadline; Kaggle test 1, 16 Nov.)
      objects  the object layer + search index, if its 25 min fit before the copies and uploads. Asked for, it is held back
               first: the tour walks on the room box without a mesh, but has no "find anything" without the index
      mesh     the finest grid (voxels, then 384, 256) whose predicted time x 1.3 fits in what is left after that; its time
               limit is 2.5 x the prediction (at least MESH_FLOOR_MIN), never more than what is left"""
    o = bool(objects) and left_min >= TAIL_MIN['after'] + TAIL_MIN['objects']
    spare = left_min - TAIL_MIN['after'] - (TAIL_MIN['objects'] if o else 0)
    ladder = sorted({v for v in (voxels,) + MESH_VOX if v <= voxels}, reverse=True)
    vox = next((v for v in ladder if mesh_minutes(frames, v) * 1.3 <= spare), 0) if mesh else 0
    limit = min(spare, max(MESH_FLOOR_MIN, 2.5 * mesh_minutes(frames, vox))) if vox else 0
    return dict(mesh=bool(vox), mesh_voxels=vox, mesh_timeout_min=round(limit, 2), objects=o)


# ---- checkpoints (torch only here; the planner above needs none)
def _splats(ckpt):
    import torch
    s = torch.load(ckpt, map_location='cpu', weights_only=True)['splats']
    return {k: v.float().numpy() for k, v in s.items()}


def ckpt_to_splat(ckpt, dst):
    """A trainer checkpoint as a .splat (to_splat's 32-byte records, base colour only, not levelled): the snapshot that
    survives if the session dies. ~32 MB per million splats."""
    import numpy as np
    from to_splat import REC
    s = _splats(ckpt); n = len(s['means']); a = 1 / (1 + np.exp(-s['opacities'].reshape(-1)))
    scale = np.exp(s['scales']); rgb = 0.5 + 0.28209479 * s['sh0'].reshape(n, 3)
    q = s['quats'] / np.linalg.norm(s['quats'], axis=1, keepdims=True)
    o = np.argsort(-scale.prod(1) * a); rec = np.zeros(n, dtype=REC)
    rec['p'], rec['s'] = s['means'][o], scale[o].astype(np.float32)
    rec['c'] = (np.clip(np.concatenate([rgb, a[:, None]], 1), 0, 1) * 255).astype(np.uint8)[o]
    rec['r'] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)[o]; rec.tofile(dst); return n


def ckpt_to_ply(ckpt, dst):
    """A trainer checkpoint as the .ply the trainer itself would have written (x y z, f_dc, f_rest channel-major, opacity,
    scale, rot), for a trainer that had to be killed: QA, the tour and the object layer read this layout."""
    import numpy as np
    from plyfile import PlyData, PlyElement
    s = _splats(ckpt); n = len(s['means']); rest = s['shN'].transpose(0, 2, 1).reshape(n, -1)
    cols = ([('x', s['means'][:, 0]), ('y', s['means'][:, 1]), ('z', s['means'][:, 2])]
            + [(f'f_dc_{i}', s['sh0'].reshape(n, 3)[:, i]) for i in range(3)] + [(f'f_rest_{i}', rest[:, i]) for i in range(rest.shape[1])]
            + [('opacity', s['opacities'].reshape(-1))] + [(f'scale_{i}', s['scales'][:, i]) for i in range(3)]
            + [(f'rot_{i}', s['quats'][:, i]) for i in range(4)])
    v = np.zeros(n, dtype=[(k, '<f4') for k, _ in cols])
    for k, x in cols: v[k] = x
    PlyData([PlyElement.describe(v, 'vertex')]).write(dst); return n
