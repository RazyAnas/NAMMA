"""Fast training for gsplat's examples/simple_trainer.py (v1.5.3), patched in like room_prior / geo_prior / budget_patch
(plan of 26 Nov: a room took 5.8 h, 3.4 h of it splat training). Each part is off unless its environment variable is set, so
the patched trainer behaves exactly as before without them.
  NM_TRAIN_EVERY=k    train on every k-th training frame only. The held-out frames (every 8th of ALL frames) stay the same, so
                      runs with different frame counts are scored on the same photos.
  NM_RES=s:f          resolution schedule: photos and cameras shrunk to s (0..1) of their size for the first f (0..1) of the
                      steps, full size after (DashGaussian's idea as a plain schedule; most early steps only place big shapes).
  NM_DENSE=<npz>      dense start: the points of dense_init.py (xyz, rgb in the trainer's world) are added to the SfM points
                      the splats start from, so training does not spend its first thousands of steps growing splats.
  NM_CACHE=1          each loader worker keeps its decoded frames (as bytes) instead of reading + undistorting them every step;
                      only for runs of at most 350 training frames (memory).
Also fixes the import that made two-GPU training crash (HANDOFF_V3: "BilateralGrid not imported in the workers"): the colour
grid was imported only under __main__, which spawned workers never run.
  python fast_patch.py <simple_trainer.py>   ->  prints True / False (False: this trainer version has different lines)"""
import os, sys

MARK = '# fast_patch'
HERE = os.path.dirname(os.path.abspath(__file__))
# (text that must occur exactly once, its replacement)
EDITS = [
    ('        self.valset = Dataset(self.parser, split="val")\n',
     '        self.valset = Dataset(self.parser, split="val")\n'
     '        import fast_patch; fast_patch.thin(self.trainset); fast_patch.cache(self.trainset)  # fast_patch\n'),
    ('        raise ValueError("Please specify a correct init_type: sfm or random")\n',
     '        raise ValueError("Please specify a correct init_type: sfm or random")\n'
     "    if os.environ.get('NM_DENSE'):  # fast_patch\n"
     '        import fast_patch; points, rgbs = fast_patch.dense_points(points, rgbs)\n'),
    ('        pbar = tqdm.tqdm(range(init_step, max_steps))\n',
     '        import fast_patch; _nm_res = fast_patch.res_plan(max_steps)  # fast_patch\n'
     '        pbar = tqdm.tqdm(range(init_step, max_steps))\n'),
    ('            height, width = pixels.shape[1:3]\n\n            if cfg.pose_noise:\n',
     '            if _nm_res is not None and step < _nm_res[1] and masks is None:  # fast_patch\n'
     '                _nm_p = points if cfg.depth_loss else None\n'
     '                pixels, Ks, _nm_p = fast_patch.shrink(pixels, Ks, _nm_res[0], _nm_p)\n'
     '                if cfg.depth_loss: points = _nm_p\n'
     '            height, width = pixels.shape[1:3]\n\n            if cfg.pose_noise:\n'),
    ('    runner = Runner(local_rank, world_rank, world_size, cfg)\n',
     '    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:  # fast_patch: spawned workers (2 GPUs) never ran the __main__ imports\n'
     '        global BilateralGrid, color_correct, slice, total_variation_loss\n'
     '        if cfg.use_fused_bilagrid: from fused_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss\n'
     '        else: from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss\n'
     '    runner = Runner(local_rank, world_rank, world_size, cfg)\n'),
]


def patch(trainer_py):
    """Insert the fast-training hooks (idempotent). Returns False and leaves the file alone unless every edit point is found once."""
    s = open(trainer_py).read()
    if MARK in s: return True
    if any(s.count(a) != 1 for a, _ in EDITS): return False
    for a, b in EDITS: s = s.replace(a, b)
    s = f'import os, sys; sys.path.insert(0, {HERE!r})  {MARK}: where this module lives\n' + s
    open(trainer_py, 'w').write(s); return True


# ---- used by the patched trainer
def thin(ds):
    """Keep every k-th training frame (NM_TRAIN_EVERY). Only the training set: the held-out set is not touched."""
    k = int(os.environ.get('NM_TRAIN_EVERY') or 1)
    if k > 1:
        n = len(ds.indices); ds.indices = ds.indices[::k]
        print(f'FAST training on every {k}. frame: {len(ds.indices)} of {n} training frames', flush=True)


def res_plan(max_steps, spec=None):
    """NM_RES 's:f' -> (scale, step at which full size starts); None = always full size."""
    spec = os.environ.get('NM_RES') if spec is None else spec
    if not spec: return None
    s, f = (float(x) for x in spec.split(':')); s = min(1.0, max(0.1, s)); f = min(1.0, max(0.0, f))
    if s >= 1.0 or f <= 0.0: return None
    until = int(round(f * max_steps)); print(f'FAST resolution schedule: {s} of full size until step {until} of {max_steps}', flush=True)
    return s, until


def small_size(H, W, s): return max(8, int(round(H * s))), max(8, int(round(W * s)))


def shrink(pixels, Ks, s, points=None):
    """One training batch at scale s: photo [B, H, W, 3] averaged down (area), intrinsics [B, 3, 3] scaled by the real size
    ratios, SfM-depth sample positions [B, M, 2] (pixel indices) moved to the same places in the small picture."""
    import torch.nn.functional as F
    H, W = pixels.shape[1:3]; h, w = small_size(H, W, s)
    if (h, w) == (H, W): return pixels, Ks, points
    px = F.interpolate(pixels.permute(0, 3, 1, 2), size=(h, w), mode='area').permute(0, 2, 3, 1).contiguous()
    K = Ks.clone(); K[:, 0, :] = K[:, 0, :] * (w / W); K[:, 1, :] = K[:, 1, :] * (h / H)
    if points is not None:
        r = points.new_tensor([w / W, h / H]); points = ((points + 0.5) * r - 0.5).clamp(min=0)
        points = points.minimum(points.new_tensor([w - 1.0, h - 1.0]))
    return px, K, points


def dense_points(points, rgbs):
    """SfM start points + the dense ones (NM_DENSE: npz with xyz float32 [N, 3] in the trainer's world, rgb uint8 [N, 3])."""
    import numpy as np, torch
    d = np.load(os.environ['NM_DENSE']); xyz = torch.from_numpy(d['xyz']).float(); rgb = torch.from_numpy(d['rgb']).float() / 255.0
    print(f'FAST dense start: {len(xyz)} points added to {len(points)} SfM points', flush=True)
    return torch.cat([points, xyz]), torch.cat([rgbs, rgb])


NOMINAL = 30000                                    # the step count gsplat's schedules are written for; --steps-scaler shrinks them together
TOKENS = ('gs', 'prior', 'geo', 'pose', 'packed', 'vadam', '2gpu')       # kinds handled elsewhere (run.py start / the speed check)


def recipe(kind, steps, fused=True, dense=None):
    """Trainer settings for a variant kind such as 'fbg+thin3+res0.5_0.6+dense' (run.py VARIANTS / BENCH). Tokens:
      fbg          fused colour grid (the plain one if it did not build)        nobg   no colour grid
      thin<k>      train on every k-th training frame                           cache  frames kept in memory
      res<s>_<f>   s of full size for the first f of the steps                  dense  dense start points (if they were made)
    Returns None for a kind without any of them (the stock recipe), else env (text for the command line), flags, grid (the
    colour-grid flag), scaler (for --steps-scaler with --max-steps NOMINAL: exactly `steps` steps) and notes."""
    env, flags, grid, notes, fast = {}, [], '--use-bilateral-grid', [], False
    for t in kind.split('+'):
        if t == 'fbg':
            fast = True
            if fused: grid = '--use-fused-bilagrid'
            else: notes.append('fused colour grid not installed: plain grid')
        elif t == 'nobg': fast = True; grid = ''
        elif t == 'cache': fast = True; env['NM_CACHE'] = '1'
        elif t.startswith('thin') and t[4:].isdigit(): fast = True; env['NM_TRAIN_EVERY'] = str(int(t[4:]))
        elif t.startswith('res') and '_' in t:
            fast = True; s, f = t[3:].split('_'); env['NM_RES'] = f'{float(s)}:{float(f)}'
        elif t == 'dense':
            fast = True
            if dense: env['NM_DENSE'] = dense; flags.append(f"--init-scale {float(os.environ.get('DENSE_SCALE') or 0.5)}")
            else: notes.append('no dense points: sparse start')
        elif t not in TOKENS: raise ValueError(f'unknown variant kind {t!r} in {kind!r}')
    if not fast: return None
    return dict(env=' '.join(f'{k}={v}' for k, v in env.items()), flags=' '.join(flags), grid=grid, scaler=(int(steps) + 0.5) / NOMINAL, notes=notes)


CACHE_MAX = 350


def cache(ds):
    """NM_CACHE=1: a decoded frame is kept (photo as bytes, 3.5 MB at 1440 px) by the loader worker that read it."""
    if os.environ.get('NM_CACHE') != '1' or len(ds.indices) > CACHE_MAX: return False
    cls = type(ds)
    if getattr(cls, '_nm_cached', False): return True
    import torch
    orig = cls.__getitem__

    def get(self, item):
        store = self.__dict__.setdefault('_nm_store', {})
        if item not in store:
            d = orig(self, item); d['image'] = d['image'].to(torch.uint8); store[item] = d
        d = dict(store[item]); d['image'] = d['image'].float(); return d

    cls.__getitem__ = get; cls._nm_cached = True; print('FAST frames are kept in memory after the first read', flush=True); return True


if __name__ == '__main__':
    print(patch(sys.argv[1]))
