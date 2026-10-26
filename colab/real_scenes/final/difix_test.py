import gc, glob, json, os, shutil, subprocess, sys, time
"""Kaggle job anasrazy/namma-difix: Difix3D+ (NVIDIA, CVPR 2025 oral, github.com/nv-tlabs/Difix3D; NVIDIA licence = research /
non-commercial only, fine for this personal project) on a finished run. Its own gsplat trainer (examples/gsplat/
simple_trainer_difix3d.py), started from the run's splats: every few thousand steps it renders views beside the filmed path,
cleans them with the single-step Difix model (SD-Turbo based, guided by the nearest real frame) and trains on them at 0.3 weight
next to the real frames (progressive 3D update). Real frames keep full weight, so filmed views should not get worse.
  views      train cameras (never the held-out ones) shifted ~50 cm left/right (same distance as v3_qa's 'side' and the FreeFix test)
  proof      held-out frames: PSNR / SSIM / LPIPS before and after, exposure-fair like run's metrics.json (must not drop);
             off-path renders before | Difix | after in difix_sheet.jpg
  coords     Difix's COLMAP loader normalises without gsplat 1.5.3's upside-down fix: cameras moved into the run's frame
  caveat     Difix was trained on 112 of DL3DV-Benchmark's 140 scenes; run 4's hotel room is one of the 140, so it may have
             seen this room. The fair test is the owner's own home.
  out        model.ply (run's coordinates, gsplat PLY layout -> tour), difix_report.json, sheet
Inputs (v3/difix_gh.py): namma-qwen-in (the run's model.ply + sparse) and the run's own frames dataset (never other images)."""
T0 = time.time(); W = '/kaggle/temp/dx'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
CFG = json.loads(os.environ.get('JOB_CFG', '{}')); REP = dict(cfg=CFG)
SHIFT, NEVAL, NSHEET = float(CFG.get('shift', 0.15)), int(CFG.get('eval_views', 16)), int(CFG.get('sheet_views', 8))
FT, EVERY, STRIDE = int(CFG.get('finetune_steps', 12000)), int(CFG.get('fix_every', 2000)), int(CFG.get('target_stride', 3))
os.environ['HF_HOME'] = f'{W}/hf'; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r

def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)
def save_report(): json.dump(REP, open(f'{O}/difix_report.json', 'w'), indent=1)

sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
sh(f'git clone -q --depth 1 https://github.com/nv-tlabs/Difix3D {W}/difix', check=True)
F = f'{W}/repo/colab/real_scenes/final'; DX = f'{W}/difix'; DG = f'{DX}/examples/gsplat'; EX = f'{W}/gsplat/examples'
tr = f'{DG}/simple_trainer_difix3d.py'; src = open(tr).read()                                    # small patches, each must apply
for a, b in [('self.difix.to("cuda")', 'self.difix.to(os.environ.get("DIFIX_DEVICE", "cuda"))'),                 # Difix on the 2nd T4
             ('self.parser.camtoworlds[self.valset.indices]', 'self.target_poses'),                               # our off-path views
             ('distance=0.5)', 'distance=float(os.environ.get("DIFIX_STEP", "0.5")))'),
             ('image=image, ref_image=ref_image,', 'image=_dsz(image), ref_image=_dsz(ref_image),')]:            # Difix's 1024 x 576
    assert a in src, a; src = src.replace(a, b)
src = src.replace('\n@dataclass\nclass Config:', '\ndef _dsz(im):\n    k = 1024 / max(im.size)\n'
                  '    return im.resize((round(im.size[0] * k / 8) * 8, round(im.size[1] * k / 8) * 8), Image.LANCZOS)\n\n@dataclass\nclass Config:', 1)
assert '_dsz(im)' in src; open(tr, 'w').write(src)
dc = f'{DG}/datasets/colmap.py'; src = open(dc).read()                         # Difix trains on every 8th frame and tests on the
a = ('                self.indices = indices[indices % self.parser.test_every == 0]\n'          # rest; ours is the other way round:
     '            else:\n                self.indices = indices[indices % self.parser.test_every != 0]')  # every 8th frame is held out
assert src.count(a) == 1, 'Difix split changed'; open(dc, 'w').write(src.replace(a, a.replace('== 0]', '@@').replace('!= 0]', '== 0]').replace('@@', '!= 0]')))
sh(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat && touch {EX}/datasets/__init__.py; '
   f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; '
   f'pip install -q -c {W}/pin.txt ninja gsplat==1.5.3 plyfile viser nerfview==0.0.2 tyro tensorly tensorboard "torchmetrics[image]" '
   '"imageio[ffmpeg]" scikit-learn matplotlib pyyaml einops accelerate 2>&1 | tail -3; pip uninstall -y -q torchao; '
   f'pip install -q -c {W}/pin.txt diffusers==0.25.1 transformers==4.38.0 huggingface-hub==0.25.1 peft==0.9.0 2>&1 | tail -3; '  # Difix's pins
   'pip install -q --no-deps "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
   'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
tgz = glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True)                          # namma-prep's compiled gsplat
if tgz:
    sh(f'mkdir -p ~/.cache && tar xzf {tgz[0]} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'python {F}/gsplat_warm.py', check=True)

# 1 the run's frames (its own dataset only), cameras and model
import numpy as np, torch
sys.path.insert(0, F); import frames
from PIL import Image
D = f'{W}/data'; ply = glob.glob('/kaggle/input/**/model.ply', recursive=True); assert ply, 'model.ply not attached'
zs = glob.glob('/kaggle/input/**/sparse.zip', recursive=True)
if zs: shutil.unpack_archive(zs[0], D)
else: shutil.copytree(os.path.dirname(glob.glob('/kaggle/input/**/sparse/**/cameras.bin', recursive=True)[0]), f'{D}/sparse/0')
FDS = CFG.get('frames_dataset', 'namma-dl3dv').split('/')[-1]
for z in glob.glob(f'/kaggle/input/**/{FDS}/**/frames.zip', recursive=True): shutil.unpack_archive(z, f'{W}/fr')
cnt = {}
for f in [g for g in glob.glob('/kaggle/input/**/*', recursive=True) if f'/{FDS}/' in g] + glob.glob(f'{W}/fr/**/*', recursive=True):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '/sparse' not in f: cnt[os.path.dirname(f)] = cnt.get(os.path.dirname(f), 0) + 1
assert cnt, f'no frames found in the attached dataset {FDS}'
say('frames', frames.from_images(max(cnt, key=cnt.get), f'{D}/images', int(CFG.get('nframes', 420)), int(CFG.get('long', 1440))))
# the run's frame (gsplat 1.5.3 normalisation) in a separate process: its examples/datasets clashes with Difix's
sh(f'cd {EX} && python -c "import json; from datasets.colmap import Parser; p = Parser(\'{D}\', factor=1, normalize=True, test_every=8); '
   f'json.dump(dict(T=p.transform.tolist(), scale=float(p.scene_scale), n=len(p.image_names), c=p.camtoworlds[:, :3, 3].tolist()), open(\'{W}/gs.json\', \'w\'))"', check=True)
gs = json.load(open(f'{W}/gs.json')); T_g = np.array(gs['T'])
assert len(glob.glob(f'{D}/images/*.jpg')) == gs['n'], 'frames != cameras: wrong frames'
from plyfile import PlyData
v = PlyData.read(ply[0])['vertex'].data; t_ = lambda *k: torch.tensor(np.stack([v[x] for x in k], 1), dtype=torch.float32)
nr = len([n for n in v.dtype.names if n.startswith('f_rest_')]) // 3
splats = dict(means=t_('x', 'y', 'z'), scales=t_('scale_0', 'scale_1', 'scale_2'), quats=t_('rot_0', 'rot_1', 'rot_2', 'rot_3'),
              opacities=torch.tensor(np.asarray(v['opacity']), dtype=torch.float32), sh0=t_('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None],
              shN=t_(*[f'f_rest_{i}' for i in range(3 * nr)]).reshape(-1, 3, nr).transpose(1, 2).contiguous())
say(len(v), 'splats, SH degree', int(np.sqrt(nr + 1)) - 1); assert nr == 15, 'expects SH degree 3'

# 2 Difix3D's trainer, its cameras moved into the run's frame
sys.path[:0] = [DG, DX, f'{F}/fused_ssim_shim']; os.chdir(DG)
if torch.cuda.device_count() > 1: os.environ['DIFIX_DEVICE'] = 'cuda:1'
import datasets.colmap as dxc
from datasets.normalize import transform_cameras, transform_points
class RunFrameParser(dxc.Parser):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        B = T_g @ np.linalg.inv(self.transform)                                             # Difix frame -> run's frame
        self.camtoworlds = transform_cameras(B, self.camtoworlds); self.points = transform_points(B, self.points)
        self.transform = B @ self.transform; self.scene_scale = gs['scale']
dxc.Parser = RunFrameParser
import simple_trainer_difix3d as std
std.Parser = RunFrameParser
from gsplat.strategy import DefaultStrategy
from lib_bilagrid import color_correct
RES = f'{W}/res'; S0 = 29999
cfg = std.Config(data_dir=D, data_factor=1, result_dir=RES, test_every=8, normalize_world_space=True, disable_viewer=True, sh_degree=3,
                 max_steps=S0 + FT, eval_steps=[], save_steps=[], fix_steps=list(range(S0 + 1, S0 + FT, EVERY)),
                 novel_data_lambda=float(CFG.get('novel_lambda', 0.3)), strategy=DefaultStrategy(verbose=False))
run = std.Runner(0, 0, 1, cfg)
assert np.allclose(run.parser.camtoworlds[:, :3, 3], np.array(gs['c']), atol=1e-4), 'camera frames differ'
for k in run.splats.keys(): run.splats[k].data = splats[k].to(run.device)
C = run.parser.camtoworlds; tri = np.array(run.trainset.indices); assert all(i % 8 for i in tri), 'held-out frame in training'
side = SHIFT * float(np.linalg.norm(C[:, :3, 3] - C[:, :3, 3].mean(0), axis=1).max())
targets = []
for i in tri[::STRIDE]:
    for sgn in (1, -1):
        c = C[i].copy(); c[:3, 3] += sgn * side * c[:3, 0]; targets.append(c)                    # camera's own left/right
run.target_poses = np.stack(targets); os.environ['DIFIX_STEP'] = str(side / 2)              # reach the targets in 2 rounds
REP.update(side_units=round(side, 4), targets=len(targets), train_views=len(tri), heldout_views=len(run.valset))
say('Difix3D runner ready; side', round(side, 4), 'units;', len(targets), 'off-path targets;', len(tri), 'train,', len(run.valset), 'held-out')

@torch.no_grad()
def render(c2w):
    K = torch.from_numpy(list(run.parser.Ks_dict.values())[0]).float().to(run.device)[None]
    w, h = list(run.parser.imsize_dict.values())[0]
    rgb, _, _ = run.rasterize_splats(camtoworlds=torch.from_numpy(c2w).float().to(run.device)[None], Ks=K, width=w, height=h, sh_degree=3)
    return rgb[0].clamp(0, 1)

@torch.no_grad()
def heldout_eval(tag):
    m = {k: [] for k in ('psnr', 'ssim', 'lpips', 'cc_psnr', 'cc_ssim', 'cc_lpips')}
    for j in range(min(NEVAL, len(run.valset))):
        d = run.valset[j]; gt = (d['image'].float() / 255).to(run.device)[None].permute(0, 3, 1, 2)
        pr = render(d['camtoworld'].numpy())[None].permute(0, 3, 1, 2); cc = color_correct(pr.permute(0, 2, 3, 1), gt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for p_, pre in ((pr, ''), (cc, 'cc_')):
            m[pre + 'psnr'].append(float(run.psnr(p_, gt))); m[pre + 'ssim'].append(float(run.ssim(p_, gt))); m[pre + 'lpips'].append(float(run.lpips(p_, gt)))
    m = {k: round(float(np.mean(x)), 4) for k, x in m.items()}; REP[f'heldout_{tag}'] = m; say('held-out', tag, m); save_report()

def u8(t): return (t.detach().cpu().numpy() * 255).astype(np.uint8)
sheet_ix = np.linspace(0, len(targets) - 1, NSHEET).astype(int)
heldout_eval('before'); before = [u8(render(run.target_poses[i])) for i in sheet_ix]
t1 = time.time(); run.train(step=S0); REP['train_minutes'] = round((time.time() - t1) / 60, 1)
heldout_eval('after'); after = [u8(render(run.target_poses[i])) for i in sheet_ix]
last = max(int(x) for x in os.listdir(f'{RES}/renders/novel'))                                  # Difix output of the last round
fixed = [np.asarray(Image.open(f'{RES}/renders/novel/{last}/Fixed/{i:04d}.png').convert('RGB')) for i in sheet_ix]
rows = [np.concatenate([b, f, a], 1) for b, f, a in zip(before, fixed, after)]
Image.fromarray(np.concatenate(rows, 0)).resize((rows[0].shape[1] // 2, len(rows) * rows[0].shape[0] // 2)).save(f'{O}/difix_sheet.jpg', quality=88)
for k, i in enumerate(sheet_ix): Image.fromarray(rows[k]).save(f'{O}/view{k}_before_difix_after.jpg', quality=90)

# 3 refined splats -> model.ply in the run's coordinates (the gsplat PLY layout make_tour.py reads)
from plyfile import PlyElement
s = {k: p.detach().cpu() for k, p in run.splats.items()}; n = len(s['means'])
shN = s['shN'].transpose(1, 2).reshape(n, -1).numpy(); cols = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(shN.shape[1])] + \
      ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
arr = np.concatenate([s['means'].numpy(), s['sh0'][:, 0].numpy(), shN, s['opacities'].numpy()[:, None], s['scales'].numpy(), s['quats'].numpy()], 1)
vv = np.zeros(n, [(c, 'f4') for c in cols])
for j, c in enumerate(cols): vv[c] = arr[:, j]
os.makedirs(f'{O}/full', exist_ok=True); PlyData([PlyElement.describe(vv, 'vertex')]).write(f'{O}/full/model.ply')
shutil.rmtree(f'{RES}/ckpts', ignore_errors=True)
REP.update(splats_after=n, minutes=round((time.time() - T0) / 60, 1), ok=True); save_report(); say('DIFIX DONE', json.dumps(REP))
