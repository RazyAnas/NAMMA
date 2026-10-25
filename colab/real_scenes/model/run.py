# Kaggle job anasrazy/namma-model (2x T4, ~35 min): THE 3D MODEL on cameras that agree. Every splat before was trained on camera
# positions whose walk passes disagreed (look-alike merges), which explains the fog and needles. Now:
#   1. align the walk passes (align_passes: video bridges + whole-group check + ICP) -> group 0 = 162 frames agreeing at ~31 dB;
#   2. group-0 data: cameras + dense depth + real colours; fused point cloud (init), floor plan + walkable grid, real-photo viewer;
#   3. two splat fits in parallel (one per GPU): A = default densification, SH3 (tile reflections), depth anchor;
#      B = MCMC 1.5M splats, SH3, depth anchor, anisotropy cap (needles). Held-out renders + stats in the log;
#   4. .splat copies (8x smaller) for SuperSplat. Output: /kaggle/working/{aligned,model,viewer}.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time(); W = '/kaggle/temp/model'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:150]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(c[:80])
sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F)
cache = os.path.dirname(glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True)[0])
dn = glob.glob('/kaggle/input/**/dense/da3_sfm.npz', recursive=True); assert dn, 'no namma-dense output attached'
D = f'{W}/dense'; os.makedirs(D, exist_ok=True); os.symlink(dn[0], f'{D}/da3_sfm.npz'); os.symlink(f'{cache}/frames', f'{D}/frames')
sh(f'cd {F} && {sys.executable} align_passes.py {D} - {W}/al --workers 4 --bridge {cache} 2>&1 | grep -v Warning', check=True)
os.makedirs(f'{O}/aligned', exist_ok=True); shutil.copy(f'{W}/al/da3_sfm.npz', f'{O}/aligned/'); shutil.copy(f'{W}/al/align_report.json', O)

# group 0 with real colours (the frames added by namma-dense have no DA3 colours)
import numpy as np
from PIL import Image
z = np.load(f'{W}/al/da3_sfm.npz'); d = {k: z[k] for k in z.files}; files = sorted(glob.glob(f'{cache}/frames/*.jpg'))
g0 = (d['group'] == 0) & (d['depth'].reshape(len(d['idx']), -1).max(1) > 0); G = {k: v[g0] for k, v in d.items()}
Hd, Wd = G['depth'].shape[1:]
G['rgb'] = np.stack([np.asarray(Image.open(files[int(g)]).convert('RGB').resize((Wd, Hd), Image.BILINEAR)) for g in G['idx']])
S = f'{W}/g0'; os.makedirs(S, exist_ok=True); np.savez(f'{S}/da3_sfm.npz', **G); os.symlink(f'{cache}/frames', f'{S}/frames')
print(f'group 0: {g0.sum()} frames, ids {int(G["idx"].min())}-{int(G["idx"].max())}', flush=True)

sh('pip -q install open3d trimesh scipy matplotlib plyfile 2>&1 | tail -1')
import fuse_nav as FN
os.makedirs(f'{O}/model', exist_ok=True); FN.fuse(f'{S}/da3_sfm.npz', f'{O}/model')
try: print(json.dumps({k: v for k, v in FN.navigate(f'{O}/model/cloud.npz', f'{O}/model', up=FN.camera_up(f'{S}/da3_sfm.npz')).items() if k != 'demo_path_cells'}), flush=True)
except Exception as e: print('floor plan failed:', e, flush=True)
sh(f'cd {F} && {sys.executable} photo_viewer.py {S} - {O}/viewer')

# gsplat build from namma-prep (as run.py MODE=train), then two fits in parallel
sh('pip -q install ninja "gsplat==1.5.3" 2>&1 | tail -1', check=True)
sh(f'mkdir -p ~/.cache && tar xzf {cache}/gsplat_ext.tgz -C ~/.cache')
sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'{sys.executable} {F}/gsplat_warm.py', check=True)
V = [dict(name='A', out='splat.ply', kw=dict(steps=15000, max_seconds=900, depth_w=0.05, sh=3)),     # 15 min each, both GPUs at once
     dict(name='B', out='splat_mcmc.ply', kw=dict(steps=15000, max_seconds=900, depth_w=0.05, sh=3, mcmc_cap=1500000, aniso=0.1))]
procs = []
for gpu, v in enumerate(V):
    job = json.dumps(dict(args=[f'{S}/da3_sfm.npz', f'{O}/model/cloud.npz', f'{cache}/frames', f'{O}/model/{v["out"]}'], kw=v['kw']))
    lg = open(f'{O}/model/train_{v["name"]}.log', 'w')
    procs.append((v, lg.name, subprocess.Popen([sys.executable, f'{F}/train_worker.py', job], env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                               stdout=lg, stderr=subprocess.STDOUT)))
for v, lg, p in procs:
    p.wait(); print(f'---- splat {v["name"]} (exit {p.returncode}) ----'); print(''.join(l for l in open(lg) if 'step ' not in l or '000 loss' in l)[-3000:], flush=True)
    ply = f'{O}/model/{v["out"]}'
    if os.path.exists(ply): sh(f'{sys.executable} {F}/to_splat.py {ply} {ply[:-4]}.splat')
print(f'done in {time.time()-T0:.0f}s; files:', sorted(os.listdir(f'{O}/model')), flush=True)
