"""From aligned cameras to the deliverables: largest group of passes that agree -> fused cloud (init) + floor plan + photo viewer
-> two splat fits in parallel (one per GPU) -> levelled .splat files. Used by namma-model and namma-all.
build(aligned_npz, frames_dir, gsplat_tgz, out, variants)"""
import glob, json, os, subprocess, sys, time
import numpy as np
from PIL import Image
F = os.path.dirname(os.path.abspath(__file__))

V2 = [dict(name='A', out='splat.ply', kw=dict(steps=20000, max_seconds=1200, depth_w=0.05, sh=3, aniso=0.1)),   # v2: no needles
      dict(name='B', out='splat_mcmc.ply', kw=dict(steps=20000, max_seconds=1200, depth_w=0.05, sh=3, mcmc_cap=1000000, aniso=0.1))]

def sh(c, check=False):
    print(f'\n[{time.strftime("%H:%M:%S")}] $ {c[:150]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(c[:80])

def build(aligned_npz, frames_dir, gsplat_tgz, out, variants=V2, tmp='/kaggle/temp/build'):
    t0 = time.time(); os.makedirs(tmp, exist_ok=True)
    z = np.load(aligned_npz); d = {k: z[k] for k in z.files}; files = sorted(glob.glob(f'{frames_dir}/*.jpg'))
    has = d['depth'].reshape(len(d['idx']), -1).max(1) > 0
    grp = d['group'] if 'group' in d else np.zeros(len(d['idx']), int)
    g0 = (grp == 0) & has; G = {k: v[g0] for k, v in d.items()}; Hd, Wd = G['depth'].shape[1:]
    G['rgb'] = np.stack([np.asarray(Image.open(files[int(g)]).convert('RGB').resize((Wd, Hd), Image.BILINEAR)) for g in G['idx']])
    S = f'{tmp}/g0'; os.makedirs(S, exist_ok=True); np.savez(f'{S}/da3_sfm.npz', **G)
    if not os.path.exists(f'{S}/frames'): os.symlink(frames_dir, f'{S}/frames')
    print(f'MODEL GROUP: {g0.sum()} of {len(g0)} frames (ids {int(G["idx"].min())}-{int(G["idx"].max())})', flush=True)
    sh('pip -q install open3d trimesh scipy matplotlib plyfile 2>&1 | tail -1')
    sys.path.insert(0, F); import fuse_nav as FN
    os.makedirs(f'{out}/model', exist_ok=True); FN.fuse(f'{S}/da3_sfm.npz', f'{out}/model')
    try: print(json.dumps({k: v for k, v in FN.navigate(f'{out}/model/cloud.npz', f'{out}/model', up=FN.camera_up(f'{S}/da3_sfm.npz')).items() if k != 'demo_path_cells'}), flush=True)
    except Exception as e: print('floor plan failed:', e, flush=True)
    sh(f'cd {F} && {sys.executable} photo_viewer.py {S} - {out}/viewer')
    sh('pip -q install ninja "gsplat==1.5.3" 2>&1 | tail -1', check=True)                 # same build as namma-prep's cache
    sh(f'mkdir -p ~/.cache && tar xzf {gsplat_tgz} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
    sh(f'{sys.executable} {F}/gsplat_warm.py', check=True)
    procs = []
    for gpu, v in enumerate(variants):
        job = json.dumps(dict(args=[f'{S}/da3_sfm.npz', f'{out}/model/cloud.npz', frames_dir, f'{out}/model/{v["out"]}'], kw=v['kw']))
        lg = open(f'{out}/model/train_{v["name"]}.log', 'w')
        procs.append((v, lg.name, subprocess.Popen([sys.executable, f'{F}/train_worker.py', job], env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu % 2)},
                                                   stdout=lg, stderr=subprocess.STDOUT)))
    for v, lg, p in procs:
        p.wait(); print(f'---- splat {v["name"]} (exit {p.returncode}) ----'); print(''.join(l for l in open(lg) if 'step ' not in l or '000 loss' in l)[-3000:], flush=True)
        ply = f'{out}/model/{v["out"]}'
        if os.path.exists(ply): sh(f'{sys.executable} {F}/to_splat.py {ply} {ply[:-4]}.splat {out}/model/nav.json')   # levelled, floor at y = 0
    print(f'MODEL DONE in {time.time()-t0:.0f}s; files:', sorted(os.listdir(f'{out}/model')), flush=True)
