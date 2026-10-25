# ONE-SHOT Kaggle job (UNTESTED on GPU, written in a container with no GPU/HF/Kaggle access): video -> walkable model.
# Stage 1 sharpest frames -> 2 Depth Anything 3 (poses+depth+gaussians, feed-forward, ~1 min) -> 3 fuse + floor plan + A* (tested on CPU)
# -> 4 package for the viewer. Matrix-3D gap filling stays a separate job (colab/real_scenes/matrix3d/kaggle) and is fed pano_raw from here.
# Every stage logs and continues, so a failed stage never loses the earlier outputs in /kaggle/working.
import glob, json, os, shutil, subprocess, sys, time, traceback
T0 = time.time(); O = '/kaggle/working'; W = '/kaggle/temp/final'; os.makedirs(W, exist_ok=True)
def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r
MODE = os.environ.get('MODE', 'full')   # full: everything | prep: data + gsplat build -> /kaggle/working/cache | train: cache -> splats
RUN = {'full': {'1', '2', '2b', '3', '4'}, 'prep': {'1', '2', '2b', '3', 'cache'}, 'train': {'load', '4'}, 'ffgs': {'load', 'ffgs'}}[MODE]
print('MODE', MODE, flush=True)
def stage(name):
    def deco(f):
        if name.split()[0] not in RUN: return
        try: f(); print(f'== {name}: ok [{time.time()-T0:.0f}s]', flush=True)
        except Exception: print(f'== {name}: FAILED'); traceback.print_exc()
    return deco
sh('ls -R /kaggle/input | head -40')
if MODE not in ('train', 'ffgs'):
    _v = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mov', recursive=True))
    assert _v, 'no video under /kaggle/input: attach dataset anasrazy/namma-room-2026-10-22 (see ls above)'
    VIDEO = _v[0]; print('VIDEO', VIDEO)
NFRAMES = int(os.environ.get('NFRAMES', 600))      # ~2.7 per second of video: density is the lever (lobby: 6/s -> 26.9 dB)
DA3_EVERY = int(os.environ.get('DA3_EVERY', 3))    # dense depth for every 3rd frame (200 frames, ~12 min); SfM uses all
if not os.path.exists(f'{W}/repo'): sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
GSPLAT = 'gsplat==1.5.3'   # pinned: the cached CUDA build must match the installed sources
sys.path.insert(0, f'{W}/repo/colab/real_scenes/final')

@stage('1 frames')
def _():
    sh('pip -q install opencv-python-headless', check=True)
    import cv2, numpy as np
    cap = cv2.VideoCapture(VIDEO); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); win = max(1, n // NFRAMES); os.makedirs(f'{W}/frames', exist_ok=True)
    best, bs, k = None, -1, 0                                   # sequential decode (seeking in 4K H.264 is what took 16 min)
    for i in range(n):
        if not cap.grab(): break
        if i % 3 == 0:
            ok, f = cap.retrieve()
            if ok:
                g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)), cv2.COLOR_BGR2GRAY)
                sc = cv2.Laplacian(g, cv2.CV_64F).var()         # sharpness = variance of Laplacian
                if sc > bs: best, bs = f, sc
        if (i + 1) % win == 0 and best is not None and k < NFRAMES:
            h, w = best.shape[:2]; s = 1008 / max(h, w); cv2.imwrite(f'{W}/frames/{k:04d}.jpg', cv2.resize(best, (int(w * s), int(h * s)))); k += 1; best, bs = None, -1
    print(k, 'frames')

@stage('2 depth anything 3')
def _():
    # Kaggle runs Python 3.13.15, DA3's metadata says <=3.13 -> ignore it; no xformers (DA3 falls back to torch attention).
    sh('pip -q install --no-deps --ignore-requires-python "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" 2>&1 | tail -3', check=True)
    sh('pip -q install einops omegaconf huggingface_hub safetensors pillow imageio plyfile trimesh scipy addict moviepy 2>&1 | tail -2')
    import numpy as np, torch, importlib
    for _ in range(25):                                         # self-heal: install whatever DA3 turns out to import
        try: from depth_anything_3.api import DepthAnything3; break
        except ModuleNotFoundError as e:
            pkg = {'cv2': 'opencv-python-headless', 'PIL': 'pillow', 'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}.get(e.name.split('.')[0], e.name.split('.')[0])
            print('missing', e.name, '-> pip', pkg); sh(f'pip -q install {pkg} 2>&1 | tail -1')
    model = DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE').to('cuda').eval()
    files = sorted(glob.glob(f'{W}/frames/*.jpg'))[::DA3_EVERY]; fid = np.array([int(os.path.basename(p)[:4]) for p in files])
    import gc, chunk_align as CA
    CH, STEP = int(os.environ.get('CHUNK', 8)), int(os.environ.get('STEP', 5))   # 8 views fit a T4; 30 and 15 ran out of memory
    chunks = []
    for a in range(0, max(1, len(files) - CH + STEP), STEP):
        idx = list(range(a, min(a + CH, len(files))))
        if len(idx) < 3: break
        pred = None
        for attempt in range(2):
            try:
                with torch.no_grad(): pred = model.inference([files[i] for i in idx]); break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' not in str(e).lower() and 'cudnn' not in str(e).lower(): raise
            gc.collect(); torch.cuda.empty_cache()        # free OUTSIDE the except block (the traceback pins the tensors there)
        if pred is None: print('chunk', a, 'skipped (OOM)'); continue
        ext = np.concatenate([pred.extrinsics, np.tile([[[0, 0, 0, 1.]]], (len(idx), 1, 1))], 1)
        chunks.append(dict(idx=idx, depth=np.asarray(pred.depth), K=np.asarray(pred.intrinsics), c2w=np.linalg.inv(ext), rgb=np.asarray(pred.processed_images)))
        print('chunk', a, 'ok', len(chunks), flush=True); gc.collect(); torch.cuda.empty_cache()
    D, Ks, C, have = CA.merge(chunks)                       # each chunk is in its own frame; fit a similarity on the shared frames
    rgb = np.zeros(D.shape + (3,), np.uint8)
    for c in chunks:
        for j, g in enumerate(c['idx']): rgb[g] = c['rgb'][j]
    k = np.where(have)[0]; np.savez(f'{W}/da3.npz', depth=D[k], K=Ks[k], c2w=C[k], rgb=rgb[k], idx=fid[k])
    print('merged', len(k), 'of', len(files), 'frames; median depth', float(np.median(D[k][D[k] > 0])), '(metric)')

@stage('2b SfM cameras (hloc SuperPoint+LightGlue, proven on this video: 574/578)')
def _():
    # DA3's cameras come from 40 chained chunks and drift (training had to move them 7.6 cm / 2.6 deg on average). SfM gives one
    # globally consistent set; DA3 then only supplies metric scale and dense depth (sfm_align.py). Python 3.10 venv: the combo
    # (torch 2.1.2 cu118 + hloc + LightGlue) that solved this video before.
    E = f'{W}/sfmenv/bin/python'; P = f'--python {E}'
    sh(f'pip -q install uv 2>&1 | tail -1; unset UV_SYSTEM_PYTHON UV_PYTHON PIP_TARGET; uv venv -q -p 3.10 {W}/sfmenv && '
       f'uv pip install -q {P} torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 > {W}/sfm_pip.log 2>&1; '
       f'git clone -q --recursive https://github.com/cvg/Hierarchical-Localization {W}/hloc && '
       f'uv pip install -q {P} "numpy<2" -e {W}/hloc "lightglue @ git+https://github.com/cvg/LightGlue" >> {W}/sfm_pip.log 2>&1; tail -3 {W}/sfm_pip.log')
    sh(f'{E} -c "import hloc, lightglue, pycolmap; print(\'hloc ok, pycolmap\', pycolmap.__version__)"', check=True)
    sh(f'cd {W} && {E} {W}/repo/colab/real_scenes/final/sfm_hloc.py {W}/frames {W}/sfm 2>&1 | grep -v "it/s" | grep -v "s/it" | tail -25')
    import sfm_align; sfm_align.align(f'{W}/sfm/sfm.npz', f'{W}/da3.npz', f'{W}/da3_sfm.npz')

@stage('3 fuse + floor plan + path')
def _():
    sh('pip -q install open3d trimesh scipy matplotlib 2>&1 | tail -1')
    import fuse_nav as F
    src = f'{W}/da3_sfm.npz' if os.path.exists(f'{W}/da3_sfm.npz') else f'{W}/da3.npz'; print('geometry from', src)
    F.fuse(src, f'{O}/model'); print(json.dumps({k: v for k, v in F.navigate(f'{O}/model/cloud.npz', f'{O}/model', up=F.camera_up(src)).items() if k != 'demo_path_cells'}))
    import build_viewer; print(build_viewer.build(f'{O}/model'), '-> model/viewer.html')
@stage('load cache from namma-prep (frames, cameras + depth, cloud, compiled gsplat)')
def _():
    global CACHE
    hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no cache: run namma-prep first'
    CACHE = os.path.dirname(hits[0]); print('cache', CACHE, len(os.listdir(f'{CACHE}/frames')), 'frames')
    sh(f'pip -q install ninja "{GSPLAT}" 2>&1 | tail -1', check=True)
    # restore the build, then make it newer than the freshly installed sources so ninja does not rebuild (~5 min saved)
    sh(f'mkdir -p ~/.cache && tar xzf {CACHE}/gsplat_ext.tgz -C ~/.cache && find ~/.cache/torch_extensions -type f -exec touch {{}} +')
    sh(f'python {W}/repo/colab/real_scenes/final/gsplat_warm.py', check=True)   # once, before two GPUs load it at the same time
    os.makedirs(f'{O}/model', exist_ok=True)
    for x in ('cloud.npz', 'walk.npz'): shutil.copy(f'{CACHE}/{x}', f'{O}/model/{x}')

@stage('4 splat fits: W1 (600 views) and W2 (+ strong depth anchor), with diagnostics')
def _():
    import torch
    if MODE != 'train':
        sh(f'nvcc --version | tail -1; pip -q install ninja "{GSPLAT}" 2>&1 | tail -1', check=True)
    import train_splat as T, fuse_nav as F
    steps = int(os.environ.get('STEPS', 15000)); f, o = f'{W}/frames', f'{O}/model'
    def run(name, *a, **k):
        try: T.train(*a, **k)
        except Exception: print(f'variant {name} FAILED (earlier files are intact)'); traceback.print_exc()
    s = f'{W}/da3_sfm.npz' if os.path.exists(f'{W}/da3_sfm.npz') else f'{W}/da3.npz'; cs = f'{O}/model/cloud.npz'
    # Last run: held-out renders were fog (fan/window/door gone) although PSNR read 18.8 dB. Two candidate causes, one change each:
    # W1 = 3x denser views (600), same recipe as before; W2 = W1 + a stronger depth anchor that never fades fully (fights floaters).
    # Both write diagnostics: renders at step 0 and at the end (train + held-out views), splat size/opacity stats, held-out depth error.
    if MODE == 'train':
        s, cs, f = f'{CACHE}/da3_sfm.npz', f'{CACHE}/cloud.npz', f'{CACHE}/frames'
        variants = json.loads(os.environ['VARIANTS'])                     # [{"name":..., "out":..., "kw":{...}}], one per GPU
        procs = []
        for gpu, v in enumerate(variants):
            job = json.dumps(dict(args=[s, cs, f, f'{o}/{v["out"]}'], kw=v['kw']))
            log = open(f'{o}/train_{v["name"]}.log', 'w')
            procs.append((v['name'], log.name, subprocess.Popen([sys.executable, f'{W}/repo/colab/real_scenes/final/train_worker.py', job],
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu % 2)}, stdout=log, stderr=subprocess.STDOUT)))
        for name, lg, p in procs:
            p.wait(); print(f'---- variant {name} (exit {p.returncode}) ----'); print(''.join(l for l in open(lg) if 'step ' not in l or ' 0 loss' in l or '000 loss' in l), flush=True)
        return
    run('W1', s, cs, f, f'{o}/splat.ply', steps=steps, max_seconds=1800, depth_w=0.05)
    run('W2', s, cs, f, f'{o}/splat_depth.ply', steps=steps, max_seconds=1800, depth_w=0.2, depth_floor=0.25)

@stage('ffgs: feed-forward Gaussians from DA3 on the real frames + SfM cameras (no training)')
def _():
    sh('pip -q install --no-deps --ignore-requires-python "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" 2>&1 | tail -3', check=True)
    sh('pip -q install einops omegaconf huggingface_hub safetensors pillow imageio plyfile trimesh scipy addict moviepy pycolmap evo e3nn 2>&1 | tail -2')
    for _ in range(25):
        try: from depth_anything_3.api import DepthAnything3; break
        except ModuleNotFoundError as e:
            pkg = {'cv2': 'opencv-python-headless', 'PIL': 'pillow', 'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}.get(e.name.split('.')[0], e.name.split('.')[0])
            print('missing', e.name, '-> pip', pkg); sh(f'pip -q install {pkg} 2>&1 | tail -1')
    import ffgs; ffgs.run(CACHE, f'{O}/model/world.ply', every=int(os.environ.get('FFGS_EVERY', 6)))

@stage('cache (build gsplat once, save everything the training jobs need)')
def _():
    sh(f'nvcc --version | tail -1; pip -q install ninja "{GSPLAT}" 2>&1 | tail -1', check=True)
    sh(f'python {W}/repo/colab/real_scenes/final/gsplat_warm.py', check=True)
    C = f'{O}/cache'; os.makedirs(C, exist_ok=True)
    src = f'{W}/da3_sfm.npz' if os.path.exists(f'{W}/da3_sfm.npz') else f'{W}/da3.npz'
    shutil.copy(src, f'{C}/da3_sfm.npz'); shutil.copytree(f'{W}/frames', f'{C}/frames', dirs_exist_ok=True)
    for x in ('cloud.npz', 'walk.npz'): shutil.copy(f'{O}/model/{x}', f'{C}/{x}')
    sh(f'tar czf {C}/gsplat_ext.tgz -C ~/.cache torch_extensions && du -sh {C}/* | tail -6')

print('files:', os.listdir(O))
