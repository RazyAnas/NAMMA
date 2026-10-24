# ONE-SHOT Kaggle job (UNTESTED on GPU, written in a container with no GPU/HF/Kaggle access): video -> walkable model.
# Stage 1 sharpest frames -> 2 Depth Anything 3 (poses+depth+gaussians, feed-forward, ~1 min) -> 3 fuse + floor plan + A* (tested on CPU)
# -> 4 package for the viewer. Matrix-3D gap filling stays a separate job (colab/real_scenes/matrix3d/kaggle) and is fed pano_raw from here.
# Every stage logs and continues, so a failed stage never loses the earlier outputs in /kaggle/working.
import glob, json, os, subprocess, sys, time, traceback
T0 = time.time(); O = '/kaggle/working'; W = '/kaggle/temp/final'; os.makedirs(W, exist_ok=True)
def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r
def stage(name):
    def deco(f):
        try: f(); print(f'== {name}: ok [{time.time()-T0:.0f}s]', flush=True)
        except Exception: print(f'== {name}: FAILED'); traceback.print_exc()
    return deco
sh('ls -R /kaggle/input | head -40')
_v = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mov', recursive=True))
assert _v, 'no video under /kaggle/input: attach dataset anasrazy/namma-room-2026-10-22 (see ls above)'
VIDEO = _v[0]; print('VIDEO', VIDEO)
NFRAMES = int(os.environ.get('NFRAMES', 60))
sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
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
    files = sorted(glob.glob(f'{W}/frames/*.jpg'))
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
    k = np.where(have)[0]; np.savez(f'{W}/da3.npz', depth=D[k], K=Ks[k], c2w=C[k], rgb=rgb[k], idx=k)
    print('merged', len(k), 'of', len(files), 'frames; median depth', float(np.median(D[k][D[k] > 0])), '(metric)')

@stage('3 fuse + floor plan + path')
def _():
    sh('pip -q install open3d trimesh scipy matplotlib 2>&1 | tail -1')
    import fuse_nav as F
    F.fuse(f'{W}/da3.npz', f'{O}/model'); print(json.dumps({k: v for k, v in F.navigate(f'{O}/model/cloud.npz', f'{O}/model').items() if k != 'demo_path_cells'}))
    import build_viewer; print(build_viewer.build(f'{O}/model'), '-> model/viewer.html')
@stage('4 splat fit (gsplat, ~12 min cap)')
def _():
    import torch
    sh('nvcc --version | tail -1; pip -q install ninja 2>&1 | tail -1')
    tv = torch.__version__.split('+')[0].split('.'); cu = (torch.version.cuda or '').replace('.', '')
    if sh(f'pip -q install gsplat --index-url https://docs.gsplat.studio/whl/pt{tv[0]}{tv[1]}cu{cu} 2>&1 | tail -2') != 0 or sh('python -c "import gsplat"') != 0:
        sh('pip -q install gsplat 2>&1 | tail -2', check=True)
    import train_splat; train_splat.train(f'{W}/da3.npz', f'{O}/model/cloud.npz', f'{W}/frames', f'{O}/model/splat.ply', steps=int(os.environ.get('STEPS', 3000)))

print('files:', os.listdir(O))
