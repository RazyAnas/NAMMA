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
VIDEO = (glob.glob('/kaggle/input/*/video.mp4') or glob.glob('/kaggle/input/*/*.mp4'))[0]
NFRAMES = int(os.environ.get('NFRAMES', 60))

@stage('1 frames')
def _():
    sh('apt-get install -y -q ffmpeg >/dev/null 2>&1; pip -q install opencv-python-headless trimesh scipy matplotlib open3d', check=True)
    import cv2, numpy as np
    cap = cv2.VideoCapture(VIDEO); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); win = n // NFRAMES; os.makedirs(f'{W}/frames', exist_ok=True)
    for k in range(NFRAMES):                                   # sharpest frame (variance of Laplacian) per window
        best, bs = None, -1
        for i in range(k * win, (k + 1) * win, 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            if not ok: continue
            s = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if s > bs: best, bs = f, s
        if best is not None:
            h, w = best.shape[:2]; sc = 756 / max(h, w); cv2.imwrite(f'{W}/frames/{k:04d}.jpg', cv2.resize(best, (int(w * sc), int(h * sc))))
    print(len(os.listdir(f'{W}/frames')), 'frames')

@stage('2 depth anything 3')
def _():
    sh('pip -q install xformers --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1; '
       'pip -q install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" 2>&1 | tail -2', check=True)
    import numpy as np, torch, cv2
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE').to('cuda').eval()
    files = sorted(glob.glob(f'{W}/frames/*.jpg'))
    pred = model.inference(files)                             # depth (N,H,W), extrinsics (N,3,4) w2c, intrinsics (N,3,3)
    ext = np.concatenate([pred.extrinsics, np.tile([[[0, 0, 0, 1.]]], (len(files), 1, 1))], 1)
    np.savez(f'{W}/da3.npz', depth=pred.depth, K=pred.intrinsics, c2w=np.linalg.inv(ext), rgb=pred.processed_images)
    print('depth', pred.depth.shape, 'median depth', float(np.median(pred.depth)), '(DA3 nested = metric)')

@stage('3 fuse + floor plan + path')
def _():
    sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
    sys.path.insert(0, f'{W}/repo/colab/real_scenes/final'); import fuse_nav as F
    F.fuse(f'{W}/da3.npz', f'{O}/model'); print(json.dumps({k: v for k, v in F.navigate(f'{O}/model/cloud.npz', f'{O}/model').items() if k != 'demo_path_cells'}))
print('files:', os.listdir(O))
