# Kaggle job anasrazy/namma-dense (T4, ~10 min): Depth Anything 3 on every VERIFIED frame (namma-prep did every 3rd), scaled onto
# the SfM cameras from the neighbours' aligned depth (final/densify.py), then the real-photo viewer with ~3x more photos.
# Outputs: /kaggle/working/dense/da3_sfm.npz (for later jobs) and /kaggle/working/viewer (download only this + the log).
import glob, os, subprocess, sys
W = '/kaggle/temp/dense'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no namma-prep cache attached'
cache = os.path.dirname(hits[0]); F = f'{W}/repo/colab/real_scenes/final'; print('cache', cache, flush=True)
r = subprocess.call([sys.executable, 'densify.py', cache, 'good_frames.json', f'{W}/d'], cwd=F)
if not r:
    subprocess.call(f'mkdir -p /kaggle/working/dense && cp {W}/d/da3_sfm.npz /kaggle/working/dense/', shell=True)
    r = subprocess.call([sys.executable, 'photo_viewer.py', f'{W}/d', '-', '/kaggle/working/viewer'], cwd=F)
sys.exit(r)
