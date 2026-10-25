# Kaggle job anasrazy/namma-ibr (CPU, ~5 min, no GPU quota). Was PROOF 1 of docs/PHOTOREAL_PLAN.md; now the camera check.
# Rebuild held-out real frames from their nearest real neighbours through the SfM cameras + depth (namma-prep cache), and
# measure PSNR/SSIM against the real photos. Also renders views 30 cm off the path (magenta = never filmed from there).
import glob, os, subprocess, sys
W = '/kaggle/temp/ibr'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no namma-prep cache attached'
cache = os.path.dirname(hits[0]); print('cache', cache, flush=True)
# Run 3 (camera-jump cut + iterative drop): 319/596 frames kept, all >= 20 dB, held-out 30.7 dB / SSIM 0.91 -> final/good_frames.json.
# Run 4 (aspect fix): bridges worked on 5/9 gaps; 94-121 moved 1.0 m -> cross 9 -> 36.8 dB. But look-alike 'already placed'
# links still joined passes that disagree with the rest (13-17 dB). Run 5: bridges first, every join checked against the WHOLE group.
# Run 2: 5 groups, but every accepted link was the current placement (nothing moved). Run 3: also follow the video itself across
# each gap between consecutive passes (chained PnP over the dropped frames): a continuous walk cannot jump.
# Run 1 of align_passes (103 depth frames): only 2 of 10 passes joined; links were one-directional and 20 dB let a wrong
# placement through. Now (fixed): both link directions, 23 dB, passes grouped instead of dropped, on namma-dense's 318 depth frames.
F = f'{W}/repo/colab/real_scenes/final'; A = '/kaggle/temp/aligned'; O = '/kaggle/working'; S = '/kaggle/temp/dsrc'
dn = glob.glob('/kaggle/input/**/dense/da3_sfm.npz', recursive=True); assert dn, 'no namma-dense output attached'
os.makedirs(S, exist_ok=True); os.symlink(dn[0], f'{S}/da3_sfm.npz'); os.symlink(f'{cache}/frames', f'{S}/frames')
r = subprocess.call([sys.executable, 'align_passes.py', S, '-', A, '--workers', '4', '--bridge', cache], cwd=F)
if not r:
    subprocess.call(f'cp {A}/align_report.json {O}/', shell=True)
    subprocess.call([sys.executable, 'ibr_check.py', A, f'{O}/ibr', '--workers', '4', '--cross'], cwd=F)
    r = subprocess.call([sys.executable, 'photo_viewer.py', A, '-', f'{O}/viewer'], cwd=F)
sys.exit(r)
