# Kaggle job anasrazy/namma-ibr (CPU, ~5 min, no GPU quota). Was PROOF 1 of docs/PHOTOREAL_PLAN.md; now the camera check.
# Rebuild held-out real frames from their nearest real neighbours through the SfM cameras + depth (namma-prep cache), and
# measure PSNR/SSIM against the real photos. Also renders views 30 cm off the path (magenta = never filmed from there).
import glob, os, subprocess, sys
W = '/kaggle/temp/ibr'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no namma-prep cache attached'
cache = os.path.dirname(hits[0]); print('cache', cache, flush=True)
# Run 3 (camera-jump cut + iterative drop): 319/596 frames kept, all >= 20 dB, held-out 30.7 dB / SSIM 0.91 -> final/good_frames.json.
# Cross-pass check: passes disagree (11-17 dB rebuilt from other passes; door rebuilt as window). Now: re-place each pass as a
# whole, judged by real pixels (align_passes.py), re-run the cross check on the result, and build the viewer from it (CPU ~15 min).
F = f'{W}/repo/colab/real_scenes/final'; A = '/kaggle/temp/aligned'; O = '/kaggle/working'
r = subprocess.call([sys.executable, 'align_passes.py', cache, 'good_frames.json', A, '--workers', '4'], cwd=F)
if not r:
    subprocess.call(f'cp {A}/align_report.json {O}/', shell=True)
    subprocess.call([sys.executable, 'ibr_check.py', A, f'{O}/ibr', '--workers', '4', '--cross'], cwd=F)
    r = subprocess.call([sys.executable, 'photo_viewer.py', A, '-', f'{O}/viewer'], cwd=F)
sys.exit(r)
