# Kaggle job anasrazy/namma-ibr (CPU, ~5 min, no GPU quota). Was PROOF 1 of docs/PHOTOREAL_PLAN.md; now the camera check.
# Rebuild held-out real frames from their nearest real neighbours through the SfM cameras + depth (namma-prep cache), and
# measure PSNR/SSIM against the real photos. Also renders views 30 cm off the path (magenta = never filmed from there).
import glob, os, subprocess, sys
W = '/kaggle/temp/ibr'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no namma-prep cache attached'
cache = os.path.dirname(hits[0]); print('cache', cache, flush=True)
# Proof 1 is done (27.0 dB held-out, 4/16 views with wrong cameras); the camera check found 89 bad frames -> final/good_frames.json.
# Now: re-check with ONLY the verified cameras (expect no dips, higher median), then redo the held-out + 30 cm off-path images.
F = f'{W}/repo/colab/real_scenes/final'; keep = f'{F}/good_frames.json'
r = subprocess.call([sys.executable, 'ibr_check.py', cache, '/kaggle/working/ibr', '--workers', '4', '--keep', keep], cwd=F)
sys.exit(r or subprocess.call([sys.executable, 'ibr.py', cache, '/kaggle/working/ibr', '--n', '16', '--keep', keep], cwd=F))
