# Kaggle job anasrazy/namma-ibr (CPU, ~5 min, no GPU quota): PROOF 1 of docs/PHOTOREAL_PLAN.md.
# Rebuild held-out real frames from their nearest real neighbours through the SfM cameras + depth (namma-prep cache), and
# measure PSNR/SSIM against the real photos. Also renders views 30 cm off the path (magenta = never filmed from there).
import glob, os, subprocess, sys
W = '/kaggle/temp/ibr'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no namma-prep cache attached'
cache = os.path.dirname(hits[0]); print('cache', cache, flush=True)
sys.exit(subprocess.call([sys.executable, f'{W}/repo/colab/real_scenes/final/ibr.py', cache, '/kaggle/working/ibr', '--n', '16', '--k', '4']))
