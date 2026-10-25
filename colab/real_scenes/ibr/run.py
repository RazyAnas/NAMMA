# Kaggle job anasrazy/namma-ibr (CPU, ~5 min, no GPU quota). Was PROOF 1 of docs/PHOTOREAL_PLAN.md; now the camera check.
# Rebuild held-out real frames from their nearest real neighbours through the SfM cameras + depth (namma-prep cache), and
# measure PSNR/SSIM against the real photos. Also renders views 30 cm off the path (magenta = never filmed from there).
import glob, os, subprocess, sys
W = '/kaggle/temp/ibr'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
hits = glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True); assert hits, 'no namma-prep cache attached'
cache = os.path.dirname(hits[0]); print('cache', cache, flush=True)
# Run 2 (507-frame keep-list): held-out 27.0 -> 30.4 dB, but 28 kept frames newly failed: a misplaced STRETCH vouches for itself.
# Run 3: start from all frames; cut the walk at camera jumps, drop short pieces, then iteratively drop < 20 dB (ibr_check --prune).
F = f'{W}/repo/colab/real_scenes/final'; O = '/kaggle/working/ibr'
r = subprocess.call([sys.executable, 'ibr_check.py', cache, O, '--workers', '4', '--prune', '6'], cwd=F)
sys.exit(r or subprocess.call([sys.executable, 'ibr.py', cache, O, '--n', '16', '--keep', f'{O}/good_frames.json'], cwd=F))
