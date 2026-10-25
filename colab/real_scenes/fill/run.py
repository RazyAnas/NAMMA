# Kaggle job anasrazy/namma-fill (T4, ~25 min): PROOF 4 = step 6 of the video method. A drone path drifting 50 cm off the walked
# path in pass 453-488 (its cameras agree: 28 dB), real pixels re-projected from that pass's photos, Wan 2.1 VACE 1.3B fills only
# the holes, real pixels composited back on top, generated pixels flagged. An honesty strip of REAL pixels is hidden from the
# model to score how close its guesses are. Logic: final/fill_holes.py. Output: /kaggle/working/fill (mp4s, sheet.jpg, report).
import glob, os, subprocess, sys
W = '/kaggle/temp/fill'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
cache = os.path.dirname(glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True)[0])
dn = glob.glob('/kaggle/input/**/dense/da3_sfm.npz', recursive=True); assert dn, 'no namma-dense output attached'
S = f'{W}/src'; os.makedirs(S, exist_ok=True); os.symlink(dn[0], f'{S}/da3_sfm.npz'); os.symlink(f'{cache}/frames', f'{S}/frames')
subprocess.call('pip -q install -U diffusers transformers accelerate ftfy "imageio[ffmpeg]" 2>&1 | tail -2', shell=True)
F = f'{W}/repo/colab/real_scenes/final'
sys.exit(subprocess.call([sys.executable, 'fill_holes.py', S, '/kaggle/working/fill', '--pass', '453', '488', '--offset', '0.5'], cwd=F))
