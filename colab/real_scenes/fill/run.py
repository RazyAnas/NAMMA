# Kaggle job anasrazy/namma-fill (T4, ~35 min). PROOF 4, run 2. Run 1 (one pass as source, 50 cm off): the model invented a
# glass door, a screen and a small figure in the big hole behind the pillar. Run 2: real pixels from EVERY aligned pass first,
# the model fills only small gaps, large unfilmed regions stay 'not filmed' (never invented), and a sweep measures holes vs
# distance from the walked path (one pass vs all aligned passes). Logic: final/fill_holes.py. Output: /kaggle/working/fill.
import glob, os, subprocess, sys
W = '/kaggle/temp/fill'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
F = f'{W}/repo/colab/real_scenes/final'
cache = os.path.dirname(glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True)[0])
S = f'{W}/src'; os.makedirs(S, exist_ok=True); os.symlink(f'{cache}/frames', f'{S}/frames')
al = glob.glob('/kaggle/input/**/aligned/da3_sfm.npz', recursive=True)
if al: os.symlink(al[0], f'{S}/da3_sfm.npz'); print('aligned cameras from namma-ibr', al[0], flush=True)
else:                                                                                  # namma-ibr output has none yet: align here (CPU, ~8 min)
    dn = glob.glob('/kaggle/input/**/dense/da3_sfm.npz', recursive=True); assert dn, 'no namma-dense output attached'
    D = f'{W}/dense'; os.makedirs(D, exist_ok=True); os.symlink(dn[0], f'{D}/da3_sfm.npz'); os.symlink(f'{cache}/frames', f'{D}/frames')
    subprocess.run([sys.executable, 'align_passes.py', D, '-', f'{W}/al', '--workers', '4', '--bridge', cache], cwd=F, check=True)
    os.symlink(f'{W}/al/da3_sfm.npz', f'{S}/da3_sfm.npz')
subprocess.call('pip -q install -U diffusers transformers accelerate ftfy "imageio[ffmpeg]" 2>&1 | tail -2', shell=True)
sys.exit(subprocess.call([sys.executable, 'fill_holes.py', S, '/kaggle/working/fill', '--pass', '453', '488', '--offset', '0.3'], cwd=F))
