# Kaggle job anasrazy/namma-resfm (~20 min): re-solve the cameras of the namma-prep frames with walk-order pairs only (no
# retrieval: it linked the corridor's look-alike ends), put the cached depth on the new cameras, and score every frame with the
# real-pixel rebuild (ibr_check) so before/after is one number. New cache -> /kaggle/working/cache (same layout as namma-prep).
import glob, os, shutil, subprocess, sys, time
T0 = time.time(); W = '/kaggle/temp/resfm'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
def sh(c, check=True):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:150]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise SystemExit(f'failed: {c[:100]}')
sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo')
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F)
old = os.path.dirname(glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True)[0]); print('old cache', old, flush=True)
E_ = f'{W}/sfmenv/bin/python'; P = f'--python {E_}'
sh(f'pip -q install uv 2>&1 | tail -1; unset UV_SYSTEM_PYTHON UV_PYTHON PIP_TARGET; uv venv -q -p 3.10 {W}/sfmenv && '
   f'uv pip install -q {P} torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 > {W}/pip.log 2>&1; '
   f'git clone -q --recursive https://github.com/cvg/Hierarchical-Localization {W}/hloc && '
   f'uv pip install -q {P} "numpy<2" -e {W}/hloc "lightglue @ git+https://github.com/cvg/LightGlue" >> {W}/pip.log 2>&1; tail -2 {W}/pip.log')
sh(f'cd {W} && {E_} {F}/sfm_hloc.py {old}/frames {W}/sfm 15 0 2>&1 | grep -v "it/s" | grep -v "s/it" | grep -iE "pairs|registered|reprojection|Largest|error" | tail -12')
C = f'{O}/cache'; os.makedirs(C, exist_ok=True)
import sfm_align; sfm_align.align(f'{W}/sfm/sfm.npz', f'{old}/da3_sfm.npz', f'{C}/da3_sfm.npz')   # cached depth re-scaled onto the new cameras
shutil.copytree(f'{old}/frames', f'{C}/frames', dirs_exist_ok=True)
for x in ('gsplat_ext.tgz', 'cloud.npz', 'walk.npz'):
    if os.path.exists(f'{old}/{x}'): shutil.copy(f'{old}/{x}', f'{C}/{x}')
sh(f'pip -q install scipy matplotlib pillow 2>&1 | tail -1')
sh(f'cd {F} && python ibr_check.py {C} {O}/check --workers 4')
