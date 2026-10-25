# Kaggle job anasrazy/namma-model (2x T4, ~40 min): THE 3D MODEL on cameras that agree. Every splat before was trained on camera
# positions whose walk passes disagreed (look-alike merges), which explains the fog and needles. Now:
#   1. align the walk passes (align_passes: video bridges + whole-group check + ICP) -> group 0 = 162 frames agreeing at ~31 dB;
#   2. group-0 data: cameras + dense depth + real colours; fused point cloud (init), floor plan + walkable grid, real-photo viewer;
#   3. two splat fits in parallel (one per GPU): A = default densification, SH3 (tile reflections), depth anchor;
#      B = MCMC 1.5M splats, SH3, depth anchor, anisotropy cap (needles). Held-out renders + stats in the log;
#   4. .splat copies (8x smaller) for SuperSplat. Output: /kaggle/working/{aligned,model,viewer}.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time(); W = '/kaggle/temp/model'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:150]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(c[:80])
sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F)
cache = os.path.dirname(glob.glob('/kaggle/input/**/cache/da3_sfm.npz', recursive=True)[0])
dn = glob.glob('/kaggle/input/**/dense/da3_sfm.npz', recursive=True); assert dn, 'no namma-dense output attached'
D = f'{W}/dense'; os.makedirs(D, exist_ok=True); os.symlink(dn[0], f'{D}/da3_sfm.npz'); os.symlink(f'{cache}/frames', f'{D}/frames')
sh(f'cd {F} && {sys.executable} align_passes.py {D} - {W}/al --workers 4 --bridge {cache} 2>&1 | grep -v Warning', check=True)
os.makedirs(f'{O}/aligned', exist_ok=True); shutil.copy(f'{W}/al/da3_sfm.npz', f'{O}/aligned/'); shutil.copy(f'{W}/al/align_report.json', O)

import build_model                                                                   # group 0 -> cloud, floor plan, viewer, two fits, levelled .splat
build_model.build(f'{W}/al/da3_sfm.npz', f'{cache}/frames', f'{cache}/gsplat_ext.tgz', O)
