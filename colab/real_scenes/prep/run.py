# Kaggle job anasrazy/namma-prep: run ONCE. Video -> 600 frames -> Depth Anything 3 -> SfM cameras -> cloud + floor plan, and builds
# gsplat's CUDA code. Everything lands in /kaggle/working/cache, which namma-train mounts (kernel_sources). Logic: final/run.py MODE=prep.
import os, subprocess
W = '/kaggle/temp/final'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
os.environ['MODE'] = 'prep'
exec(compile(open(f'{W}/repo/colab/real_scenes/final/run.py').read(), 'final/run.py', 'exec'))
