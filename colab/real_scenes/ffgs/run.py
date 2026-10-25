# Kaggle job anasrazy/namma-ffgs (~10 min): feed-forward Gaussians from Depth Anything 3 on the real frames, placed with our SfM
# cameras (from the namma-prep cache). No per-scene training. Output: model/world.ply. Logic: final/run.py MODE=ffgs, final/ffgs.py.
import os, subprocess
W = '/kaggle/temp/final'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
os.environ['MODE'] = 'ffgs'
exec(compile(open(f'{W}/repo/colab/real_scenes/final/run.py').read(), 'final/run.py', 'exec'))
