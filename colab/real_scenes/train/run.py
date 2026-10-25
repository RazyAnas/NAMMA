# Kaggle job anasrazy/namma-train: ~15 min per experiment. Mounts namma-prep's cache, trains two variants at once (one per T4).
# To try something new, change VARIANTS below and re-run the same command. Logic: final/run.py MODE=train.
import json, os, subprocess
VARIANTS = [
    # 600 views, opacity reset now active (gsplat's own never fired), light depth anchor
    {"name": "X1", "out": "splat.ply",       "kw": {"steps": 15000, "max_seconds": 1500, "depth_w": 0.05}},
    # same + strong depth anchor that never fades (against floaters)
    {"name": "X2", "out": "splat_depth.ply", "kw": {"steps": 15000, "max_seconds": 1500, "depth_w": 0.2, "depth_floor": 0.25}},
]
W = '/kaggle/temp/final'; os.makedirs(W, exist_ok=True)
subprocess.run(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', shell=True, check=True)
os.environ['MODE'] = 'train'; os.environ['VARIANTS'] = json.dumps(VARIANTS)
exec(compile(open(f'{W}/repo/colab/real_scenes/final/run.py').read(), 'final/run.py', 'exec'))
