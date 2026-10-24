# OpenSplat project that renders every TRAIN view (named test_<frame>.jpg) so we can build MODP's error maps.
import sys, os, json, shutil, numpy as np, pycolmap
from scipy.spatial.transform import Rotation
W, d = sys.argv[1:3]; only = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None  # optional frame subset
st = json.load(open(f'{W}/clean/stats.json')); P = json.load(open(f'{W}/clean/poses_w2c.json'))
shutil.rmtree(d, ignore_errors=True); os.makedirs(f'{d}/images'); os.makedirs(f'{d}/txt'); os.makedirs(f'{d}/sparse/0')
open(f'{d}/txt/cameras.txt', 'w').write(f"1 PINHOLE {st['width']} {st['height']} {st['K'][0][0]} {st['K'][1][1]} {st['K'][0][2]} {st['K'][1][2]}\n")
with open(f'{d}/txt/images.txt', 'w') as fh:
    items = [(f, '') for f in st['train_frames'][:4]] + [(f, 'test_') for f in st['train_frames'] if only is None or f in only]
    for i, (f, pre) in enumerate(items):
        w2c = np.array(P[str(f)]); q = Rotation.from_matrix(w2c[:3, :3]).as_quat(); t = w2c[:3, 3]; name = f'{pre}{f:05d}.jpg'
        fh.write(f'{i+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
        os.symlink(f'{W}/data/color_raw/{f:05d}.jpg', f'{d}/images/{name}')
open(f'{d}/txt/points3D.txt', 'w').write(''.join(f'{i+1} {x} {y} {z} 128 128 128 0\n' for i, (x, y, z) in enumerate(np.random.default_rng(0).uniform(-2, 2, (1000, 3)))))
pycolmap.Reconstruction(f'{d}/txt').write_binary(f'{d}/sparse/0'); print('ok')
