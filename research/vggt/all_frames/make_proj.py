# make_whole_proj.py with poses read from cameras.json (same train/test lists, same camera, same txt->binary path).
# Usage: python make_proj.py <init.ply> <out dir>
import sys, os, json, shutil, numpy as np, pycolmap
from plyfile import PlyData
from scipy.spatial.transform import Rotation
ply, d = sys.argv[1:3]; D0 = '/tmp/claude-0/work/data/namma_wholeroom_gpu/data'
cams = json.load(open(f'{D0}/clean/cameras.json'))
v = PlyData.read(ply)['vertex']; pts = np.stack([v[k] for k in ('x', 'y', 'z', 'red', 'green', 'blue')], 1)
shutil.rmtree(d, ignore_errors=True); os.makedirs(f'{d}/images'); os.makedirs(f'{d}/txt'); os.makedirs(f'{d}/sparse/0')
with open(f'{d}/txt/cameras.txt', 'w') as fh: fh.write('1 PINHOLE 640 480 525 525 319.5 239.5\n')
with open(f'{d}/txt/images.txt', 'w') as fh:
    for i, (c, pre) in enumerate([(c, '') for c in cams['train']] + [(c, 'test_') for c in cams['test']]):
        f = c['frame']; w2c = np.array(c['w2c']); q = Rotation.from_matrix(w2c[:3, :3]).as_quat(); t = w2c[:3, 3]; name = f'{pre}{f:05d}.jpg'
        fh.write(f'{i+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
        os.symlink(f'{D0}/images/clean/{f:05d}.jpg', f'{d}/images/{name}')
with open(f'{d}/txt/points3D.txt', 'w') as fh:
    for i, p in enumerate(pts): fh.write(f'{i+1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(p[3])} {int(p[4])} {int(p[5])} 0\n')
pycolmap.Reconstruction(f'{d}/txt').write_binary(f'{d}/sparse/0')
print(len(cams['train']), 'train', len(cams['test']), 'test', len(pts), 'points')
