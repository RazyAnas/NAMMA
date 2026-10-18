# COLMAP project for OpenSplat: our Sim3-aligned global-SfM poses, train images + test_ held-out images, an init cloud.
# Usage: python make_whole_proj.py <work> <init.ply> <out dir> [poses_w2c.json, default clean/poses_w2c.json]
import sys, os, json, shutil, numpy as np, pycolmap
from plyfile import PlyData
from scipy.spatial.transform import Rotation
W, ply, d = sys.argv[1:4]; pf = sys.argv[4] if len(sys.argv) > 4 else f'{W}/clean/poses_w2c.json'
st = json.load(open(f'{W}/clean/stats.json')); poses = json.load(open(pf))
v = PlyData.read(ply)['vertex']; pts = np.stack([v[k] for k in ('x', 'y', 'z', 'red', 'green', 'blue')], 1)
shutil.rmtree(d, ignore_errors=True); os.makedirs(f'{d}/images'); os.makedirs(f'{d}/txt'); os.makedirs(f'{d}/sparse/0')
with open(f'{d}/txt/cameras.txt', 'w') as fh: fh.write(f"1 PINHOLE {st['width']} {st['height']} {st['K'][0][0]} {st['K'][1][1]} {st['K'][0][2]} {st['K'][1][2]}\n")
with open(f'{d}/txt/images.txt', 'w') as fh:
    for i, (f, pre) in enumerate([(f, '') for f in st['train_frames']] + [(f, 'test_') for f in st['test_frames']]):
        w2c = np.array(poses[str(f)]); q = Rotation.from_matrix(w2c[:3, :3]).as_quat(); t = w2c[:3, 3]; name = f'{pre}{f:05d}.jpg'
        fh.write(f'{i+1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
        os.symlink(f'{W}/data/color_raw/{f:05d}.jpg', f'{d}/images/{name}')
with open(f'{d}/txt/points3D.txt', 'w') as fh:
    for i, p in enumerate(pts): fh.write(f'{i+1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(p[3])} {int(p[4])} {int(p[5])} 0\n')
pycolmap.Reconstruction(f'{d}/txt').write_binary(f'{d}/sparse/0')
print(len(st['train_frames']), 'train', len(st['test_frames']), 'test', len(pts), 'points')
