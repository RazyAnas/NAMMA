"""3DGS .ply -> .splat (32 bytes per splat: position, scale, RGBA, rotation; SH degree 0 colour), about 8x smaller, opens in SuperSplat.
Usage: python to_splat.py in.ply out.splat"""
import sys, numpy as np
from plyfile import PlyData

def convert(src, dst):
    v = PlyData.read(src)['vertex'].data; n = len(v)
    pos = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
    scale = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)).astype(np.float32)
    rgb = 0.5 + 0.28209479 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1); a = 1 / (1 + np.exp(-v['opacity']))
    col = (np.clip(np.concatenate([rgb, a[:, None]], 1), 0, 1) * 255).astype(np.uint8)
    q = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
    rot = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    o = np.argsort(-scale.prod(1) * a)                                                # big, opaque splats first (format convention)
    rec = np.zeros(n, dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
    rec['p'], rec['s'], rec['c'], rec['r'] = pos[o], scale[o], col[o], rot[o]; rec.tofile(dst)
    print(f'{src}: {n} splats -> {dst} ({rec.nbytes / 1e6:.0f} MB)', flush=True)

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
