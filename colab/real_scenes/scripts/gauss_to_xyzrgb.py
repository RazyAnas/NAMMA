# Positions + colours of a 3DGS PLY as a plain xyzrgb cloud, so OpenSplat starts from these points with its own
# neighbour-distance scales (the "MODP placement + our scale" setting that won the 1k-step test).
# Usage: python gauss_to_xyzrgb.py <in 3dgs.ply> <out.ply>
import sys, numpy as np
src, dst = sys.argv[1:3]
with open(src, 'rb') as fh:
    hdr = []
    while True:
        l = fh.readline().decode().strip(); hdr.append(l)
        if l == 'end_header': break
    n = int([h for h in hdr if h.startswith('element vertex')][0].split()[-1]); props = [h.split()[-1] for h in hdr if h.startswith('property')]
    g = np.frombuffer(fh.read(), np.float32).reshape(n, len(props))
i = props.index('f_dc_0'); rgb = np.clip((g[:, i:i + 3] * 0.28209479177387814 + 0.5) * 255, 0, 255).astype(np.uint8)
a = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
a['x'], a['y'], a['z'] = g[:, 0], g[:, 1], g[:, 2]; a['red'], a['green'], a['blue'] = rgb.T
with open(dst, 'wb') as fh:
    fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n'
             'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode()); fh.write(a.tobytes())
print(n, 'points')
