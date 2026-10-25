"""Instant continuous 3D asset (no training): every real photo of one aligned group, back-projected through its depth into
small coloured splats, merged on a 1 cm grid (the closest photo wins a cell: it saw the surface sharpest). Depth edges are dropped
(no streaks between a pillar and the wall behind it). Output .splat (opens in SuperSplat) + .ply point cloud.
Input: a photo-viewer folder (cams.json + f/*.jpg + f/*.bin), so it runs on a laptop in about a minute (numpy + Pillow only).
Usage: python3 viewer_to_splat.py <viewer dir> <out.splat> [--group 0] [--voxel 0.01] [--viewer]"""
import argparse, json, os, numpy as np
from PIL import Image

def points(vdir, group=0, voxel=0.01, edge=1.08):
    """Merged coloured points of one group (world frame of the viewer, y down), finest footprint per voxel; also returns sizes."""
    j = json.load(open(f'{vdir}/cams.json')); s = j['stride']; Hd, Wd = j['Hd'], j['Wd']; h, w = -(-Hd // s), -(-Wd // s)
    cams = [c for c in j['cams'] if c.get('grp', 0) == group]; P, C, Z = [], [], []
    for c in cams:
        f = f"{vdir}/f/{c['g']:04d}"
        if not os.path.exists(f + '.bin'): continue
        z = np.fromfile(f + '.bin', '<u2').reshape(h, w).astype(np.float32) / 1000
        img = np.asarray(Image.open(f + '.jpg').convert('RGB').resize((w, h), Image.BILINEAR))
        zz = np.pad(z, 1, mode='edge'); nb = np.stack([zz[1:-1, :-2], zz[1:-1, 2:], zz[:-2, 1:-1], zz[2:, 1:-1]])
        ok = (z > 0) & (nb.min(0) > 0) & (nb.max(0) < edge * z) & (nb.min(0) * edge > z)   # smooth surface only (no depth edges)
        v, u = np.nonzero(ok); d = z[v, u]; fx, _, cx, _, fy, cy = c['K']; px, py = u * s, v * s
        X = np.stack([(px - cx) / fx * d, (py - cy) / fy * d, d], 1); M = np.array(c['c2w']).reshape(3, 4)
        P.append(X @ M[:, :3].T + M[:, 3]); C.append(img[v, u]); Z.append(d * s / fx)   # splat size = pixel footprint
    P, C, Z = np.concatenate(P), np.concatenate(C), np.concatenate(Z)
    key = np.floor(P / voxel).astype(np.int64); o = np.argsort(Z)                         # finest footprint first
    _, first = np.unique(key[o], axis=0, return_index=True); k = o[first]
    return P[k], C[k], np.maximum(Z[k], voxel * 0.6), len(cams)

def viewer_clouds(vdir, voxel=0.02):
    """cloud_<group>.bin for the photo viewer: n x float32 xyz, then n x uint8 rgb. The viewer draws it behind the photos."""
    j = json.load(open(f'{vdir}/cams.json'))
    for g in sorted({c.get('grp', 0) for c in j['cams']}):
        P, C, _, nc = points(vdir, g, voxel)
        with open(f'{vdir}/cloud_{g}.bin', 'wb') as f: np.array([len(P)], '<u4').tofile(f); P.astype('<f4').tofile(f); C.astype('u1').tofile(f)
        print(f'viewer cloud group {g}: {nc} photos -> {len(P)} points ({voxel*100:.0f} cm)', flush=True)

def build(vdir, out, group=0, voxel=0.01, edge=1.08):
    P, C, Z, ncam = points(vdir, group, voxel, edge)
    P = P * np.array([1, -1, -1])                                                          # camera frame is y-down: show upright
    n = len(P); rec = np.zeros(n, dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
    rec['p'] = P; rec['s'] = np.repeat(Z[:, None] * 0.7, 3, 1); rec['c'][:, :3] = C; rec['c'][:, 3] = 235; rec['r'] = [255, 128, 128, 128]
    rec = rec[np.argsort(-Z)]; rec.tofile(out)
    with open(out[:-6] + '.ply', 'wb') as fp:                                             # plain coloured point cloud too
        fp.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
        pc = np.zeros(n, dtype=[('p', '<f4', 3), ('c', 'u1', 3)]); pc['p'] = rec['p']; pc['c'] = rec['c'][:, :3]; pc.tofile(fp)
    print(f'group {group}: {ncam} photos -> {n} splats ({voxel*100:.0f} cm grid) -> {out} ({rec.nbytes/1e6:.0f} MB) + {out[:-6]}.ply')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('vdir'); ap.add_argument('out'); ap.add_argument('--group', type=int, default=0)
    ap.add_argument('--voxel', type=float, default=0.01); ap.add_argument('--viewer', action='store_true', help='also write cloud_<g>.bin for the photo viewer')
    a = ap.parse_args(); build(a.vdir, a.out, a.group, a.voxel)
    if a.viewer: viewer_clouds(a.vdir)
