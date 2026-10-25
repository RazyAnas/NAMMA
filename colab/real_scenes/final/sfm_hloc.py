"""Run INSIDE the Python 3.10 SfM venv: hloc SuperPoint + LightGlue cameras for a folder of frames, exported to one npz.
Pairs: every frame with its next 10 (walk order) + NetVLAD top-15 (loop closures). One shared PINHOLE camera.
Usage: python sfm_hloc.py <frames_dir> <out_dir> [seq=10] [retrieval=15]  ->  <out_dir>/sfm.npz
retrieval=0: walk-order pairs only. In a repetitive corridor retrieval links the two look-alike ends of the corridor and pulls
whole stretches into the wrong place (found by ibr_check: frames 86-108 at ~11 dB, a door rebuilt as a window)."""
import sys, numpy as np, pycolmap
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
frames, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 10; RET = int(sys.argv[4]) if len(sys.argv) > 4 else 15
names = sorted(p.name for p in frames.glob('*.jpg'))
fc, mc, rc = extract_features.confs['superpoint_max'], match_features.confs['superpoint+lightglue'], extract_features.confs['netvlad']
pairs = {(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, min(len(names), i + SEQ + 1))}
if RET:
    ret = extract_features.main(rc, frames, out)
    pairs_from_retrieval.main(ret, out / 'pairs_ret.txt', num_matched=RET)
    for line in open(out / 'pairs_ret.txt'):
        a, b = line.split()
        if a != b: pairs.add(tuple(sorted((a, b))))
(out / 'pairs.txt').write_text('\n'.join(f'{a} {b}' for a, b in sorted(pairs)))
print('pairs', len(pairs), flush=True)
feats = extract_features.main(fc, frames, out)
matches = match_features.main(mc, out / 'pairs.txt', fc['output'], out)
rec = reconstruction.main(out / 'sfm', frames, out / 'pairs.txt', feats, matches,
                          camera_mode=pycolmap.CameraMode.SINGLE, image_options=dict(camera_model='PINHOLE'))
print('registered', len(rec.images), 'of', len(names), 'images,', len(rec.points3D), 'points', flush=True)
N, K, W2C, WH, OBS = [], [], [], [], []
for im in rec.images.values():
    cam = rec.cameras[im.camera_id]; cfw = im.cam_from_world
    cfw = cfw() if callable(cfw) else cfw                       # property in older pycolmap, method in newer
    M = np.eye(4); M[:3] = cfw.matrix()
    uvz = []
    for p in im.points2D:
        if p.has_point3D():
            xc = M[:3, :3] @ rec.points3D[p.point3D_id].xyz + M[:3, 3]
            if xc[2] > 0: uvz.append((p.xy[0], p.xy[1], xc[2]))
    N.append(im.name); K.append(cam.calibration_matrix()); W2C.append(M); WH.append((cam.width, cam.height)); OBS.append(np.array(uvz, np.float32))
pts = np.array([p.xyz for p in rec.points3D.values()], np.float32); rgb = np.array([p.color for p in rec.points3D.values()], np.uint8)
np.savez(out / 'sfm.npz', names=np.array(N), K=np.array(K), w2c=np.array(W2C), wh=np.array(WH), obs=np.array(OBS, dtype=object), pts=pts, rgb=rgb)
