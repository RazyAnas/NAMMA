# Collision mesh + NavMesh for the walking demo from one real scene: back-project the anchored monocular depth
# (clean/boot_depth, uint16 mm) of the training frames with their poses (clean/poses_w2c.json), the same world frame
# the splat is trained in, then run NAMMA's collision and NavMesh code (needs the NAMMA app package on sys.path).
# Floor: RANSAC plane among points at least 0.9 m below the median camera height (people film from standing height),
# normal within 25 deg of the mean camera image-up direction; its normal is the up axis.
# --path-floor: when the depth gives no usable floor (the lobby: white walls, no floor peak), use a flat floor 1.45 m
# below the cameras (the height its scale was set from) under the camera path widened by 0.6 m: you walk where the camera walked.
# DEPTH_DIR=vggt_boot_depth uses the VGGT-anchored depth (the hybrid: VGGT for the floor, DA-V2 for the splat).
# Usage: python build_walk_scene.py <scene dir with clean/> <NAMMA app root> <out dir> [max_depth_m=6] [--path-floor]
import sys, json, os
from pathlib import Path
import numpy as np, cv2, open3d as o3d
path_floor = '--path-floor' in sys.argv; argv = [a for a in sys.argv if a != '--path-floor']
W, app_root, out = Path(argv[1]), argv[2], Path(argv[3]); max_d = float(argv[4]) if len(argv) > 4 else 6.0
sys.path.insert(0, app_root)
from app.config import Settings
from app.services.geometry import build_collision_candidate
from app.services.navigation import approve_navigation_mesh, build_navigation_mesh, calibrate_collision_scale
st = json.load(open(W / 'clean/stats.json')); K = np.array(st['K']); w2c = json.load(open(W / 'clean/poses_w2c.json'))
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; s = 4
pts, centres, ups = [], [], []
for f in st['train_frames']:
    p = W / f"clean/{os.environ.get('DEPTH_DIR', 'boot_depth')}/{f:05d}.png"
    if str(f) not in w2c or not p.exists(): continue
    T = np.linalg.inv(np.array(w2c[str(f)])); D = cv2.imread(str(p), -1).astype(np.float64)[::s, ::s] / 1000.
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]] * s; ok = (D > 0.1) & (D < max_d); z = D[ok]
    X = np.column_stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z]) @ T[:3, :3].T + T[:3, 3]
    pts.append(X); centres.append(T[:3, 3]); ups.append(-T[:3, 1])
up = np.mean(ups, 0); up /= np.linalg.norm(up); centres = np.array(centres)
cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(0.02)
cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
if path_floor:
    a = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    g = np.mgrid[-0.6:0.6:0.05, -0.6:0.6:0.05].reshape(2, -1).T
    F = np.concatenate([c - 1.45 * up + g[:, :1] * a + g[:, 1:] * b for c in centres])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(F)).voxel_down_sample(0.05)
geo = out / 'geometry'; (geo / 'dense').mkdir(parents=True, exist_ok=True)
o3d.io.write_point_cloud(str(geo / 'dense/fused.ply'), cloud)
settings = Settings(_env_file=None)
build_collision_candidate(geo / 'dense/fused.ply', geo / 'collision', settings, source_description='dense/fused.ply')
calibrate_collision_scale(geo, 1.0, 1.0, 'metric poses (SfM aligned to ARCore, or camera-height scale)')
P = np.asarray(cloud.points); low = P[P @ up < np.median(centres @ up) - 0.9 + (0.5 if path_floor else 0)]; rng = np.random.default_rng(0); best = (0, None)
for _ in range(2000):
    q = low[rng.choice(len(low), 3, replace=False)]; n = np.cross(q[1] - q[0], q[2] - q[0])
    if np.linalg.norm(n) < 1e-9: continue
    n /= np.linalg.norm(n); n = n if n @ up > 0 else -n
    if n @ up < np.cos(np.radians(25)): continue
    c = (np.abs((low - q[0]) @ n) < 0.03).sum()
    if c > best[0]: best = (c, (n, q[0]))
n, q0 = best[1]; inl = low[np.abs((low - q0) @ n) < 0.03]; m = inl.mean(0)
n = np.linalg.svd(inl - m, full_matrices=False)[2][2]; up = n if n @ up > 0 else -n; floor = float(m @ up)
cam_h = centres @ up
(geo / 'navigation').mkdir(exist_ok=True)
(geo / 'navigation/preparation.json').write_text(json.dumps({'status': 'floor_review_required',
    'plane': {'normal': up.tolist(), 'offset': -floor, 'accepted': True, 'confidence': 1.0}}))
nav = build_navigation_mesh(geo, geo, settings, camera_centers_source=centres, source_to_viewer=np.eye(4).tolist())
approve_navigation_mesh(geo, 'real-scene', 'Our poses and anchored monocular depth')
print(json.dumps({k: nav[k] for k in ('width', 'height', 'walkable_count', 'component_count')} | {
    'standable': nav['movement']['standable_count'], 'floor': floor, 'up': up.round(3).tolist(),
    'camera_height_above_floor_m': round(float(np.median(cam_h) - floor), 2), 'points': len(cloud.points)}))
