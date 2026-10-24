# Walkable floor from per-frame depth, with NAMMA's own collision + NavMesh code (navigation patches 0001-0003),
# exactly the steps of scripts/build_walk_scene_from_rgbd.py minus the reference mesh.
# Usage: build_floor.py <depth_dir uint16 mm PNGs named FFFFF.png> <poses json {frame: c2w}> <out dir> [noise]
import sys, json
from pathlib import Path
import numpy as np, open3d as o3d
from PIL import Image
sys.path.insert(0, '/tmp/claude-0/work/navrepo')
from app.config import Settings
from app.services.geometry import build_collision_candidate
from app.services.navigation import approve_navigation_mesh, build_navigation_mesh, calibrate_collision_scale
ddir, pjson, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
poses = {int(k): np.array(v) for k, v in json.load(open(pjson)).items()}
rng = np.random.default_rng(3); fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5; stride = 4
clouds, centres = [], []
for png in sorted(ddir.glob('*.png')):
    i = int(png.stem)
    if i not in poses: continue
    depth = np.asarray(Image.open(png), dtype=np.float64)[::stride, ::stride] / 1000.0
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]] * stride; valid = depth > 0.1
    z = depth[valid] * (1.0 + rng.normal(0.0, noise, valid.sum())) if noise > 0 else depth[valid]
    if noise > 0:
        pull = rng.random(len(z)) < 0.002; z[pull] *= rng.uniform(0.3, 0.95, pull.sum())
    x = (u[valid] - cx) / fx * z; y = (v[valid] - cy) / fy * z; P = poses[i]
    clouds.append(np.column_stack([x, y, z]) @ P[:3, :3].T + P[:3, 3]); centres.append(P[:3, 3])
points, centres = np.concatenate(clouds), np.asarray(centres)
geometry = out / 'geometry'; (geometry / 'dense').mkdir(parents=True, exist_ok=True)
cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.01)
o3d.io.write_point_cloud(str(geometry / 'dense' / 'fused.ply'), cloud)
settings = Settings(_env_file=None)
build_collision_candidate(geometry / 'dense' / 'fused.ply', geometry / 'collision', settings, source_description='dense/fused.ply')
calibrate_collision_scale(geometry, 1.0, 1.0, 'metric poses')
up = np.array([0.0, 1.0, 0.0]); heights = np.asarray(cloud.points) @ up
low = heights[heights < np.percentile(heights, 30)]
hist, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.02, 0.01)); floor = float(edges[np.argmax(hist)] + 0.005)
(geometry / 'navigation').mkdir(exist_ok=True)
(geometry / 'navigation' / 'preparation.json').write_text(json.dumps({'status': 'floor_review_required',
    'plane': {'normal': up.tolist(), 'offset': -floor, 'accepted': True, 'confidence': 1.0}}))
nav = build_navigation_mesh(geometry, geometry, settings, camera_centers_source=centres, source_to_viewer=np.eye(4).tolist())
approve_navigation_mesh(geometry, 'rgbd-test', 'floor test')
json.dump(nav, open(out / 'navmesh.json', 'w'))
print(json.dumps({'frames': len(centres), 'standable': nav['movement']['standable_count'], 'walkable': nav['walkable_count'], 'floor_y': floor}))
