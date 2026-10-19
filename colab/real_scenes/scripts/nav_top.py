# Top-down check image of a built walk scene: fused depth cloud (grey) with the walkable NavMesh cells (green).
# Usage: python nav_top.py <geometry dir> <out.png>
import sys, json, zlib, base64, numpy as np, open3d as o3d, cv2
g = sys.argv[1]; n = json.load(open(f'{g}/navigation/navmesh.json')); W, H, cs = n['width'], n['height'], n['cell_size_m']
dec = lambda b: np.frombuffer(zlib.decompress(base64.b64decode(b)), np.uint8).reshape(H, W)
walk = dec(n['walkable_grid_base64']); ax, ay = np.array(n['basis']['axis_x']), np.array(n['basis']['axis_y']); ox, oy = n['origin_floor_coordinates']
P = np.asarray(o3d.io.read_point_cloud(f'{g}/dense/fused.ply').points); up = np.cross(ax, ay); up *= np.sign(up @ np.array(n['basis'].get('up', up)))
u = ((P @ ax - ox) / cs * 8).astype(int); v = ((P @ ay - oy) / cs * 8).astype(int); ok = (u >= 0) & (v >= 0) & (u < W * 8) & (v < H * 8)
img = np.zeros((H * 8, W * 8), np.float32); np.add.at(img, (v[ok], u[ok]), 1); img = np.clip(img / np.percentile(img[img > 0], 95), 0, 1)
out = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
m = cv2.resize(walk, (W * 8, H * 8), interpolation=cv2.INTER_NEAREST) > 0; out[m] = (0.5 * out[m] + [0, 120, 0]).astype(np.uint8)
cv2.imwrite(sys.argv[2], out); print(n['walkable_count'], n['movement']['standable_count'])
