"""Tiny synthetic street world (game.json + assets) to test the client build and publish chain.

Usage: python make_synthetic_world.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement

out = Path(sys.argv[1])
(out / "ibl").mkdir(parents=True, exist_ok=True)
xs, zs = np.meshgrid(np.arange(-20, 100, 0.25), np.arange(-20, 20, 0.25), indexing="ij")
g = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], 1)
col = np.full((len(g), 3), 0.35)
col[np.abs(g[:, 2]) < 0.15] = [0.9, 0.9, 0.85]
walls = []
for z0 in (-12, 12):
    wx, wy = np.meshgrid(np.arange(-20, 100, 0.25), np.arange(0, 12, 0.25), indexing="ij")
    walls.append(np.stack([wx.ravel(), wy.ravel(), np.full(wx.size, z0)], 1))
W = np.concatenate(walls)
wc = np.stack([0.7 + 0.2 * np.sin(W[:, 0]), 0.6 + 0.1 * (W[:, 1] % 3 < 1.5), np.full(len(W), 0.5)], 1)
P, C = np.concatenate([g, W]), np.concatenate([col, wc])
names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]
data = np.concatenate([P, (C - 0.5) / 0.28209479177387814, np.full((len(P), 1), 4.0),
                       np.full((len(P), 3), np.log(0.14)), np.tile([1, 0, 0, 0], (len(P), 1))], 1).astype(np.float32)
arr = np.empty(len(data), dtype=[(n, "f4") for n in names])
for j, n in enumerate(names):
    arr[n] = data[:, j]
PlyData([PlyElement.describe(arr, "vertex")]).write(str(out / "scene.ply"))
ground = trimesh.creation.box(extents=[120, 0.2, 40]).apply_translation([40, -0.1, 0])
walls_m = [trimesh.creation.box(extents=[120, 12, 0.4]).apply_translation([40, 6, z]) for z in (-12, 12)]
trimesh.util.concatenate([ground] + walls_m).export(out / "collider.glb")
ground.export(out / "ground.glb")
# textured base layer (stands in for the baked street surface)
tex = np.zeros((256, 256, 3), np.uint8)
yy, xx = np.mgrid[0:256, 0:256]
tex[((xx // 32 + yy // 32) % 2) == 0] = [180, 170, 150]
tex[((xx // 32 + yy // 32) % 2) == 1] = [90, 85, 80]
surf = trimesh.util.concatenate([ground] + walls_m)
uv = np.stack([surf.vertices[:, 0] / 8.0 % 1.0, (surf.vertices[:, 1] + surf.vertices[:, 2]) / 8.0 % 1.0], 1)
surf.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
surf.export(out / "surface.glb")
sky = np.zeros((512, 1024, 3), np.uint8)
v = np.linspace(0, 1, 512)[:, None]
sky[..., 0] = (120 + 100 * v).astype(np.uint8)
sky[..., 1] = (170 + 60 * v).astype(np.uint8)
sky[..., 2] = 240
Image.fromarray(sky).save(out / "sky.jpg")
Image.fromarray(sky).save(out / "ibl" / "p0000.jpg")
json.dump({"units": "m", "up": "y",
           "files": {"splats": "scene.ply", "collider": "collider.glb", "ground_mesh": "ground.glb", "sky": "sky.jpg",
                     "surface": "surface.glb"},
           "spawn": {"pos": [0, 1.7, 0], "look_at": [10, 1.7, 0]},
           "route": [[float(x), 1.7, 0.0] for x in range(0, 90, 10)],
           "ibl": [{"file": "ibl/p0000.jpg", "pos": [0, 2, 0]}]}, open(out / "game.json", "w"))
print("synthetic world:", len(P), "splats")
