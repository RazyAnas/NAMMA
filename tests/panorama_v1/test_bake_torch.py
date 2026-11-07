import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
xatlas = pytest.importorskip("xatlas")
trimesh = pytest.importorskip("trimesh")
from PIL import Image  # noqa: E402

import export_game as eg  # noqa: E402


def test_bake_reproduces_checkerboard_wall(tmp_path):
    # wall: plane z = 5 (world = game frame here: R = I, origin = 0, scale = 1), seen by one camera at origin
    wall = trimesh.Trimesh(vertices=[[-2, -2, 5], [2, -2, 5], [2, 2, 5], [-2, 2, 5]], faces=[[0, 1, 2], [0, 2, 3]])
    wall = wall.subdivide().subdivide()
    w = h = 200
    f = 250.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    ys, xs = np.mgrid[0:h, 0:w]
    X = (xs + 0.5 - w / 2) / f * 5
    Y = (ys + 0.5 - h / 2) / f * 5
    checker = ((np.floor(X * 2) + np.floor(Y * 2)) % 2 == 0)
    img = np.where(checker[..., None], [230, 230, 230], [20, 20, 20]).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "v.png")
    np.save(tmp_path / "d.npy", np.full((h, w), 5.0, np.float32))
    cam = {"name": "v", "path": str(tmp_path / "v.png"), "depth": str(tmp_path / "d.npy"), "width": w, "height": h,
           "K": K, "w2c": np.eye(4), "split": "train", "pano": "p0"}
    rep = eg.bake_textured_mesh(wall, [cam], np.eye(3), np.zeros(3), 1.0, tmp_path / "s.glb", atlas=256, device="cpu",
                                log=lambda *a: None)
    assert rep["texel_coverage"] > 0.9
    tex = np.asarray(Image.open(tmp_path / "s.preview.jpg").convert("L"))
    # a checkerboard survives baking: strongly bimodal texture
    assert (tex < 70).mean() > 0.25 and (tex > 180).mean() > 0.25
