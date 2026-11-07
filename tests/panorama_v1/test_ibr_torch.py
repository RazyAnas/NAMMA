import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

import densify  # noqa: E402


def test_ibr_from_the_real_camera_reproduces_its_photo(tmp_path):
    h, w = 60, 80
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / "v.png")
    np.save(tmp_path / "d.npy", np.full((h, w), 4.0, np.float32))
    K = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    cam = {"name": "v", "path": str(tmp_path / "v.png"), "depth": str(tmp_path / "d.npy"), "width": w,
           "height": h, "K": K, "w2c": np.eye(4), "split": "train", "pano": "p0"}
    real = densify.RealViews([cam], device="cpu")
    out, mask = densify.ibr_composite(real, np.eye(4), K, w, h, np.full((h, w), 4.0), np.zeros(3), 1.0)
    assert mask[5:-5, 5:-5].all()
    assert np.abs(out[5:-5, 5:-5].astype(int) - img[5:-5, 5:-5].astype(int)).max() <= 2


def test_ibr_rejects_occluded_pixels(tmp_path):
    h, w = 40, 40
    Image.fromarray(np.full((h, w, 3), 200, np.uint8)).save(tmp_path / "v.png")
    np.save(tmp_path / "d.npy", np.full((h, w), 2.0, np.float32))   # real view saw a wall at 2 m
    K = np.array([[30.0, 0, w / 2], [0, 30.0, h / 2], [0, 0, 1]])
    cam = {"name": "v", "path": str(tmp_path / "v.png"), "depth": str(tmp_path / "d.npy"), "width": w,
           "height": h, "K": K, "w2c": np.eye(4), "split": "train", "pano": "p0"}
    real = densify.RealViews([cam], device="cpu")
    _, mask = densify.ibr_composite(real, np.eye(4), K, w, h, np.full((h, w), 6.0), np.zeros(3), 1.0)
    assert not mask.any()          # novel view's surface at 6 m is hidden behind the 2 m wall
