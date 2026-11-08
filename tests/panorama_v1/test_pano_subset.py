import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "colab" / "real_scenes" / "panorama_v1"))
import train_splats  # noqa: E402


def cams_along_route(n=40, views=4, spacing=10.0):
    cams = []
    for i in range(n):
        for k in range(views):
            w2c = np.eye(4)
            w2c[:3, 3] = -np.array([i * spacing, 0.0, 0.0])
            cams.append({"pano": f"{i:04d}_abc", "name": f"pano_camera{k}/{i:04d}_abc.jpg", "w2c": w2c.tolist(),
                         "split": "val" if i % 8 == 0 else "train"})
    return cams


def test_subset_keeps_views_and_nearby_points():
    cams = cams_along_route()
    pts = np.zeros((400, 6), np.float32)
    pts[:, 0] = np.linspace(-50, 450, 400)
    kept, kp = train_splats.pano_subset(cams, pts, "16-39")
    assert {c["pano"][:4] for c in kept} == {f"{i:04d}" for i in range(16, 40)}
    assert len(kept) == 24 * 4
    assert kp[:, 0].min() >= 160 - 40 - 1e-3 and kp[:, 0].max() <= 390 + 40 + 1e-3
