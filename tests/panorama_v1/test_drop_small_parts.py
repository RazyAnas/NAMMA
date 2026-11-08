"""tsdf.drop_small_parts keeps exactly the parts trimesh.split would keep, without building one mesh per fragment."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "colab/real_scenes/panorama_v1"))
import tsdf  # noqa: E402


def test_drops_fragments_keeps_big_parts():
    rng = np.random.default_rng(0)
    big = trimesh.creation.icosphere(subdivisions=4)            # 5120 faces
    mid = trimesh.creation.icosphere(subdivisions=3)            # 1280 faces
    mid.apply_translation([5, 0, 0])
    frags = []
    for _ in range(50):
        b = trimesh.creation.icosphere(subdivisions=1, radius=0.01)  # 80 faces each
        b.apply_translation(rng.uniform(-3, 3, 3))
        frags.append(b)
    m = trimesh.util.concatenate([big, mid] + frags)
    m = trimesh.Trimesh(m.vertices, m.faces, vertex_colors=np.full((len(m.vertices), 4), 200, np.uint8), process=False)
    out = tsdf.drop_small_parts(m, 500)
    assert len(out.faces) == 5120 + 1280
    assert len(out.vertices) == len(big.vertices) + len(mid.vertices)
    assert len(out.visual.vertex_colors) == len(out.vertices)
    assert out.faces.max() < len(out.vertices)


def test_nothing_to_drop_returns_same_mesh():
    m = trimesh.creation.icosphere(subdivisions=3)
    assert tsdf.drop_small_parts(m, 500) is m
