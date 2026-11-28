import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "colab" / "real_scenes" / "panorama_v1"))
pytest.importorskip("xatlas")
import export_game  # noqa: E402


def messy_mesh(pieces=400, n=6, seed=0):
    """Many small bumpy disconnected patches, like a TSDF street mesh after simplification."""
    rng = np.random.default_rng(seed)
    V, F = [], []
    for k in range(pieces):
        xs, zs = np.meshgrid(np.arange(n), np.arange(n))
        v = np.stack([xs.ravel(), rng.normal(0, 0.3, n * n), zs.ravel()], 1).astype(np.float32)
        v = v @ np.linalg.qr(rng.normal(size=(3, 3)))[0].astype(np.float32) + rng.uniform(-50, 50, 3)
        i = np.arange(n * n).reshape(n, n)
        a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
        F.append(np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]) + len(V) * n * n)
        V.append(v)
    return np.concatenate(V), np.concatenate(F).astype(np.int64)


def test_uv_unwrap_valid():
    V, F = messy_mesh()
    t = time.time()
    vmap, idx, uvs = export_game.uv_unwrap(V, F, 2048, timeout_s=120, log=lambda *a: None)
    assert time.time() - t < 60
    assert idx.reshape(-1, 3).shape[0] == len(F)
    assert vmap.max() < len(V) and uvs.min() >= 0 and uvs.max() <= 1


def test_uv_unwrap_times_out():
    import subprocess

    V, F = messy_mesh(pieces=2000)
    with pytest.raises(subprocess.TimeoutExpired):
        export_game.uv_unwrap(V, F, 2048, timeout_s=0.01, log=lambda *a: None)
