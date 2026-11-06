import math

import numpy as np

import train_splats as ts


def _cams(n=40, spacing=10.0):
    cams = []
    for i in range(n):
        for k in range(4):
            w2c = np.eye(4)
            w2c[:3, 3] = -np.array([i * spacing, 0.0, 0.0])
            cams.append({"pano": f"{i:04d}", "w2c": w2c.tolist(), "kind": "real"})
    return cams


def test_cap_is_about_three_metres_on_a_street():
    cams = _cams()
    centre, scale = ts.normalisation(cams)
    cap_m = ts.max_splat_size(cams, scale) / scale
    assert math.isclose(cap_m, 3.0, rel_tol=1e-6)
    assert ts.max_splat_size(cams, scale) < 0.15  # the old fixed cap was ~20 m here
