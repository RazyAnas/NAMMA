import numpy as np

from app.services.monocular import fit_depth_alignment


def test_fit_depth_alignment_recovers_direct_affine_depth() -> None:
    predicted = np.linspace(0.5, 5.0, 100)
    geometric = 2.5 * predicted + 0.7
    result = fit_depth_alignment(predicted, geometric)
    assert result["mode"] == "direct"
    assert np.isclose(result["scale"], 2.5, atol=1e-4)
    assert np.isclose(result["shift"], 0.7, atol=1e-4)


def test_fit_depth_alignment_recovers_inverse_depth() -> None:
    predicted = np.linspace(0.3, 3.0, 100)
    geometric = 4.0 / predicted + 0.2
    result = fit_depth_alignment(predicted, geometric)
    assert result["mode"] == "inverse"
    assert result["median_relative_error"] < 1e-5
