"""sky_clean: buildings baked into the learned dome are replaced by sky; real sky texels are kept."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "colab/real_scenes/panorama_v1"))
import sky_clean  # noqa: E402


def street_dome(h=128, w=256, seed=0):
    """Blue sky on top, a noisy dark 'building' band from 20 deg elevation down, grey road below."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    img[:] = [0.55, 0.72, 0.92]
    lat = (0.5 - (np.arange(h) + 0.5) / h) * 180
    band = (lat < 20)[:, None] & np.ones((1, w), bool)
    img[band] = 0.25 + 0.3 * rng.random((band.sum(), 3))
    return img, band


def test_heuristic_removes_buildings_and_keeps_sky():
    img, band = street_dome()
    out = sky_clean.clean_sky(img)
    h = img.shape[0]
    above = band[: h // 2]
    # where the building was (above the horizon) it is now smooth and sky-coloured
    blg = out[: h // 2][above]
    assert blg.std(0).max() < 0.05
    assert (blg[:, 2] >= blg[:, 0]).mean() > 0.95
    # the open sky at the top is untouched
    assert np.allclose(out[:10], img[:10], atol=1e-6)


def test_observed_mask_is_respected():
    img, band = street_dome()
    observed = ~band
    out = sky_clean.clean_sky(img, observed)
    assert np.allclose(out[observed[: img.shape[0]]], img[observed], atol=1e-6)
    assert out[band].std(0).max() < 0.05


def test_no_sky_seen_falls_back_to_haze():
    img = np.full((64, 128, 3), 0.3, np.float32)
    out = sky_clean.clean_sky(img, np.zeros((64, 128), bool))
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
