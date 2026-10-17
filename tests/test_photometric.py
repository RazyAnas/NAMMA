import numpy as np

from app.services.photometric import solve_gains


def test_solve_gains_recovers_exposure_and_white_balance():
    rng = np.random.default_rng(1)
    frames, points = 30, 400
    true_gain = rng.normal(0, 0.4, (frames, 3))
    true_gain -= true_gain.mean(axis=0)
    radiance = rng.normal(-1.5, 0.5, (points, 3))
    frame_index = rng.integers(0, frames, 6000)
    point_index = rng.integers(0, points, 6000)
    observed = true_gain[frame_index] + radiance[point_index] + rng.normal(0, 0.02, (6000, 3))
    outliers = rng.random(6000) < 0.05
    observed[outliers] += 1.5  # specular highlights / bad matches
    gains, _ = solve_gains(frame_index, point_index, observed, frames, points)
    assert np.abs(gains - true_gain).max() < 0.05
