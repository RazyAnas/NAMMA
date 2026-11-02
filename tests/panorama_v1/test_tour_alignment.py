from __future__ import annotations

import numpy as np
import pytest

from make_tour import fit_similarity_2d, robust_fit


def test_fit_similarity_recovers_scale_rotation_and_translation():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(30, 2))
    angle = np.radians(37)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    target = 2.75 * source @ rotation + np.array([13.0, -8.0])
    scale, fitted_rotation, translation = fit_similarity_2d(source, target)
    assert scale == pytest.approx(2.75, rel=1e-6)
    assert np.allclose(fitted_rotation, rotation, atol=1e-6)
    assert np.allclose(translation, [13.0, -8.0], atol=1e-6)


def test_robust_fit_rejects_a_gps_outlier():

    rng = np.random.default_rng(11)
    source = rng.normal(size=(20, 2))
    target = 1.8 * source + np.array([4.0, 9.0])
    target[-1] += np.array([500.0, -400.0])
    scale, rotation, translation, inliers, rms = robust_fit(source, target)
    assert scale == pytest.approx(1.8, rel=1e-3)
    assert inliers.sum() == 19
    assert rms < 1e-3

def test_load_panorama_metadata_uses_prepare_output_directory(tmp_path):
    import json
    from make_tour import load_panorama_metadata

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "views_manifest.json").write_text(
        json.dumps({"schema_version": 1, "views": []}), encoding="utf-8"
    )
    (data_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "panoramas": [{"pano_id": "p1"}]}),
        encoding="utf-8"
    )
    # Parent-level sentinels make this test catch the previous one-directory-too-high bug.
    (tmp_path / "views_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"panoramas": [{"pano_id": "wrong-parent"}]}), encoding="utf-8"
    )

    views, source = load_panorama_metadata(data_dir)
    assert views["schema_version"] == 1
    assert source["panoramas"][0]["pano_id"] == "p1"
