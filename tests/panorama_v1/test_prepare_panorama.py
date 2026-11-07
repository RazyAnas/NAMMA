from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from prepare_panorama import load_manifest, make_views, perspective_maps


def test_perspective_map_centre_points_to_equirectangular_centre():
    mx, my, focal = perspective_maps(
        width=768, height=576, source_width=1536, source_height=768,
        yaw_deg=0.0, hfov_deg=60.0
    )
    assert mx.shape == (576, 768)
    assert my.shape == (576, 768)
    assert focal == pytest.approx(768 / (2 * np.tan(np.radians(30))))
    assert mx[288, 383] == pytest.approx(768, abs=2)
    assert my[288, 383] == pytest.approx(384, abs=2)
    assert np.isfinite(mx).all() and np.isfinite(my).all()


def test_yaw_rotates_centre_sample_around_panorama():
    mx, my, _ = perspective_maps(
        width=384, height=288, source_width=1536, source_height=768,
        yaw_deg=90.0, hfov_deg=60.0
    )
    assert mx[144, 191] == pytest.approx(1152, abs=3)
    assert my[144, 191] == pytest.approx(384, abs=3)


def test_make_views_writes_expected_view_count(tmp_path: Path):
    source = tmp_path / "source"
    pano_dir = source / "panoramas"
    pano_dir.mkdir(parents=True)
    # An equirectangular image with a different colour in each horizontal band.
    h, w = 128, 256
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :w // 2] = (20, 80, 180)
    image[:, w // 2:] = (180, 80, 20)
    for i in range(8):
        assert cv2.imwrite(str(pano_dir / f"p{i}.jpg"), image)
    manifest = {
        "schema_version": 1,
        "scene_name": "test",
        "panoramas": [
            {"pano_id": f"p{i}", "image": f"panoramas/p{i}.jpg", "sequence": i}
            for i in range(8)
        ]
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "derived"
    report = make_views(source, out, image_size=384, view_count=8, hfov_deg=60, max_panoramas=8)
    assert report["source_panoramas"] == 8
    assert report["perspective_views"] == 64
    assert len(list((out / "images").glob("*.jpg"))) == 64
    assert (out / "views_manifest.json").is_file()


def test_manifest_rejects_path_traversal(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "panoramas": [
            {"pano_id": str(i), "image": "../outside.jpg"}
            for i in range(8)
        ]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        load_manifest(tmp_path)
