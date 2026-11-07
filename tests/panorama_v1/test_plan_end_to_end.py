from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from make_tour import build_plan


def test_build_plan_writes_route_settings_from_synthetic_camera_poses(tmp_path, monkeypatch):
    panoramas = [
        {
            "pano_id": f"p{i}",
            "image": f"panoramas/p{i}.jpg",
            "sequence": i,
            "lat": 37.0 + i * 0.00001,
            "lon": -122.0,
        }
        for i in range(8)
    ]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "views_manifest.json").write_text(
        json.dumps({"schema_version": 1, "views": []}), encoding="utf-8"
    )
    (data_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "panoramas": panoramas}), encoding="utf-8"
    )

    class FakeColmapParser:
        def __init__(self, _data_dir, factor=1, normalize=True, test_every=8):
            assert factor == 1
            assert normalize is True
            self.image_names = []
            camera_poses = []
            for pano_index in range(8):
                for view_index in range(12):
                    name = f"p{pano_index:05d}_v{view_index:02d}.jpg"
                    self.image_names.append(name)
                    pose = np.eye(4, dtype=np.float64)
                    # A proper 180-degree rotation around Z: local camera-down is world -Y.
                    pose[:3, :3] = np.diag([-1.0, -1.0, 1.0])
                    pose[:3, 3] = [float(pano_index), 1.5, 0.0]
                    camera_poses.append(pose)
            self.camtoworlds = np.stack(camera_poses)
            self.points = np.array(
                [[float(i), float(y), float(z)]
                 for i in range(8)
                 for y in (0.2, 1.0, 2.5)
                 for z in (-0.5, 0.0, 0.5)],
                dtype=np.float64,
            )

    datasets_module = types.ModuleType("datasets")
    datasets_module.__path__ = []
    colmap_module = types.ModuleType("datasets.colmap")
    colmap_module.Parser = FakeColmapParser
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "datasets.colmap", colmap_module)

    run_dir = tmp_path / "run"
    tour_dir = tmp_path / "tour"
    examples_dir = tmp_path / "gsplat-examples"
    examples_dir.mkdir()

    result = build_plan(data_dir, run_dir, tour_dir, examples_dir)

    plan = json.loads((tour_dir / "plan.json").read_text(encoding="utf-8"))
    settings = json.loads((tour_dir / "settings.json").read_text(encoding="utf-8"))
    tour = json.loads((tour_dir / "tour.json").read_text(encoding="utf-8"))

    assert result["panoramas_registered"] == 8
    assert result["gps_aligned"] is True
    assert result["gps_inliers"] >= 4
    assert len(plan["rotate"]) == 3
    assert len(plan["translate"]) == 3
    assert len(plan["box"]) == 6
    assert len(settings["cameras"]) == 1
    assert len(settings["animTracks"][0]["keyframes"]["times"]) >= 2
    assert tour["pipeline"] == "panorama_v1"
    assert tour["spots"] > 0
