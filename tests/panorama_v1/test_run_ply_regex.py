import re
from pathlib import Path

RUN = Path(__file__).resolve().parents[2] / "colab" / "real_scenes" / "panorama_v1" / "run.py"


def test_exported_ply_regex_matches_gsplat_names():
    src = RUN.read_text(encoding="utf-8")
    m = re.search(r're\.search\((r"point_cloud_[^"]+")', src)
    assert m, "PLY ordering regex not found in run.py"
    pattern = eval(m.group(1))
    found = re.search(pattern, "point_cloud_29999.ply")
    assert found and found.group(1) == "29999"
