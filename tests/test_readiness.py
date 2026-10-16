from pathlib import Path
from types import SimpleNamespace

from app.services.readiness import project_readiness


def test_readiness_reports_first_incomplete_gate(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    paths = {
        "root": tmp_path,
        "source": tmp_path / "source",
        "frames": tmp_path / "frames",
        "tracking": tmp_path / "tracking",
        "training": tmp_path / "training",
        "artifacts": tmp_path / "artifacts",
        "logs": tmp_path / "logs",
    }
    paths["artifacts"].mkdir()
    project = SimpleNamespace(
        id="demo",
        status="imported",
        source_path=str(source),
        frame_count=0,
    )
    result = project_readiness(paths, project)
    assert result["ready_for_visual_roaming"] is False
    assert result["ready_for_obstacle_aware_roaming"] is False
    assert result["gates"][0]["complete"] is True
    assert result["gates"][1]["complete"] is False
    assert result["next_action"] == "Extract frames from the source video."
