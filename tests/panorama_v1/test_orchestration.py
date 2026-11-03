from __future__ import annotations

import json
from pathlib import Path

import gh_run
from gh_run import KAGGLE_CODE_FILES, stage_kaggle_code


def test_kaggle_staging_contains_every_runtime_module_and_shim(tmp_path: Path):
    stage_kaggle_code(tmp_path)

    assert set(KAGGLE_CODE_FILES) == {
        "prepare_panorama.py",
        "pairing.py",
        "sfm_hloc.py",
        "make_tour.py",
    }
    for name in KAGGLE_CODE_FILES:
        assert (tmp_path / name).is_file(), f"Kaggle input omitted {name}"
    assert (tmp_path / "fused_ssim_shim" / "fused_ssim.py").is_file()


def test_push_event_job_discovery_reads_added_and_modified_paths(tmp_path: Path, monkeypatch):
    repo = tmp_path
    job_relative = Path("colab/real_scenes/panorama_v1/jobs/test-scene.json")
    job_file = repo / job_relative
    job_file.parent.mkdir(parents=True)
    job_file.write_text(json.dumps({"job_id": "test-scene"}), encoding="utf-8")
    event_file = repo / "event.json"
    event_file.write_text(json.dumps({
        "commits": [{
            "added": [job_relative.as_posix()],
            "modified": []
        }],
        "head_commit": {"added": [], "modified": []}
    }), encoding="utf-8")

    monkeypatch.setattr(gh_run, "REPO_ROOT", repo)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.delenv("PANORAMA_JOB_FILE", raising=False)

    assert gh_run.job_file_from_event() == job_file.resolve()


def test_quality_and_live_deploy_workflows_are_manual_only():
    root = Path(__file__).resolve().parents[2]
    quality = (root / ".github/workflows/namma-panorama-quality-test.yml").read_text(encoding="utf-8")
    deploy = (root / ".github/workflows/namma-panorama-deploy.yml").read_text(encoding="utf-8")
    main_pipeline = (root / ".github/workflows/namma-panorama.yml").read_text(encoding="utf-8")

    quality_triggers = quality.split("permissions:", 1)[0]
    deploy_triggers = deploy.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in quality_triggers
    assert "push:" not in quality_triggers
    assert "workflow_dispatch:" in deploy_triggers
    assert "push:" not in deploy_triggers

    # Code edits must not invoke the runtime as if the commit itself carried a job.
    push_block = main_pipeline.split("  workflow_dispatch:", 1)[0]
    assert 'jobs/*.json' in push_block
    assert "poll_inbox.py" not in push_block
    assert "gh_run.py" not in push_block


def test_browser_status_poll_treats_not_found_as_pending_and_preserves_null_gps():
    root = Path(__file__).resolve().parents[2]
    client = (root / "colab/real_scenes/panorama_v1/selector/panorama-client.js").read_text(encoding="utf-8")

    assert "function isMissingStatusFile(error)" in client
    assert "if (!isMissingStatusFile(error)) throw error;" in client
    assert "lat: item.lat," in client
    assert "lon: item.lng" in client
    assert "lat: Number(item.lat)" not in client
