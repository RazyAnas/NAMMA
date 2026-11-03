import json
import subprocess
from pathlib import Path

import gh_run


def test_job_file_from_event_falls_back_to_checked_out_commit(tmp_path, monkeypatch):
    relative = "colab/real_scenes/panorama_v1/jobs/api-submitted.json"
    manifest = tmp_path / relative
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({"commits": [], "head_commit": None}), encoding="utf-8")

    monkeypatch.setattr(gh_run, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PANORAMA_JOB_FILE", "")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    def fake_git_show(args, cwd, capture_output, text, check):
        assert args == [
            "git", "show", "--format=", "--name-only", "--diff-filter=ACMR", "abc123"
        ]
        assert cwd == tmp_path
        return subprocess.CompletedProcess(
            args, 0, stdout=relative + "\n", stderr=""
        )

    monkeypatch.setattr(gh_run.subprocess, "run", fake_git_show)

    assert gh_run.job_file_from_event() == manifest.resolve()
