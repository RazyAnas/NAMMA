import json
import subprocess
from pathlib import Path

import gh_run


def test_job_file_from_event_uses_checked_out_commit_diff(tmp_path, monkeypatch):
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

    def fake_git_diff_tree(args, cwd, capture_output, text, check):
        assert args == [
            "git", "diff-tree", "--no-commit-id", "--name-only", "-r",
            "--diff-filter=ACMR", "abc123"
        ]
        assert cwd == tmp_path
        return subprocess.CompletedProcess(args, 0, stdout=relative + "\n", stderr="")

    monkeypatch.setattr(gh_run.subprocess, "run", fake_git_diff_tree)
    assert gh_run.job_file_from_event() == manifest.resolve()


def test_job_file_from_event_ignores_old_manifests_in_aggregate_push_payload(tmp_path, monkeypatch):
    old_relative = "colab/real_scenes/panorama_v1/jobs/old-job.json"
    current_relative = "colab/real_scenes/panorama_v1/jobs/current-job.json"
    old_manifest = tmp_path / old_relative
    current_manifest = tmp_path / current_relative
    old_manifest.parent.mkdir(parents=True)
    old_manifest.write_text("{}", encoding="utf-8")
    current_manifest.write_text("{}", encoding="utf-8")

    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({
        "commits": [{"added": [old_relative], "modified": []}],
        "head_commit": {"added": [current_relative], "modified": []}
    }), encoding="utf-8")

    monkeypatch.setattr(gh_run, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PANORAMA_JOB_FILE", "")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_SHA", "def456")

    def fake_git_diff_tree(args, cwd, capture_output, text, check):
        return subprocess.CompletedProcess(
            args, 0, stdout=current_relative + "\n", stderr=""
        )

    monkeypatch.setattr(gh_run.subprocess, "run", fake_git_diff_tree)
    assert gh_run.job_file_from_event() == current_manifest.resolve()


def test_job_file_from_event_falls_back_to_head_commit_when_git_diff_fails(tmp_path, monkeypatch):
    relative = "colab/real_scenes/panorama_v1/jobs/current-job.json"
    manifest = tmp_path / relative
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({
        "commits": [],
        "head_commit": {"added": [relative], "modified": []}
    }), encoding="utf-8")

    monkeypatch.setattr(gh_run, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PANORAMA_JOB_FILE", "")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_SHA", "ghi789")

    def fake_git_diff_tree(args, cwd, capture_output, text, check):
        return subprocess.CompletedProcess(
            args, 128, stdout="", stderr="fatal: not a git repository"
        )

    monkeypatch.setattr(gh_run.subprocess, "run", fake_git_diff_tree)
    assert gh_run.job_file_from_event() == manifest.resolve()
