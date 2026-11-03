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
        "gsplat_colmap_compat.py",
        "gsplat_trainer_compat.py",
        "ssim_preflight.py",
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
    monkeypatch.delenv("GITHUB_SHA", raising=False)
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


def test_static_exporter_preflight_imports_the_existing_read_only_builder():
    exporter = gh_run.preflight_static_exporter()
    assert callable(getattr(exporter, "build", None))


def test_viewer_and_exporter_preflights_precede_kaggle_upload_and_kernel_push():
    root = Path(__file__).resolve().parents[2]
    source = (root / "colab/real_scenes/panorama_v1/gh_run.py").read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    main = next(node for node in tree.body
                if isinstance(node, __import__("ast").FunctionDef) and node.name == "main")

    def index(name):
        for i, statement in enumerate(main.body):
            if any(
                isinstance(node, __import__("ast").Call)
                and isinstance(node.func, __import__("ast").Name)
                and node.func.id == name
                for node in __import__("ast").walk(statement)
            ):
                return i
        raise AssertionError(f"main() does not call {name}")

    assert index("ensure_viewer_toolchain") < index("upload_input_bundle")
    assert index("preflight_static_exporter") < index("upload_input_bundle")
    assert index("upload_input_bundle") < index("push_kernel")


def test_publish_tour_reuses_preflighted_viewer_and_exporter():
    root = Path(__file__).resolve().parents[2]
    source = (root / "colab/real_scenes/panorama_v1/gh_run.py").read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    publish = next(node for node in tree.body
                   if isinstance(node, __import__("ast").FunctionDef) and node.name == "publish_tour")
    calls = {
        node.func.id for node in __import__("ast").walk(publish)
        if isinstance(node, __import__("ast").Call) and isinstance(node.func, __import__("ast").Name)
    }
    assert "ensure_viewer_toolchain" in calls
    assert "preflight_static_exporter" in calls


def test_status_updates_preserve_current_kernel_and_clear_it_for_a_new_retry(tmp_path: Path, monkeypatch):
    import huggingface_hub

    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({
        "job_id": "test-job",
        "pipeline": "panorama_v1",
        "status": "training",
        "kaggle_dataset": "user/input",
        "kaggle_kernel": "user/latest-kernel",
    }), encoding="utf-8")

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == "user/results"
        assert kwargs["filename"] == "test-job/status.json"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["force_download"] is True
        return str(status_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class FakeApi:
        payload = None

        def upload_file(self, path_or_fileobj, **kwargs):
            self.payload = json.loads(path_or_fileobj.decode("utf-8"))
            assert kwargs["path_in_repo"] == "test-job/status.json"

    api = FakeApi()
    gh_run.update_status(api, "user/results", "test-job", "failed", error="kernel failed")
    assert api.payload["status"] == "failed"
    assert api.payload["kaggle_kernel"] == "user/latest-kernel"
    assert api.payload["error"] == "kernel failed"

    status_file.write_text(json.dumps(api.payload), encoding="utf-8")
    gh_run.update_status(api, "user/results", "test-job", "staging")
    assert api.payload["status"] == "staging"
    assert "kaggle_kernel" not in api.payload
    assert "kaggle_dataset" not in api.payload
    assert "error" not in api.payload
