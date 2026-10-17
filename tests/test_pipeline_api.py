import asyncio

import numpy as np
import pytest

from app.pipeline import backends
from app.pipeline.stages.tracking import umeyama
from app.pipeline.tasks import task


pytestmark = pytest.mark.anyio


@task("test.echo")
def echo(ctx):
    ctx.progress("echoing", 0.5)
    ctx.log("hello from the test task")
    return {"echo": ctx.params.get("value"), "project": ctx.project.id}


@pytest.fixture
def no_gpu(monkeypatch):
    real = backends.capabilities
    monkeypatch.setattr(backends, "capabilities", lambda: {**real(), "cuda": False, "mps": False, "opensplat": False})


async def _project(client) -> str:
    response = await client.post(
        "/api/v1/projects", data={"name": "Pipeline room"}, files={"video": ("room.mp4", b"video", "video/mp4")}
    )
    return response.json()["id"]


async def _wait(client, job_id: str) -> dict:
    for _ in range(200):
        detail = (await client.get(f"/api/v1/pipeline/jobs/{job_id}")).json()
        if detail["status"] in {"succeeded", "failed"}:
            return detail
        await asyncio.sleep(0.05)
    raise AssertionError("job did not finish")


async def test_stage_catalog_covers_every_task(client):
    stages = (await client.get("/api/v1/pipeline/stages")).json()
    kinds = {stage["kind"] for stage in stages}
    for kind in ("capture.bundle", "tracking.sfm", "tracking.ar_poses", "points.monocular", "splats.train", "export.splat_pack"):
        assert kind in kinds
    assert all(stage["status"] in {"verified", "wired"} for stage in stages)


async def test_backends_report_devices(client):
    payload = (await client.get("/api/v1/pipeline/backends")).json()
    assert payload["capabilities"]["cpu"] is True
    assert payload["recommended_device"] in {"cpu", "mps", "cuda"}


async def test_findings_keep_measured_and_quoted_apart(client):
    payload = (await client.get("/api/v1/pipeline/findings")).json()
    labels = {item["label"] for items in payload["stages"].values() for item in items}
    assert labels == {"M", "Q"}
    assert payload["dead_ends"]


async def test_task_runs_in_background_with_status_result_and_log(client):
    project_id = await _project(client)
    response = await client.post(f"/api/v1/pipeline/projects/{project_id}/run", json={"stages": ["test.echo"], "params": {"test.echo": {"value": 7}}})
    assert response.status_code == 202
    detail = await _wait(client, response.json()["id"])
    assert detail["status"] == "succeeded", detail
    assert detail["result"]["stages"]["test.echo"] == {"echo": 7, "project": project_id}
    assert any("hello from the test task" in line for line in detail["log_tail"])
    jobs = (await client.get(f"/api/v1/pipeline/projects/{project_id}/jobs")).json()
    assert jobs[0]["kind"] == "pipeline.run"


async def test_cuda_only_trainer_is_refused_without_a_gpu(client, no_gpu):
    project_id = await _project(client)
    response = await client.post(f"/api/v1/pipeline/projects/{project_id}/splats/train", json={"trainer": "splatfacto"})
    assert response.status_code == 503
    assert response.json()["detail"]["missing"] == ["cuda"]


async def test_device_parameter_adds_its_requirement(client, no_gpu):
    project_id = await _project(client)
    response = await client.post(f"/api/v1/pipeline/projects/{project_id}/tracking/deblur", json={"device": "mps"})
    assert response.status_code == 503
    assert "mps" in response.json()["detail"]["missing"]


async def test_full_run_skips_stages_this_machine_cannot_run(client, no_gpu):
    project_id = await _project(client)
    response = await client.post(f"/api/v1/pipeline/projects/{project_id}/run", json={"stages": ["splats.train"]})
    detail = await _wait(client, response.json()["id"])
    assert detail["status"] == "succeeded", detail
    assert detail["result"]["stages"]["splats.train"] == {"skipped": True, "missing": ["opensplat"]}


async def test_stage_errors_fail_the_job_with_a_message(client):
    project_id = await _project(client)
    response = await client.post(f"/api/v1/pipeline/projects/{project_id}/tracking/ar-poses", json={})
    detail = await _wait(client, response.json()["id"])
    assert detail["status"] == "failed"
    assert "ARCore" in detail["error_message"]


async def test_walk_scene_export_needs_outputs(client):
    project_id = await _project(client)
    assert (await client.get(f"/api/v1/pipeline/projects/{project_id}/export/walk-scene")).status_code == 409


def test_umeyama_recovers_a_similarity_transform():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(30, 3))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    target = 2.5 * points @ rotation.T + np.array([1.0, -2.0, 0.5])
    scale, r, t = umeyama(points, target)
    assert scale == pytest.approx(2.5)
    assert np.allclose(r, rotation)
    assert np.allclose(t, [1.0, -2.0, 0.5])
