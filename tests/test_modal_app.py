"""modal_app.py's web/GPU split, run in-process against a stand-in for the `modal` package.

What it checks: an upload on the web side spawns the GPU function with the job's rows, the GPU side
publishes live state that the phone's status and history calls show, and the end state (including a
render container that died) lands in the web side's history. What it cannot check: Modal itself
(image build, volumes, GPUs); the deploy workflow does that.
"""

import importlib
import io
import json
import sys
import threading
import time
import types
import zipfile

import pytest

from app.pipeline import tasks
from app.pipeline.stages import phone


pytestmark = pytest.mark.anyio


class _Chain:
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _Volume:
    def __init__(self):
        self.commits = self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


class _Call:
    calls = {}

    def __init__(self, function, args):
        self.object_id = f"fc-{len(self.calls)}"
        self.result, self.error = None, None
        _Call.calls[self.object_id] = self
        self.thread = threading.Thread(target=self._run, args=(function, args))
        self.thread.start()

    def _run(self, function, args):
        _fake.current = self.object_id
        try:
            self.result = function(*args)
        except Exception as exc:  # noqa: BLE001
            self.error = exc

    @classmethod
    def from_id(cls, call_id):
        return cls.calls[call_id]

    def get(self, timeout=None):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError
        if self.error:
            raise self.error
        return self.result


def _function(**_):
    def wrap(function):
        function.spawn = lambda *args: _Call(function, args)
        return function
    return wrap


_fake = types.SimpleNamespace(current=None)


def _fake_modal():
    module = types.ModuleType("modal")
    module.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    module.App = lambda name: types.SimpleNamespace(function=_function)
    module.Volume = types.SimpleNamespace(from_name=lambda name, **k: _Volume())
    module.Dict = types.SimpleNamespace(from_name=lambda name, **k: {})
    module.Secret = types.SimpleNamespace(from_name=lambda name: object())
    module.FunctionCall = _Call
    module.concurrent = lambda **k: (lambda f: f)
    module.asgi_app = lambda: (lambda f: f)
    module.current_function_call_id = lambda: _fake.current
    return module


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    sys.modules.pop("modal_app", None)
    module = importlib.import_module("modal_app")
    monkeypatch.setattr(module, "_enter_repo", lambda: None)
    monkeypatch.setattr(phone, "REMOTE", None)
    monkeypatch.setattr(tasks, "REMOTE_RUNNER", None)
    module.web()  # installs the hooks, as the Modal web container does
    yield module
    phone.REMOTE = None
    tasks.REMOTE_RUNNER = None


def _bundle() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("meta.json", json.dumps({"format": "namma-sensor-bundle", "mode": "ar"}))
    return buffer.getvalue()


async def _send(client) -> str:
    data = _bundle()
    start = (await client.post("/api/v1/remote/uploads", json={"filename": "lobby.zip", "size": len(data), "name": "Lobby"})).json()
    await client.put(f"/api/v1/remote/uploads/{start['upload_id']}/chunks/0", content=data)
    return (await client.post(f"/api/v1/remote/uploads/{start['upload_id']}/finish", json={"quality": "fast"})).json()["render_id"]


async def _until(client, render_id, done, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        status = (await client.get(f"/api/v1/remote/renders/{render_id}")).json()
        if done(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"timed out: {status}")


async def test_cloud_render_reports_progress_and_lands_in_history(client, cloud, monkeypatch):
    gate = threading.Event()

    def fake_run_task(job_id):  # stands in for the GPU pipeline: one running step, then a result
        from app.config import get_settings
        from app.database import SessionLocal
        from app.models import Job

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            job.status, job.stage, job.progress = "running", "splat:gaussian_training", 0.5
            db.commit()
            directory = tasks.job_dir(get_settings(), job.project_id, job_id)
        (directory / "steps.json").write_text(json.dumps([{"key": "splat", "label": "Photoreal splat", "state": "running"}]))
        gate.wait(10)
        (directory / "outputs").mkdir(exist_ok=True)
        (directory / "outputs" / "splat.splat").write_bytes(b"\0" * 32)
        (directory / "outputs" / "result.json").write_text(json.dumps({"best": "splat.splat", "quality": "fast"}))
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            job.status, job.stage, job.progress = "succeeded", "complete", 1.0
            db.commit()

    monkeypatch.setattr(tasks, "run_task", fake_run_task)
    hello = (await client.get("/api/v1/remote/hello")).json()
    assert hello["cloud"] and hello["photoreal"] and hello["history"]
    render_id = await _send(client)
    assert cloud.data.commits >= 1  # the upload was committed before the GPU side started
    running = await _until(client, render_id, lambda s: s["steps"] and s["steps"][0]["state"] == "running")
    assert running["status"] == "running" and running["progress"] == 0.5
    history = (await client.get("/api/v1/remote/renders")).json()["renders"]
    assert history[0]["render_id"] == render_id and history[0]["name"] == "Lobby" and history[0]["status"] == "running"
    gate.set()
    done = await _until(client, render_id, lambda s: s["status"] == "succeeded")
    assert done["best"] == "splat.splat" and "splat.splat" in done["outputs"] and cloud.data.reloads >= 1
    assert (await client.get("/api/v1/remote/renders")).json()["renders"][0]["status"] == "succeeded"


async def test_a_dead_render_container_marks_the_render_failed(client, cloud, monkeypatch):
    def crash(job_id):
        raise RuntimeError("GPU container lost")

    monkeypatch.setattr(tasks, "run_task", crash)
    monkeypatch.setattr(cloud, "STALE_SECONDS", 0)
    render_id = await _send(client)
    _Call.calls[cloud.live[render_id]["call_id"]].thread.join(10)
    cloud.live[render_id] = {**cloud.live[render_id], "status": "running", "updated": 0}  # its last heartbeat, long ago
    failed = await _until(client, render_id, lambda s: s["status"] == "failed")
    assert "GPU container lost" in failed["error"]
