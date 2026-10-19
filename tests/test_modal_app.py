"""modal_app.py's web/GPU split, run in-process against a stand-in for the `modal` package.

What it checks: an upload on the web side starts a GPU sandbox running scripts/cloud_render.py with the
job's rows, the phone's status and history calls show the sandbox's live state, and the end state lands
in the web side's history; a sandbox that dies, and an account Modal refuses a GPU (no card yet), end
as failed renders the phone can explain and try again. What it cannot check: Modal itself (image build,
volumes, GPUs); the deploy workflow does that.
"""

import importlib
import io
import json
from pathlib import Path
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


class _Sandbox:
    boxes = {}
    refuse = None  # set to Modal's error text to refuse the GPU, as for an account with no card

    def __init__(self, args, kwargs):
        self.object_id = f"sb-{len(self.boxes)}"
        self.kwargs = kwargs
        self.live = Path(_fake.tmp) / f"{self.object_id}.json"
        self.code = None
        _Sandbox.boxes[self.object_id] = self
        payload = json.loads(args[-1])
        self.thread = threading.Thread(target=self._run, args=(payload,))
        self.thread.start()
        self.filesystem = types.SimpleNamespace(read_text=lambda path: self.live.read_text())
        self.stderr = types.SimpleNamespace(read=lambda: "")

    def _run(self, payload):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import cloud_render

        try:
            self.code = cloud_render.main(payload, live=self.live)
        except Exception:  # noqa: BLE001
            self.code = 1

    @classmethod
    def create(cls, *args, **kwargs):
        if cls.refuse:
            raise RuntimeError(cls.refuse)
        return cls(args, kwargs)

    @classmethod
    def from_id(cls, sandbox_id):
        return cls.boxes[sandbox_id]

    def poll(self):
        return None if self.thread.is_alive() else self.code


def _function(**_):
    return lambda function: function


_fake = types.SimpleNamespace(tmp=None)


def _fake_modal():
    module = types.ModuleType("modal")
    module.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    module.App = lambda name: types.SimpleNamespace(function=_function)
    module.Volume = types.SimpleNamespace(from_name=lambda name, **k: _Volume())
    module.Dict = types.SimpleNamespace(from_name=lambda name, **k: {})
    module.Secret = types.SimpleNamespace(from_name=lambda name: object())
    module.Sandbox = _Sandbox
    module.concurrent = lambda **k: (lambda f: f)
    module.asgi_app = lambda: (lambda f: f)
    return module


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    _fake.tmp = tmp_path
    _Sandbox.refuse = None
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    sys.modules.pop("modal_app", None)
    module = importlib.import_module("modal_app")
    monkeypatch.setattr(module, "_enter_repo", lambda: None)
    monkeypatch.setattr(module, "POLL_SECONDS", 0)
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


async def _until(client, render_id, done, seconds=20):
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
        gate.wait(20)
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
    assert done["best"] == "splat.splat" and "splat.splat" in done["outputs"]
    assert (await client.get("/api/v1/remote/renders")).json()["renders"][0]["status"] == "succeeded"


async def test_a_dead_render_sandbox_marks_the_render_failed(client, cloud, monkeypatch):
    def killed(job_id):  # the sandbox is gone before it could save anything
        raise RuntimeError("GPU container lost")

    monkeypatch.setattr(tasks, "run_task", killed)
    monkeypatch.setattr(cloud, "SAVE_GRACE_SECONDS", 0)
    render_id = await _send(client)
    box = _Sandbox.boxes[cloud.live[render_id]["sandbox_id"]]
    box.thread.join(10)
    assert box.kwargs["gpu"] == cloud.GPU and "/data" in box.kwargs["volumes"]
    failed = await _until(client, render_id, lambda s: s["status"] == "failed")
    assert "stopped" in failed["error"] and failed["action"] is None
    assert cloud.data.reloads >= 1  # looked for the saved result on the volume before giving up


async def test_no_card_on_modal_is_explained_on_the_phone_and_can_be_tried_again(client, cloud, monkeypatch):
    def fake_run_task(job_id):
        from app.config import get_settings
        from app.database import SessionLocal
        from app.models import Job

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            directory = tasks.job_dir(get_settings(), job.project_id, job_id)
            (directory / "outputs").mkdir(exist_ok=True)
            (directory / "outputs" / "splat.splat").write_bytes(b"\0" * 32)
            (directory / "outputs" / "result.json").write_text(json.dumps({"best": "splat.splat", "quality": "fast"}))
            job.status, job.stage, job.progress = "succeeded", "complete", 1.0
            db.commit()

    monkeypatch.setattr(tasks, "run_task", fake_run_task)
    _Sandbox.refuse = "Please add a payment method to use T4 GPU functions."
    render_id = await _send(client)
    failed = (await client.get(f"/api/v1/remote/renders/{render_id}")).json()
    assert failed["status"] == "failed" and failed["action"]["retry"]
    assert failed["action"]["url"].startswith("https://modal.com/") and "card" in failed["action"]["message"]
    history = (await client.get("/api/v1/remote/renders")).json()["renders"]
    assert history[0]["action"]["kind"] == "billing"
    _Sandbox.refuse = None  # the person added a card
    retried = (await client.post(f"/api/v1/remote/renders/{render_id}/retry")).json()
    assert retried["status"] in ("queued", "running") and retried["error"] is None
    done = await _until(client, render_id, lambda s: s["status"] == "succeeded")
    assert done["best"] == "splat.splat" and done["action"] is None
    again = await client.post(f"/api/v1/remote/renders/{render_id}/retry")
    assert again.status_code == 409
