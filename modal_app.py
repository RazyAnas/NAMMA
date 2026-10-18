"""The phone's render server on Modal: "Send for render" with no laptop and no Colab tab.

Two functions share one image:

* `web` (CPU, a single container that sleeps when nobody calls it): the same FastAPI app the laptop
  runs, so the phone page, pairing key, chunked upload and scan history all work unchanged. It keeps
  the history in sqlite on the `namma-db` volume and the uploads and results on `namma-data`.
* `render` (GPU, one container per render, billed only while it renders): runs the `phone.render`
  job on a private copy of the job's rows and publishes its live state (steps, log tail, time left)
  to the `namma-live` dict every few seconds; the results land on `namma-data`.

The web container never shares the sqlite file with a GPU container (sqlite on a shared volume is not
safe), and a render keeps going when the phone closes the page or the web container goes to sleep.

Deploy (no laptop): .github/workflows/modal-deploy.yml runs `modal deploy modal_app.py` on every push
with the MODAL_TOKEN_ID / MODAL_TOKEN_SECRET / NAMMA_REMOTE_KEY repository secrets. The phone link is
https://<workspace>--namma-web.modal.run/record/ ; the page pairs with the server it is served from.

    NAMMA_GPU=T4 modal deploy modal_app.py     # cheaper, slower GPU (default L4)
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import modal

GPU = os.environ.get("NAMMA_GPU", "L4")
ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/NAMMA"
DEPTH_MODEL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
# gsplat compiles its CUDA kernels on first use; the build step below does that once, on a GPU, for these archs.
CUDA_ARCHS = "7.5;8.0;8.6;8.9"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1", "build-essential")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "fastapi==0.115.14", "starlette==0.46.2", "uvicorn[standard]", "python-multipart", "pydantic-settings",
        "sqlalchemy", "numpy<2.3", "scipy", "opencv-python-headless", "imageio", "imageio-ffmpeg", "open3d",
        "trimesh", "plyfile<1.1.3", "pycolmap==4.2.1", "onnxruntime-gpu", "pyyaml", "rich", "httpx",
        "ninja", "pytorch-msssim", "jaxtyping", "gsplat==1.5.3",
    )
    .env({"TORCH_CUDA_ARCH_LIST": CUDA_ARCHS, "NAMMA_MODELS_DIR": "/models", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "python -c \"from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat kernels ready')\"",
        gpu="T4",
    )
    .run_commands(f"mkdir -p /models && curl -fsSL -o /models/depth_anything_v2_vits.onnx {DEPTH_MODEL}")
    .add_local_dir(ROOT, REMOTE_ROOT, ignore=[".git", "data", "logs", "tmp", "tools", "gpu-handout", "research",
                                              "**/__pycache__", "tests/.test-data", "node_modules"])
)

app = modal.App("namma")
data = modal.Volume.from_name("namma-data", create_if_missing=True)
database = modal.Volume.from_name("namma-db", create_if_missing=True)
live = modal.Dict.from_name("namma-live", create_if_missing=True)
secret = modal.Secret.from_name("namma")  # NAMMA_REMOTE_KEY, created by the deploy workflow

COMMON_ENV = {"DATA_DIR": "/data", "LOG_DIR": "/tmp/logs"}  # nothing held open on /data, so the web side can reload it
STALE_SECONDS = 90  # no heartbeat for this long: ask Modal whether the render container is still alive


def _enter_repo() -> None:
    import sys

    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)


# ---- GPU side -------------------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, volumes={"/data": data}, secrets=[secret], timeout=4 * 3600,
              env={**COMMON_ENV, "DATABASE_URL": "sqlite:////tmp/render.db"})
def render(job: dict) -> dict:
    """Runs one phone.render job (rows copied from the web container's history) and returns its end state."""
    _enter_repo()
    data.reload()
    from app.database import SessionLocal, init_db
    from app.models import Job, Project
    from app.pipeline.stages.phone import live_state
    from app.pipeline.tasks import job_dir, run_task
    from app.config import get_settings

    init_db()
    with SessionLocal() as db:
        db.merge(Project(**job["project"]))
        db.merge(Job(id=job["id"], project_id=job["project"]["id"], kind="phone.render", log_path=job.get("log_path")))
        db.commit()
    directory = job_dir(get_settings(), job["project"]["id"], job["id"])
    started = time.time()
    call_id = modal.current_function_call_id()
    done = threading.Event()

    def state() -> dict:
        with SessionLocal() as db:
            row = db.get(Job, job["id"])
            current = {"status": row.status, "stage": row.stage, "progress": row.progress, "error": row.error_message}
        return {**current, **live_state(directory, current["progress"] or 0.0, started), "started": started,
                "updated": time.time(), "call_id": call_id}

    def publish() -> None:
        while not done.wait(4):
            try:
                live[job["id"]] = state()
            except Exception as exc:  # a missed heartbeat only delays the phone's progress bar
                print(f"live update failed: {exc}")

    threading.Thread(target=publish, daemon=True).start()
    try:
        run_task(job["id"])
    finally:
        done.set()
        final = state()
        data.commit()  # results must be on the volume before the web container is told the job ended
        live[job["id"]] = final
    return final


# ---- Web side -------------------------------------------------------------------------------------

class CloudRenders:
    """phone.REMOTE on the web container: sends jobs to `render` and reads their live state back."""

    def hello(self) -> dict:
        return {"machine": f"the NAMMA cloud ({GPU} GPU, only while rendering)", "trainer": "gsplat",
                "photoreal": True, "dense_points": True, "cuda": True, "cloud": True}

    def submit(self, row) -> None:
        payload = {
            "id": row.id,
            "log_path": row.log_path,
            "project": {c: getattr(row.project, c) for c in ("id", "name", "status", "source_filename", "source_path",
                                                             "source_sha256", "source_size_bytes", "created_at")},
        }
        data.commit()  # the upload and params.json have to be visible to the GPU container
        call = render.spawn(payload)
        live[row.id] = {"status": "queued", "stage": "starting a GPU", "progress": 0.0, "call_id": call.object_id,
                        "updated": time.time(), "steps": [], "log_tail": [], "eta_seconds": None}

    def refresh(self, db, row) -> dict | None:
        from datetime import datetime, timezone

        state = live.get(row.id)
        if state is None:
            return None
        if state.get("status") not in ("succeeded", "failed") and time.time() - state.get("updated", 0) > STALE_SECONDS:
            state = self._check_container(row.id, state)
        row.stage = (state.get("stage") or row.stage)[:64]
        row.progress = state.get("progress") or row.progress
        if state.get("started") and row.started_at is None:
            row.started_at = datetime.fromtimestamp(state["started"], timezone.utc)
        if state.get("status") in ("running", "succeeded", "failed"):
            row.status = state["status"]
        if state.get("status") in ("succeeded", "failed"):
            row.error_message = state.get("error")
            row.finished_at = datetime.now(timezone.utc)
            try:
                data.reload()  # pick up the results the GPU container committed
            except Exception as exc:  # open files: the next poll tries again
                print(f"reload failed: {exc}")
                row.status = "running"
        db.commit()
        return state

    def _check_container(self, job_id: str, state: dict) -> dict:
        call_id = state.get("call_id")
        if not call_id:
            return state
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return state  # still running (or still waiting for a GPU)
        except Exception as exc:
            result = {**state, "status": "failed", "error": f"The render container stopped: {exc}"[:2000]}
        live[job_id] = result
        return result


@app.function(image=image, volumes={"/data": data, "/db": database}, secrets=[secret], max_containers=1,
              scaledown_window=300, timeout=3600,
              env={**COMMON_ENV, "DATABASE_URL": "sqlite:////db/namma.db", "JOBS_RUN_ELSEWHERE": "true"})
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    _enter_repo()
    from app.main import app as fastapi_app
    from app.pipeline import tasks
    from app.pipeline.stages import phone

    cloud = CloudRenders()
    phone.REMOTE = cloud
    tasks.REMOTE_RUNNER = cloud.submit
    return fastapi_app
