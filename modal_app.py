"""The phone's render server on Modal: "Send for render" with no laptop and no Colab tab.

One deployed function and one sandbox per render, sharing one image:

* `web` (CPU, a single container that sleeps when nobody calls it): the same FastAPI app the laptop
  runs, so the phone page, pairing key, chunked upload and scan history all work unchanged. It keeps
  the history in sqlite on the `namma-db` volume and the uploads and results on `namma-data`.
* each render is a GPU sandbox started by `web` (billed only while it renders) running
  scripts/cloud_render.py on a private copy of the job's rows; `web` reads its live state (steps, log
  tail, time left) from the sandbox while it runs and its end state from `namma-data` when it exits.

Nothing GPU is deployed, so the server deploys on any Modal account, card or not. A GPU is asked for only
when a render starts; if the account cannot have one yet (Modal wants a card on file before GPUs), the
phone says so with a link to Modal's billing page and a "Try again" button, and the scan stays saved.

The web container never shares the sqlite file with a GPU container (sqlite on a shared volume is not
safe), and a render keeps going when the phone closes the page or the web container goes to sleep.

Deploy (no laptop, no stored secrets): run .github/workflows/modal-deploy.yml from GitHub's Actions tab,
open the Modal sign-in link it shows, approve with whichever Modal account should pay, and it deploys
there (scripts/modal_login.py). The phone link is https://<workspace>--namma-web.modal.run/record/ ; the
first phone that opens it pairs with it by itself (a random key, kept on the `namma-db` volume).

    NAMMA_GPU=L4 modal deploy modal_app.py     # faster, dearer GPU (default T4)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

GPU = os.environ.get("NAMMA_GPU", "T4")
ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/NAMMA"
DEPTH_MODEL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
# gsplat compiles its CUDA kernels on first use; the build step below does that once, on a GPU, for these archs.
CUDA_ARCHS = "7.5;8.0;8.6;8.9"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    # open3d needs EGL, X11, libusb and libudev even headless (the first deploy crash-looped on libEGL.so.1).
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libegl1", "libx11-6", "libusb-1.0-0", "libudev1", "libglib2.0-0",
                 "libgomp1", "build-essential")
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
    # copy=True bakes the repo into the image: render sandboxes start from the image alone.
    .add_local_dir(ROOT, REMOTE_ROOT, copy=True, ignore=[".git", "data", "logs", "tmp", "tools", "gpu-handout",
                                                         "research", "**/__pycache__", "tests/.test-data", "node_modules"])
    # A missing system library fails the build here instead of crash-looping the deployed server.
    .run_commands(f"cd {REMOTE_ROOT} && python -c 'import app.main, scripts.cloud_render'")
)

app = modal.App("namma")
data = modal.Volume.from_name("namma-data", create_if_missing=True)
database = modal.Volume.from_name("namma-db", create_if_missing=True)
live = modal.Dict.from_name("namma-live", create_if_missing=True)

COMMON_ENV = {"DATA_DIR": "/data", "LOG_DIR": "/tmp/logs"}  # nothing held open on /data, so the web side can reload it
RENDER_ENV = {**COMMON_ENV, "DATABASE_URL": "sqlite:////tmp/render.db"}  # a private copy of the job's rows
RENDER_TIMEOUT = 4 * 3600
POLL_SECONDS = 3  # the phone and its history poll often; ask the sandbox at most this often per render
SAVE_GRACE_SECONDS = 300  # after the sandbox exits, how long its results may take to show up on the volume
BILLING_URL = "https://modal.com/settings"


def _enter_repo() -> None:
    import sys

    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_ROOT)



# ---- GPU side: one sandbox per render ------------------------------------------------------------

def needs_card(message: str) -> bool:
    """Modal's refusal for accounts with no payment method: "Please add a payment method to use T4 GPU functions"."""
    return "payment method" in message.lower()


def card_action() -> dict:
    return {
        "kind": "billing",
        "message": f"Your Modal account needs a card on file before it can use a {GPU} GPU (Modal's rule; your monthly "
                   "free credit still applies). Add one under Settings, Billing, then tap Try again. Your scan is saved.",
        "url": BILLING_URL,
        "label": "Add a card on Modal",
        "retry": True,
    }


def start_sandbox(payload: dict):
    return modal.Sandbox.create(
        "python", f"{REMOTE_ROOT}/scripts/cloud_render.py", json.dumps(payload),
        image=image, gpu=GPU, volumes={"/data": data}, timeout=RENDER_TIMEOUT, workdir=REMOTE_ROOT, env=RENDER_ENV,
    )


# ---- Web side -------------------------------------------------------------------------------------

class CloudRenders:
    """phone.REMOTE on the web container: starts a GPU sandbox per job and reads its state back."""

    def hello(self) -> dict:
        return {"machine": f"the NAMMA cloud ({GPU} GPU, only while rendering)", "trainer": "gsplat",
                "photoreal": True, "dense_points": True, "cuda": True, "cloud": True, "retry": True}

    def submit(self, row) -> None:
        payload = {
            "id": row.id,
            "log_path": row.log_path,
            "project": {c: getattr(row.project, c) for c in ("id", "name", "status", "source_filename", "source_path",
                                                             "source_sha256", "source_size_bytes")},
        }
        payload["project"]["created_at"] = row.project.created_at.isoformat() if row.project.created_at else None
        data.commit()  # the upload and params.json have to be visible to the GPU sandbox
        try:
            sandbox = start_sandbox(payload)
        except Exception as exc:  # most often: no card on the Modal account yet
            live[row.id] = self._failed(str(exc))
            return
        live[row.id] = {"status": "queued", "stage": "starting a GPU", "progress": 0.0, "sandbox_id": sandbox.object_id,
                        "updated": time.time(), "checked": 0.0, "steps": [], "log_tail": [], "eta_seconds": None}

    def action(self, row) -> dict | None:
        state = live.get(row.id)
        return state.get("action") if state else None

    def refresh(self, db, row) -> dict | None:
        from datetime import datetime, timezone

        state = live.get(row.id)
        if state is None:
            return None
        if state.get("status") not in ("succeeded", "failed") and time.time() - state.get("checked", 0) > POLL_SECONDS:
            state = self._check_sandbox(row, state)
            live[row.id] = state
        row.stage = (state.get("stage") or row.stage)[:64]
        row.progress = state.get("progress") or row.progress
        if state.get("started") and row.started_at is None:
            row.started_at = datetime.fromtimestamp(state["started"], timezone.utc)
        if state.get("status") in ("running", "succeeded", "failed"):
            row.status = state["status"]
        if state.get("status") in ("succeeded", "failed"):
            row.error_message = state.get("error")
            row.finished_at = datetime.now(timezone.utc)
        db.commit()
        return state

    @staticmethod
    def _failed(message: str) -> dict:
        state = {"status": "failed", "stage": "failed", "progress": 0.0, "error": message[:2000], "updated": time.time(),
                 "steps": [], "log_tail": [], "eta_seconds": None}
        if needs_card(message):
            state.update(error="Needs a card on your Modal account before it can use a GPU.", action=card_action())
        return state

    def _check_sandbox(self, row, state: dict) -> dict:
        from app.config import get_settings
        from app.pipeline.tasks import job_dir

        state = {**state, "checked": time.time()}
        sandbox = modal.Sandbox.from_id(state["sandbox_id"])
        code = sandbox.poll()
        if code is None:
            try:
                return {**state, **json.loads(sandbox.filesystem.read_text("/tmp/live.json")), "checked": time.time()}
            except Exception:  # still starting, or between two writes
                return state
        final_path = job_dir(get_settings(), row.project_id, row.id) / "cloud_state.json"
        try:
            data.reload()  # pick up the results the sandbox saved
        except Exception as exc:  # open files: the next poll tries again
            print(f"reload failed: {exc}")
        if final_path.is_file():
            final = json.loads(final_path.read_text())
            if final.get("status") in ("succeeded", "failed"):
                return {**final, "checked": time.time()}
            return self._failed(f"The GPU sandbox stopped (exit code {code}) before the render finished.")
        exited = state.get("exited") or time.time()
        if time.time() - exited < SAVE_GRACE_SECONDS:
            return {**state, "exited": exited, "stage": state.get("stage") or "saving the result"}
        why = ""
        try:
            why = sandbox.stderr.read()[-1500:]
        except Exception:
            pass
        return self._failed(why if needs_card(why) else f"The GPU sandbox stopped (exit code {code}) without saving "
                                                         f"a result. {why}".strip())


@app.function(image=image, volumes={"/data": data, "/db": database}, max_containers=1, scaledown_window=300, timeout=3600,
              env={**COMMON_ENV, "DATABASE_URL": "sqlite:////db/namma.db", "JOBS_RUN_ELSEWHERE": "true",
                   "NAMMA_REMOTE_KEY_FILE": "/db/pairing_key", "NAMMA_GPU": GPU})
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
