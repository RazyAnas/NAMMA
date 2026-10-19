"""Runs one phone render inside the Modal GPU sandbox that modal_app.py starts (no Modal calls in here).

The web container passes the job's rows as JSON. This copies them into a private sqlite file, runs the
`phone.render` task, keeps its live state (steps, log tail, time left) in /tmp/live.json for the web
container to read while it runs, and leaves the end state in the job directory on the /data volume.

    python scripts/cloud_render.py '{"id": ..., "log_path": ..., "project": {...}}'
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
LIVE = Path(os.environ.get("NAMMA_LIVE_FILE", "/tmp/live.json"))
FINAL_NAME = "cloud_state.json"  # in the job directory, read by modal_app.CloudRenders once the sandbox has exited


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def main(job: dict, live: Path = LIVE) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from datetime import datetime

    from app.config import get_settings
    from app.database import SessionLocal, init_db
    from app.models import Job, Project
    from app.pipeline.stages.phone import live_state
    from app.pipeline.tasks import job_dir, run_task

    init_db()
    project = dict(job["project"])
    if isinstance(project.get("created_at"), str):
        project["created_at"] = datetime.fromisoformat(project["created_at"])
    with SessionLocal() as db:
        db.merge(Project(**project))
        db.merge(Job(id=job["id"], project_id=project["id"], kind="phone.render", log_path=job.get("log_path")))
        db.commit()
    directory = job_dir(get_settings(), project["id"], job["id"])
    started = time.time()
    done = threading.Event()

    def state() -> dict:
        with SessionLocal() as db:
            row = db.get(Job, job["id"])
            current = {"status": row.status, "stage": row.stage, "progress": row.progress, "error": row.error_message}
        return {**current, **live_state(directory, current["progress"] or 0.0, started), "started": started,
                "updated": time.time()}

    def publish() -> None:
        while not done.wait(4):
            try:
                write_atomic(live, state())
            except Exception as exc:  # a missed update only delays the phone's progress bar
                print(f"live update failed: {exc}", flush=True)

    threading.Thread(target=publish, daemon=True).start()
    try:
        run_task(job["id"])
    finally:
        done.set()
        final = state()
        write_atomic(directory / FINAL_NAME, final)
        write_atomic(live, final)
        subprocess.run(["sync", str(get_settings().absolute_data_dir())], check=False)  # flush the results to the volume
    return 0 if final["status"] == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main(json.loads(sys.argv[1])))
