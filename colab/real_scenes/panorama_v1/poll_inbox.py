"""Pick up browser-submitted panorama jobs from the private HF inbox.

GitHub scheduled workflows run from the default branch, so this worker becomes
active after the panorama PR is merged. It processes at most one pending job
per run, and the workflow concurrency group keeps Kaggle jobs serialized.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
INBOX = os.environ.get("PANORAMA_INBOX_DATASET", "anasrazy/namma-panorama-inbox")
OUTPUT = os.environ.get("PANORAMA_OUTPUT_DATASET", "anasrazy/namma-panorama-out")
ACTIVE = {"staging", "queued", "training", "publishing"}
STALE_AFTER_SECONDS = 8 * 60 * 60


def load_remote_json(api, repo_id: str, path: str, token: str):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        local = hf_hub_download(
            repo_id=repo_id, filename=path, repo_type="dataset",
            token=token, local_dir="panorama-inbox-cache"
        )
    except EntryNotFoundError:
        return None
    return json.loads(Path(local).read_text(encoding="utf-8"))


def is_stale(status: dict) -> bool:
    updated = str(status.get("updated_at_utc", ""))
    if not updated:
        return True
    try:
        from datetime import datetime, timezone
        stamp = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - stamp).total_seconds() > STALE_AFTER_SECONDS
    except Exception:
        return True


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN GitHub Actions secret is required.")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=INBOX, repo_type="dataset")
    candidates = sorted(
        path for path in files
        if path.endswith("/job.json") and len(path.split("/")) == 2
    )
    if not candidates:
        print("No browser-submitted panorama jobs are waiting in the inbox.")
        return 0

    for remote_job_path in candidates:
        raw = load_remote_json(api, INBOX, remote_job_path, token)
        if not isinstance(raw, dict):
            continue
        job_id = str(raw.get("job_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{2,50}", job_id):
            print(f"Skipping invalid job ID in {remote_job_path}.")
            continue
        expected_path = f"{job_id}/job.json"
        if remote_job_path != expected_path:
            print(f"Skipping mismatched job path {remote_job_path}.")
            continue

        status = load_remote_json(api, OUTPUT, f"{job_id}/status.json", token) or {}
        phase = str(status.get("status", "pending")).lower()
        if phase in {"complete", "failed"}:
            continue
        if phase in ACTIVE and not is_stale(status):
            print(f"Job {job_id} is already {phase}; leaving it alone.")
            continue

        local_job = HERE / "jobs" / f"{job_id}.json"
        local_job.parent.mkdir(parents=True, exist_ok=True)
        local_job.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        relative_job = local_job.relative_to(REPO_ROOT).as_posix()
        env = dict(os.environ)
        env["PANORAMA_JOB_FILE"] = relative_job
        print(f"Dispatching browser-submitted panorama job {job_id}.")
        completed = subprocess.run(
            [sys.executable, str(HERE / "gh_run.py")],
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        return int(completed.returncode)

    print("No new panorama jobs are waiting; all inbox jobs are complete or active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
