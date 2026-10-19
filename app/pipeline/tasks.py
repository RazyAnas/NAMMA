"""Background task runner shared by every pipeline stage router.

Each stage registers plain functions with `@task(kind, requires=...)`. A task receives a
`TaskContext` with the project's folders, the request parameters and a log, and returns a JSON
result. Jobs reuse the existing `jobs` table, so `/api/v1/jobs/{id}` keeps working; parameters,
results and logs live next to the project under `jobs/<job id>/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Callable

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, Settings, get_settings
from app.database import SessionLocal
from app.models import Job, JobStatus, Project
from app.pipeline.backends import missing_requirements
from app.services import jobs as legacy_jobs
from app.services.storage import ensure_project_dirs


class TaskError(RuntimeError):
    pass


@dataclass
class TaskSpec:
    kind: str
    function: Callable[["TaskContext"], dict]
    requires: tuple[str, ...] = ()
    description: str = ""


TASKS: dict[str, TaskSpec] = {}


def task(kind: str, requires: tuple[str, ...] = (), description: str = ""):
    def register(function: Callable[["TaskContext"], dict]):
        TASKS[kind] = TaskSpec(kind, function, requires, description or (function.__doc__ or "").strip())
        return function

    return register


@dataclass
class TaskContext:
    job_id: str
    project: Project
    db: Session
    paths: dict[str, Path]
    params: dict[str, Any]
    settings: Settings
    job_dir: Path
    notes: list[str] = field(default_factory=list)

    @property
    def log_path(self) -> Path:
        return self.job_dir / "log.txt"

    def log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}\n")

    def progress(self, stage: str, value: float) -> None:
        job = self.db.get(Job, self.job_id)
        job.stage = stage[:64]
        job.progress = max(0.0, min(1.0, value))
        self.db.commit()
        self.log(f"stage={stage} progress={value:.2f}")

    def override(self, **values: Any) -> Settings:
        """Settings with request overrides, so each job can try a variant without editing .env."""
        clean = {key: value for key, value in values.items() if value is not None}
        return self.settings.model_copy(update=clean) if clean else self.settings

    def run(self, command: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
        self.log("$ " + " ".join(str(part) for part in command))
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment.update(env or {})
        with self.log_path.open("a", encoding="utf-8", errors="replace") as handle:
            result = subprocess.run(
                [str(part) for part in command],
                cwd=cwd or PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if result.returncode != 0:
            raise TaskError(f"{Path(str(command[0])).name} exited with {result.returncode}; see the job log.")

    @staticmethod
    def python() -> str:
        return sys.executable


def job_dir(settings: Settings, project_id: str, job_id: str) -> Path:
    directory = ensure_project_dirs(settings, project_id)["root"] / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Runs a submitted job somewhere else instead of this process's worker thread. modal_app.py sets it so
# phone renders go to a GPU container while the web container only answers the phone.
REMOTE_RUNNER: Callable[[Job], None] | None = None


def submit_task(db: Session, project_id: str, kind: str, params: dict[str, Any] | None = None) -> Job:
    spec = TASKS.get(kind)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline task {kind}.")
    params = params or {}
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    missing = missing_requirements(spec.requires, params)
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"message": f"{kind} cannot run on this machine.", "missing": missing},
        )
    active = db.scalar(select(Job).where(Job.project_id == project_id, Job.status.in_(["queued", "running"])))
    if active:
        raise HTTPException(status_code=409, detail=f"Job {active.id} is already active for this project.")
    job = Job(project_id=project_id, kind=kind)
    db.add(job)
    db.commit()
    db.refresh(job)
    directory = job_dir(get_settings(), project_id, job.id)
    (directory / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    job.log_path = str(directory / "log.txt")
    db.commit()
    dispatch(job)
    return job


def dispatch(job: Job) -> None:
    """Hands a queued job to whatever runs jobs here: another machine (REMOTE_RUNNER) or this one's worker pool."""
    if REMOTE_RUNNER is not None:
        REMOTE_RUNNER(job)
    else:
        legacy_jobs.executor.submit(run_task, job.id)


def run_task(job_id: str) -> None:
    settings = get_settings()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        project = db.get(Project, job.project_id) if job else None
        if job is None or project is None:
            return
        with legacy_jobs._active_lock:
            if project.id in legacy_jobs._active_projects:
                job.status = JobStatus.FAILED
                job.error_message = "Another job is already running for this project."
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
                return
            legacy_jobs._active_projects.add(project.id)
        directory = job_dir(settings, project.id, job.id)
        params = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        context = TaskContext(job.id, project, db, ensure_project_dirs(settings, project.id), params, settings, directory)
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            job.progress = 0.02
            db.commit()
            result = TASKS[job.kind].function(context) or {}
            if context.notes:
                result.setdefault("notes", context.notes)
            (directory / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            job.status = JobStatus.SUCCEEDED
            job.stage = "complete"
            job.progress = 1.0
        except Exception as exc:
            context.log(traceback.format_exc())
            message = str(exc)[:8000]
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error_message = message
            (directory / "result.json").write_text(json.dumps({"error": message}, indent=2), encoding="utf-8")
        finally:
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            with legacy_jobs._active_lock:
                legacy_jobs._active_projects.discard(project.id)


def read_job_files(settings: Settings, job: Job, log_lines: int = 40) -> dict:
    directory = ensure_project_dirs(settings, job.project_id)["root"] / "jobs" / job.id
    payload: dict[str, Any] = {"params": None, "result": None, "log_tail": []}
    for name in ("params", "result"):
        path = directory / f"{name}.json"
        if path.is_file():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    log = directory / "log.txt"
    if log.is_file():
        payload["log_tail"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-log_lines:]
    return payload
