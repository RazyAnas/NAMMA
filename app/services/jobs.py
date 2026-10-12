from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading

from app.config import get_settings
from app.database import SessionLocal
from app.models import Job, JobStatus, Project, ProjectStatus
from app.services.colmap import run_colmap
from app.services.storage import ensure_project_dirs
from app.services.video import extract_frames


settings = get_settings()
executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="namma-job")
_active_projects: set[str] = set()
_active_lock = threading.Lock()


def submit_job(job_id: str) -> None:
    executor.submit(_run_job, job_id)


def _run_job(job_id: str) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        project = db.get(Project, job.project_id)
        if project is None:
            return

        with _active_lock:
            if project.id in _active_projects:
                job.status = JobStatus.FAILED
                job.error_message = "Another job is already running for this project."
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
                return
            _active_projects.add(project.id)

        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            job.progress = 0.05
            db.commit()

            paths = ensure_project_dirs(settings, project.id)
            project.status = ProjectStatus.PREPROCESSING
            project.error_message = None
            job.stage = "extracting_frames"
            db.commit()

            frames = extract_frames(Path(project.source_path), paths["frames"], settings)
            project.frame_count = len(frames)
            job.progress = 0.35
            db.commit()

            if job.kind == "reconstruct":
                project.status = ProjectStatus.TRACKING
                job.stage = "colmap_reconstruction"
                job.log_path = str(paths["logs"] / "colmap.log")
                job.progress = 0.45
                db.commit()
                project.registered_image_count = run_colmap(
                    paths["frames"], paths["tracking"], Path(job.log_path), settings
                )

            project.status = ProjectStatus.READY
            job.status = JobStatus.SUCCEEDED
            job.stage = "complete"
            job.progress = 1.0
        except Exception as exc:
            message = str(exc)[:8000]
            project.status = ProjectStatus.FAILED
            project.error_message = message
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error_message = message
        finally:
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            with _active_lock:
                _active_projects.discard(project.id)
