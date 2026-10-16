from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading

from app.config import get_settings
from app.database import SessionLocal
from app.models import Job, JobStatus, Project, ProjectStatus
from app.services.colmap import run_colmap
from app.services.dataset import export_gaussian_dataset
from app.services.gaussian import run_gaussian_training
from app.services.geometry import run_dense_geometry
from app.services.monocular import run_monocular_geometry
from app.services.navigation import build_navigation_mesh
from app.services.quality import read_quality_report, score_reconstruction, write_quality_report
from app.services.storage import ensure_project_dirs
from app.services.video import extract_frames


settings = get_settings()
executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="namma-job")
_active_projects: set[str] = set()
_active_lock = threading.Lock()


def recover_interrupted_jobs() -> int:
    """Mark in-process jobs left active by a previous server exit as failed."""
    recovered = 0
    with SessionLocal() as db:
        jobs = db.query(Job).filter(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])).all()
        for job in jobs:
            job.status = JobStatus.FAILED
            job.stage = "interrupted"
            job.error_message = "Backend stopped before this job completed; retry the operation."
            job.finished_at = datetime.now(timezone.utc)
            recovered += 1
        db.commit()
    return recovered


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

        previous_project_status = project.status
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            job.progress = 0.05
            db.commit()

            paths = ensure_project_dirs(settings, project.id)
            project.error_message = None
            if job.kind in {"extract_frames", "reconstruct"}:
                project.status = ProjectStatus.PREPROCESSING
                job.stage = "extracting_frames"
                db.commit()
                frames = extract_frames(Path(project.source_path), paths["frames"], settings)
                project.frame_count = len(frames)
                job.progress = 0.35
                db.commit()

            if job.kind == "extract_frames":
                project.status = ProjectStatus.IMPORTED
            elif job.kind == "reconstruct":
                project.status = ProjectStatus.TRACKING
                job.stage = "colmap_reconstruction"
                job.log_path = str(paths["logs"] / "colmap.log")
                job.progress = 0.45
                db.commit()
                project.registered_image_count = run_colmap(
                    paths["frames"], paths["tracking"], Path(job.log_path), settings
                )

                project.status = ProjectStatus.QUALITY_CHECK
                job.stage = "quality_scoring"
                job.progress = 0.82
                db.commit()
                report = score_reconstruction(
                    paths["tracking"] / "colmap" / "sparse", project.frame_count, settings
                )
                write_quality_report(report, paths["artifacts"] / "quality-report.json")

                job.stage = "exporting_gaussian_dataset"
                job.progress = 0.9
                db.commit()
                export_gaussian_dataset(
                    paths["tracking"] / "colmap" / "sparse" / report["selected_model_id"],
                    paths["frames"],
                    paths["training"] / "dataset",
                )
                project.status = (
                    ProjectStatus.DATASET_READY if report["passed"] else ProjectStatus.QUALITY_FAILED
                )
            elif job.kind == "train_gaussian":
                project.status = ProjectStatus.TRAINING
                job.stage = "gaussian_training"
                job.log_path = str(paths["logs"] / "gaussian-training.log")
                job.progress = 0.15
                db.commit()
                run_gaussian_training(
                    paths["training"] / "dataset",
                    paths["artifacts"] / "gaussian",
                    Path(job.log_path),
                    settings,
                )
                project.status = ProjectStatus.READY
            elif job.kind == "build_collision":
                project.status = ProjectStatus.GEOMETRY
                job.stage = "dense_stereo"
                job.log_path = str(paths["logs"] / "dense-geometry.log")
                job.progress = 0.1
                db.commit()
                report = read_quality_report(paths["artifacts"] / "quality-report.json")
                geometry_runner = (
                    run_monocular_geometry
                    if settings.collision_geometry_backend == "monocular"
                    else run_dense_geometry
                )
                geometry_runner(
                    paths["frames"],
                    paths["tracking"] / "colmap" / "sparse" / report["selected_model_id"],
                    paths["artifacts"] / "geometry",
                    Path(job.log_path),
                    settings,
                )
                project.status = previous_project_status
            elif job.kind == "build_navmesh":
                project.status = ProjectStatus.GEOMETRY
                job.stage = "navigation_mesh_generation"
                job.log_path = str(paths["logs"] / "navigation-mesh.log")
                job.progress = 0.15
                db.commit()
                quality = read_quality_report(paths["artifacts"] / "quality-report.json")
                build_navigation_mesh(
                    paths["artifacts"] / "geometry",
                    paths["tracking"] / "colmap" / "sparse" / quality["selected_model_id"],
                    settings,
                )
                project.status = previous_project_status

            job.status = JobStatus.SUCCEEDED
            job.stage = "complete"
            job.progress = 1.0
        except Exception as exc:
            message = str(exc)[:8000]
            project.status = (
                previous_project_status if job.kind in {"build_collision", "build_navmesh"} else ProjectStatus.FAILED
            )
            project.error_message = message
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error_message = message
        finally:
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            with _active_lock:
                _active_projects.discard(project.id)
