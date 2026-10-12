import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Job, Project
from app.schemas import HealthRead, JobRead, ProjectRead
from app.services.jobs import submit_job
from app.services.storage import ensure_project_dirs, safe_filename, save_upload


router = APIRouter(prefix="/api/v1")


async def settings_dependency() -> Settings:
    return get_settings()


@router.get("/health", response_model=HealthRead)
async def health(settings: Settings = Depends(settings_dependency)) -> HealthRead:
    return HealthRead(status="ok", ffmpeg=settings.resolve_ffmpeg(), colmap=settings.colmap_status())


@router.post("/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
async def create_project(
    name: str = Form(min_length=1, max_length=200),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> Project:
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Project name cannot be blank.")
    project_id = str(uuid.uuid4())
    filename = safe_filename(video.filename)
    paths = ensure_project_dirs(settings, project_id)
    destination = paths["source"] / filename
    try:
        size, digest = await save_upload(video, destination, settings.max_upload_mb * 1024 * 1024)
    except ValueError as exc:
        shutil.rmtree(paths["root"], ignore_errors=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    project = Project(
        id=project_id,
        name=clean_name,
        source_filename=filename,
        source_path=str(destination),
        source_size_bytes=size,
        source_sha256=digest,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[ProjectRead])
async def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return list(db.scalars(select(Project).order_by(Project.created_at.desc())))


@router.get("/projects/{project_id}", response_model=ProjectRead)
async def get_project(project_id: str, db: Session = Depends(get_db)) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _start_job(project_id: str, kind: str, db: Session) -> Job:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    active = db.scalar(
        select(Job).where(Job.project_id == project_id, Job.status.in_(["queued", "running"]))
    )
    if active:
        raise HTTPException(status_code=409, detail=f"Job {active.id} is already active for this project.")
    job = Job(project_id=project_id, kind=kind)
    db.add(job)
    db.commit()
    db.refresh(job)
    submit_job(job.id)
    return job


@router.post("/projects/{project_id}/extract-frames", response_model=JobRead, status_code=202)
async def start_frame_extraction(project_id: str, db: Session = Depends(get_db)) -> Job:
    return _start_job(project_id, "extract_frames", db)


@router.post("/projects/{project_id}/reconstruct", response_model=JobRead, status_code=202)
async def start_reconstruction(project_id: str, db: Session = Depends(get_db)) -> Job:
    return _start_job(project_id, "reconstruct", db)


@router.get("/jobs/{job_id}", response_model=JobRead)
async def get_job(job_id: str, db: Session = Depends(get_db)) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
