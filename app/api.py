import json
from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Job, Project
from app.schemas import HealthRead, JobRead, ProjectRead
from app.services.jobs import submit_job
from app.services.gaussian import GaussianTrainingError, read_gaussian_manifest
from app.services.geometry import GeometryError, read_collision_manifest
from app.services.quality import read_quality_report
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


@router.get("/projects/{project_id}/quality")
async def get_reconstruction_quality(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> dict:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    try:
        return read_quality_report(
            ensure_project_dirs(settings, project_id)["artifacts"] / "quality-report.json"
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{project_id}/training-dataset")
async def get_training_dataset_manifest(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> dict:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    manifest = ensure_project_dirs(settings, project_id)["training"] / "dataset" / "manifest.json"
    if not manifest.is_file():
        raise HTTPException(status_code=404, detail="Gaussian training dataset is not available.")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _gaussian_artifact(project_id: str, db: Session, settings: Settings) -> tuple[dict, Path]:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    output_dir = ensure_project_dirs(settings, project_id)["artifacts"] / "gaussian"
    try:
        manifest = read_gaussian_manifest(output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GaussianTrainingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return manifest, output_dir


@router.get("/projects/{project_id}/gaussian-artifact")
async def get_gaussian_artifact(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> dict:
    manifest, _ = _gaussian_artifact(project_id, db, settings)
    return manifest


@router.get("/projects/{project_id}/gaussian-model", response_class=FileResponse)
async def download_gaussian_model(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> FileResponse:
    manifest, output_dir = _gaussian_artifact(project_id, db, settings)
    model = output_dir / manifest["model_path"]
    return FileResponse(model, media_type="application/octet-stream", filename=f"{project_id}.ply")


def _collision_artifact(project_id: str, db: Session, settings: Settings) -> tuple[dict, Path]:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    output_dir = ensure_project_dirs(settings, project_id)["artifacts"] / "geometry"
    try:
        manifest = read_collision_manifest(output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeometryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return manifest, output_dir


@router.get("/projects/{project_id}/collision-artifact")
async def get_collision_artifact(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> dict:
    manifest, _ = _collision_artifact(project_id, db, settings)
    return manifest


@router.get("/projects/{project_id}/collision-model", response_class=FileResponse)
async def download_collision_model(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> FileResponse:
    manifest, output_dir = _collision_artifact(project_id, db, settings)
    model = output_dir / "collision" / manifest["model_path"]
    return FileResponse(model, media_type="model/gltf-binary", filename=f"{project_id}-collision.glb")


@router.post("/projects/{project_id}/build-collision", response_model=JobRead, status_code=202)
async def start_collision_build(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> Job:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    try:
        report = read_quality_report(
            ensure_project_dirs(settings, project_id)["artifacts"] / "quality-report.json"
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="Run reconstruction before collision geometry.") from exc
    if not report["passed"]:
        raise HTTPException(status_code=409, detail="Reconstruction quality gate did not pass.")
    if settings.collision_geometry_backend == "monocular" and not settings.monocular_depth_command:
        raise HTTPException(status_code=503, detail="Monocular depth predictor is not configured.")
    if settings.collision_geometry_backend == "colmap" and not settings.resolve_colmap():
        raise HTTPException(status_code=503, detail="Native COLMAP is required for dense stereo.")
    return _start_job(project_id, "build_collision", db)


@router.post("/projects/{project_id}/train-gaussian", response_model=JobRead, status_code=202)
async def start_gaussian_training(
    project_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
) -> Job:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    try:
        report = read_quality_report(
            ensure_project_dirs(settings, project_id)["artifacts"] / "quality-report.json"
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="Run reconstruction before Gaussian training.") from exc
    if not report["passed"]:
        raise HTTPException(
            status_code=409,
            detail={"message": "Reconstruction did not pass the training quality gate.", "warnings": report["warnings"]},
        )
    return _start_job(project_id, "train_gaussian", db)


@router.get("/jobs/{job_id}", response_model=JobRead)
async def get_job(job_id: str, db: Session = Depends(get_db)) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
