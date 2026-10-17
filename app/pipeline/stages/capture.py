"""Stage 1, capture: phone sensor bundles (/record/), recorded video and live streaming (/capture/)."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Project
from app.pipeline.tasks import TaskContext, submit_task, task
from app.schemas import JobRead
from app.services.sensor_bundle import ingest
from app.services.storage import ensure_project_dirs, safe_filename, save_upload
from app.services.video import extract_frames


router = APIRouter(prefix="/api/v1/pipeline", tags=["1 capture"])


class FrameSelectionRequest(BaseModel):
    selection: str = Field("sharpest", pattern="^(sharpest|uniform)$")
    fps: float | None = Field(None, gt=0, le=30)
    candidates_per_keyframe: int | None = Field(None, ge=1, le=30)
    max_width: int | None = Field(None, ge=320, le=4096)


@task("capture.bundle", requires=("ffmpeg",))
def ingest_bundle(ctx: TaskContext) -> dict:
    """Unpack a /record/ bundle: frames, ARCore poses as a COLMAP model, pose priors, IMU report."""
    capture_dir = ctx.paths["root"] / "capture"
    ctx.progress("unpacking_bundle", 0.1)
    report = ingest(Path(ctx.project.source_path), capture_dir, int(ctx.params.get("stride", 10)))
    ctx.progress("copying_frames", 0.8)
    for stale in ctx.paths["frames"].glob("frame_*.jpg"):
        stale.unlink()
    frames = sorted((capture_dir / "frames").glob("frame_*.jpg"))
    for frame in frames:
        shutil.copy2(frame, ctx.paths["frames"] / frame.name)
    video = capture_dir / str(report.get("remuxed_video") or "video.mp4")
    if video.is_file():
        ctx.project.source_path = str(video)
    ctx.project.frame_count = len(frames)
    ctx.db.commit()
    return {
        "frames": len(frames),
        "frames_with_pose": report.get("frames_with_pose", 0),
        "known_poses_model": str(capture_dir / "colmap_known_poses"),
        "pose_priors": str(capture_dir / "pose_priors.jsonl"),
        "report": report,
    }


@task("capture.frames", requires=("ffmpeg",))
def select_frames(ctx: TaskContext) -> dict:
    """Decode the video and keep the sharpest frame of every window (lucky imaging)."""
    settings = ctx.override(
        frame_selection=ctx.params.get("selection"),
        frame_sample_fps=ctx.params.get("fps"),
        frame_candidates_per_keyframe=ctx.params.get("candidates_per_keyframe"),
        frame_max_width=ctx.params.get("max_width"),
    )
    ctx.progress("extracting_frames", 0.1)
    frames = extract_frames(Path(ctx.project.source_path), ctx.paths["frames"], settings)
    ctx.project.frame_count = len(frames)
    ctx.db.commit()
    report = ctx.paths["frames"].parent / "frame-selection.json"
    return {"frames": len(frames), "selection": settings.frame_selection, "report": str(report) if report.is_file() else None}


@router.post("/capture/sensor-bundle", response_model=JobRead, status_code=202)
async def upload_sensor_bundle(
    name: str = Form(min_length=1, max_length=200),
    bundle: UploadFile = File(..., description="namma-*.zip saved by the /record/ page"),
    stride: int = Form(10, ge=1, le=120),
    db: Session = Depends(get_db),
):
    """Create a project from a phone sensor bundle and unpack it in the background."""
    settings = get_settings()
    project_id = str(uuid.uuid4())
    paths = ensure_project_dirs(settings, project_id)
    destination = paths["source"] / safe_filename(bundle.filename or "bundle.zip")
    try:
        size, digest = await save_upload(bundle, destination, settings.max_upload_mb * 1024 * 1024, {".zip"})
    except ValueError as exc:
        shutil.rmtree(paths["root"], ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = Project(
        id=project_id,
        name=name.strip(),
        source_filename=destination.name,
        source_path=str(destination),
        source_size_bytes=size,
        source_sha256=digest,
    )
    db.add(project)
    db.commit()
    return submit_task(db, project_id, "capture.bundle", {"stride": stride})


@router.post("/projects/{project_id}/capture/frames", response_model=JobRead, status_code=202)
async def start_frame_selection(project_id: str, request: FrameSelectionRequest, db: Session = Depends(get_db)):
    """Pick keyframes from the uploaded video (default: sharpest of every 4 decoded frames)."""
    return submit_task(db, project_id, "capture.frames", request.model_dump())
