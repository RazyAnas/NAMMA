"""Stage 4, splats: Gaussian splat training (OpenSplat on Mac/CPU, splatfacto on CUDA) and GPU results."""

from __future__ import annotations

from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, get_settings
from app.database import get_db
from app.models import Project, ProjectStatus
from app.pipeline.tasks import TaskContext, TaskError, submit_task, task
from app.schemas import JobRead
from app.services.gaussian import run_gaussian_training
from app.services.storage import ensure_project_dirs, safe_filename, save_upload


router = APIRouter(prefix="/api/v1/pipeline", tags=["4 splats"])

TRAINER_REQUIREMENTS = {"opensplat": ("opensplat",), "splatfacto": ("cuda",), "command": ("gaussian_trainer_command",)}


class TrainRequest(BaseModel):
    trainer: str = Field("opensplat", pattern="^(opensplat|splatfacto|command)$")
    iterations: int = Field(7000, ge=100, le=100000)
    downscale: float = Field(1.0, ge=1.0, le=8.0)
    init: str = Field("sparse", pattern="^(sparse|monocular)$", description="monocular = points from points/monocular")
    max_init_points: int = Field(200000, ge=1000, le=5000000)


def monocular_dataset(ctx: TaskContext, max_points: int) -> Path:
    """Copy of the splat dataset whose initial points are the fused monocular cloud."""
    import open3d as o3d
    import pycolmap

    source = ctx.paths["training"] / "dataset"
    cloud_path = ctx.paths["artifacts"] / "geometry" / "dense" / "fused-monocular.ply"
    if not cloud_path.is_file():
        raise TaskError("init=monocular needs points/monocular first.")
    target = ctx.paths["training"] / "dataset-monocular"
    if target.exists():
        shutil.rmtree(target)
    (target / "sparse").mkdir(parents=True)
    (target / "images").symlink_to((source / "images").resolve(), target_is_directory=True)
    reconstruction = pycolmap.Reconstruction(str(source / "sparse" / "0"))
    for point_id in list(reconstruction.points3D):
        reconstruction.delete_point3D(point_id)
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    xyz, rgb = np.asarray(cloud.points), np.asarray(cloud.colors)
    if len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    for point, colour in zip(xyz, (rgb * 255).round().astype(np.uint8)):
        reconstruction.add_point3D(point, pycolmap.Track(), colour)
    (target / "sparse" / "0").mkdir()
    reconstruction.write(str(target / "sparse" / "0"))
    ctx.log(f"monocular init: {len(xyz)} points")
    return target


@task("splats.train")
def train_splats(ctx: TaskContext) -> dict:
    """Train a Gaussian splat on the exported dataset with the chosen trainer."""
    p = ctx.params
    dataset = ctx.paths["training"] / "dataset"
    if not (dataset / "sparse" / "0").is_dir():
        raise TaskError("Run tracking/sfm first; it exports the splat dataset.")
    if p.get("init") == "monocular":
        ctx.progress("monocular_init", 0.05)
        dataset = monocular_dataset(ctx, int(p.get("max_init_points", 200000)))
    trainer = p.get("trainer", "opensplat")
    iterations = int(p.get("iterations", 7000))
    if trainer == "opensplat":
        command = (f'"{ctx.python()}" scripts/train_opensplat.py --dataset {{dataset}} --output {{output}} '
                   f'--iterations {iterations} --downscale {p.get("downscale", 1.0)}')
    elif trainer == "splatfacto":
        command = f'"{ctx.python()}" scripts/train_splatfacto.py --dataset {{dataset}} --output {{output}} --iterations {iterations}'
    else:
        command = ctx.settings.gaussian_trainer_command
    ctx.project.status = ProjectStatus.TRAINING
    ctx.progress("gaussian_training", 0.1)
    manifest = run_gaussian_training(dataset, ctx.paths["artifacts"] / "gaussian", ctx.log_path, ctx.override(gaussian_trainer_command=command))
    ctx.project.status = ProjectStatus.READY
    ctx.db.commit()
    return {"trainer": trainer, "iterations": iterations, "init": p.get("init", "sparse"), "manifest": manifest}


@task("splats.gpu_results")
def ingest_gpu_results(ctx: TaskContext) -> dict:
    """Score the friend's GPU run (gsplat 30k steps, MODP) into final_comparison.md/.png."""
    archive = Path(ctx.params["archive"])
    out = ctx.paths["artifacts"] / "gpu-results" / archive.stem
    ctx.run([ctx.python(), PROJECT_ROOT / "gpu-handout" / "ingest_results.py", archive, out])
    return {"output": str(out), "table": str(out / "final_comparison.md"), "image": str(out / "final_comparison.png")}


@router.post("/projects/{project_id}/splats/train", response_model=JobRead, status_code=202)
async def start_training(project_id: str, request: TrainRequest, db: Session = Depends(get_db)):
    """OpenSplat runs on Metal or CPU (set OPENSPLAT_BIN); splatfacto needs CUDA and answers 503 without it."""
    from app.pipeline.backends import missing_requirements

    missing = missing_requirements(TRAINER_REQUIREMENTS[request.trainer])
    if missing:
        raise HTTPException(status_code=503, detail={"message": f"{request.trainer} cannot run here.", "missing": missing})
    return submit_task(db, project_id, "splats.train", request.model_dump())


@router.post("/projects/{project_id}/splats/gpu-results", response_model=JobRead, status_code=202)
async def upload_gpu_results(project_id: str, archive: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload namma_results_*.zip from the GPU handout and build the comparison table."""
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    settings = get_settings()
    destination = ensure_project_dirs(settings, project_id)["root"] / "uploads" / safe_filename(archive.filename or "results.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    await save_upload(archive, destination, settings.max_upload_mb * 1024 * 1024, {".zip"})
    return submit_task(db, project_id, "splats.gpu_results", {"archive": str(destination)})
