"""Stage 4, splats: Gaussian splat training (OpenSplat on Mac/CPU, splatfacto on CUDA), MODP on CUDA, and GPU results."""

from __future__ import annotations

from pathlib import Path
import json
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

TRAINER_REQUIREMENTS = {
    "opensplat": ("opensplat",),
    "gsplat": ("cuda", "gsplat"),
    "splatfacto": ("cuda",),
    "command": ("gaussian_trainer_command",),
}


class TrainRequest(BaseModel):
    trainer: str = Field("opensplat", pattern="^(opensplat|gsplat|splatfacto|command)$")
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
    elif trainer == "gsplat":
        command = (f'"{ctx.python()}" scripts/train_gsplat.py --dataset {{dataset}} --output {{output}} '
                   f'--steps {iterations} --downscale {int(p.get("downscale", 1.0))}')
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


class ModpRequest(BaseModel):
    poses: str = Field("sfm", pattern="^(sfm|ar)$", description="sfm = COLMAP model, ar = ARCore poses (metric)")
    points: str = Field("dense", pattern="^(sparse|dense)$", description="sparse = stock MODP input; dense = monocular points")
    post_steps: int | None = Field(None, ge=0, le=30000, description="MODP post-refinement steps; None = paper config (1000)")
    camera_opt: bool = Field(True, description="MODP's own pose refinement (paper config: on)")
    resize: float = Field(1.0, gt=0.1, le=1.0, description="image scale before MODP (0.5 for 1080p phone frames)")
    cam_height_m: float = Field(1.45, gt=0.3, le=3.0, description="sets metric scale for SfM poses (MODP assumes metres)")


@task("splats.modp", requires=("modp", "onnxruntime", "colmap"))
def run_modp(ctx: TaskContext) -> dict:
    """MODP (Meta, SIGGRAPH 2025) online Gaussian mapping, code unchanged, fed our poses and points; scored on held-out frames."""
    from app.pipeline.backends import depth_model_path, modp_paths
    from app.pipeline.stages.points import frames_for
    from app.pipeline.stages.tracking import model_dir

    p = ctx.params
    poses, points = p.get("poses", "sfm"), p.get("points", "dense")
    code, python = modp_paths()
    scripts = PROJECT_ROOT / "scripts" / "modp"
    root = ctx.paths["artifacts"] / "modp"
    scene, track = root / "inputs", f"{poses}_poses"
    run, result = root / "runs" / f"{poses}_{points}", root / "results" / f"{poses}_{points}"
    shutil.rmtree(result, ignore_errors=True)
    result.mkdir(parents=True)
    ctx.progress("modp_inputs", 0.05)
    prep = [ctx.python(), scripts / "prep_modp_inputs.py", "--frames", frames_for(ctx, poses), "--models", model_dir(ctx, poses),
            "--out", scene, "--track", track, "--resize", p.get("resize", 1.0), "--depth-model", depth_model_path("vits")]
    prep += ["--metric"] if poses == "ar" else ["--cam-height", p.get("cam_height_m", 1.45)]
    ctx.run(prep)
    make = [ctx.python(), scripts / "make_modp_dataset.py", scene / track, points, run]
    if p.get("post_steps") is not None:
        make += ["--post-steps", p["post_steps"]]
    if not p.get("camera_opt", True):
        make += ["--no-cam-opt"]
    ctx.run(make)
    (result / "run_meta.json").write_text(json.dumps({"scene": "project", "track": str(scene / track), "points": points,
                                                      "label": f"MODP, {poses} poses, {points} points"}))
    # MODP runs in its own CUDA 11.8 environment; modp_vars.sh (written by setup_modp.sh) holds its compiler paths
    env_vars = python.parent.parent / "modp_vars.sh"
    prefix = ["bash", "-c", f'. "{env_vars}" && exec "$@"', "_"] if env_vars.is_file() else []
    ctx.progress("modp_mapping", 0.2)
    ctx.run(prefix + [python, scripts / "run_modp.py", run, result], cwd=code)
    ctx.progress("modp_scoring", 0.95)
    ctx.run(prefix + [python, scripts / "score.py", result.parent])
    rows = {row["run"]: row for row in json.loads((result.parent / "summary.json").read_text())}
    score = rows.get(result.name, {})
    return {"poses": poses, "points": points, "model": str(result / "point_cloud.ply"), "renders": str(result / "renders"),
            "heldout": {k: score.get(k) for k in ("heldout_frames", "psnr", "ssim", "lpips")},
            "modp_native_eval": score.get("native"), "run_info": json.loads((result / "run_info.json").read_text())}


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


@router.post("/projects/{project_id}/splats/modp", response_model=JobRead, status_code=202)
async def start_modp(project_id: str, request: ModpRequest, db: Session = Depends(get_db)):
    """Meta's MODP on this project's poses and points (Linux + NVIDIA only; 503 on a Mac, where splats/train still works)."""
    return submit_task(db, project_id, "splats.modp", request.model_dump())


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
