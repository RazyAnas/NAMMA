"""Stage 2, tracking: camera poses from SfM, from the phone's ARCore poses, deblurring and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT
from app.database import get_db
from app.models import ProjectStatus
from app.pipeline.backends import capabilities, restormer_paths
from app.pipeline.tasks import TaskContext, TaskError, submit_task, task
from app.schemas import JobRead
from app.services.colmap import run_colmap
from app.services.dataset import export_gaussian_dataset
from app.services.photometric import compensate_dataset
from app.services.quality import read_quality_report, score_reconstruction, write_quality_report


router = APIRouter(prefix="/api/v1/pipeline", tags=["2 tracking"])


class SfmRequest(BaseModel):
    mapper: str = Field("global", pattern="^(global|incremental)$")
    contrast_normalize: bool = True
    blur_robust_features: bool = True
    exhaustive_loop_closure: bool = True
    photometric_compensation: bool = True
    use_deblurred_frames: bool = False


class ArPosesRequest(BaseModel):
    refine_with_bundle_adjustment: bool = False
    sequential_overlap: int = Field(15, ge=2, le=100)
    max_features: int = Field(8192, ge=512, le=32768)


class DeblurRequest(BaseModel):
    device: str = Field("cpu", pattern="^(cpu|mps|cuda)$")
    threads: int = Field(4, ge=1, le=64)


class EvaluateRequest(BaseModel):
    reference: str = Field("ar", pattern="^(ar|file)$", description="ar = the phone's ARCore poses; file = ground truth")
    ground_truth_path: str | None = Field(None, description="JSON {image name: [x, y, z]} camera centres in metres")
    model: str = Field("sfm", pattern="^(sfm|ar)$")


class LiveTrackerRequest(BaseModel):
    dataset_dir: str = Field(description="ICL-NUIM style folder (data/color_raw, models/*.onnx, work/)")
    grid: bool = True
    first: int = 1200
    last: int = 2869
    step: int = 8
    window: int = 8


def model_dir(ctx: TaskContext, source: str = "sfm") -> Path:
    """The sparse model a later stage should use: `sfm` (the quality-gated COLMAP model) or `ar`."""
    if source == "ar":
        path = ctx.paths["tracking"] / "ar" / "sparse"
        if not (path / "images.bin").is_file():
            raise TaskError("Run tracking/ar-poses first.")
        return path
    report = read_quality_report(ctx.paths["artifacts"] / "quality-report.json")
    return ctx.paths["tracking"] / "colmap" / "sparse" / report["selected_model_id"]


@task("tracking.sfm", requires=("colmap",))
def structure_from_motion(ctx: TaskContext) -> dict:
    """COLMAP/GLOMAP SfM with the robustness fixes, quality gate, splat dataset and exposure solve."""
    settings = ctx.override(
        colmap_mapper=ctx.params.get("mapper"),
        colmap_contrast_normalize=ctx.params.get("contrast_normalize"),
        colmap_blur_robust_features=ctx.params.get("blur_robust_features"),
        colmap_exhaustive_loop_closure=ctx.params.get("exhaustive_loop_closure"),
        colmap_use_gpu=capabilities()["cuda"],
    )
    frames = ctx.paths["frames"]
    if ctx.params.get("use_deblurred_frames"):
        frames = ctx.paths["root"] / "frames-deblurred"
        if not any(frames.glob("*.jpg")):
            raise TaskError("No deblurred frames; run tracking/deblur first.")
    frame_count = sum(1 for _ in frames.glob("*.jpg"))
    if frame_count == 0:
        raise TaskError("No frames; run capture/frames or upload a sensor bundle first.")
    ctx.project.status = ProjectStatus.TRACKING
    ctx.progress("colmap_reconstruction", 0.1)
    registered = run_colmap(frames, ctx.paths["tracking"], ctx.log_path, settings)
    ctx.project.registered_image_count = registered
    ctx.progress("quality_scoring", 0.8)
    report = score_reconstruction(ctx.paths["tracking"] / "colmap" / "sparse", frame_count, settings)
    write_quality_report(report, ctx.paths["artifacts"] / "quality-report.json")
    ctx.progress("exporting_gaussian_dataset", 0.88)
    sparse = ctx.paths["tracking"] / "colmap" / "sparse" / report["selected_model_id"]
    export_gaussian_dataset(sparse, frames, ctx.paths["training"] / "dataset")
    photometric = None
    if ctx.params.get("photometric_compensation", True):
        ctx.progress("exposure_compensation", 0.94)
        photometric = compensate_dataset(
            ctx.paths["training"] / "dataset" / "sparse" / "0",
            ctx.paths["training"] / "dataset" / "images",
            ctx.paths["artifacts"] / "photometric-report.json",
        )
    ctx.project.status = ProjectStatus.DATASET_READY if report["passed"] else ProjectStatus.QUALITY_FAILED
    ctx.db.commit()
    return {
        "registered_images": registered,
        "frames": frame_count,
        "mapper": settings.colmap_mapper,
        "quality": report,
        "photometric": {k: photometric[k] for k in photometric if k not in {"frames", "gains_rgb"}} if isinstance(photometric, dict) else None,
    }


def _remap_known_model(source: Path, destination: Path, database: Path) -> None:
    """Rewrite the ARCore model's image ids to the database's, which point_triangulator requires."""
    import pycolmap

    db = pycolmap.Database.open(str(database))
    ids = {image.name: image.image_id for image in db.read_all_images()}
    db.close()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy(source / "cameras.txt", destination)
    shutil.copy(source / "points3D.txt", destination)
    lines = []
    for line in (source / "images.txt").read_text().splitlines():
        parts = line.split()
        if not line.startswith("#") and len(parts) >= 10:
            parts[0] = str(ids[parts[9]])
            line = " ".join(parts)
        lines.append(line)
    (destination / "images.txt").write_text("\n".join(lines) + "\n")


@task("tracking.ar_poses", requires=("colmap",))
def triangulate_ar_poses(ctx: TaskContext) -> dict:
    """Skip SfM: triangulate points from the phone's metric ARCore poses (optionally refine with BA)."""
    import pycolmap

    known = ctx.paths["root"] / "capture" / "colmap_known_poses"
    frames = ctx.paths["root"] / "capture" / "frames"
    if not (known / "images.txt").is_file():
        raise TaskError("This project has no ARCore poses; upload a /record/ bundle recorded in AR mode.")
    work = ctx.paths["tracking"] / "ar"
    work.mkdir(parents=True, exist_ok=True)
    database = work / "database.db"
    database.unlink(missing_ok=True)
    model = pycolmap.Reconstruction(str(known))
    camera = next(iter(model.cameras.values()))
    names = sorted(image.name for image in model.images.values())
    ctx.progress("features", 0.1)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = int(ctx.params.get("max_features", 8192))
    extraction.sift.domain_size_pooling = True
    extraction.sift.peak_threshold = 0.004
    pycolmap.extract_features(
        database,
        frames,
        image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(
            camera_model="PINHOLE", camera_params=",".join(str(v) for v in camera.params)
        ),
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )
    ctx.progress("matching", 0.5)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = int(ctx.params.get("sequential_overlap", 15))
    pycolmap.match_sequential(database, pairing_options=pairing, device=pycolmap.Device.cpu)
    ctx.progress("triangulating", 0.75)
    _remap_known_model(known, work / "known", database)
    sparse = work / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    reconstruction = pycolmap.triangulate_points(pycolmap.Reconstruction(str(work / "known")), database, frames, sparse)
    result = {
        "images": reconstruction.num_reg_images(),
        "points": reconstruction.num_points3D(),
        "reprojection_error_px": round(reconstruction.compute_mean_reprojection_error(), 3),
        "metric_scale": True,
    }
    if ctx.params.get("refine_with_bundle_adjustment"):
        ctx.progress("bundle_adjustment", 0.9)
        before = {image.name: image.projection_center() for image in reconstruction.images.values()}
        pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
        shifts = [np.linalg.norm(image.projection_center() - before[image.name]) for image in reconstruction.images.values()]
        result["refined_reprojection_error_px"] = round(reconstruction.compute_mean_reprojection_error(), 3)
        result["median_camera_shift_cm"] = round(float(np.median(shifts)) * 100, 2)
        ctx.notes.append("BA here has a free gauge, so the camera shift includes a small global drift.")
    reconstruction.write(str(sparse))
    return result


@task("tracking.deblur", requires=("torch", "restormer"))
def deblur_frames(ctx: TaskContext) -> dict:
    """Restormer motion deblur on every keyframe (about 27 s per 640x480 frame on CPU; MPS on a Mac)."""
    code, weights = restormer_paths()
    destination = ctx.paths["root"] / "frames-deblurred"
    ctx.progress("deblurring", 0.1)
    ctx.run([
        ctx.python(), PROJECT_ROOT / "research" / "deblur_restormer.py",
        "--input", ctx.paths["frames"], "--output", destination,
        "--restormer", code, "--weights", weights,
        "--device", ctx.params.get("device", "cpu"), "--threads", ctx.params.get("threads", 4),
    ])
    return {"frames": sum(1 for _ in destination.glob("*.jpg")), "output": str(destination)}


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, R, t) mapping source points onto target points."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(xt.T @ xs / len(source))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / xs.var(0).sum())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def ar_camera_centres(ctx: TaskContext) -> dict[str, np.ndarray]:
    priors = ctx.paths["root"] / "capture" / "pose_priors.jsonl"
    if not priors.is_file():
        raise TaskError("No ARCore pose priors in this project.")
    centres = {}
    for line in priors.read_text().splitlines():
        meta = json.loads(line)
        if meta.get("position") is not None:
            centres[meta["name"]] = np.asarray(meta["position"], dtype=float)
    return centres


@task("tracking.evaluate", requires=("colmap",))
def evaluate_trajectory(ctx: TaskContext) -> dict:
    """ATE RMSE after Sim3 alignment against ARCore poses or a ground-truth file."""
    import pycolmap

    if ctx.params.get("reference", "ar") == "ar":
        reference = ar_camera_centres(ctx)
    else:
        path = Path(ctx.params.get("ground_truth_path") or "")
        if not path.is_file():
            raise TaskError("ground_truth_path must point to a JSON file of camera centres.")
        reference = {name: np.asarray(xyz, dtype=float) for name, xyz in json.loads(path.read_text()).items()}
    reconstruction = pycolmap.Reconstruction(str(model_dir(ctx, ctx.params.get("model", "sfm"))))
    names = [image.name for image in reconstruction.images.values() if image.name in reference]
    if len(names) < 3:
        raise TaskError("Fewer than 3 frames are shared with the reference.")
    estimated = np.array([reconstruction.find_image_with_name(n).projection_center() for n in names])
    target = np.array([reference[n] for n in names])
    scale, rotation, translation = umeyama(estimated, target)
    errors = np.linalg.norm((scale * (rotation @ estimated.T)).T + translation - target, axis=1)
    return {
        "frames_compared": len(names),
        "ate_rmse_cm": round(float(np.sqrt((errors ** 2).mean())) * 100, 2),
        "ate_median_cm": round(float(np.median(errors)) * 100, 2),
        "sim3_scale": round(scale, 6),
        "reference": ctx.params.get("reference", "ar"),
        "caveat": "ARCore poses are a reference, not ground truth." if ctx.params.get("reference", "ar") == "ar" else None,
    }


@task("tracking.live_tracker", requires=("onnxruntime",))
def live_tracker(ctx: TaskContext) -> dict:
    """Research tracker: Depth Anything as a pseudo depth sensor + PnP + windowed BA (ICL layout only)."""
    p = ctx.params
    ctx.run([
        ctx.python(), PROJECT_ROOT / "research" / "live_track3.py",
        p["dataset_dir"], int(p.get("grid", True)), p.get("first", 1200), p.get("last", 2869), p.get("step", 8), p.get("window", 8),
    ])
    return {"log": str(ctx.log_path), "note": "Measured 7.9 cm ATE on ICL living room 1 (ORB-SLAM3 3.3-6.5 cm)."}


@router.post("/projects/{project_id}/tracking/sfm", response_model=JobRead, status_code=202)
async def start_sfm(project_id: str, request: SfmRequest, db: Session = Depends(get_db)):
    """Camera poses with COLMAP (global mapper, CLAHE + DSP-SIFT, exposure solve). CPU is fine."""
    return submit_task(db, project_id, "tracking.sfm", request.model_dump())


@router.post("/projects/{project_id}/tracking/ar-poses", response_model=JobRead, status_code=202)
async def start_ar_poses(project_id: str, request: ArPosesRequest, db: Session = Depends(get_db)):
    """Use the phone's metric ARCore poses directly and triangulate points (sensor bundles only)."""
    return submit_task(db, project_id, "tracking.ar_poses", request.model_dump())


@router.post("/projects/{project_id}/tracking/deblur", response_model=JobRead, status_code=202)
async def start_deblur(project_id: str, request: DeblurRequest, db: Session = Depends(get_db)):
    """Restormer deblur of the keyframes before SfM (needs PyTorch and the Restormer weights)."""
    return submit_task(db, project_id, "tracking.deblur", request.model_dump())


@router.post("/projects/{project_id}/tracking/evaluate", response_model=JobRead, status_code=202)
async def start_evaluation(project_id: str, request: EvaluateRequest, db: Session = Depends(get_db)):
    """Trajectory error of the SfM or AR model against ARCore poses or a ground-truth file."""
    return submit_task(db, project_id, "tracking.evaluate", request.model_dump())


@router.post("/projects/{project_id}/tracking/live-tracker", response_model=JobRead, status_code=202)
async def start_live_tracker(project_id: str, request: LiveTrackerRequest, db: Session = Depends(get_db)):
    """Research-only monocular tracker; runs on an ICL-NUIM style folder, not on project frames yet."""
    if not Path(request.dataset_dir).is_dir():
        raise HTTPException(status_code=400, detail="dataset_dir does not exist.")
    return submit_task(db, project_id, "tracking.live_tracker", request.model_dump())
