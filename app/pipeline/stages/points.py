"""Stage 3, points: dense monocular depth (Depth Anything V2 via ONNX) anchored to the sparse model.

Measured on ICL living room 1: 98.8% of pixels get depth vs 0.05% for SfM points, and splats
initialised from these points reached 34.7 dB vs 31.0 dB from sparse points (2k steps, CPU).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from fastapi import APIRouter, Depends
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.pipeline.stages.tracking import model_dir
from app.pipeline.tasks import TaskContext, TaskError, submit_task, task
from app.schemas import JobRead


router = APIRouter(prefix="/api/v1/pipeline", tags=["3 points"])


class MonocularRequest(BaseModel):
    model: str = Field("vits", pattern="^(vits|vitb)$")
    provider: str = Field("auto", pattern="^(auto|cpu|coreml|cuda)$")
    poses: str = Field("sfm", pattern="^(sfm|ar)$", description="sfm = COLMAP model, ar = ARCore-triangulated model")
    pixel_stride: int = Field(4, ge=1, le=32)
    fusion_voxel_m: float = Field(0.02, gt=0, le=0.5)


class ModpExportRequest(BaseModel):
    pixel_stride: int = Field(4, ge=1, le=32)
    max_points_per_frame: int = Field(4000, ge=100, le=100000)


def frames_for(ctx: TaskContext, poses: str) -> Path:
    return ctx.paths["root"] / "capture" / "frames" if poses == "ar" else ctx.paths["frames"]


@task("points.monocular", requires=("onnxruntime", "open3d", "colmap"))
def monocular_points(ctx: TaskContext) -> dict:
    """Depth Anything V2 per frame, scale/shift fitted to sparse points, fused cloud + collision mesh."""
    from app.services.monocular import run_monocular_geometry

    poses = ctx.params.get("poses", "sfm")
    command = (
        f'"{ctx.python()}" scripts/predict_depth_onnx.py --images {{images}} --output {{output}} '
        f'--model {ctx.params.get("model", "vits")} --provider {ctx.params.get("provider", "auto")}'
    )
    settings = ctx.override(
        monocular_depth_command=command,
        monocular_pixel_stride=ctx.params.get("pixel_stride"),
        monocular_fusion_voxel_size=ctx.params.get("fusion_voxel_m"),
    )
    ctx.progress("depth_and_fusion", 0.1)
    artifact = run_monocular_geometry(
        frames_for(ctx, poses), model_dir(ctx, poses), ctx.paths["artifacts"] / "geometry", ctx.log_path, settings
    )
    (ctx.paths["artifacts"] / "geometry" / "points-source.json").write_text(json.dumps({"poses": poses}))
    depth_manifest = json.loads((ctx.paths["artifacts"] / "geometry" / "monocular-depth" / "manifest.json").read_text())
    errors = [frame["median_relative_error"] for frame in artifact["alignment"]["frames"]]
    return {
        "poses": poses,
        "fused_cloud": str(ctx.paths["artifacts"] / "geometry" / "dense" / "fused-monocular.ply"),
        "collision_mesh": artifact.get("model_path"),
        "frames": len(errors),
        "median_alignment_error": round(float(np.median(errors)), 4) if errors else None,
        "seconds_per_frame": depth_manifest.get("seconds_per_frame"),
        "providers": depth_manifest.get("providers"),
        "anchoring": "per-frame scale/shift (global fallback); boot and multi-view variants are research-only",
    }


@task("points.export_modp", requires=("colmap",))
def export_modp(ctx: TaskContext) -> dict:
    """Write a MODP 'aria'-layout dataset whose per-frame points come from monocular depth."""
    import cv2
    import pycolmap

    geometry = ctx.paths["artifacts"] / "geometry"
    manifest_path = geometry / "collision" / "manifest.json"
    if not manifest_path.is_file():
        raise TaskError("Run points/monocular first.")
    alignment = {f["image"]: f for f in json.loads(manifest_path.read_text())["alignment"]["frames"]}
    source = geometry / "points-source.json"
    poses = json.loads(source.read_text())["poses"] if source.is_file() else "sfm"
    reconstruction = pycolmap.Reconstruction(str(model_dir(ctx, poses)))
    out = ctx.paths["artifacts"] / "modp"
    if out.exists():
        shutil.rmtree(out)
    (out / "rectified").mkdir(parents=True)
    (out / "point_cloud").mkdir()
    stride = int(ctx.params.get("pixel_stride", 4))
    limit = int(ctx.params.get("max_points_per_frame", 4000))
    rng = np.random.default_rng(0)
    cameras = []
    for index, image in enumerate(sorted(reconstruction.images.values(), key=lambda im: im.name)):
        fit = alignment.get(image.name)
        depth_path = geometry / "monocular-depth" / f"{Path(image.name).stem}.npy"
        if fit is None or not depth_path.is_file():
            continue
        raw = np.load(depth_path)
        values = raw if fit["mode"] == "direct" else 1.0 / np.maximum(raw, 1e-6)
        depth = (fit["scale"] * values + fit["shift"])[::stride, ::stride]
        k = np.asarray(reconstruction.cameras[image.camera_id].calibration_matrix())
        v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
        z = depth[v, u]
        u, v = u * stride, v * stride
        camera_points = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], 1)
        w2c = np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]])
        c2w = np.linalg.inv(w2c)
        world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        if len(world) > limit:
            world = world[rng.choice(len(world), limit, replace=False)]
        name = f"frame_{index}.jpg"
        cv2.imwrite(str(out / "rectified" / name), cv2.imread(str(frames_for(ctx, poses) / image.name)))
        np.savetxt(out / "point_cloud" / f"point_cloud_{index}.txt", world, fmt="%.5f")
        cameras.append({"image": name, "T_camera_world": w2c.tolist()})
    (out / "trajectory.json").write_text(json.dumps({"cameras": cameras}))
    ctx.notes.append("Images are written as-is; MODP expects undistorted (rectified) frames.")
    return {"frames": len(cameras), "poses": poses, "output": str(out)}


@router.post("/projects/{project_id}/points/monocular", response_model=JobRead, status_code=202)
async def start_monocular_points(project_id: str, request: MonocularRequest, db: Session = Depends(get_db)):
    """Dense points from one camera (no LiDAR). Runs on CPU or Core ML; also builds the collision mesh."""
    return submit_task(db, project_id, "points.monocular", request.model_dump())


@router.post("/projects/{project_id}/points/export-modp", response_model=JobRead, status_code=202)
async def start_modp_export(project_id: str, request: ModpExportRequest, db: Session = Depends(get_db)):
    """Package frames, poses and monocular points for MODP (MODP itself needs a CUDA GPU)."""
    return submit_task(db, project_id, "points.export_modp", request.model_dump())
