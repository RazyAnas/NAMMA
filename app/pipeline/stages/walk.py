"""Stage 5, walking: collision mesh, metric scale, floor, NavMesh (human-approved) for the game-style walker.

The NavMesh adds `movement` and `climb_top` layers used by viewer/src/walker.js (Rapier character
controller: steps, slopes, jumping onto furniture). Measured on ICL with noisy depth: 73-74% of free
floor walkable vs 17% before; landings on furniture tops 0% -> 87%.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import api as legacy_api
from app.config import get_settings
from app.database import get_db
from app.models import Project
from app.pipeline.stages.points import frames_for
from app.pipeline.stages.tracking import ar_camera_centres, model_dir, umeyama
from app.pipeline.tasks import TaskContext, submit_task, task
from app.schemas import JobRead, NavigationApprovalRequest
from app.services.navigation import calibrate_collision_scale
from app.services.storage import ensure_project_dirs


router = APIRouter(prefix="/api/v1/pipeline", tags=["5 walking"])


class CollisionRequest(BaseModel):
    backend: str = Field("monocular", pattern="^(monocular|colmap)$", description="colmap = PatchMatch dense stereo, CUDA only")
    poses: str = Field("sfm", pattern="^(sfm|ar)$")


class AutoScaleRequest(BaseModel):
    poses: str = Field("sfm", pattern="^(sfm|ar)$", description="the poses the collision mesh was built from")


@task("walk.collision_dense", requires=("cuda", "colmap", "open3d"))
def dense_collision(ctx: TaskContext) -> dict:
    """COLMAP PatchMatch dense stereo collision mesh (upstream's original CUDA path)."""
    from app.services.geometry import run_dense_geometry

    poses = ctx.params.get("poses", "sfm")
    return run_dense_geometry(frames_for(ctx, poses), model_dir(ctx, poses), ctx.paths["artifacts"] / "geometry", ctx.log_path, ctx.settings)


@router.post("/projects/{project_id}/walk/collision", response_model=JobRead, status_code=202)
async def start_collision(project_id: str, request: CollisionRequest, db: Session = Depends(get_db)):
    """Collision mesh. `monocular` is the CPU path (same job as points/monocular); `colmap` needs CUDA."""
    if request.backend == "monocular":
        return submit_task(db, project_id, "points.monocular", {"poses": request.poses})
    return submit_task(db, project_id, "walk.collision_dense", {"poses": request.poses})


@router.post("/projects/{project_id}/walk/auto-scale")
async def auto_scale(project_id: str, request: AutoScaleRequest, db: Session = Depends(get_db)) -> dict:
    """Metric scale from the phone's ARCore poses instead of a tape measure (sensor bundles only)."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    settings = get_settings()
    paths = ensure_project_dirs(settings, project_id)
    geometry = paths["artifacts"] / "geometry"
    context = TaskContext("", project, db, paths, {}, settings, paths["root"])
    try:
        if request.poses == "ar":
            scale, frames = 1.0, None
        else:
            import pycolmap

            reference = ar_camera_centres(context)
            reconstruction = pycolmap.Reconstruction(str(model_dir(context, "sfm")))
            names = [im.name for im in reconstruction.images.values() if im.name in reference]
            if len(names) < 3:
                raise HTTPException(status_code=409, detail="Fewer than 3 SfM frames have ARCore poses.")
            estimated = np.array([reconstruction.find_image_with_name(n).projection_center() for n in names])
            scale, _, _ = umeyama(estimated, np.array([reference[n] for n in names]))
            frames = len(names)
        result = calibrate_collision_scale(geometry, float(scale), 1.0, "ARCore poses (Sim3 fit of camera centres)")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"metres_per_unit": scale, "frames_used": frames, "calibration": result}


@router.post("/projects/{project_id}/walk/prepare")
async def prepare(project_id: str, db: Session = Depends(get_db)) -> dict:
    """Detect the floor plane (same as /api/v1/projects/{id}/prepare-navigation)."""
    return await legacy_api.prepare_project_navigation(project_id, db, get_settings())


@router.post("/projects/{project_id}/walk/navmesh", response_model=JobRead, status_code=202)
async def navmesh(project_id: str, db: Session = Depends(get_db)):
    """Build the NavMesh with movement and climb_top layers (needs scale + floor first)."""
    return await legacy_api.start_navigation_mesh_build(project_id, db, get_settings())


@router.post("/projects/{project_id}/walk/approve")
async def approve(project_id: str, request: NavigationApprovalRequest, db: Session = Depends(get_db)) -> dict:
    """A person confirms the NavMesh before it is used (kept from upstream's design)."""
    return await legacy_api.approve_project_navigation(project_id, request, db, get_settings())
