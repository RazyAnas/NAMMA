"""Pipeline overview: machine backends, stage catalog, findings, job status and the one-call full run."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Job, Project
from app.pipeline.backends import describe, missing_requirements
from app.pipeline.findings import DEAD_ENDS, FINDINGS
from app.pipeline.tasks import TASKS, TaskContext, TaskError, read_job_files, submit_task, task
from app.schemas import JobRead


router = APIRouter(prefix="/api/v1/pipeline", tags=["0 pipeline"])

# "verified": ran end to end through the API in tests/test_pipeline_api.py or on the real phone bundle.
# "wired": endpoint and job exist, not yet run through the API (needs a tool, GPU or dataset not present).
STATUS = {
    "capture.bundle": "verified",
    "capture.frames": "verified",
    "tracking.sfm": "verified",
    "tracking.ar_poses": "verified",
    "tracking.evaluate": "verified",
    "tracking.deblur": "wired",
    "tracking.live_tracker": "wired",
    "points.monocular": "verified",
    "points.export_modp": "verified",
    "splats.train": "wired",
    "splats.gpu_results": "wired",
    "walk.collision_dense": "wired",
    "export.splat_pack": "wired",
    "pipeline.run": "verified",
}

DEFAULT_PLAN = ["capture.frames", "tracking.sfm", "points.monocular", "splats.train"]


class RunRequest(BaseModel):
    stages: list[str] | None = Field(None, description=f"task kinds in order; default {DEFAULT_PLAN}")
    params: dict[str, dict] = Field(default_factory=dict, description="per-stage parameters keyed by task kind")
    skip_unavailable: bool = Field(True, description="skip stages this machine cannot run instead of failing")


@task("pipeline.run")
def run_pipeline(ctx: TaskContext) -> dict:
    """Run several stages in order inside one job; stages this machine can't run are skipped and listed."""
    stages = ctx.params.get("stages") or list(DEFAULT_PLAN)
    if ctx.params.get("default_plan") and (ctx.paths["root"] / "capture" / "frames").is_dir():
        stages = [kind for kind in stages if kind != "capture.frames"]  # a sensor bundle already gave frames
    results = {}
    for index, kind in enumerate(stages):
        spec = TASKS.get(kind)
        if spec is None or kind == "pipeline.run":
            raise TaskError(f"Unknown stage {kind}.")
        params = ctx.params.get("params", {}).get(kind, {})
        requires = spec.requires
        if kind == "splats.train":
            from app.pipeline.stages.splats import TRAINER_REQUIREMENTS

            requires = requires + TRAINER_REQUIREMENTS[params.get("trainer", "opensplat")]
        missing = missing_requirements(requires, params)
        if missing:
            if not ctx.params.get("skip_unavailable", True):
                raise TaskError(f"{kind} needs {missing}.")
            results[kind] = {"skipped": True, "missing": missing}
            ctx.log(f"skip {kind}: missing {missing}")
            continue
        ctx.progress(kind, index / len(stages))
        sub = TaskContext(ctx.job_id, ctx.project, ctx.db, ctx.paths, params, ctx.settings, ctx.job_dir)
        results[kind] = spec.function(sub) or {}
        if sub.notes:
            results[kind]["notes"] = sub.notes
    return {"stages": results}


@router.get("/backends")
async def backends() -> dict:
    """CPU / Apple silicon / CUDA availability and which tools this machine has."""
    return describe()


@router.get("/stages")
async def stages() -> list[dict]:
    """Every pipeline task: what it does, what it needs, whether it can run here, and verified vs wired."""
    return [
        {
            "kind": spec.kind,
            "description": spec.description,
            "requires": list(spec.requires),
            "missing_here": missing_requirements(spec.requires),
            "status": STATUS.get(spec.kind, "wired"),
        }
        for spec in TASKS.values()
    ]


@router.get("/findings")
async def findings() -> dict:
    """Measured (M) and quoted (Q) results per stage from the research threads, plus dead ends."""
    return {"labels": {"M": "measured here, CPU only", "Q": "quoted from a paper"}, "stages": FINDINGS, "dead_ends": DEAD_ENDS}


@router.get("/jobs/{job_id}")
async def job_detail(job_id: str, db: Session = Depends(get_db)) -> dict:
    """Job status plus its parameters, result and the last lines of its log."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {**JobRead.model_validate(job).model_dump(mode="json"), **read_job_files(get_settings(), job)}


@router.get("/projects/{project_id}/jobs", response_model=list[JobRead])
async def project_jobs(project_id: str, db: Session = Depends(get_db)):
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return list(db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())))


@router.post("/projects/{project_id}/run", response_model=JobRead, status_code=202)
async def run(project_id: str, request: RunRequest, db: Session = Depends(get_db)):
    """One call for the whole pipeline: frames -> SfM -> monocular points -> splat (skips what can't run)."""
    payload = request.model_dump()
    payload["default_plan"] = request.stages is None
    return submit_task(db, project_id, "pipeline.run", payload)
