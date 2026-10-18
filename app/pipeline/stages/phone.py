"""Stage 7, phone render: the phone's "Send for render" button lands here.

The /record/ page (served from GitHub Pages or this laptop) uploads its bundle in chunks, because a
Cloudflare quick tunnel refuses request bodies over 100 MB. The laptop (or a Colab GPU running this
same app) then runs the heavy pipeline as one job and the phone polls it and downloads the result:

    POST /api/v1/remote/uploads                      {filename, size}       -> {upload_id, chunk_bytes}
    PUT  /api/v1/remote/uploads/{id}/chunks/{index}  raw bytes              -> {received}
    POST /api/v1/remote/uploads/{id}/finish          {quality}              -> {render_id, project_id}
    GET  /api/v1/remote/renders/{render_id}          status, steps, outputs
    GET  /api/v1/remote/renders/{render_id}/files/{name}   splat.splat | dense.ply | sparse.ply | result.json

With AR bundles the job is: unpack -> triangulate on ARCore poses + BA -> dense Depth Anything points
-> splat training from those points (gsplat on CUDA, OpenSplat on a Mac) -> a compact `.splat` for the
phone. Plain videos go through SfM instead of ARCore poses. Steps the machine can't run are skipped
and the best result so far (dense points, then sparse points) is still returned.

Set NAMMA_REMOTE_KEY to require the key (header `X-Namma-Key` or `?key=`) on every remote call;
scripts/serve_render.py always sets one because a tunnel URL is public.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Job, Project
from app.pipeline.backends import capabilities, describe, missing_requirements
from app.pipeline.stages.export import read_splat_ply, walk_scene_files
from app.pipeline.tasks import TASKS, TaskContext, TaskError, job_dir, read_job_files, submit_task, task
from app.services.storage import ALLOWED_VIDEO_SUFFIXES, ensure_project_dirs, safe_filename


router = APIRouter(prefix="/api/v1/remote", tags=["7 phone render"])

CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 32 * 1024 * 1024
OUTPUT_FILES = ("splat.splat", "dense.ply", "sparse.ply", "result.json")

# stride = keep every n-th AR frame; iterations/downscale feed the splat trainer.
QUALITY = {
    "fast": {"stride": 15, "iterations": 2000, "downscale": 2, "pixel_stride": 6, "max_splats": 400000},
    "balanced": {"stride": 10, "iterations": 7000, "downscale": 2, "pixel_stride": 4, "max_splats": 700000},
    "best": {"stride": 5, "iterations": 15000, "downscale": 1, "pixel_stride": 4, "max_splats": 1000000},
}
# Best with OpenSplat runs the measured CPU recipe (scripts/train_opensplat_best.py) at the step count it was measured
# with: on a CPU, more steps cost hours, and the recipe gained +0.9 dB at 3,000 steps.
OPENSPLAT_BEST_ITERATIONS = 3000


def require_key(request: Request) -> None:
    expected = os.environ.get("NAMMA_REMOTE_KEY")
    if not expected:
        return
    given = request.headers.get("x-namma-key") or request.query_params.get("key") or ""
    if not secrets.compare_digest(given, expected):
        raise HTTPException(status_code=401, detail="Wrong or missing pairing key; open the link the laptop printed.")


def uploads_root() -> Path:
    root = get_settings().absolute_data_dir() / "remote" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def upload_dir(upload_id: str) -> Path:
    if not upload_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Bad upload id.")
    directory = uploads_root() / upload_id
    if not (directory / "upload.json").is_file():
        raise HTTPException(status_code=404, detail="Upload not found.")
    return directory


# ---- Output conversion ---------------------------------------------------------------------------

def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary PLY with float xyz and uchar rgb (rgb in 0..1 or 0..255)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    colours = np.clip(rgb * 255 if rgb.size and rgb.max() <= 1.0 else rgb, 0, 255).round().astype(np.uint8)
    records = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    records["p"], records["c"] = xyz, colours
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path.write_bytes(header.encode() + records.tobytes())


def splat_to_dot_splat(ply: Path, destination: Path, camera_centres: np.ndarray | None, max_splats: int) -> dict:
    """3DGS PLY -> the 32-byte `.splat` records most web viewers (Spark included) load.

    Record: position f32x3, scale f32x3 (linear), colour + opacity u8x4, rotation u8x4 (w x y z, q*128+128).
    Floaters far outside the camera path are dropped; the biggest, most opaque splats are kept first.
    """
    g = read_splat_ply(ply)
    pos = np.stack([g["x"], g["y"], g["z"]], 1)
    scale = np.exp(np.stack([g[f"scale_{i}"] for i in range(3)], 1))
    quat = np.stack([g[f"rot_{i}"] for i in range(4)], 1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
    rgb = np.clip(0.5 + 0.28209479177387814 * np.stack([g[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    opacity = 1 / (1 + np.exp(-g["opacity"]))
    keep = (opacity > 0.03) & np.all(np.isfinite(pos), 1) & np.all(np.isfinite(scale), 1)
    if camera_centres is not None and len(camera_centres):
        low, high = camera_centres.min(0), camera_centres.max(0)
        pad = max(1.5, 0.6 * float((high - low).max()))
        keep &= np.all((pos > low - pad) & (pos < high + pad), 1)
    idx = np.where(keep)[0]
    idx = idx[np.argsort(-(opacity[idx] * scale[idx].prod(1) ** (1 / 3)))][:max_splats]
    records = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    records["p"], records["s"] = pos[idx], scale[idx]
    records["c"][:, :3] = (rgb[idx] * 255).round()
    records["c"][:, 3] = (opacity[idx] * 255).round()
    records["r"] = np.clip(quat[idx] * 128 + 128, 0, 255).round()
    destination.write_bytes(records.tobytes())
    return {"splats_in": int(len(pos)), "splats_out": int(len(idx)), "bytes": destination.stat().st_size}


# ---- The render job ------------------------------------------------------------------------------

class StepContext(TaskContext):
    """A stage run inside the render job: its own log folder, progress mapped into the job's range."""

    def __init__(self, parent: TaskContext, key: str, params: dict, low: float, high: float):
        directory = parent.job_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        super().__init__(parent.job_id, parent.project, parent.db, parent.paths, params, parent.settings, directory)
        self.parent, self.key, self.low, self.high = parent, key, low, high

    def progress(self, stage: str, value: float) -> None:
        self.parent.progress(f"{self.key}:{stage}", self.low + (self.high - self.low) * max(0.0, min(1.0, value)))


def quality_is_best(quality: dict) -> bool:
    return quality.get("stride") == QUALITY["best"]["stride"] and quality.get("downscale") == QUALITY["best"]["downscale"]


def render_plan(kind: str, quality: dict, trainer: str | None) -> list[dict]:
    q = quality
    if kind == "ar":
        steps = [
            {"key": "unpack", "label": "Unpack frames and ARCore poses", "task": "capture.bundle", "params": {"stride": q["stride"]}},
            {"key": "poses", "label": "Triangulate on ARCore poses + bundle adjustment", "task": "tracking.ar_poses",
             "params": {"refine_with_bundle_adjustment": True}},
            {"key": "dataset", "label": "Export the splat training set", "task": "phone.dataset", "params": {"poses": "ar"}},
            {"key": "dense", "label": "Dense points (Depth Anything V2)", "task": "points.monocular",
             "params": {"poses": "ar", "pixel_stride": q["pixel_stride"]}, "optional": True},
        ]
    else:
        unpack = ({"key": "unpack", "label": "Unpack frames", "task": "capture.bundle", "params": {"stride": max(2, q["stride"] // 2)}}
                  if kind == "bundle" else
                  {"key": "unpack", "label": "Pick the sharpest frames", "task": "capture.frames", "params": {"selection": "sharpest"}})
        steps = [
            unpack,
            {"key": "poses", "label": "Structure from motion (COLMAP global mapper)", "task": "tracking.sfm", "params": {}},
            {"key": "dense", "label": "Dense points (Depth Anything V2)", "task": "points.monocular",
             "params": {"poses": "sfm", "pixel_stride": q["pixel_stride"]}, "optional": True},
        ]
    best = quality_is_best(q) and trainer == "opensplat"
    label = f"Photoreal splat ({trainer or 'no trainer here'}{', best recipe' if best else ''})"
    steps.append({"key": "splat", "label": label, "task": "splats.train",
                  "params": {"trainer": trainer or "opensplat", "iterations": OPENSPLAT_BEST_ITERATIONS if best else q["iterations"],
                             "downscale": q["downscale"], "init": "monocular", "recipe": "best" if best else "plain"},
                  "optional": True, "needs_trainer": True})
    return steps


def pick_trainer(preferred: str | None = None) -> str | None:
    from app.pipeline.stages.splats import TRAINER_REQUIREMENTS

    order = [preferred] if preferred else []
    order += ["gsplat", "opensplat", "command"]
    for name in order:
        if name in TRAINER_REQUIREMENTS and not missing_requirements(TRAINER_REQUIREMENTS[name]):
            return name
    return None


@task("phone.dataset", requires=("colmap",))
def export_dataset(ctx: TaskContext) -> dict:
    """Splat training set (images + sparse/0) from the ARCore-triangulated model."""
    from app.pipeline.stages.points import frames_for
    from app.pipeline.stages.tracking import model_dir
    from app.services.dataset import export_gaussian_dataset

    poses = ctx.params.get("poses", "ar")
    export_gaussian_dataset(model_dir(ctx, poses), frames_for(ctx, poses), ctx.paths["training"] / "dataset")
    return {"dataset": str(ctx.paths["training"] / "dataset"), "images": len(list((ctx.paths["training"] / "dataset" / "images").iterdir()))}


def _write_steps(ctx: TaskContext, steps: list[dict]) -> None:
    public = [{k: step.get(k) for k in ("key", "label", "state", "seconds", "summary", "reason")} for step in steps]
    (ctx.job_dir / "steps.json").write_text(json.dumps(public, indent=2, default=str))


def _summary(kind: str, result: dict) -> dict:
    keep = {
        "capture.bundle": ("frames", "frames_with_pose"),
        "capture.frames": ("frames",),
        "tracking.ar_poses": ("images", "points", "reprojection_error_px", "refined_reprojection_error_px", "median_camera_shift_cm"),
        "tracking.sfm": ("registered_images", "points", "mean_reprojection_error", "selected_model_id"),
        "phone.dataset": ("images",),
        "points.monocular": ("frames", "median_alignment_error", "seconds_per_frame"),
        "splats.train": ("trainer", "iterations", "init"),
    }.get(kind, ())
    return {k: result[k] for k in keep if k in result}


def _camera_centres(ctx: TaskContext) -> np.ndarray | None:
    import pycolmap

    dataset = ctx.paths["training"] / "dataset" / "sparse" / "0"
    if not dataset.is_dir():
        return None
    reconstruction = pycolmap.Reconstruction(str(dataset))
    return np.array([im.projection_center() for im in reconstruction.images.values()])


def collect_outputs(ctx: TaskContext, max_splats: int) -> dict:
    """Best result first: trained splat, then dense monocular points, then sparse points."""
    import pycolmap

    out = ctx.job_dir / "outputs"
    out.mkdir(exist_ok=True)
    made: dict[str, dict] = {}
    splat = walk_scene_files(ctx.paths["artifacts"])["splat.ply"]
    if splat is not None:
        made["splat.splat"] = splat_to_dot_splat(splat, out / "splat.splat", _camera_centres(ctx), max_splats)
        train_info = next(iter(sorted(splat.parent.rglob("train_info.json"))), None)
        if train_info:
            made["splat.splat"]["train_info"] = json.loads(train_info.read_text())
    dense = ctx.paths["artifacts"] / "geometry" / "dense" / "fused-monocular.ply"
    if dense.is_file():
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(dense))
        xyz, rgb = np.asarray(cloud.points), np.asarray(cloud.colors)
        if len(xyz) > 800000:
            pick = np.random.default_rng(0).choice(len(xyz), 800000, replace=False)
            xyz, rgb = xyz[pick], rgb[pick]
        write_points_ply(out / "dense.ply", xyz, rgb)
        made["dense.ply"] = {"points": int(len(xyz))}
    for candidate in (ctx.paths["training"] / "dataset" / "sparse" / "0", ctx.paths["tracking"] / "ar" / "sparse"):
        if (candidate / "points3D.bin").is_file():
            reconstruction = pycolmap.Reconstruction(str(candidate))
            points = [p for p in reconstruction.points3D.values() if p.track.length() >= 2]
            if points:
                write_points_ply(out / "sparse.ply", np.array([p.xyz for p in points]), np.array([p.color for p in points]))
                made["sparse.ply"] = {"points": len(points)}
            break
    return made


@task("phone.render")
def phone_render(ctx: TaskContext) -> dict:
    """Everything the phone's "Send for render" button asks for, in one job (see the module docstring)."""
    quality_name = ctx.params.get("quality", "balanced")
    quality = {**QUALITY[quality_name], **{k: v for k, v in (ctx.params.get("overrides") or {}).items() if k in QUALITY["fast"]}}
    trainer = pick_trainer(ctx.params.get("trainer"))
    steps = render_plan(ctx.params.get("kind", "ar"), quality, trainer)
    for step in steps:
        step["state"] = "pending"
    _write_steps(ctx, steps)
    started = time.time()
    count = len(steps) + 1
    for index, step in enumerate(steps):
        spec = TASKS[step["task"]]
        requires = spec.requires
        if step["task"] == "splats.train":
            from app.pipeline.stages.splats import TRAINER_REQUIREMENTS

            requires = requires + TRAINER_REQUIREMENTS[step["params"]["trainer"]]
        missing = missing_requirements(requires, step["params"])
        if step.get("needs_trainer") and trainer is None:
            missing = ["a splat trainer (gsplat + CUDA, or OpenSplat)"]
        if missing:
            if not step.get("optional"):
                step["state"], step["reason"] = "failed", f"needs {', '.join(missing)}"
                _write_steps(ctx, steps)
                raise TaskError(f"{step['label']} can't run here: needs {', '.join(missing)}.")
            step["state"], step["reason"] = "skipped", f"needs {', '.join(missing)}"
            ctx.log(f"skip {step['key']}: missing {missing}")
            _write_steps(ctx, steps)
            continue
        step["state"] = "running"
        _write_steps(ctx, steps)
        sub = StepContext(ctx, step["key"], step["params"], index / count, (index + 1) / count)
        t0 = time.time()
        try:
            result = spec.function(sub) or {}
        except Exception as exc:  # optional steps degrade the result instead of failing the render
            step["seconds"] = round(time.time() - t0, 1)
            if not step.get("optional"):
                step["state"], step["reason"] = "failed", str(exc)[:300]
                _write_steps(ctx, steps)
                raise
            step["state"], step["reason"] = "failed", str(exc)[:300]
            ctx.log(f"optional step {step['key']} failed: {exc}")
            _write_steps(ctx, steps)
            continue
        step["state"], step["seconds"] = "done", round(time.time() - t0, 1)
        step["summary"] = _summary(step["task"], result)
        _write_steps(ctx, steps)
    ctx.progress("packing_results", len(steps) / count)
    outputs = collect_outputs(ctx, int(quality["max_splats"]))
    best = next((name for name in OUTPUT_FILES if name in outputs), None)
    result = {
        "quality": quality_name,
        "settings": quality,
        "trainer": trainer,
        "best": best,
        "outputs": outputs,
        "steps": [{k: s.get(k) for k in ("key", "state", "seconds", "summary", "reason")} for s in steps],
        "total_seconds": round(time.time() - started, 1),
        "machine": describe()["platform"],
        "frame": "WebXR local-floor (metres, y up)" if ctx.params.get("kind", "ar") == "ar" else "COLMAP (arbitrary scale)",
    }
    (ctx.job_dir / "outputs" / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# ---- Endpoints -----------------------------------------------------------------------------------

class UploadStart(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    size: int = Field(gt=0)
    name: str | None = Field(None, max_length=200)
    sha256: str | None = Field(None, pattern="^[0-9a-f]{64}$")


class UploadFinish(BaseModel):
    quality: str = Field("balanced", pattern="^(fast|balanced|best)$")
    trainer: str | None = Field(None, pattern="^(gsplat|opensplat|command)$")
    overrides: dict = Field(default_factory=dict, description="per-render tweaks of the quality preset")


@router.get("/hello", dependencies=[Depends(require_key)])
async def hello() -> dict:
    """What the phone shows after "Test connection": this machine and what it can render."""
    caps = capabilities()
    trainer = pick_trainer()
    return {
        "name": get_settings().app_name,
        "machine": describe()["platform"],
        "trainer": trainer,
        "photoreal": trainer is not None,
        "dense_points": not missing_requirements(TASKS["points.monocular"].requires),
        "cuda": caps.get("cuda", False),
        "chunk_bytes": CHUNK_BYTES,
        "qualities": list(QUALITY),
    }


@router.post("/uploads", dependencies=[Depends(require_key)])
async def start_upload(request: UploadStart) -> dict:
    suffix = Path(request.filename).suffix.lower()
    if suffix not in {".zip"} | ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Send a /record/ bundle (.zip) or a video, not '{suffix}'.")
    limit = get_settings().max_upload_mb * 1024 * 1024
    if request.size > limit:
        raise HTTPException(status_code=413, detail=f"Recording is larger than {get_settings().max_upload_mb} MB.")
    upload_id = str(uuid.uuid4())
    directory = uploads_root() / upload_id
    directory.mkdir(parents=True)
    meta = {**request.model_dump(), "filename": safe_filename(request.filename), "chunk_bytes": CHUNK_BYTES,
            "chunks": -(-request.size // CHUNK_BYTES), "created": time.time()}
    (directory / "upload.json").write_text(json.dumps(meta))
    return {"upload_id": upload_id, "chunk_bytes": CHUNK_BYTES, "chunks": meta["chunks"]}


@router.put("/uploads/{upload_id}/chunks/{index}", dependencies=[Depends(require_key)])
async def put_chunk(upload_id: str, index: int, request: Request) -> dict:
    directory = upload_dir(upload_id)
    meta = json.loads((directory / "upload.json").read_text())
    if not 0 <= index < meta["chunks"]:
        raise HTTPException(status_code=400, detail="Chunk index out of range.")
    body = await request.body()
    expected = min(meta["chunk_bytes"], meta["size"] - index * meta["chunk_bytes"])
    if len(body) > MAX_CHUNK_BYTES or len(body) != expected:
        raise HTTPException(status_code=400, detail=f"Chunk {index} has {len(body)} bytes, expected {expected}.")
    part = directory / f"chunk_{index:05d}"
    temporary = part.with_suffix(".tmp")
    temporary.write_bytes(body)
    temporary.replace(part)  # a retried chunk simply overwrites
    received = len(list(directory.glob("chunk_*[0-9]")))
    return {"index": index, "received": received, "chunks": meta["chunks"]}


@router.get("/uploads/{upload_id}", dependencies=[Depends(require_key)])
async def upload_status(upload_id: str) -> dict:
    """Which chunks already arrived, so a phone that lost signal resumes instead of starting over."""
    directory = upload_dir(upload_id)
    meta = json.loads((directory / "upload.json").read_text())
    have = sorted(int(p.name.split("_")[1]) for p in directory.glob("chunk_*[0-9]"))
    return {"upload_id": upload_id, "chunks": meta["chunks"], "received": have, "render_id": meta.get("render_id")}


@router.post("/uploads/{upload_id}/finish", dependencies=[Depends(require_key)])
async def finish_upload(upload_id: str, request: UploadFinish, db: Session = Depends(get_db)) -> dict:
    directory = upload_dir(upload_id)
    meta = json.loads((directory / "upload.json").read_text())
    if meta.get("render_id"):
        return {"render_id": meta["render_id"], "project_id": meta["project_id"]}
    missing = [i for i in range(meta["chunks"]) if not (directory / f"chunk_{i:05d}").is_file()]
    if missing:
        raise HTTPException(status_code=409, detail={"message": "Some chunks are missing.", "missing": missing[:50]})
    settings = get_settings()
    project_id = str(uuid.uuid4())
    paths = ensure_project_dirs(settings, project_id)
    destination = paths["source"] / meta["filename"]
    digest = hashlib.sha256()
    with destination.open("wb") as target:
        for i in range(meta["chunks"]):
            data = (directory / f"chunk_{i:05d}").read_bytes()
            digest.update(data)
            target.write(data)
    size = destination.stat().st_size
    if size != meta["size"] or (meta.get("sha256") and digest.hexdigest() != meta["sha256"]):
        shutil.rmtree(paths["root"], ignore_errors=True)
        raise HTTPException(status_code=400, detail="The assembled recording does not match what the phone sent; send again.")
    kind = "video"
    if destination.suffix.lower() == ".zip":
        import zipfile

        try:
            with zipfile.ZipFile(destination) as archive:
                bundle_meta = json.loads(archive.read("meta.json"))
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            shutil.rmtree(paths["root"], ignore_errors=True)
            raise HTTPException(status_code=400, detail="This zip is not a /record/ bundle.") from exc
        kind = "ar" if bundle_meta.get("mode") == "ar" else "bundle"
    project = Project(id=project_id, name=(meta.get("name") or Path(meta["filename"]).stem)[:200], source_filename=destination.name,
                      source_path=str(destination), source_size_bytes=size, source_sha256=digest.hexdigest())
    db.add(project)
    db.commit()
    job = submit_task(db, project_id, "phone.render", {"kind": kind, "quality": request.quality, "trainer": request.trainer,
                                                       "overrides": request.overrides})
    meta.update(render_id=job.id, project_id=project_id, kind=kind)
    (directory / "upload.json").write_text(json.dumps(meta))
    for chunk in directory.glob("chunk_*"):
        chunk.unlink()
    return {"render_id": job.id, "project_id": project_id, "kind": kind}


def _render_job(db: Session, render_id: str) -> Job:
    job = db.get(Job, render_id)
    if job is None or job.kind != "phone.render":
        raise HTTPException(status_code=404, detail="Render not found.")
    return job


@router.get("/renders/{render_id}", dependencies=[Depends(require_key)])
async def render_status(render_id: str, db: Session = Depends(get_db)) -> dict:
    job = _render_job(db, render_id)
    directory = job_dir(get_settings(), job.project_id, job.id)
    steps_path = directory / "steps.json"
    files = read_job_files(get_settings(), job, log_lines=12)
    outputs = {}
    for name in OUTPUT_FILES:
        path = directory / "outputs" / name
        if path.is_file():
            outputs[name] = {"bytes": path.stat().st_size, "url": f"/api/v1/remote/renders/{render_id}/files/{name}"}
    return {
        "render_id": job.id,
        "project_id": job.project_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "error": job.error_message,
        "steps": json.loads(steps_path.read_text()) if steps_path.is_file() else [],
        "outputs": outputs,
        "best": (files.get("result") or {}).get("best"),
        "log_tail": files.get("log_tail", []),
    }


@router.get("/renders/{render_id}/files/{name}", dependencies=[Depends(require_key)])
async def render_file(render_id: str, name: str, db: Session = Depends(get_db)):
    if name not in OUTPUT_FILES:
        raise HTTPException(status_code=404, detail="Unknown output.")
    job = _render_job(db, render_id)
    path = job_dir(get_settings(), job.project_id, job.id) / "outputs" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} is not ready.")
    media = "application/json" if name.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=name)
