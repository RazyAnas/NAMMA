"""Stage 6, export: the walk-demo bundle, compact splats for the browser, and a list of every output."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Project
from app.pipeline.stages.tracking import model_dir
from app.pipeline.tasks import TaskContext, TaskError, submit_task, task
from app.schemas import JobRead
from app.services.storage import ensure_project_dirs


router = APIRouter(prefix="/api/v1/pipeline", tags=["6 export"])


class SplatPackRequest(BaseModel):
    max_splats: int = Field(500000, ge=1000, le=5000000)


def walk_scene_files(artifacts: Path) -> dict[str, Path | None]:
    files: dict[str, Path | None] = {"splat.ply": None, "navmesh.json": None, "collision.glb": None}
    gaussian = artifacts / "gaussian" / "manifest.json"
    if gaussian.is_file():
        files["splat.ply"] = artifacts / "gaussian" / json.loads(gaussian.read_text())["model_path"]
    navmesh = artifacts / "geometry" / "navigation" / "navmesh.json"
    files["navmesh.json"] = navmesh if navmesh.is_file() else None
    collision = artifacts / "geometry" / "collision" / "manifest.json"
    if collision.is_file():
        files["collision.glb"] = artifacts / "geometry" / "collision" / json.loads(collision.read_text())["model_path"]
    return {name: path if path and path.is_file() else None for name, path in files.items()}


def read_splat_ply(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode().split("\n")
    count = int(next(line for line in header if line.startswith("element vertex")).split()[-1])
    props = [line.split()[-1] for line in header if line.startswith("property")]
    table = np.frombuffer(raw[end : end + count * 4 * len(props)], "<f4").reshape(count, len(props))
    return {name: table[:, i] for i, name in enumerate(props)}


@task("export.splat_pack", requires=("colmap",))
def splat_pack(ctx: TaskContext) -> dict:
    """Trained splat -> 32-byte records in lossless PNGs (what the browser viewer artifacts load)."""
    import cv2
    import pycolmap

    ply = walk_scene_files(ctx.paths["artifacts"])["splat.ply"]
    if ply is None:
        raise TaskError("Train a splat first.")
    g = read_splat_ply(ply)
    pos = np.stack([g["x"], g["y"], g["z"]], 1)
    scale = np.exp(np.stack([g[f"scale_{i}"] for i in range(3)], 1))
    quat = np.stack([g[f"rot_{i}"] for i in range(4)], 1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rgb = np.clip(0.5 + 0.28209479 * np.stack([g[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    opacity = 1 / (1 + np.exp(-g["opacity"]))
    reconstruction = pycolmap.Reconstruction(str(model_dir(ctx, "sfm")))
    images = sorted(reconstruction.images.values(), key=lambda im: im.name)
    centres = np.array([im.projection_center() for im in images])
    low, high = centres.min(0), centres.max(0)
    pad = 0.6 * (high - low).max()
    keep = (opacity > 0.03) & np.all(np.isfinite(pos), 1) & np.all((pos > low - pad) & (pos < high + pad), 1)
    idx = np.where(keep)[0]
    idx = idx[np.argsort(-(opacity[idx] * scale[idx].prod(1) ** (1 / 3)))][: int(ctx.params.get("max_splats", 500000))]
    records = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    records["p"], records["s"] = pos[idx], scale[idx]
    records["c"][:, :3] = (rgb[idx] * 255).round()
    records["c"][:, 3] = (opacity[idx] * 255).round()
    records["r"] = np.clip(quat[idx] * 128 + 128, 0, 255).round()
    out = ctx.paths["artifacts"] / "web-splats"
    out.mkdir(parents=True, exist_ok=True)
    data, width, per_file, files = np.frombuffer(records.tobytes(), np.uint8), 4096, 4096 * 3 * 1000, []
    for start in range(0, len(data), per_file):
        chunk = data[start : start + per_file]
        rows = -(-len(chunk) // (width * 3))
        buffer = np.zeros(rows * width * 3, np.uint8)
        buffer[: len(chunk)] = chunk
        name = f"splats_{len(files)}.png"
        cv2.imwrite(str(out / name), buffer.reshape(rows, width, 3)[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        files.append(name)
    # COLMAP cameras look along +z with +y down, so "up" is minus the mean camera y axis.
    up = -np.mean([im.cam_from_world().rotation.matrix()[1] for im in images], axis=0)
    camera = reconstruction.cameras[images[0].camera_id]
    (out / "splats.json").write_text(json.dumps({
        "count": int(len(idx)), "files": files, "up": (up / np.linalg.norm(up)).tolist(), "centre": centres.mean(0).tolist(),
        "w2c": [np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]]).round(5).tolist() for im in images],
        "fx": float(camera.params[0]), "width": camera.width, "height": camera.height,
    }))
    return {"splats_in": int(len(pos)), "splats_out": int(len(idx)), "output": str(out), "files": files}


@router.get("/projects/{project_id}/export/walk-scene")
async def download_walk_scene(project_id: str, db: Session = Depends(get_db)):
    """Zip of data/splat.ply, data/navmesh.json and data/collision.glb for the walking demo."""
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    files = walk_scene_files(ensure_project_dirs(get_settings(), project_id)["artifacts"])
    missing = [name for name, path in files.items() if path is None]
    if len(missing) == len(files):
        raise HTTPException(status_code=409, detail="Nothing to export yet.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            if path is not None:
                archive.write(path, f"data/{name}")
        archive.writestr("data/README.txt", "Copy data/ over the walk demo's data/ folder.\nMissing: " + (", ".join(missing) or "none") + "\n")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{project_id}-walk-scene.zip"'})


@router.post("/projects/{project_id}/export/splat-pack", response_model=JobRead, status_code=202)
async def start_splat_pack(project_id: str, request: SplatPackRequest, db: Session = Depends(get_db)):
    """Compact the splat for the browser viewer (drops floaters outside the camera hull)."""
    return submit_task(db, project_id, "export.splat_pack", request.model_dump())


@router.get("/projects/{project_id}/export/outputs")
async def list_outputs(project_id: str, db: Session = Depends(get_db)) -> dict:
    """Every file a stage has produced for this project, with sizes."""
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    paths = ensure_project_dirs(get_settings(), project_id)
    outputs = {}
    for key in ("artifacts", "tracking", "training"):
        for path in sorted(paths[key].rglob("*")):
            if path.is_file() and "monocular-depth" not in path.parts and "feature-images" not in path.parts:
                outputs[str(path.relative_to(paths["root"]))] = path.stat().st_size
    return {"root": str(paths["root"]), "walk_scene": {k: bool(v) for k, v in walk_scene_files(paths["artifacts"]).items()}, "files": outputs}
