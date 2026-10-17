"""Live capture: a phone browser streams keyframes over a WebSocket while the server maps them."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import secrets
import statistics
import threading
import uuid

import cv2
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Job, Project
from app.schemas import JobRead
from app.services.frame_quality import measure_frame
from app.services.jobs import submit_job
from app.services.live_mapper import KeyframeMeta, LiveMapper, write_keyframe_log
from app.services.live_protocol import parse_packet
from app.services.storage import ensure_project_dirs

router = APIRouter(prefix="/api/v1/live-sessions")

LIVE_SOURCE = "live-capture"


@dataclass
class LiveSession:
    id: str
    project_id: str
    token: str
    mapper: LiveMapper
    frame_count: int = 0
    sharpness_history: list[float] = field(default_factory=list)
    finished: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, LiveSession] = {}


class LiveSessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class LiveSessionRead(BaseModel):
    session_id: str
    project_id: str
    token: str
    capture_path: str


def _session(session_id: str, token: str) -> LiveSession:
    session = _sessions.get(session_id)
    if session is None or not secrets.compare_digest(session.token, token):
        raise HTTPException(status_code=404, detail="Live session not found.")
    return session


@router.post("", response_model=LiveSessionRead, status_code=status.HTTP_201_CREATED)
async def create_live_session(
    request: LiveSessionCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> LiveSessionRead:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Session name cannot be blank.")
    project_id = str(uuid.uuid4())
    paths = ensure_project_dirs(settings, project_id)
    project = Project(
        id=project_id,
        name=name,
        source_filename=LIVE_SOURCE,
        source_path=str(paths["frames"]),
        source_size_bytes=0,
        source_sha256="",
    )
    db.add(project)
    db.commit()
    session = LiveSession(
        id=str(uuid.uuid4()),
        project_id=project_id,
        token=secrets.token_urlsafe(24),
        mapper=LiveMapper(paths["frames"], paths["tracking"] / "live", settings),
    )
    session.mapper.start()
    _sessions[session.id] = session
    return LiveSessionRead(
        session_id=session.id,
        project_id=project_id,
        token=session.token,
        capture_path=f"/capture/?session={session.id}&token={session.token}",
    )


@router.get("/{session_id}")
async def get_live_session(session_id: str, token: str) -> dict:
    session = _session(session_id, token)
    return {"session_id": session.id, "project_id": session.project_id, "finished": session.finished, **session.mapper.snapshot()}


@router.post("/{session_id}/finish", response_model=JobRead, status_code=202)
async def finish_live_session(session_id: str, token: str, db: Session = Depends(get_db)) -> Job:
    """Stop streaming and run the full offline pass (SfM, quality gate, exposure compensation)."""
    session = _session(session_id, token)
    with session.lock:
        if session.finished:
            raise HTTPException(status_code=409, detail="Live session already finished.")
        session.finished = True
    session.mapper.stop(wait=False)
    project = db.get(Project, session.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    if session.frame_count < 2:
        raise HTTPException(status_code=409, detail="At least two keyframes are needed.")
    project.frame_count = session.frame_count
    job = Job(project_id=project.id, kind="reconstruct")
    db.add(job)
    db.commit()
    db.refresh(job)
    submit_job(job.id)
    return job


def _vector(value, length: int) -> list[float] | None:
    if isinstance(value, list) and len(value) == length and all(isinstance(v, (int, float)) for v in value):
        return [float(v) for v in value]
    return None


@router.websocket("/{session_id}/stream")
async def stream_keyframes(websocket: WebSocket, session_id: str, token: str) -> None:
    session = _sessions.get(session_id)
    if session is None or not secrets.compare_digest(session.token, token) or session.finished:
        await websocket.close(code=4404)
        return
    settings = get_settings()
    await websocket.accept()
    paths = ensure_project_dirs(settings, session.project_id)
    log_path = paths["tracking"] / "keyframes.jsonl"
    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                header, image = parse_packet(data, settings.live_max_packet_bytes)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                await websocket.send_json({"type": "ack", "accepted": False, "reason": str(exc)})
                continue
            quality = measure_frame(image)
            history = session.sharpness_history[-30:]
            reference = statistics.median(history) if len(history) >= 5 else None
            if reference is not None and quality.sharpness < settings.live_min_sharpness_ratio * reference:
                await websocket.send_json(
                    {"type": "ack", "accepted": False, "reason": "blurry", **_progress(session)}
                )
                continue
            with session.lock:
                if session.finished:
                    break
                session.frame_count += 1
                name = f"frame_{session.frame_count:06d}.jpg"
            cv2.imwrite(str(paths["frames"] / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            session.sharpness_history.append(quality.sharpness)
            meta = KeyframeMeta(
                name=name,
                timestamp=float(header.get("t", 0.0)),
                sharpness=quality.sharpness,
                position=_vector(header.get("position"), 3),
                gravity=_vector(header.get("gravity"), 3),
            )
            write_keyframe_log(log_path, meta)
            session.mapper.add_keyframe(meta)
            await websocket.send_json(
                {
                    "type": "ack",
                    "accepted": True,
                    "luma": round(quality.mean_luma, 3),
                    "clipped": round(quality.clipped_fraction, 3),
                    **_progress(session),
                }
            )
    except WebSocketDisconnect:
        return


def _progress(session: LiveSession) -> dict:
    snapshot = session.mapper.snapshot()
    return {
        "keyframes": session.frame_count,
        "registered": snapshot["registered"],
        "points": snapshot["points"],
        "lost_recent": len(snapshot["lost_frames"]),
    }
