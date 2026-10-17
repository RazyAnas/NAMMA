from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import get_settings
from app.database import init_db
from app.live import router as live_router
from app.pipeline import routers as pipeline_routers
from app.services.jobs import recover_interrupted_jobs


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recover_interrupted_jobs()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(live_router)
for pipeline_router in pipeline_routers:
    app.include_router(pipeline_router)

viewer_directory = Path(__file__).resolve().parent / "static" / "viewer"
if viewer_directory.is_dir():
    app.mount("/viewer", StaticFiles(directory=viewer_directory, html=True), name="viewer")

capture_directory = Path(__file__).resolve().parent / "static" / "capture"
app.mount("/capture", StaticFiles(directory=capture_directory, html=True), name="capture")

record_directory = Path(__file__).resolve().parent / "static" / "record"
app.mount("/record", StaticFiles(directory=record_directory, html=True), name="record")


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/v1/health",
        "pipeline": "/api/v1/pipeline/stages",
        "viewer": "/viewer/",
        "capture": "/capture/",
        "record": "/record/",
    }
