from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import get_settings
from app.database import init_db
from app.services.jobs import recover_interrupted_jobs


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recover_interrupted_jobs()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(router)

viewer_directory = Path(__file__).resolve().parent / "static" / "viewer"
if viewer_directory.is_dir():
    app.mount("/viewer", StaticFiles(directory=viewer_directory, html=True), name="viewer")


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/v1/health",
        "viewer": "/viewer/",
    }
