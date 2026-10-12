from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.config import get_settings
from app.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/v1/health"}
