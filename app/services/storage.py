from hashlib import sha256
from pathlib import Path
import re

from fastapi import UploadFile

from app.config import Settings


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "video.mp4").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "video.mp4"


def project_root(settings: Settings, project_id: str) -> Path:
    return settings.absolute_data_dir() / "projects" / project_id


def ensure_project_dirs(settings: Settings, project_id: str) -> dict[str, Path]:
    root = project_root(settings, project_id)
    paths = {
        "root": root,
        "source": root / "source",
        "frames": root / "frames",
        "tracking": root / "tracking",
        "training": root / "training",
        "artifacts": root / "artifacts",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> tuple[int, str]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video type '{suffix or 'unknown'}'.")

    digest = sha256()
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as target:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Video exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
                digest.update(chunk)
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("Uploaded video is empty.")
    return total, digest.hexdigest()
