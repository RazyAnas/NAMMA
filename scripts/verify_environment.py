"""Smoke-check local dependencies; does not claim the research backend is ready."""
import importlib
import shutil
import sqlite3
import sys
from pathlib import Path

MODULES = [
    "fastapi", "uvicorn", "python_multipart", "pydantic_settings", "dotenv",
    "sqlalchemy", "numpy", "scipy", "cv2", "PIL", "imageio", "imageio_ffmpeg",
    "open3d", "trimesh", "plyfile", "yaml", "tqdm", "rich", "httpx", "pytest",
]
NATIVE_MODULES = ["pycolmap"]


def main():
    failures = []
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as exc:
            failures.append(name)
            cause = f" Cause: {exc.__cause__}" if exc.__cause__ else ""
            print(f"FAIL {name}: {exc}{cause}")
    for name in NATIVE_MODULES:
        try:
            module = importlib.import_module(name)
            print(f"OK {name} {getattr(module, '__version__', '')}".rstrip())
        except Exception as exc:
            failures.append(name)
            cause = f" Cause: {exc.__cause__}" if exc.__cause__ else ""
            print(f"FAIL {name}: {exc}{cause}")
    with sqlite3.connect(":memory:") as db:
        assert db.execute("SELECT 1").fetchone() == (1,)
    print("OK SQLite")
    if "dotenv" not in failures:
        from dotenv import dotenv_values
        settings = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
        assert settings["BLOCK_UNKNOWN_SPACE"] == "true"
        assert settings["HOST"] == "127.0.0.1"
        print("OK .env local defaults")
    if "imageio_ffmpeg" not in failures:
        import imageio_ffmpeg
        print(f"Bundled FFmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
        print(f"FFmpeg version: {imageio_ffmpeg.get_ffmpeg_version()}")
    for name in ["colmap", "nvcc", "cmake", "node"]:
        print(f"External {name}: {shutil.which(name) or 'not on PATH'}")
    print("MODP/SLAM/CUDA backend: separate setup; not validated by this check.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
