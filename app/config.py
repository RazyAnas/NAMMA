from functools import lru_cache
from pathlib import Path
import shutil

import imageio_ffmpeg
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "NAMMA"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    database_url: str = "sqlite:///./data/namma.db"
    max_concurrent_jobs: int = 1
    max_upload_mb: int = 2048
    frame_sample_fps: float = 2.0
    frame_max_width: int = 1280
    max_keyframes: int = 500
    colmap_bin: str = "colmap"
    ffmpeg_bin: str = "ffmpeg"
    colmap_use_gpu: bool = True

    def absolute_data_dir(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir

    def absolute_log_dir(self) -> Path:
        return self.log_dir if self.log_dir.is_absolute() else PROJECT_ROOT / self.log_dir

    def resolve_ffmpeg(self) -> str:
        configured = shutil.which(self.ffmpeg_bin)
        if configured:
            return configured
        configured_path = Path(self.ffmpeg_bin)
        if configured_path.is_file():
            return str(configured_path.resolve())
        return imageio_ffmpeg.get_ffmpeg_exe()

    def resolve_colmap(self) -> str | None:
        configured = shutil.which(self.colmap_bin)
        if configured:
            return configured
        configured_path = Path(self.colmap_bin)
        if configured_path.is_file():
            return str(configured_path.resolve())
        bundled = PROJECT_ROOT / "tools" / "colmap" / "COLMAP.bat"
        return str(bundled) if bundled.is_file() else None

    def colmap_status(self) -> str | None:
        executable = self.resolve_colmap()
        if executable:
            return executable
        try:
            import pycolmap

            return f"pycolmap {pycolmap.__version__}"
        except Exception:
            return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
