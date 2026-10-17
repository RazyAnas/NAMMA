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
    # "sharpest" keeps the sharpest of every N decoded frames; "uniform" keeps every frame at the sample rate.
    frame_selection: str = "sharpest"
    frame_candidates_per_keyframe: int = 4
    # Remove per-frame exposure / white-balance / flicker changes before Gaussian training.
    photometric_compensation: bool = True
    colmap_bin: str = "colmap"
    ffmpeg_bin: str = "ffmpeg"
    colmap_use_gpu: bool = True
    colmap_exhaustive_loop_closure: bool = True
    colmap_exhaustive_max_images: int = 500
    # "global" (GLOMAP-style, more accurate on the ICL-NUIM benchmark) or "incremental".
    colmap_mapper: str = "global"
    # Measured on blurred, auto-exposed ICL-NUIM footage: 22 cm -> 7.5 cm tracking error together.
    colmap_contrast_normalize: bool = True
    colmap_blur_robust_features: bool = True
    quality_min_registered_images: int = 20
    quality_min_registered_ratio: float = 0.5
    quality_min_sparse_points: int = 1000
    quality_max_reprojection_error: float = 2.0
    gaussian_trainer_command: str | None = None
    live_max_packet_bytes: int = 8 * 1024 * 1024
    live_min_sharpness_ratio: float = 0.5
    live_max_features: int = 4096
    live_match_window: int = 8
    live_loop_radius_m: float = 0.75
    live_prior_position_sigma_m: float = 0.05
    live_preview_points: int = 4000
    dense_max_image_size: int = 1280
    dense_gpu_index: int = -1
    dense_geom_consistency: bool = False
    dense_source_images: int = 10
    dense_window_step: int = 2
    dense_num_samples: int = 10
    dense_num_iterations: int = 3
    dense_min_num_pixels: int = 5
    dense_max_reprojection_error: float = 2.0
    collision_min_dense_points: int = 5000
    collision_voxels_across_diagonal: int = 600
    collision_poisson_depth: int = 9
    collision_density_trim_quantile: float = 0.02
    collision_target_triangles: int = 100000
    collision_geometry_backend: str = "monocular"
    monocular_depth_command: str | None = None
    monocular_min_alignment_points: int = 12
    monocular_max_alignment_error: float = 0.4
    monocular_pixel_stride: int = 4
    monocular_fusion_voxel_size: float = 0.02
    navigation_floor_distance_ratio: float = 0.005
    navigation_floor_ransac_iterations: int = 1000
    navigation_floor_candidate_count: int = 8
    navigation_min_floor_points: int = 1000
    navigation_floor_min_inlier_ratio: float = 0.05
    navigation_min_floor_confidence: float = 0.55
    navigation_cell_size_m: float = 0.10
    navigation_floor_support_below_m: float = 0.08
    navigation_floor_support_above_m: float = 0.12
    navigation_min_floor_points_per_cell: int = 2
    navigation_obstacle_min_height_m: float = 0.15
    navigation_max_support_height_m: float = 1.0
    navigation_step_clearance_m: float = 0.30
    navigation_max_step_m: float = 0.25
    navigation_max_climb_height_m: float = 1.2
    navigation_min_obstacle_points_per_cell: int = 3
    navigation_obstacle_neighbor_radius_m: float = 0.08
    navigation_min_obstacle_neighbor_voxels: int = 12
    navigation_grid_max_dimension: int = 512
    navigation_component_min_cells: int = 20
    agent_radius_meters: float = 0.30
    agent_height_meters: float = 1.70
    safety_margin_meters: float = 0.10
    max_slope_degrees: float = 35.0

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
            return self._prefer_colmap_executable(Path(configured))
        configured_path = Path(self.colmap_bin)
        if configured_path.is_file():
            return self._prefer_colmap_executable(configured_path)
        bundled = PROJECT_ROOT / "tools" / "colmap" / "COLMAP.bat"
        return self._prefer_colmap_executable(bundled) if bundled.is_file() else None

    @staticmethod
    def _prefer_colmap_executable(path: Path) -> str:
        """Avoid fragile cmd.exe quoting when using the official Windows bundle."""
        resolved = path.resolve()
        if resolved.suffix.lower() in {".bat", ".cmd"}:
            native_executable = resolved.parent / "bin" / "colmap.exe"
            if native_executable.is_file():
                return str(native_executable)
        return str(resolved)

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
