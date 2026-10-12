from pathlib import Path
import contextlib
import io
import subprocess

from app.config import Settings


class ColmapError(RuntimeError):
    pass


def run_colmap(frames_dir: Path, tracking_dir: Path, log_path: Path, settings: Settings) -> int:
    executable = settings.resolve_colmap()
    if executable:
        return _run_colmap_cli(executable, frames_dir, tracking_dir, log_path, settings)
    return _run_pycolmap(frames_dir, tracking_dir, log_path)


def _run_colmap_cli(
    executable: str,
    frames_dir: Path,
    tracking_dir: Path,
    log_path: Path,
    settings: Settings,
) -> int:

    workspace = tracking_dir / "colmap"
    workspace.mkdir(parents=True, exist_ok=True)
    arguments = [
        executable,
        "automatic_reconstructor",
        "--workspace_path",
        str(workspace),
        "--image_path",
        str(frames_dir),
        "--data_type",
        "VIDEO",
        "--quality",
        "MEDIUM",
        "--camera_model",
        "SIMPLE_RADIAL",
        "--single_camera",
        "1",
        "--sparse",
        "1",
        "--dense",
        "0",
        "--use_gpu",
        "1" if settings.colmap_use_gpu else "0",
    ]
    command = ["cmd.exe", "/d", "/s", "/c", *arguments] if executable.lower().endswith((".bat", ".cmd")) else arguments

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        result = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise ColmapError(f"COLMAP failed with exit code {result.returncode}. Log tail: {tail}")

    sparse_root = workspace / "sparse"
    models = [path for path in sparse_root.iterdir() if path.is_dir()] if sparse_root.exists() else []
    if not models:
        raise ColmapError("COLMAP finished but produced no sparse reconstruction model.")
    return max(_registered_image_count(model) for model in models)


def _run_pycolmap(frames_dir: Path, tracking_dir: Path, log_path: Path) -> int:
    try:
        import pycolmap
    except Exception as exc:
        raise ColmapError(
            "Neither a COLMAP executable nor a usable PyCOLMAP backend is available. "
            f"Set COLMAP_BIN or install/allow PyCOLMAP. Import error: {exc}"
        ) from exc

    workspace = tracking_dir / "colmap"
    sparse_root = workspace / "sparse"
    workspace.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    if database_path.exists():
        database_path.unlink()

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            pycolmap.extract_features(
                database_path,
                frames_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options=pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL"),
                device=pycolmap.Device.cpu,
            )
            pycolmap.match_sequential(database_path, device=pycolmap.Device.cpu)
            maps = pycolmap.incremental_mapping(database_path, frames_dir, sparse_root)
            if not maps:
                raise ColmapError("PyCOLMAP produced no sparse reconstruction model.")
            for model_id, reconstruction in maps.items():
                model_path = sparse_root / str(model_id)
                model_path.mkdir(parents=True, exist_ok=True)
                reconstruction.write(model_path)
    except ColmapError:
        raise
    except Exception as exc:
        raise ColmapError(f"PyCOLMAP reconstruction failed: {exc}") from exc
    finally:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output.getvalue(), encoding="utf-8", errors="replace")

    return max(int(reconstruction.num_reg_images()) for reconstruction in maps.values())


def _registered_image_count(model_dir: Path) -> int:
    images_txt = model_dir / "images.txt"
    if images_txt.is_file():
        lines = [line for line in images_txt.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
        return len(lines) // 2

    images_bin = model_dir / "images.bin"
    if not images_bin.is_file():
        return 0
    # The first uint64 in COLMAP's images.bin is the registered image count.
    import struct

    with images_bin.open("rb") as handle:
        data = handle.read(8)
    return int(struct.unpack("<Q", data)[0]) if len(data) == 8 else 0
