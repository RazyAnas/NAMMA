from pathlib import Path
import shutil
import subprocess

from app.config import Settings


class VideoProcessingError(RuntimeError):
    pass


def extract_frames(video_path: Path, frames_dir: Path, settings: Settings) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink()

    vf = f"fps={settings.frame_sample_fps},scale='min({settings.frame_max_width},iw)':-2"
    output_pattern = frames_dir / "frame_%06d.jpg"
    command = [
        settings.resolve_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        str(settings.max_keyframes),
        "-q:v",
        "2",
        str(output_pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        detail = result.stderr.strip()[-2000:] or "FFmpeg exited without an error message."
        raise VideoProcessingError(f"Frame extraction failed: {detail}")

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        raise VideoProcessingError("Frame extraction produced fewer than two frames.")
    return frames
