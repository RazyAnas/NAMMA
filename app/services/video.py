from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

from app.config import Settings
from app.services.frame_quality import measure_file, select_sharpest, summarize, write_report


class VideoProcessingError(RuntimeError):
    pass


def extract_frames(video_path: Path, frames_dir: Path, settings: Settings) -> list[Path]:
    """Extract keyframes at `frame_sample_fps`.

    With `frame_selection=sharpest`, FFmpeg decodes `frame_candidates_per_keyframe` times as
    many frames and only the sharpest of each group is kept, so blurred frames from fast
    motion are replaced by a nearby sharp one without reducing overlap.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink()

    window = max(1, settings.frame_candidates_per_keyframe) if settings.frame_selection == "sharpest" else 1
    candidates_dir = frames_dir.parent / "frame-candidates" if window > 1 else frames_dir
    if window > 1:
        shutil.rmtree(candidates_dir, ignore_errors=True)
        candidates_dir.mkdir(parents=True)

    vf = f"fps={settings.frame_sample_fps * window},scale='min({settings.frame_max_width},iw)':-2"
    output_pattern = candidates_dir / "frame_%06d.jpg"
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
        str(settings.max_keyframes * window),
        "-q:v",
        "2",
        str(output_pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        if window > 1:
            shutil.rmtree(candidates_dir, ignore_errors=True)
        detail = result.stderr.strip()[-2000:] or "FFmpeg exited without an error message."
        raise VideoProcessingError(f"Frame extraction failed: {detail}")

    candidates = sorted(candidates_dir.glob("frame_*.jpg"))
    scores = [measure_file(path) for path in candidates]
    selected = select_sharpest(scores, window)
    if window > 1:
        renamed = []
        for index, item in enumerate(selected, start=1):
            name = f"frame_{index:06d}.jpg"
            shutil.move(str(candidates_dir / item.name), frames_dir / name)
            renamed.append(replace(item, name=name))
        selected = renamed
        shutil.rmtree(candidates_dir, ignore_errors=True)
    write_report(summarize(scores, selected), frames_dir.parent / "artifacts" / "frame-quality.json")

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        raise VideoProcessingError("Frame extraction produced fewer than two frames.")
    return frames
