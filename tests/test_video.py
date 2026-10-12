from pathlib import Path
import subprocess

from app.config import Settings
from app.services.video import extract_frames


def test_extract_frames_with_bundled_ffmpeg(tmp_path: Path):
    settings = Settings(frame_sample_fps=2, frame_max_width=320, max_keyframes=10)
    video = tmp_path / "sample.mp4"
    create = [
        settings.resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=2",
        "-pix_fmt", "yuv420p", str(video),
    ]
    subprocess.run(create, check=True)
    frames = extract_frames(video, tmp_path / "frames", settings)
    assert len(frames) == 4
    assert all(frame.stat().st_size > 0 for frame in frames)
