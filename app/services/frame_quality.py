"""Per-frame sharpness and exposure measurements shared by video import and live capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    name: str
    sharpness: float
    mean_luma: float
    clipped_fraction: float
    dark_fraction: float


def measure_frame(image: np.ndarray, name: str = "") -> FrameQuality:
    """Score a BGR frame.

    Sharpness is the Laplacian variance divided by the intensity variance, so a brighter
    or darker exposure of the same view scores the same and only blur lowers it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    height, width = gray.shape
    if width > 320:
        gray = cv2.resize(gray, (320, max(1, round(height * 320 / width))), interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return FrameQuality(
        name=name,
        sharpness=float(laplacian.var() / (gray.var() + 1e-6)),
        mean_luma=float(gray.mean() / 255.0),
        clipped_fraction=float((gray >= 250).mean()),
        dark_fraction=float((gray <= 5).mean()),
    )


def measure_file(path: Path) -> FrameQuality:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable frame: {path}")
    return measure_frame(image, path.name)


def select_sharpest(scores: list[FrameQuality], window: int) -> list[FrameQuality]:
    """Keep the sharpest frame from each run of `window` consecutive candidates.

    This keeps the original spacing (one frame per window, so overlap is unchanged)
    while skipping frames smeared by fast motion or hand shake.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    return [max(scores[start : start + window], key=lambda item: item.sharpness) for start in range(0, len(scores), window)]


def summarize(scores: list[FrameQuality], selected: list[FrameQuality]) -> dict:
    """Capture-level warnings for blur and lighting drift."""
    warnings: list[str] = []
    if selected:
        sharp = np.array([item.sharpness for item in selected])
        luma = np.array([max(item.mean_luma, 1e-3) for item in selected])
        reference = float(np.median(sharp))
        blurry = int((sharp < 0.5 * reference).sum())
        if blurry > 0.2 * len(selected):
            warnings.append(f"{blurry} of {len(selected)} kept frames are still blurry; move the camera more slowly.")
        stops = float(np.log2(luma.max() / luma.min()))
        if stops > 1.0:
            warnings.append(
                f"Brightness changed by {stops:.1f} stops during capture; exposure compensation will be applied."
            )
        clipped = int(sum(item.clipped_fraction > 0.1 for item in selected))
        if clipped > 0.2 * len(selected):
            warnings.append(f"{clipped} frames have over 10% blown-out highlights.")
    return {
        "schema_version": 1,
        "candidate_count": len(scores),
        "selected_count": len(selected),
        "selected": [asdict(item) for item in selected],
        "warnings": warnings,
    }


def normalize_contrast(source: Path, destination: Path) -> Path:
    """Write contrast-equalised copies of every JPEG frame for feature detection.

    Auto-exposure on a fast-moving phone leaves frames washed out or dark, and SIFT's fixed
    contrast threshold then finds few features. A 1-99 percentile stretch followed by CLAHE
    on lightness (a staple of medical imaging) restores them without changing geometry.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for path in sorted(source.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0].astype(np.float32)
        low, high = np.percentile(lightness, [1, 99])
        stretched = np.clip((lightness - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        lab[:, :, 0] = clahe.apply(stretched)
        cv2.imwrite(str(destination / path.name), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return destination


def write_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
