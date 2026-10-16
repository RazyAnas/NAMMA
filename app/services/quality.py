from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pycolmap

from app.config import Settings


@dataclass(frozen=True)
class ModelMetrics:
    model_id: str
    path: str
    registered_images: int
    sparse_points: int
    mean_reprojection_error: float | None


def score_reconstruction(
    sparse_root: Path,
    frame_count: int,
    settings: Settings,
) -> dict:
    models: list[ModelMetrics] = []
    for model_dir in sorted((path for path in sparse_root.iterdir() if path.is_dir()), key=lambda p: p.name):
        reconstruction = pycolmap.Reconstruction(str(model_dir))
        point_count = int(reconstruction.num_points3D())
        error = float(reconstruction.compute_mean_reprojection_error()) if point_count else None
        models.append(
            ModelMetrics(
                model_id=model_dir.name,
                path=f"sparse/{model_dir.name}",
                registered_images=int(reconstruction.num_reg_images()),
                sparse_points=point_count,
                mean_reprojection_error=error,
            )
        )

    if not models:
        raise RuntimeError("No COLMAP sparse models are available for quality scoring.")

    best = max(models, key=lambda model: (model.registered_images, model.sparse_points))
    ratio = best.registered_images / frame_count if frame_count else 0.0
    warnings: list[str] = []
    if best.registered_images < settings.quality_min_registered_images:
        warnings.append(
            f"Only {best.registered_images} images registered; minimum is {settings.quality_min_registered_images}."
        )
    if ratio < settings.quality_min_registered_ratio:
        warnings.append(
            f"Only {ratio:.1%} of extracted frames registered; minimum is {settings.quality_min_registered_ratio:.0%}."
        )
    if best.sparse_points < settings.quality_min_sparse_points:
        warnings.append(
            f"Only {best.sparse_points} sparse points were reconstructed; minimum is {settings.quality_min_sparse_points}."
        )
    if best.mean_reprojection_error is None:
        warnings.append("The selected model has no triangulated 3D points.")
    elif best.mean_reprojection_error > settings.quality_max_reprojection_error:
        warnings.append(
            f"Mean reprojection error is {best.mean_reprojection_error:.3f}px; maximum is "
            f"{settings.quality_max_reprojection_error:.3f}px."
        )
    if len(models) > 1:
        warnings.append(f"Reconstruction is fragmented into {len(models)} disconnected models.")

    return {
        "schema_version": 1,
        "passed": not warnings,
        "frame_count": frame_count,
        "registered_ratio": ratio,
        "selected_model_id": best.model_id,
        "selected_model_path": best.path,
        "model_count": len(models),
        "models": [asdict(model) for model in models],
        "thresholds": {
            "min_registered_images": settings.quality_min_registered_images,
            "min_registered_ratio": settings.quality_min_registered_ratio,
            "min_sparse_points": settings.quality_min_sparse_points,
            "max_reprojection_error": settings.quality_max_reprojection_error,
        },
        "warnings": warnings,
    }


def write_quality_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def read_quality_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("Reconstruction quality report is not available.")
    return json.loads(path.read_text(encoding="utf-8"))
