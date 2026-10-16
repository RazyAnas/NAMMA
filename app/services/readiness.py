from __future__ import annotations

import json
from pathlib import Path

from app.services.gaussian import GaussianTrainingError, read_gaussian_manifest
from app.services.geometry import GeometryError, read_collision_manifest
from app.services.navigation import NavigationPreparationError, validate_navigation_mesh
from app.services.quality import read_quality_report


def project_readiness(paths: dict[str, Path], project) -> dict:
    """Return an auditable, ordered list of gates for demo readiness."""
    gates: list[dict] = []

    def gate(name: str, complete: bool, detail: str) -> None:
        gates.append({"name": name, "complete": bool(complete), "detail": detail})

    gate("video_uploaded", Path(project.source_path).is_file(), "Source video is stored.")
    gate(
        "frames_extracted",
        project.frame_count > 0,
        f"{project.frame_count} frames extracted." if project.frame_count else "Extract frames from the source video.",
    )

    quality = None
    try:
        quality = read_quality_report(paths["artifacts"] / "quality-report.json")
        selected_model = next(
            (
                model
                for model in quality.get("models", [])
                if str(model.get("model_id")) == str(quality.get("selected_model_id"))
            ),
            {},
        )
        gate(
            "reconstruction_quality",
            bool(quality.get("passed")),
            f"{selected_model.get('registered_images', 0)} registered images; quality gate {'passed' if quality.get('passed') else 'failed'}.",
        )
    except FileNotFoundError:
        gate("reconstruction_quality", False, "Run camera tracking and reconstruction quality scoring.")

    try:
        gaussian = read_gaussian_manifest(paths["artifacts"] / "gaussian")
        gate("photorealistic_model", True, f"Gaussian model available ({gaussian.get('size_bytes', 0)} bytes).")
    except (FileNotFoundError, GaussianTrainingError):
        gate("photorealistic_model", False, "Train and export the Gaussian model.")

    geometry_dir = paths["artifacts"] / "geometry"
    collision = None
    try:
        collision = read_collision_manifest(geometry_dir)
        gate("collision_geometry", True, "Collision geometry artifact is present and size-verified.")
    except (FileNotFoundError, GeometryError):
        gate("collision_geometry", False, "Build collision geometry.")

    scale_ready = bool(collision and collision.get("scale_calibrated", False))
    gate("metric_scale", scale_ready, "Metric scale calibrated." if scale_ready else "Provide one known real-world distance.")

    preparation_path = geometry_dir / "navigation" / "preparation.json"
    preparation = None
    if preparation_path.is_file():
        try:
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            preparation = None
    floor_ready = bool(
        preparation
        and preparation.get("status") != "stale"
        and preparation.get("plane", {}).get("accepted", False)
    )
    gate("floor_detection", floor_ready, "Floor candidate passed confidence checks." if floor_ready else "Run or review floor detection.")

    navmesh_path = geometry_dir / "navigation" / "navmesh.json"
    navmesh = None
    validation = None
    if navmesh_path.is_file():
        try:
            navmesh = json.loads(navmesh_path.read_text(encoding="utf-8"))
            validation = validate_navigation_mesh(geometry_dir)
        except (json.JSONDecodeError, NavigationPreparationError, FileNotFoundError, GeometryError):
            validation = {"passed": False, "errors": ["Navigation artifact could not be validated."]}
    navmesh_valid = bool(validation and validation.get("passed"))
    gate("navigation_validation", navmesh_valid, "Navigation artifact passed integrity and connectivity checks." if navmesh_valid else "Generate and validate the navigation mesh.")
    approved = bool(navmesh_valid and navmesh and navmesh.get("navmesh_ready") and navmesh.get("status") == "approved")
    gate("navigation_approval", approved, "Obstacle-aware roaming is approved." if approved else "Review and approve the validated navigation artifact.")

    first_pending = next((item for item in gates if not item["complete"]), None)
    return {
        "project_id": project.id,
        "project_status": project.status,
        "ready_for_visual_roaming": all(
            next(item for item in gates if item["name"] == name)["complete"]
            for name in ("video_uploaded", "frames_extracted", "reconstruction_quality", "photorealistic_model")
        ),
        "ready_for_obstacle_aware_roaming": approved,
        "next_action": first_pending["detail"] if first_pending else "Ready for the Round 1 demo.",
        "gates": gates,
        "navigation_validation": validation,
    }
