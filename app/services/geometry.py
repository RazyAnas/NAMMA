from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import open3d as o3d
import trimesh

from app.config import Settings


class GeometryError(RuntimeError):
    pass


def _run_command(arguments: list[str], log_file) -> None:
    log_file.write(f"\n=== {arguments[1]} ===\n")
    log_file.write(subprocess.list2cmdline(arguments) + "\n")
    log_file.flush()
    result = subprocess.run(
        arguments,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GeometryError(f"COLMAP {arguments[1]} failed with exit code {result.returncode}.")


def run_dense_geometry(
    frames_dir: Path,
    sparse_model_dir: Path,
    output_dir: Path,
    log_path: Path,
    settings: Settings,
) -> dict:
    """Build dense MVS geometry and a simplified visual-only collision candidate."""
    executable = settings.resolve_colmap()
    if not executable:
        raise GeometryError("A native COLMAP executable is required for dense stereo.")
    if not sparse_model_dir.is_dir():
        raise GeometryError(f"Sparse COLMAP model does not exist: {sparse_model_dir}")

    workspace = output_dir / "dense"
    artifact_dir = output_dir / "collision"
    if workspace.exists():
        shutil.rmtree(workspace)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    workspace.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    fused_path = workspace / "fused.ply"

    commands = [
        [
            executable,
            "image_undistorter",
            "--image_path",
            str(frames_dir),
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            str(workspace),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(settings.dense_max_image_size),
            "--num_patch_match_src_images",
            str(settings.dense_source_images),
        ],
        [
            executable,
            "patch_match_stereo",
            "--workspace_path",
            str(workspace),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "1" if settings.dense_geom_consistency else "0",
            "--PatchMatchStereo.filter",
            "1",
            "--PatchMatchStereo.max_image_size",
            str(settings.dense_max_image_size),
            "--PatchMatchStereo.gpu_index",
            str(settings.dense_gpu_index),
            "--PatchMatchStereo.window_step",
            str(settings.dense_window_step),
            "--PatchMatchStereo.num_samples",
            str(settings.dense_num_samples),
            "--PatchMatchStereo.num_iterations",
            str(settings.dense_num_iterations),
        ],
        [
            executable,
            "stereo_fusion",
            "--workspace_path",
            str(workspace),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric" if settings.dense_geom_consistency else "photometric",
            "--output_path",
            str(fused_path),
            "--StereoFusion.min_num_pixels",
            str(settings.dense_min_num_pixels),
            "--StereoFusion.max_reproj_error",
            str(settings.dense_max_reprojection_error),
        ],
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            for command in commands:
                _run_command(command, log_file)
            log_file.write("\n=== collision_mesh ===\n")
            log_file.flush()
            manifest = build_collision_candidate(fused_path, artifact_dir, settings)
            log_file.write(json.dumps(manifest, indent=2) + "\n")
    except GeometryError:
        raise
    except Exception as exc:
        raise GeometryError(f"Dense geometry processing failed: {exc}") from exc
    return manifest


def build_collision_candidate(
    fused_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    coordinate_units: str = "colmap_arbitrary_units",
    scale_calibrated: bool = False,
    source_description: str = "../dense/fused.ply",
) -> dict:
    if not fused_path.is_file():
        raise GeometryError("COLMAP stereo fusion did not produce fused.ply.")
    cloud = o3d.io.read_point_cloud(str(fused_path), remove_nan_points=True, remove_infinite_points=True)
    if len(cloud.points) < settings.collision_min_dense_points:
        raise GeometryError(
            f"Dense cloud has only {len(cloud.points)} points; minimum is "
            f"{settings.collision_min_dense_points}."
        )

    bounds = np.asarray(cloud.get_axis_aligned_bounding_box().get_extent(), dtype=float)
    diagonal = float(np.linalg.norm(bounds))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise GeometryError("Dense cloud has invalid spatial bounds.")
    voxel_size = diagonal / settings.collision_voxels_across_diagonal
    filtered = cloud.voxel_down_sample(voxel_size)
    filtered, _ = filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(filtered.points) < settings.collision_min_dense_points:
        raise GeometryError("Dense cloud became too small after outlier filtering.")

    filtered.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=40)
    )
    filtered.orient_normals_consistent_tangent_plane(20)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        filtered,
        depth=settings.collision_poisson_depth,
        scale=1.05,
        linear_fit=False,
    )
    density_values = np.asarray(densities)
    if len(density_values):
        cutoff = np.quantile(density_values, settings.collision_density_trim_quantile)
        mesh.remove_vertices_by_mask(density_values < cutoff)
    mesh = _clean_mesh(mesh, settings.collision_target_triangles)
    if len(mesh.triangles) == 0:
        raise GeometryError("Mesh cleanup removed all collision triangles.")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "collision-preview.ply"
    glb_path = output_dir / "collision.glb"
    if not o3d.io.write_triangle_mesh(str(preview_path), mesh, write_ascii=False):
        raise GeometryError("Could not write collision preview PLY.")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    trimesh.Trimesh(vertices=vertices, faces=triangles, process=False).export(glb_path)

    digest = sha256(glb_path.read_bytes()).hexdigest()
    warnings = [
        "Poisson output can invent surfaces and must be visually reviewed before navigation.",
        "Unknown or poorly observed space is not yet classified and must remain blocked.",
    ]
    if not scale_calibrated:
        warnings.insert(0, "Monocular reconstruction scale is not calibrated to metres.")
    manifest = {
        "schema_version": 1,
        "status": "review_required",
        "format": "glb",
        "model_path": "collision.glb",
        "preview_path": "collision-preview.ply",
        "source_dense_cloud": source_description,
        "source_point_count": len(cloud.points),
        "filtered_point_count": len(filtered.points),
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "size_bytes": glb_path.stat().st_size,
        "sha256": digest,
        "coordinate_units": coordinate_units,
        "scale_calibrated": scale_calibrated,
        "safe_for_navigation": False,
        "voxel_size": voxel_size,
        "bounds_min": np.asarray(mesh.get_min_bound()).tolist(),
        "bounds_max": np.asarray(mesh.get_max_bound()).tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _clean_mesh(mesh: o3d.geometry.TriangleMesh, target_triangles: int) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def read_collision_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "collision" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Collision geometry is not available.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = output_dir / "collision" / manifest.get("model_path", "")
    if not model_path.is_file():
        raise GeometryError("Collision manifest points to a missing model.")
    if model_path.stat().st_size != manifest.get("size_bytes"):
        raise GeometryError("Collision model size does not match its manifest.")
    return manifest
