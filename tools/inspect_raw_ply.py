from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from plyfile import PlyData

REPO = "anasrazy/namma-panorama-out"
JOB = "ggps-ftp-20261103-retry-ssim-05"
OUT = Path("audit-output")
OUT.mkdir(parents=True, exist_ok=True)


def percentiles(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    q = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "min": float(q[0]), "p01": float(q[1]), "p05": float(q[2]),
        "p25": float(q[3]), "p50": float(q[4]), "p75": float(q[5]),
        "p95": float(q[6]), "p99": float(q[7]), "max": float(q[8]),
    }


def audit_camera_centres(token):
    path = hf_hub_download(
        repo_id=REPO, repo_type="dataset", filename=f"{JOB}/camera_poses.json",
        token=token
    )
    poses = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = {}
    for row in poses:
        name = str(row.get("name", ""))
        parts = name.split("_v", 1)
        if len(parts) != 2 or not parts[0].startswith("p"):
            continue
        try:
            idx = int(parts[0][1:])
            center = np.asarray(row["camera_center"], dtype=np.float64)
        except (ValueError, KeyError, TypeError):
            continue
        if center.shape == (3,) and np.isfinite(center).all():
            groups.setdefault(idx, []).append(center)

    rows = []
    for idx, centers in sorted(groups.items()):
        a = np.stack(centers)
        mean = a.mean(axis=0)
        spread = np.linalg.norm(a - mean, axis=1)
        maximum = 0.0
        for i in range(len(a)):
            if i + 1 < len(a):
                maximum = max(maximum, float(np.linalg.norm(a[i + 1:] - a[i], axis=1).max()))
        rows.append({
            "panorama_index": idx,
            "views": int(len(a)),
            "rms_spread": float(np.sqrt(np.mean(spread ** 2))),
            "max_pairwise_spread": maximum,
            "centroid": mean.tolist(),
        })
    vals = [r["rms_spread"] for r in rows]
    return {
        "pose_count": len(poses),
        "panorama_count": len(rows),
        "within_panorama_rms_spread": percentiles(vals),
        "worst_panoramas": sorted(rows, key=lambda x: x["rms_spread"], reverse=True)[:10],
        "per_panorama": rows,
    }


def main():
    token = os.environ["HF_TOKEN"]
    ply_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset", filename=f"{JOB}/model.ply",
        token=token
    )
    ply_path = Path(ply_path)
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError("PLY contains no vertex element.")
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names or [])
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise RuntimeError(f"PLY is missing XYZ coordinates; properties={names}")

    xyz = np.column_stack([
        np.asarray(vertex[k], dtype=np.float64) for k in ("x", "y", "z")
    ])
    finite = np.isfinite(xyz).all(axis=1)
    xyz_finite = xyz[finite]
    if not len(xyz_finite):
        raise RuntimeError("PLY has no finite XYZ vertices.")

    rng = np.random.default_rng(20261105)
    sample_n = min(400_000, len(xyz_finite))
    sample_idx = rng.choice(len(xyz_finite), size=sample_n, replace=False)
    sample_xyz = xyz_finite[sample_idx]

    scales = None
    scale_cols = [f"scale_{i}" for i in range(3)]
    if set(scale_cols).issubset(names):
        raw = np.column_stack([
            np.asarray(vertex[k], dtype=np.float64)[finite] for k in scale_cols
        ])
        scales = np.exp(np.clip(raw, -30, 20))
        scale_summary = {
            "storage": "log-scale; exponentiated for statistics",
            "axis_percentiles": {f"axis_{i}": percentiles(scales[:, i]) for i in range(3)},
            "anisotropy_ratio": percentiles(
                np.max(scales, axis=1) / np.maximum(np.min(scales, axis=1), 1e-12)
            ),
            "fraction_max_axis_scale_over_0_1": float(np.mean(np.max(scales, axis=1) > 0.1)),
            "fraction_max_axis_scale_over_0_5": float(np.mean(np.max(scales, axis=1) > 0.5)),
        }
    else:
        scale_summary = {"missing_scale_columns": scale_cols}

    opacity_summary = {"missing_opacity": True}
    opacity = None
    if "opacity" in names:
        raw_opacity = np.asarray(vertex["opacity"], dtype=np.float64)[finite]
        opacity = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -40, 40)))
        opacity_summary = {
            "missing_opacity": False,
            "sigmoid_percentiles": percentiles(opacity),
            "fraction_below_0_05": float(np.mean(opacity < 0.05)),
            "fraction_above_0_95": float(np.mean(opacity > 0.95)),
        }

    dc_cols = [f"f_dc_{i}" for i in range(3)]
    colors = None
    color_summary = {"missing_dc_color": True}
    if set(dc_cols).issubset(names):
        dc = np.column_stack([
            np.asarray(vertex[k], dtype=np.float64)[finite] for k in dc_cols
        ])
        # 3DGS DC coefficient converted to a neutral-direction display color.
        colors = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
        color_summary = {
            "missing_dc_color": False,
            "dc_coefficient_percentiles_per_channel": {
                f"channel_{i}": percentiles(dc[:, i]) for i in range(3)
            },
            "display_rgb_percentiles_per_channel": {
                f"channel_{i}": percentiles(colors[:, i]) for i in range(3)
            },
            "note": "DC-only approximate color; directional spherical-harmonic terms are not rendered."
        }

    covariance = np.cov(xyz_finite[::max(1, len(xyz_finite)//200000)].T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    pos_summary = {
        axis: percentiles(xyz_finite[:, i])
        for i, axis in enumerate(("x", "y", "z"))
    }
    lo = np.percentile(xyz_finite, 1, axis=0)
    hi = np.percentile(xyz_finite, 99, axis=0)
    spans = hi - lo
    occupancy = {}
    for resolution in (16, 32, 64):
        scaled = np.floor((xyz_finite - lo) / np.maximum(spans, 1e-12) * resolution)
        scaled = np.clip(scaled, 0, resolution - 1).astype(np.int32)
        cells = np.unique(scaled, axis=0)
        occupancy[str(resolution)] = {
            "occupied_cells": int(len(cells)),
            "total_cells": int(resolution ** 3),
            "occupancy_fraction": float(len(cells) / resolution ** 3),
        }

    camera_audit = audit_camera_centres(token)
    report = {
        "job_id": JOB,
        "source_file": f"{JOB}/model.ply",
        "file_bytes": int(ply_path.stat().st_size),
        "vertex_count": int(len(vertex)),
        "finite_xyz_vertices": int(finite.sum()),
        "nonfinite_xyz_vertices": int((~finite).sum()),
        "properties": names,
        "position_percentiles": pos_summary,
        "position_covariance_eigenvalues": eigenvalues.tolist(),
        "position_axis_variance_ratio": float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)),
        "central_98_percent_extent_xyz": spans.tolist(),
        "spatial_occupancy": occupancy,
        "scale": scale_summary,
        "opacity": opacity_summary,
        "color": color_summary,
        "camera_centre_consistency": camera_audit,
        "sampled_vertices_for_plots": int(sample_n),
    }
    (OUT / "raw_ply_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    sample_colors = colors[sample_idx] if colors is not None else "black"
    sample_scales = scales[sample_idx] if scales is not None else None
    if sample_scales is not None:
        sizes = np.clip(np.mean(sample_scales, axis=1) * 18.0, 0.15, 3.5)
    else:
        sizes = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    projections = [
        (0, 2, "Top view: X vs Z"),
        (0, 1, "Front view: X vs Y"),
        (2, 1, "Side view: Z vs Y"),
    ]
    for ax, (a, b, title) in zip(axes.flat[:3], projections):
        ax.scatter(sample_xyz[:, a], sample_xyz[:, b], c=sample_colors,
                   s=sizes, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel(("X", "Y", "Z")[a])
        ax.set_ylabel(("X", "Y", "Z")[b])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
    ax = axes.flat[3]
    if opacity is not None:
        ax.hist(opacity, bins=80, log=True)
        ax.set_title("Opacity after sigmoid")
        ax.set_xlabel("Opacity")
        ax.set_ylabel("Gaussian count (log scale)")
    else:
        ax.text(0.1, 0.5, "No opacity property found")
    fig.suptitle(f"NAMMA raw Gaussian PLY forensic views — {len(vertex):,} vertices")
    fig.tight_layout()
    fig.savefig(OUT / "raw_ply_spatial_views.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if scales is not None:
        for i, label in enumerate(("scale X", "scale Y", "scale Z")):
            axes[0].hist(scales[:, i], bins=100, alpha=0.5, label=label, log=True)
        axes[0].legend()
        axes[0].set_title("Gaussian axis scales (exponentiated)")
        axes[0].set_xlabel("Scale in arbitrary SfM units")
        axes[0].set_ylabel("Count (log scale)")
    else:
        axes[0].text(0.1, 0.5, "Scale properties missing")
    spread = [x["rms_spread"] for x in camera_audit["camera_centre_consistency"]["per_panorama"]]
    if spread:
        axes[1].plot(range(len(spread)), spread, marker=".")
        axes[1].set_title("Within-panorama camera-centre RMS spread")
        axes[1].set_xlabel("Panorama index")
        axes[1].set_ylabel("RMS spread (arbitrary SfM units)")
        axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "raw_ply_scale_and_camera_diagnostics.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "file_bytes": report["file_bytes"],
        "vertex_count": report["vertex_count"],
        "finite_xyz_vertices": report["finite_xyz_vertices"],
        "properties": names,
        "central_extent_xyz": spans.tolist(),
        "spatial_occupancy": occupancy,
        "camera_centre_spread": camera_audit["within_panorama_rms_spread"],
        "output_files": sorted(p.name for p in OUT.iterdir()),
    }, indent=2))

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(OUT),
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo=f"forensic/{JOB}",
        commit_message=f"Add raw PLY forensic audit for {JOB}",
    )
    print(f"UPLOADED_REPORT=forensic/{JOB}")


if __name__ == "__main__":
    main()
