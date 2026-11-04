"""Independent panorama SfM stage: ALIKED + LightGlue + NetVLAD retrieval + COLMAP.

Input images are already perspective crops. This file intentionally does not import
or modify any v3/video-pipeline module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction
from pairing import merge_retrieval_pairs, sequence_pairs


def run_sfm(images: Path, output: Path, sequence_window: int = 18,
            retrieval_count: int = 10, min_registered_fraction: float = 0.35) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in images.glob("*.jpg"))
    if len(names) < 40:
        raise RuntimeError(f"Only {len(names)} perspective images; at least 40 are required.")

    pairs = sequence_pairs(names, sequence_window)
    local_cfg = extract_features.confs["aliked-n16"]
    match_cfg = match_features.confs["aliked+lightglue"]
    global_retrieval_used = False
    if retrieval_count > 0:
        def load_global_retrieval_pairs():
            retrieval_cfg = extract_features.confs["netvlad"]
            retrieval_path = extract_features.main(retrieval_cfg, images, output)
            retrieval_pairs_path = output / "pairs_retrieval.txt"
            pairs_from_retrieval.main(
                retrieval_path, retrieval_pairs_path, num_matched=retrieval_count
            )
            retrieved = []
            for line in retrieval_pairs_path.read_text(encoding="utf-8").splitlines():
                bits = line.split()
                if len(bits) == 2:
                    retrieved.append((bits[0], bits[1]))
            return retrieved

        pairs, global_retrieval_used, retrieval_error = merge_retrieval_pairs(
            pairs, load_global_retrieval_pairs
        )
        if retrieval_error:
            print(
                "WARNING: NetVLAD global retrieval unavailable; continuing with "
                f"sequential pairs only ({retrieval_error}).",
                flush=True
            )
    else:
        print(
            "NetVLAD global retrieval disabled for this benchmark; "
            "using sequential route pairs.",
            flush=True
        )

    pairs_path = output / "pairs.txt"
    pairs_path.write_text("\\n".join(f"{a} {b}" for a, b in sorted(pairs)), encoding="utf-8")
    print(f"Matching {len(names)} perspective images with {len(pairs)} pairs", flush=True)

    features = extract_features.main(local_cfg, images, output)
    matches = match_features.main(match_cfg, pairs_path, local_cfg["output"], output)
    model_dir = output / "model"
    views_manifest_path = images.parent / "views_manifest.json"
    image_opts: dict = {"camera_model": "PINHOLE"}
    if views_manifest_path.is_file():
        vm = json.loads(views_manifest_path.read_text(encoding="utf-8"))
        fx = float(vm.get("fx_initial", 1108.5125))
        fy = float(vm.get("fy_initial", fx))
        cx = float(vm.get("cx", 640.0))
        cy = float(vm.get("cy", 480.0))
        image_opts["camera_params"] = f"{fx},{fy},{cx},{cy}"

    mapper_opts = {
        "ba_refine_focal_length": False,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": False,
        "min_num_matches": 15,
    }
    rec = reconstruction.main(
        model_dir, images, pairs_path, features, matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=image_opts,
        mapper_options=mapper_opts,
    )
    registered = len(rec.images)
    fraction = registered / max(len(names), 1)
    point_count = len(rec.points3D)
    try:
        mean_reproj = float(rec.compute_mean_reprojection_error())
    except Exception:
        mean_reproj = None
    if registered < max(40, int(len(names) * min_registered_fraction))]:
        raise RuntimeError(
            f"SfM registered only {registered}/{len(names)} views ({fraction:.1%}); "
            "the panoramas may be too far apart, blurred, or visually repetitive."
        )
    if point_count < 500:
        raise RuntimeError(f"SfM produced only {point_count} 3D points; refusing to train a misleading scene.")

    poses = []
    for image in rec.images.values():
        cfw = image.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = cfw.matrix()
        c2w = np.linalg.inv(w2c)
        poses.append({
            "name": image.name,
            "camera_center": c2w[:3, 3].tolist(),
            "camera_to_world": c2w.tolist(),
            "camera_id": int(image.camera_id)
        })
    rep = {
        "perspective_views": len(names),
        "registered_views": registered,
        "registered_fraction": round(fraction, 4),
        "points3D": point_count,
        "mean_reprojection_error_px": round(mean_reproj, 4) if mean_reproj is not None else None,
        "camera_models": len(rec.cameras),
        "global_retrieval_used": global_retrieval_used,
        "pair_count": len(pairs),
        "images_without_3d_points": int(sum(
            (im.num_points3D() if callable(im.num_points3D) else im.num_points3D) == 0
            for im in rec.images.values()
        ))
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(model_dir))
    (output / "camera_poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")
    (output / "sfm_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("PANORAMA SFM", json.dumps(rep), flush=True)
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence-window", type=int, default=18)
    p.add_argument("--retrieval", type=int, default=10)
    a = p.parse_args()
    run_sfm(a.images, a.output, a.sequence_window, a.retrieval)


if __name__ == "__main__":
    main()
