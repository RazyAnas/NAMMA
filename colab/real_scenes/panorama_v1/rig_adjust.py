"""Apply the known shared-centre constraint for perspective crops from each panorama.

The change is applied to the in-memory COLMAP reconstruction, not only to a
sidecar JSON file. gsplat reads the COLMAP binary model, so the Frame poses
must be updated before Reconstruction.write().
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np


_PANO_VIEW = re.compile(r"^(p\d+)_v\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _cam_from_world(image: Any):
    pose = image.cam_from_world
    return pose() if callable(pose) else pose


def enforce_panorama_rig_closure(reconstruction: Any) -> dict:
    """Move every crop camera to its panorama's median centre, preserving rotation.

    COLMAP stores each crop pose as world-to-camera. For each panorama, we
    compute the median of the current camera centres in world coordinates,
    retain each crop's existing rotation, and update its Frame via
    Frame.set_cam_from_world(). This updates the model subsequently consumed
    by gsplat, unlike editing camera_poses.json alone.

    The sparse points are not retriangulated here. The caller must inspect the
    post-closure reprojection error and reject a poor result before training.
    """
    import pycolmap

    groups = defaultdict(list)
    skipped_names = []
    for image in reconstruction.images.values():
        match = _PANO_VIEW.match(str(image.name))
        if not match:
            skipped_names.append(str(image.name))
            continue
        groups[match.group(1)].append(image)

    if not groups:
        raise RuntimeError("Rig closure found no panorama view names such as p00000_v00.jpg.")

    before_centres = {}
    group_stats = {}
    seen_frame_ids = set()

    for pano_id, images in sorted(groups.items()):
        images = sorted(images, key=lambda image: str(image.name))
        centres = []
        for image in images:
            pose = _cam_from_world(image)
            if pose is None:
                raise RuntimeError(f"Registered image has no camera pose: {image.name}")
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :] = np.asarray(pose.matrix(), dtype=np.float64)
            c2w = np.linalg.inv(w2c)
            center = c2w[:3, 3]
            if not np.isfinite(center).all():
                raise RuntimeError(f"Non-finite camera centre for {image.name}")
            centres.append(center)

        centers_array = np.stack(centres)
        median_center = np.median(centers_array, axis=0)
        before_spread = np.linalg.norm(centers_array - median_center, axis=1)
        before_centres[pano_id] = centers_array

        for image in images:
            frame_id = int(image.frame_id)
            if frame_id in seen_frame_ids:
                raise RuntimeError(
                    f"COLMAP frame {frame_id} is shared by multiple panorama crops; "
                    "cannot assign distinct crop rotations to one frame."
                )
            seen_frame_ids.add(frame_id)

            pose = _cam_from_world(image)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :] = np.asarray(pose.matrix(), dtype=np.float64)
            c2w = np.linalg.inv(w2c)

            # Keep the orientation estimated by SfM; replace only the centre.
            c2w[:3, 3] = median_center
            corrected_w2c = np.linalg.inv(c2w)

            frame = image.frame
            if frame is None:
                raise RuntimeError(f"COLMAP image has no associated Frame: {image.name}")
            frame.set_cam_from_world(
                int(image.camera_id),
                pycolmap.Rigid3d(corrected_w2c[:3, :]),
            )

        group_stats[pano_id] = {
            "views": len(images),
            "pre_closure_rms_spread": float(
                np.sqrt(np.mean(np.square(before_spread)))
            ),
            "pre_closure_max_spread": float(np.max(before_spread)),
        }

    # Re-read poses from the reconstruction and verify the hard constraint.
    post_residuals = []
    for pano_id, images in sorted(groups.items()):
        centers = []
        for image in images:
            pose = _cam_from_world(image)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :] = np.asarray(pose.matrix(), dtype=np.float64)
            centers.append(np.linalg.inv(w2c)[:3, 3])
        centers = np.stack(centers)
        residual = np.linalg.norm(centers - centers[0], axis=1)
        post_residuals.extend(residual.tolist())
        group_stats[pano_id]["post_closure_max_residual"] = float(np.max(residual))

    max_residual = float(max(post_residuals, default=0.0))
    if max_residual > 1e-5:
        raise RuntimeError(
            f"COLMAP rig closure verification failed: max centre residual={max_residual:.6g}"
        )

    before_rms = [x["pre_closure_rms_spread"] for x in group_stats.values()]
    return {
        "method": "median_camera_centre_per_panorama",
        "panorama_groups": len(groups),
        "registered_views_constrained": int(sum(len(x) for x in groups.values())),
        "views_with_unrecognized_names": len(skipped_names),
        "pre_closure_median_rms_spread": float(np.median(before_rms)),
        "pre_closure_max_rms_spread": float(max(before_rms)),
        "post_closure_max_residual": max_residual,
        "per_panorama": group_stats,
    }
