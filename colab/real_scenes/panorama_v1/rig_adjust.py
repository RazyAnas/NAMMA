\"\"\"Rig constraint post-processor for panorama SfM camera poses.

Forces all perspective views originating from the same physical 360 panorama
to share the exact median camera centre while preserving their relative
known rotations.
\"\"\"
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def enforce_panorama_rig_closure(sfm_out: Path) -> None:
    poses_path = sfm_out / \"camera_poses.json\"
    if not poses_path.is_file():
        return

    poses = json.loads(poses_path.read_text(encoding=\"utf-8\"))
    by_pano: dict[str, list[dict]] = {}
    for p in poses:
        name = p[\"name\"]
        prefix = name.split(\"_v\")[0] if \"_v\" in name else name
        by_pano.setdefault(prefix, []).append(p)

    updated_poses = []
    for prefix, group in by_pano.items():
        centers = np.array([np.array(item[\"camera_center\"], dtype=np.float64) for item in group])
        median_center = np.median(centers, axis=0)
        
        for item in group:
            c2w = np.array(item[\"camera_to_world\"], dtype=np.float64)
            c2w[:3, 3] = median_center
            item[\"camera_center\"] = median_center.tolist()
            item[\"camera_to_world\"] = c2w.tolist()
            updated_poses.append(item)

    poses_path.write_text(json.dumps(updated_poses, indent=2), encoding=\"utf-8\")
    print(f\"Enforced strict rig closure across {len(by_pano)} panoramas ({len(poses)} views).\", flush=True)
