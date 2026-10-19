"""Stand-in splat trainer for tests on machines without CUDA or OpenSplat.

Writes a valid 3DGS PLY (one small Gaussian per sparse point, coloured like the point) so the
render job's .splat conversion and the phone's Spark viewer can be exercised end to end. It does
not optimise anything: its "PSNR" is marked as not measured.

    GAUSSIAN_TRAINER_COMMAND="python tests/fixtures/fake_splat_trainer.py --dataset {dataset} --output {output}"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

reconstruction = pycolmap.Reconstruction(str(Path(args.dataset) / "sparse" / "0"))
points = list(reconstruction.points3D.values())
xyz = np.array([p.xyz for p in points], np.float32)
rgb = np.array([p.color for p in points], np.float32) / 255
names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
table = np.zeros((len(xyz), len(names)), np.float32)
table[:, 0:3] = xyz
table[:, 6:9] = (rgb - 0.5) / 0.28209479
table[:, 9] = 3.0  # sigmoid -> 0.95
table[:, 10:13] = np.log(0.01)
table[:, 13] = 1.0
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz)
header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
(out / "splat.ply").write_bytes(header.encode() + table.astype("<f4").tobytes())
(out / "train_info.json").write_text(json.dumps({"trainer": "fake", "splats": len(xyz), "measured": False}))
print(f"fake trainer: wrote {len(xyz)} splats")
