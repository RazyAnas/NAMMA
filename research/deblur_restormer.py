"""Restormer (CVPR 2022, MIT licence) motion deblurring of a folder of keyframes.

Measured in the robustness thread: about 27 s per 640x480 frame on a 4-core CPU; on blurred ICL
footage it took tracking from 7.5 cm to 3.3 cm ATE. Use --device mps on Apple silicon.
Needs a clone of github.com/swz30/Restormer and its GoPro weights (motion_deblurring.pth).
"""
import argparse
import importlib.util
from pathlib import Path
import time

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restormer", type=Path, required=True, help="Restormer repository clone")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("restormer_arch", args.restormer / "basicsr/models/archs/restormer_arch.py")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    model = arch.Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                           heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
                           dual_pixel_task=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu")["params"])
    model = model.to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.glob("*.jpg")):
        if (args.output / path.name).exists():
            continue
        started = time.time()
        image = cv2.imread(str(path))[:, :, ::-1].astype(np.float32) / 255
        x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device)
        with torch.inference_mode():
            y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(str(args.output / path.name), (y[:, :, ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(path.name, f"{time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
