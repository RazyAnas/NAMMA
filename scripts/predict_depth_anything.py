"""Predict relative depth maps with the Apache-2.0 Depth Anything V2 Small model."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="depth-anything/Depth-Anything-V2-Small-hf"
    )
    args = parser.parse_args()
    images = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        parser.error(f"No images found under {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()
    records = []
    for index, image_path in enumerate(images, start=1):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth.unsqueeze(1)
            prediction = functional.interpolate(
                prediction,
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        depth = prediction.float().cpu().numpy().astype(np.float32)
        destination = args.output / f"{image_path.stem}.npy"
        np.save(destination, depth, allow_pickle=False)
        records.append(
            {
                "image": image_path.name,
                "depth": destination.name,
                "width": image.width,
                "height": image.height,
                "min": float(np.nanmin(depth)),
                "max": float(np.nanmax(depth)),
            }
        )
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "license": "Apache-2.0",
        "depth_type": "relative",
        "device": str(device),
        "images": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
