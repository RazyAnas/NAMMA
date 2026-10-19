"""Predict relative depth with Depth Anything V2 through ONNX Runtime (no PyTorch, no CUDA).

Same outputs as predict_depth_anything.py (one .npy per image plus manifest.json), so it plugs into
MONOCULAR_DEPTH_COMMAND. Uses Core ML on Apple silicon when available, otherwise the CPU. Weights
come from the GitHub release fabio-sim/Depth-Anything-ONNX v2.0.0 (Hugging Face is not needed).
Measured: Small 0.51 s/frame on a 4-core CPU; Base is 4x slower for little gain.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline.backends import DEPTH_ANYTHING_URL, depth_model_path  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def ensure_weights(model: str) -> Path:
    path = depth_model_path(model)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = DEPTH_ANYTHING_URL.format(model=model)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    return path


def session(path: Path, provider: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider == "auto":
        order = [name for name in ("CoreMLExecutionProvider", "CUDAExecutionProvider") if name in available]
        providers = order + ["CPUExecutionProvider"]
    else:
        providers = [{"cpu": "CPUExecutionProvider", "coreml": "CoreMLExecutionProvider", "cuda": "CUDAExecutionProvider"}[provider]]
    return ort.InferenceSession(str(path), providers=providers)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "coreml", "cuda"])
    parser.add_argument("--size", type=int, default=518)
    args = parser.parse_args()
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        parser.error(f"No images found under {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    sess = session(ensure_weights(args.model), args.provider)
    input_name = sess.get_inputs()[0].name
    records, started = [], time.time()
    for index, image_path in enumerate(images, start=1):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) / 255.0
        height, width = image.shape[:2]
        x = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        disparity = sess.run(None, {input_name: x})[0][0]
        depth = cv2.resize(disparity, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        destination = args.output / f"{image_path.stem}.npy"
        np.save(destination, depth, allow_pickle=False)
        records.append({"image": image_path.name, "depth": destination.name, "width": width, "height": height,
                        "min": float(depth.min()), "max": float(depth.max())})
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
    manifest = {
        "schema_version": 1,
        "model": f"depth_anything_v2_{args.model} (onnx)",
        "license": "Apache-2.0" if args.model == "vits" else "CC-BY-NC-4.0",
        "depth_type": "relative_inverse",
        "providers": sess.get_providers(),
        "seconds_per_frame": round((time.time() - started) / len(images), 3),
        "images": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
