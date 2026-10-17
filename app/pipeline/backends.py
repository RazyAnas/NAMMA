"""What this machine can run: CPU, Apple silicon (Metal/MPS, Core ML) or CUDA, and which tools exist.

Every stage names its requirements, so a CUDA-only step on a MacBook answers 503 with the reason
instead of failing halfway through a job.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import platform
import shutil

from app.config import PROJECT_ROOT, get_settings


MODELS_DIR = Path(os.environ.get("NAMMA_MODELS_DIR", PROJECT_ROOT / "tools" / "models"))
DEPTH_ANYTHING_URL = (
    "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_{model}.onnx"
)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@lru_cache
def _torch_devices() -> dict[str, bool]:
    if not _has_module("torch"):
        return {"torch": False, "cuda": False, "mps": False}
    import torch

    return {
        "torch": True,
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def onnx_providers() -> list[str]:
    if not _has_module("onnxruntime"):
        return []
    import onnxruntime

    return list(onnxruntime.get_available_providers())


def opensplat_binary() -> str | None:
    configured = os.environ.get("OPENSPLAT_BIN", "opensplat")
    found = shutil.which(configured)
    if found:
        return found
    return configured if Path(configured).is_file() else None


def restormer_paths() -> tuple[Path, Path]:
    code = Path(os.environ.get("RESTORMER_DIR", PROJECT_ROOT / "tools" / "Restormer"))
    weights = Path(os.environ.get("RESTORMER_WEIGHTS", MODELS_DIR / "motion_deblurring.pth"))
    return code, weights


def depth_model_path(model: str = "vits") -> Path:
    return MODELS_DIR / f"depth_anything_v2_{model}.onnx"


def capabilities() -> dict[str, bool]:
    settings = get_settings()
    devices = _torch_devices()
    restormer_code, restormer_weights = restormer_paths()
    return {
        "cpu": True,
        "ffmpeg": bool(settings.resolve_ffmpeg()),
        "colmap": settings.colmap_status() is not None,
        "open3d": _has_module("open3d"),
        "onnxruntime": _has_module("onnxruntime"),
        "coreml": "CoreMLExecutionProvider" in onnx_providers(),
        "torch": devices["torch"],
        "cuda": devices["cuda"],
        "mps": devices["mps"],
        "opensplat": opensplat_binary() is not None,
        "gaussian_trainer_command": bool(settings.gaussian_trainer_command),
        "restormer": restormer_code.is_dir() and restormer_weights.is_file(),
        "depth_anything_vits": depth_model_path("vits").is_file(),
    }


def missing_requirements(requires: tuple[str, ...], params: dict | None = None) -> list[str]:
    """Names of requirements this machine lacks. A `device` parameter adds that device."""
    needed = list(requires)
    device = (params or {}).get("device")
    if device in {"cuda", "mps"}:
        needed.append(device)
    available = capabilities()
    return [name for name in needed if not available.get(name, False)]


def describe() -> dict:
    devices = _torch_devices()
    recommended = "cuda" if devices["cuda"] else "mps" if devices["mps"] else "cpu"
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "recommended_device": recommended,
        "onnx_providers": onnx_providers(),
        "capabilities": capabilities(),
        "models_dir": str(MODELS_DIR),
        "notes": [
            "Everything except splatfacto training and COLMAP dense stereo has a CPU or Apple silicon path.",
            "Splat training on a Mac: build OpenSplat (Metal) and set OPENSPLAT_BIN.",
            "Depth Anything V2 runs through ONNX Runtime; on a Mac the Core ML provider is used when present.",
        ],
    }
