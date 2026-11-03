"""Harden gsplat v1.5.3's first CUDA-extension build on current Kaggle images.

The v1.5.3 JIT fallback removes its build directory before compiling, but
assumes PyTorch has already created that directory.  Kaggle's current PyTorch
runtime does not do so, causing a ``FileNotFoundError`` before training starts.
This narrowly patches only the isolated gsplat checkout used by panorama_v1.
"""
from __future__ import annotations

from pathlib import Path


def patch_gsplat_cuda_jit_cache(gsplat_root: Path) -> None:
    backend_path = Path(gsplat_root) / "gsplat" / "cuda" / "_backend.py"
    if not backend_path.is_file():
        raise RuntimeError(f"gsplat CUDA backend is missing: {backend_path}")

    source = backend_path.read_text(encoding="utf-8")
    legacy = "            shutil.rmtree(build_dir)\n"
    patched = "            shutil.rmtree(build_dir, ignore_errors=True)\n"
    if patched in source:
        return
    count = source.count(legacy)
    if count != 1:
        raise RuntimeError(
            "Unrecognized gsplat CUDA JIT cleanup scope; refusing an unsafe patch."
        )
    backend_path.write_text(source.replace(legacy, patched), encoding="utf-8")
