"""Harden gsplat v1.5.3's first CUDA-extension build on current Kaggle images.

The v1.5.3 JIT fallback removes its build directory before compiling, but
assumes PyTorch has already created that directory. Kaggle's current PyTorch
runtime does not do so, causing a FileNotFoundError before training starts.

The trainer imports gsplat from the installed site-packages directory, while
NAMMA keeps a separate gsplat checkout only for its example scripts. Support
both layouts and patch the installed package after pip installation.
"""
from __future__ import annotations

from pathlib import Path


def patch_gsplat_cuda_jit_cache(gsplat_root: Path) -> None:
    root = Path(gsplat_root)
    candidates = (
        root / "gsplat" / "cuda" / "_backend.py",  # repository checkout
        root / "cuda" / "_backend.py",             # installed package root
    )
    backend_path = next((path for path in candidates if path.is_file()), None)
    if backend_path is None:
        raise RuntimeError(f"gsplat CUDA backend is missing under: {root}")

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
    updated = source.replace(legacy, patched, 1)
    compile(updated, str(backend_path), "exec")
    backend_path.write_text(updated, encoding="utf-8")
