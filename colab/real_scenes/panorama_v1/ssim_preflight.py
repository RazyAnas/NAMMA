"""Build a subprocess environment that can import NAMMA's SSIM fallback."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def trainer_preflight_env(
    shim_dir: Path, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return an environment with the embedded fused_ssim module importable."""
    shim_dir = Path(shim_dir).resolve()
    shim_module = shim_dir / "fused_ssim.py"
    if not shim_module.is_file():
        raise FileNotFoundError(f"SSIM fallback module is missing: {shim_module}")

    env = dict(os.environ if base_env is None else base_env)
    existing = env.get("PYTHONPATH", "")
    paths = [str(shim_dir)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
