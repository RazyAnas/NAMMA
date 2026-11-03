import subprocess
import sys
from pathlib import Path

from gsplat_trainer_compat import patch_gsplat_bilateral_grid_import


TRAINER_SOURCE = """import os
import time

class Config:
    use_bilateral_grid = False
    use_fused_bilagrid = False

cfg = Config()

if __name__ == "__main__":
    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss

    # Simulate cli() applying the trailing command-line override.
    if "--use-bilateral-grid" in sys.argv:
        cfg.use_bilateral_grid = True
    if cfg.use_bilateral_grid:
        assert "BilateralGrid" in globals()
        print(BilateralGrid.__name__)
"""


def test_cli_bilateral_grid_flag_imports_class_before_cli_applies_config(tmp_path: Path):
    gsplat_root = tmp_path / "gsplat"
    examples = gsplat_root / "examples"
    examples.mkdir(parents=True)
    trainer = examples / "simple_trainer.py"
    trainer.write_text(TRAINER_SOURCE, encoding="utf-8")
    (examples / "lib_bilagrid.py").write_text(
        "class BilateralGrid: pass\\n"
        "def color_correct(*args): pass\\n"
        "def slice(*args): pass\\n"
        "def total_variation_loss(*args): pass\\n",
        encoding="utf-8",
    )

    patch_gsplat_bilateral_grid_import(gsplat_root)
    patched = trainer.read_text(encoding="utf-8")
    assert "requested_bilateral" in patched
    assert '"--use-bilateral-grid" in sys.argv' in patched
    assert "import sys" in patched

    result = subprocess.run(
        [sys.executable, str(trainer), "--use-bilateral-grid"],
        cwd=examples,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "BilateralGrid"

    patch_gsplat_bilateral_grid_import(gsplat_root)
    assert trainer.read_text(encoding="utf-8") == patched
