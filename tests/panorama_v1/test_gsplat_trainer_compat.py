import subprocess
import sys
from pathlib import Path

from gsplat_trainer_compat import patch_gsplat_bilateral_grid_import


TRAINER_SOURCE = """class Runner:
    def __init__(self):
        self.grid = BilateralGrid()

def main(cfg):
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.use_bilateral_grid:
        Runner()

if __name__ == "__main__":
    class Config:
        use_bilateral_grid = True
        use_fused_bilagrid = False
    main(Config())
"""


def test_bilateral_grid_import_is_visible_to_runner_module_scope(tmp_path: Path):
    gsplat_root = tmp_path / "gsplat"
    examples = gsplat_root / "examples"
    examples.mkdir(parents=True)
    trainer = examples / "simple_trainer.py"
    trainer.write_text(TRAINER_SOURCE, encoding="utf-8")
    (examples / "lib_bilagrid.py").write_text(
        "class BilateralGrid: pass\n"
        "def color_correct(*args): pass\n"
        "def slice(*args): pass\n"
        "def total_variation_loss(*args): pass\n",
        encoding="utf-8",
    )

    patch_gsplat_bilateral_grid_import(gsplat_root)
    patched = trainer.read_text(encoding="utf-8")
    assert 'globals().update({' in patched
    assert '"BilateralGrid": BilateralGrid' in patched
    assert '"total_variation_loss": total_variation_loss' in patched

    result = subprocess.run(
        [sys.executable, str(trainer)],
        cwd=examples,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    # Applying the compatibility patch twice must be harmless.
    patch_gsplat_bilateral_grid_import(gsplat_root)
    assert trainer.read_text(encoding="utf-8") == patched
