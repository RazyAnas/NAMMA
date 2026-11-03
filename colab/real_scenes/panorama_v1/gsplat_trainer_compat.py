"""Patch gsplat v1.5.3 bilateral-grid imports into the module namespace.

The trainer imports BilateralGrid inside main(), but Runner.__init__ resolves
that symbol in module globals. Without explicitly publishing the imported
symbols, training raises NameError even when the config import branch ran.
"""
from pathlib import Path


def patch_gsplat_bilateral_grid_import(gsplat_root: Path) -> None:
    trainer_path = Path(gsplat_root) / "examples" / "simple_trainer.py"
    if not trainer_path.is_file():
        raise RuntimeError(f"gsplat trainer script is missing: {trainer_path}")

    source = trainer_path.read_text(encoding="utf-8")
    marker = "        globals().update({\n            \"BilateralGrid\": BilateralGrid,"
    if marker in source:
        required = (
            '"color_correct": color_correct',
            '"slice": slice',
            '"total_variation_loss": total_variation_loss',
        )
        if all(item in source for item in required):
            compile(source, str(trainer_path), "exec")
            return
        raise RuntimeError("Found a partial bilateral-grid global-export patch.")

    old_block = """            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies"""
    new_block = """            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

        # Runner methods resolve these names in module globals, not main() locals.
        globals().update({
            "BilateralGrid": BilateralGrid,
            "color_correct": color_correct,
            "slice": slice,
            "total_variation_loss": total_variation_loss,
        })

    # try import extra dependencies"""
    if source.count(old_block) != 1:
        raise RuntimeError(
            "Unrecognized gsplat bilateral-grid import scope; refusing an unsafe patch."
        )

    source = source.replace(old_block, new_block, 1)
    compile(source, str(trainer_path), "exec")
    trainer_path.write_text(source, encoding="utf-8")
