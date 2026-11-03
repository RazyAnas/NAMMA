"""Make gsplat bilateral-grid helpers visible in every distributed worker.

gsplat v1.5.3 imports these names inside its CLI main().  That works in the
parent process, but torch.multiprocessing uses spawn: each training worker
imports the trainer module afresh and does not execute that CLI setup block.
Runner.__init__ therefore must publish the selected helpers in that worker.
"""
from pathlib import Path


MAIN_EXPORT_MARKER = '        globals().update({\n            "BilateralGrid": BilateralGrid,'
WORKER_EXPORT_MARKER = (
    "# NAMMA: publish bilateral-grid helpers inside every spawned Runner worker."
)


def patch_gsplat_bilateral_grid_import(gsplat_root: Path) -> None:
    trainer_path = Path(gsplat_root) / "examples" / "simple_trainer.py"
    if not trainer_path.is_file():
        raise RuntimeError(f"gsplat trainer script is missing: {trainer_path}")

    source = trainer_path.read_text(encoding="utf-8")

    # Retain the parent-process export for compatibility with the original
    # single-process path.  It is not sufficient by itself for spawn workers.
    if MAIN_EXPORT_MARKER not in source:
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
    else:
        required_main = (
            '"color_correct": color_correct',
            '"slice": slice',
            '"total_variation_loss": total_variation_loss',
        )
        if not all(item in source for item in required_main):
            raise RuntimeError("Found a partial parent bilateral-grid export patch.")

    # Every torch.multiprocessing spawn worker imports this module independently.
    # Publish the imports from Runner.__init__, which runs inside each worker,
    # rather than relying only on the CLI parent process.
    if WORKER_EXPORT_MARKER not in source:
        anchor = "        self.cfg = cfg\n"
        if source.count(anchor) != 1:
            raise RuntimeError(
                "Unrecognized gsplat Runner.__init__ scope; refusing an unsafe worker patch."
            )
        worker_block = anchor + """        # NAMMA: publish bilateral-grid helpers inside every spawned Runner worker.
        if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
            if cfg.use_fused_bilagrid:
                from fused_bilagrid import (
                    BilateralGrid,
                    color_correct,
                    slice,
                    total_variation_loss,
                )
            else:
                from lib_bilagrid import (
                    BilateralGrid,
                    color_correct,
                    slice,
                    total_variation_loss,
                )
            globals().update({
                "BilateralGrid": BilateralGrid,
                "color_correct": color_correct,
                "slice": slice,
                "total_variation_loss": total_variation_loss,
            })
"""
        source = source.replace(anchor, worker_block, 1)
    else:
        required_worker = (
            "if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:",
            "from lib_bilagrid import (",
            '"BilateralGrid": BilateralGrid',
            '"total_variation_loss": total_variation_loss',
        )
        if not all(item in source for item in required_worker):
            raise RuntimeError("Found a partial spawned-worker bilateral-grid patch.")

    compile(source, str(trainer_path), "exec")
    trainer_path.write_text(source, encoding="utf-8")
