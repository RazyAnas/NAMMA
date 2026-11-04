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

    # NAMMA effective-rank needle regularizer (from final/room_prior.py)
    needle_code = """
# NAMMA effective-rank needle regularizer (from final/room_prior.py)
def _namma_needle_loss(splats):
    s, _ = torch.sort(splats["scales"], dim=1)
    return 0.01 * torch.relu(s[:, 2] - s[:, 1] - math.log(6.0)).mean()
"""
    if "def _namma_needle_loss" not in source:
        source = needle_code + "\n" + source

    compile(source, str(trainer_path), "exec")
    trainer_path.write_text(source, encoding="utf-8")
