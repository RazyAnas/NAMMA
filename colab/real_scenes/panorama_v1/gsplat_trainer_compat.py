"""Patch gsplat's bilateral-grid import ordering for CLI overrides.

In gsplat v1.5.3, the preset is selected before ``cli()`` consumes the remaining
arguments. The original conditional import can therefore miss
``--use-bilateral-grid`` and leave ``BilateralGrid`` undefined at training time.
"""
from pathlib import Path


def patch_gsplat_bilateral_grid_import(gsplat_root: Path) -> None:
    trainer_path = Path(gsplat_root) / "examples" / "simple_trainer.py"
    if not trainer_path.is_file():
        raise RuntimeError(f"gsplat trainer script is missing: {trainer_path}")

    source = trainer_path.read_text(encoding="utf-8")
    marker = "    requested_bilateral = ("
    if marker in source:
        required = (
            "import sys",
            '"--use-bilateral-grid" in sys.argv',
            '"--use-fused-bilagrid" in sys.argv',
        )
        if all(item in source for item in required):
            return
        raise RuntimeError("Found a partial bilateral-grid trainer compatibility patch.")

    import_block = "import os\nimport time\n"
    if source.count(import_block) != 1:
        raise RuntimeError(
            "Unrecognized gsplat trainer imports; refusing an unsafe sys import edit."
        )
    source = source.replace(import_block, "import os\nimport sys\nimport time\n", 1)

    old_guard = (
        "    # Import BilateralGrid and related functions based on configuration\n"
        "    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:\n"
        "        if cfg.use_fused_bilagrid:\n"
    )
    new_guard = (
        "    # The preset parser runs before cli() consumes trailing CLI flags.\n"
        "    # Inspect argv as well, or --use-bilateral-grid can leave the class unbound.\n"
        "    requested_bilateral = (\n"
        "        cfg.use_bilateral_grid\n"
        "        or cfg.use_fused_bilagrid\n"
        '        or "--use-bilateral-grid" in sys.argv\n'
        '        or "--use-fused-bilagrid" in sys.argv\n'
        "    )\n"
        "    if requested_bilateral:\n"
        '        if cfg.use_fused_bilagrid or "--use-fused-bilagrid" in sys.argv:\n'
    )
    if source.count(old_guard) != 1:
        raise RuntimeError(
            "Unrecognized gsplat bilateral-grid import block; refusing an unsafe patch."
        )
    source = source.replace(old_guard, new_guard, 1)

    compile(source, str(trainer_path), "exec")
    trainer_path.write_text(source, encoding="utf-8")
