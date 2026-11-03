import subprocess
import sys
from pathlib import Path

from gsplat_trainer_compat import patch_gsplat_bilateral_grid_import


TRAINER_SOURCE = """import multiprocessing as mp

class Config:
    use_bilateral_grid = True
    use_fused_bilagrid = False

def set_random_seed(seed):
    pass

class Runner:
    def __init__(self, local_rank, world_rank, world_size, cfg):
        set_random_seed(42 + local_rank)
        self.cfg = cfg
        self.grid = BilateralGrid()

def _spawned_worker(result_queue):
    try:
        Runner(0, 0, 1, Config())
    except Exception as exc:
        result_queue.put(f"{type(exc).__name__}: {exc}")
    else:
        result_queue.put("ok")

def main():
    cfg = Config()
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

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_spawned_worker, args=(result_queue,))
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("spawned worker timed out")
    if process.exitcode != 0:
        raise RuntimeError(f"spawned worker exited with code {process.exitcode}")
    result = result_queue.get(timeout=5)
    result_queue.close()
    if result != "ok":
        raise RuntimeError(result)

if __name__ == "__main__":
    main()
"""


def test_bilateral_grid_import_is_visible_inside_spawned_runner_workers(tmp_path: Path):
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
    assert "NAMMA: publish bilateral-grid helpers inside every spawned Runner worker." in patched
    assert '"BilateralGrid": BilateralGrid' in patched
    assert '"total_variation_loss": total_variation_loss' in patched

    result = subprocess.run(
        [sys.executable, str(trainer)],
        cwd=examples,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # Applying the compatibility patch twice must be harmless.
    patch_gsplat_bilateral_grid_import(gsplat_root)
    assert trainer.read_text(encoding="utf-8") == patched
