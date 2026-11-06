"""Subprocess entry for GPU stages that need their own Python state (fill, fixer)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fill", "fixer"])
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("/kaggle/temp/namma-exp/fixer"))
    p.add_argument("--max-views", type=int, default=160)
    a = p.parse_args()
    import fill

    log = lambda *x: print(*x, flush=True)  # noqa: E731
    if a.stage == "fill":
        rep = fill.run_fill(a.scene, a.model, a.out, device="cuda", max_views=a.max_views, log=log)
    else:
        rep = fill.run_fixer_pass(a.scene, a.model, a.out, a.work, device="cuda", max_views=a.max_views, log=log)
    print("[stage]", a.stage, json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
