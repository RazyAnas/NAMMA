"""Subprocess entry for GPU stages that need their own Python state (fill, fixer)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_priors(sdir: Path, pri: dict, log):
    """Depth memory (MoGe-2) and masks for one scene; rewrites scene.json in place."""
    import numpy as np

    import scene_io

    cams, pts, meta = scene_io.read_scene(sdir)
    refs = dict(np.load(sdir / "refs.npz")) if (sdir / "refs.npz").is_file() else {}
    run_masks(sdir, cams, meta, pri, log)  # masks first: depth must never see the blanked vehicle
    if pri.get("depth"):
        import depth_memory

        rep = depth_memory.build_memory(cams, refs, sdir / "memory", device="cuda", log=log)
        meta["depth_memory"] = rep
        log("[priors] depth memory", json.dumps(rep))
        if pri.get("dense_init", True):
            fused = np.load(sdir / "memory" / "fused_points.npy")
            if len(fused):
                pts = np.concatenate([pts, fused])
    if not pri.get("sky_mask", pri.get("depth")):
        for c in cams:
            c.pop("sky_mask", None)
    scene_io.write_scene(sdir, cams, pts, meta)


def run_masks(sdir, cams, meta, pri, log):
    import numpy as np

    import scene_io

    if pri.get("masks"):
        import masks as masks_mod

        up = scene_io.world_up_from_cameras(cams)
        vehicle, rots = None, None
        if (sdir / "vehicle_erp.npy").is_file() and pri.get("views") == "rig":
            import pycolmap.panorama as P

            vehicle = np.load(sdir / "vehicle_erp.npy")
            rots = P.get_virtual_rotations(4, (-35.0, 0.0, 35.0))
        rep = masks_mod.build_masks(cams, sdir / "masks_dyn", up, device="cuda", dynamic=pri.get("dynamic", True),
                                    log=log, nadir_deg=float(pri.get("nadir_deg", 89.0)),
                                    vehicle_erp=vehicle, rotations=rots)
        meta["masks"] = rep
        log("[priors] masks", json.dumps(rep))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fill", "fixer", "export", "priors", "vehicle"])
    p.add_argument("--priors", default="{}")
    p.add_argument("--pano-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("/kaggle/temp/namma-exp/fixer"))
    p.add_argument("--max-views", type=int, default=160)
    a = p.parse_args()
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    if a.stage == "vehicle":
        import numpy as np

        import masks as masks_mod

        keep = masks_mod.detected_vehicle_mask(a.scene, device="cuda", log=log)
        if keep is not None:
            np.save(a.out, keep)
        return
    if a.stage == "priors":
        run_priors(a.scene, json.loads(a.priors), log)
        return
    if a.stage == "export":
        import export_game

        man = json.loads(a.manifest.read_text()) if a.manifest and a.manifest.is_file() else None
        rep = export_game.export(a.scene, a.model, a.out, a.pano_dir, man, log=log)
        print("[stage] export", json.dumps({k: v for k, v in rep.items() if k not in ("route", "ibl")}), flush=True)
        return
    import fill

    if a.stage == "fill":
        rep = fill.run_fill(a.scene, a.model, a.out, device="cuda", max_views=a.max_views, log=log)
    else:
        rep = fill.run_fixer_pass(a.scene, a.model, a.out, a.work, device="cuda", max_views=a.max_views, log=log)
    print("[stage]", a.stage, json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
