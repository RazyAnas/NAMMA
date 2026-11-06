"""Kaggle entry point for the panorama hypothesis tests (H0-H7).

Config (JSON in env NAMMA_EXP, or argv[1]) example:
  {"data": {"source": "ggps", "scene": "FTP", "first": 0, "count": 40, "pano_width": 4096},
   "experiments": [
     {"name": "H1-gt-rig", "poses": "gt", "views": "rig", "gpu": 1,
      "train": {"steps": 7000, "cap": 1000000}}]}

poses: legacy (12x60deg crops + hloc SfM + rig closure, the branch head),
       gt (openMVG ground truth from GGPS), pano_sfm (pycolmap.panorama, rig-aware).
views: legacy (prepare_panorama 12x60deg horizon band) or rig (4 yaw x 3 pitch, 90deg).
Experiments run in parallel, one per GPU. Outputs: /kaggle/working/out/<name>/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORK = Path(os.environ.get("NAMMA_WORK", "/kaggle/temp/namma-exp"))
OUT = Path(os.environ.get("NAMMA_OUT", "/kaggle/working/out"))
T0 = time.time()


def say(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def sh(cmd, env=None, log=None, check=True):
    say("$", " ".join(map(str, cmd))[:300])
    if log:
        with open(log, "a", encoding="utf-8") as f:
            p = subprocess.run(list(map(str, cmd)), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.run(list(map(str, cmd)), env=env, text=True)
    if check and p.returncode:
        tail = Path(log).read_text(errors="replace")[-3000:] if log else ""
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(map(str, cmd))[:200]}\n{tail}")
    return p.returncode


# ---------------------------------------------------------------- setup
def install_env():
    marker = WORK / ".env-ok"
    if marker.exists():
        return
    WORK.mkdir(parents=True, exist_ok=True)
    pkgs = ["gsplat==1.5.3", "pycolmap==4.2.1", "plyfile", "torchmetrics[image]", "ninja", "huggingface_hub"]
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    if os.environ.get("NAMMA_NEED_MOGE"):
        sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "git+https://github.com/microsoft/MoGe.git@0744441"])
        sh([sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"])
    # compile gsplat's CUDA kernels once, before parallel GPU jobs start
    sh([sys.executable, "-c",
        "import torch; from gsplat import rasterization; d='cuda';"
        "m=torch.zeros(1,3,device=d); m[0,2]=2;"
        "rasterization(m, torch.tensor([[1.,0,0,0]],device=d), torch.full((1,3),.1,device=d),"
        "torch.ones(1,device=d), torch.ones(1,3,device=d), torch.eye(4,device=d)[None],"
        "torch.tensor([[[64.,0,32],[0,64,32],[0,0,1]]],device=d), 64, 64); print('gsplat ok')"])
    marker.touch()


# ----------------------------------------------------------------- data
def fetch_ggps(scene: str) -> Path:
    from huggingface_hub import hf_hub_download

    root = WORK / f"ggps-{scene}"
    if (root / ".ok").exists():
        return root
    z = hf_hub_download("Insta360-Research/GGPS", f"datasets/{scene}.zip", repo_type="model",
                        local_dir=str(WORK / "hf"))
    say("downloaded", z, round(Path(z).stat().st_size / 1e9, 2), "GB")
    with zipfile.ZipFile(z) as f:
        f.extractall(root)
    Path(z).unlink()
    (root / ".ok").touch()
    return root


def ggps_inputs(cfg) -> dict:
    """Pick panoramas, resize to pano_width, return paths + GT poses + GT points."""
    import numpy as np
    from PIL import Image
    import scene_io

    root = fetch_ggps(cfg.get("scene", "FTP"))
    sfm_json = sorted(root.rglob("sfm_data*.json"), key=lambda p: -p.stat().st_size)[0]
    images_dir = next(p for p in root.rglob("images") if p.is_dir())
    all_panos = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    first, count = int(cfg.get("first", 0)), int(cfg.get("count", 40))
    picked = all_panos[first:first + count]
    width = int(cfg.get("pano_width", 4096))
    pano_dir = WORK / f"panos-{cfg.get('scene', 'FTP')}-{first}-{count}-{width}"
    if not (pano_dir / ".ok").exists():
        pano_dir.mkdir(parents=True, exist_ok=True)
        for name in picked:
            im = Image.open(images_dir / name).convert("RGB")
            im.resize((width, width // 2), Image.LANCZOS).save(pano_dir / (Path(name).stem + ".jpg"), quality=95)
        (pano_dir / ".ok").touch()
    poses = scene_io.read_openmvg_poses(json.loads(sfm_json.read_text()))
    gt = {Path(n).stem + ".jpg": poses[n] for n in picked if n in poses}
    ply = next(root.rglob("colorized.ply"), None)
    sfm = json.loads(sfm_json.read_text())
    refs = scene_io.openmvg_points_by_view(sfm)
    gt_refs = {Path(n).stem: refs[n] for n in picked if n in refs}
    say(f"GGPS {len(picked)} panoramas, {len(gt)} with GT poses, pano dir {pano_dir}")
    return {"pano_dir": pano_dir, "gt_poses": gt, "gt_ply": ply, "gt_refs": gt_refs}


# ---------------------------------------------------------------- views
RIG = {"num_steps_yaw": 4, "pitches_deg": (-35.0, 0.0, 35.0), "hfov_deg": 90.0, "vfov_deg": 90.0}


def rig_views(pano_dir: Path, out_dir: Path):
    import pycolmap.panorama as P

    opts = P.PanoRenderOptions(**RIG)
    names = sorted(p.name for p in pano_dir.glob("*.jpg"))
    proc = P.render_perspective_images(names, pano_dir, out_dir / "images", out_dir / "masks", opts, False)
    cam = proc.rig_config.cameras[0].camera
    import numpy as np

    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    return proc.cams_from_pano_rotation, K, int(cam.width), int(cam.height)


def legacy_views(pano_dir: Path, out_dir: Path):
    """Same crops as prepare_panorama.py (12 x 60 deg, pitch 0, 1280x960)."""
    import cv2
    import numpy as np
    import prepare_panorama as pp
    import scene_io

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    w, h, hfov, n = 1280, 960, 60.0, 12
    rot, K = [], None
    names = sorted(p.name for p in pano_dir.glob("*.jpg"))
    for pi, name in enumerate(names):
        img = cv2.imread(str(pano_dir / name))
        for vi in range(n):
            mx, my, f = pp.perspective_maps(w, h, img.shape[1], img.shape[0], 360.0 * vi / n, hfov)
            cv2.imwrite(str(out_dir / "images" / f"p{pi:05d}_v{vi:02d}.jpg"),
                        cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    rot = [scene_io.legacy_crop_rotation(360.0 * vi / n) for vi in range(n)]
    return names, rot, K, w, h


# ---------------------------------------------------------------- poses
def scene_gt(exp, inputs, sdir: Path):
    import numpy as np
    import scene_io

    gt = inputs["gt_poses"]
    if exp["views"] == "rig":
        rots, K, w, h = rig_views(inputs["pano_dir"], sdir)
        cams = scene_io.rig_views_from_poses(gt, sdir / "images", rots, K, w, h)
    else:
        names, rots, K, w, h = legacy_views(inputs["pano_dir"], sdir)
        cams = []
        for pi, name in enumerate(names):
            for vi, R in enumerate(rots):
                T = np.eye(4)
                T[:3, :3] = R
                cams.append({"name": f"p{pi:05d}_v{vi:02d}.jpg",
                             "path": str(sdir / "images" / f"p{pi:05d}_v{vi:02d}.jpg"),
                             "pano": f"p{pi:05d}", "width": w, "height": h, "K": K,
                             "w2c": T @ gt[name]})
        cams = scene_io.assign_splits(cams)
    pts = scene_io.read_ply_points(inputs["gt_ply"])
    # keep only points near the chosen panoramas (the GT cloud covers all 354)
    from scipy.spatial import cKDTree

    C = np.stack([scene_io.camera_centre(T) for T in gt.values()])
    span = np.linalg.norm(C.max(0) - C.min(0)) + 1e-6
    d, _ = cKDTree(C).query(pts[:, :3])
    pts = pts[d < 1.5 * span]
    check = scene_io.colour_check(cams, pts)
    say("GT colour check", json.dumps(check))
    refs = dict(inputs["gt_refs"])
    if exp["views"] == "legacy":
        order = sorted(p.stem for p in inputs["pano_dir"].glob("*.jpg"))
        refs = {f"p{i:05d}": refs[n] for i, n in enumerate(order) if n in refs}
    return cams, pts, {"poses": "gt", "views": exp["views"], "colour_check": check}, refs


def scene_pano_sfm(exp, inputs, sdir: Path):
    import pycolmap
    import pycolmap.panorama as P
    import scene_io

    opts = P.PanoramaReconstructionOptions(
        matcher=P.Matcher(exp.get("matcher", "sequential")),
        mapper=P.Mapper(exp.get("mapper", "incremental")),
        use_gpu=False, show_progress=False)
    recs = P.reconstruct(inputs["pano_dir"], sdir, opts)
    if not recs:
        raise RuntimeError("pycolmap.panorama produced no reconstruction")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    say("pano_sfm:", rec.summary().replace("\n", " | ")[:600])
    cams, pts = scene_io.from_pycolmap(rec, sdir / "images", scene_io.pano_of_name)
    refs = scene_io.pycolmap_points_by_pano(rec, scene_io.pano_of_name)
    within = scene_io.within_pano_spread(cams)
    say("pano_sfm within-panorama centre spread", within)
    return cams, pts, refs, {"poses": "pano_sfm", "views": "rig", "registered_views": len(cams),
                       "points3D": int(len(rec.points3D)), "models": len(recs),
                       "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error())}


def scene_legacy(exp, inputs, sdir: Path, log: Path):
    """Branch-head behaviour: 12x60deg crops, hloc ALIKED+LightGlue SfM, median rig closure."""
    import pycolmap
    import scene_io

    legacy_views(inputs["pano_dir"], sdir)
    vm = {"fx_initial": 1108.5125, "fy_initial": 1108.5125, "cx": 640.0, "cy": 480.0}
    (sdir / "views_manifest.json").write_text(json.dumps(vm))
    venv = WORK / "sfm-env"
    py = venv / "bin" / "python"
    if not py.exists():
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"], log=log)
        sh(["uv", "venv", "-q", "-p", "3.10", str(venv)], log=log)
        sh(["uv", "pip", "install", "--python", str(py), "torch==2.1.2", "torchvision==0.16.2",
            "--index-url", "https://download.pytorch.org/whl/cu118"], log=log)
        hloc = WORK / "Hierarchical-Localization"
        if not hloc.exists():
            sh(["git", "clone", "-q", "--recursive", "https://github.com/cvg/Hierarchical-Localization", str(hloc)], log=log)
        sh(["uv", "pip", "install", "--python", str(py), "numpy<2", "opencv-python-headless", "Pillow",
            "pycolmap", "-e", str(hloc), "lightglue @ git+https://github.com/cvg/LightGlue"], log=log)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", 0))
    sh([py, HERE / "sfm_hloc.py", "--images", sdir / "images", "--output", sdir / "sfm",
        "--sequence-window", "18", "--retrieval", "0"], env=env, log=log)
    rec = pycolmap.Reconstruction(str(sdir / "sfm" / "model"))
    cams, pts = scene_io.from_pycolmap(rec, sdir / "images", scene_io.pano_of_name)
    rep = json.loads((sdir / "sfm" / "sfm_report.json").read_text())
    rep.pop("rig_closure", None)
    refs = scene_io.pycolmap_points_by_pano(rec, scene_io.pano_of_name)
    return cams, pts, {"poses": "legacy", "views": "legacy", **rep}, refs


# ------------------------------------------------------------- one experiment
def run_experiment(exp, inputs):
    import scene_io

    name = exp["name"]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    sdir = WORK / "scenes" / name
    t = time.time()
    try:
        if exp["poses"] == "gt":
            cams, pts, meta, refs = scene_gt(exp, inputs, sdir)
        elif exp["poses"] == "pano_sfm":
            cams, pts, refs, meta = scene_pano_sfm(exp, inputs, sdir)
        elif exp["poses"] == "legacy":
            cams, pts, meta, refs = scene_legacy(exp, inputs, sdir, log)
        else:
            raise ValueError(exp["poses"])
        meta["scene_minutes"] = round((time.time() - t) / 60, 1)
        pri = exp.get("priors", {})
        dev = "cuda:0"
        if pri.get("depth"):
            import depth_memory
            import torch

            torch.cuda.set_device(int(exp.get("gpu", 0)))
            rep_d = depth_memory.build_memory(cams, refs, sdir / "memory", device=f"cuda:{int(exp.get('gpu', 0))}",
                                              log=lambda *a: say(name, *a))
            meta["depth_memory"] = rep_d
            say(name, "depth memory", json.dumps(rep_d))
            if pri.get("dense_init", True):
                import numpy as np

                fused = np.load(sdir / "memory" / "fused_points.npy")
                pts = np.concatenate([pts, fused]) if len(fused) else pts
        if pri.get("masks"):
            import masks as masks_mod

            up = scene_io.world_up_from_cameras(cams)
            rep_m = masks_mod.build_masks(cams, sdir / "masks_dyn", up, device=f"cuda:{int(exp.get('gpu', 0))}",
                                          dynamic=pri.get("dynamic", True), log=lambda *a: say(name, *a))
            meta["masks"] = rep_m
        if not pri.get("sky_mask", pri.get("depth")):
            for c in cams:
                c.pop("sky_mask", None)
        scene_io.write_scene(sdir, cams, pts, meta)
        (out / "scene_meta.json").write_text(json.dumps(
            {**meta, "views": len(cams), "init_points": int(len(pts)),
             "within_pano_centre_spread": scene_io.within_pano_spread(cams)}, indent=2))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", 0))
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
        tr = exp.get("train", {})
        args = [sys.executable, HERE / "train_splats.py", "--scene", sdir, "--out", sdir / "model",
                "--steps", tr.get("steps", 7000), "--cap", tr.get("cap", 1_000_000),
                "--long-side", tr.get("long_side", 960), "--sh-degree", tr.get("sh_degree", 1),
                "--depth-weight", tr.get("depth_weight", 0.0), "--aniso-weight", tr.get("aniso_weight", 0.0),
                "--max-minutes", tr.get("max_minutes", 240)]
        if tr.get("sky"):
            args.append("--sky")
        if tr.get("carve"):
            args.append("--carve")
        sh(args, env=env, log=log)
        sh([sys.executable, HERE / "offpath.py", "--model", sdir / "model", "--scene", sdir, "--out", out],
           env=env, log=log)
        shutil.copy2(sdir / "model" / "metrics.json", out / "metrics.json")
        if exp.get("keep_ply", True):
            shutil.copy2(sdir / "model" / "model.ply", out / "model.ply")
        status = {"name": name, "status": "ok"}
    except Exception as exc:
        status = {"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[-4000:]}
    status["minutes"] = round((time.time() - t) / 60, 1)
    (out / "status.json").write_text(json.dumps(status, indent=2))
    say(name, status["status"], status.get("error", "")[:500])
    return status


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NAMMA_EXP", "{}")
    cfg = json.loads(Path(raw).read_text() if raw.endswith(".json") and Path(raw).is_file() else raw)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    if any(e.get("priors", {}).get("depth") for e in cfg["experiments"]):
        os.environ["NAMMA_NEED_MOGE"] = "1"
    install_env()
    data = cfg.get("data", {})
    inputs = ggps_inputs(data) if data.get("source", "ggps") == "ggps" else None
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda e: run_experiment(e, inputs), cfg["experiments"]))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    say("ALL DONE", json.dumps(results))


if __name__ == "__main__":
    main()
