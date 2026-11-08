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


def sh(cmd, env=None, log=None, check=True, timeout=None):
    say("$", " ".join(map(str, cmd))[:300])
    try:
        if log:
            with open(log, "a", encoding="utf-8") as f:
                p = subprocess.run(list(map(str, cmd)), env=env, stdout=f, stderr=subprocess.STDOUT, text=True,
                                   timeout=timeout)
        else:
            p = subprocess.run(list(map(str, cmd)), env=env, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timed out after {timeout / 60:.0f} min: {' '.join(map(str, cmd))[:200]}") from None
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
    pkgs = ["gsplat==1.5.3", "pycolmap==4.2.1", "plyfile", "torchmetrics[image]", "ninja", "huggingface_hub",
            "trimesh", "scikit-image", "fast-simplification", "piexif", "xatlas"]
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    if os.environ.get("NAMMA_NEED_FILL"):
        # Kaggle's torchao 0.10 makes peft refuse every LoRA (video pipeline World job hit the same)
        sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
        sh([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.33", "accelerate", "peft", "opencv-python-headless",
            "ftfy"])  # Wan VACE prompt cleaning imports ftfy (H10 densify crashed without it)
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


def remove_capture_vehicle(inputs: dict, enabled: bool = True) -> dict:
    """Blank the capture car/bike in every panorama before SfM and training.

    The vehicle never changes along the route (masks.static_vehicle_mask). Its
    pixels are replaced by a smooth, feature-free fill so SfM cannot match the
    bonnet between frames; training ignores those pixels via the same mask,
    so the road under the vehicle is rebuilt from neighbouring panoramas and,
    where none saw it, by the hole-only generative fill.
    """
    import cv2
    import numpy as np
    import masks as masks_mod
    from PIL import Image

    if not enabled:
        return inputs
    src = inputs["pano_dir"]
    keep = masks_mod.static_vehicle_mask(src)
    det = None
    det_path = WORK / f"vehicle-det-{src.name}.npy"
    if not det_path.exists():
        venv = os.environ.copy()
        venv["CUDA_VISIBLE_DEVICES"] = "0"
        sh([sys.executable, HERE / "stage.py", "vehicle", "--scene", src, "--model", src, "--out", det_path],
           env=venv, check=False)
    if det_path.exists():
        det = np.load(det_path)
    if det is not None:
        import cv2

        det = cv2.resize(det.astype("uint8"), (keep.shape[1], keep.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        keep = keep & cv2.erode(det.astype("uint8"), np.ones((7, 7), np.uint8)).astype(bool)
    frac = 1 - float(keep.mean())
    inputs["vehicle_erp"] = keep
    Image.fromarray((keep * 255).astype("uint8")).save(OUT / "vehicle_mask_erp.png")
    first = sorted(src.glob("*.jpg"))[0]
    pano = np.asarray(Image.open(first).convert("RGB").resize((keep.shape[1], keep.shape[0])))
    over = np.where(keep[..., None], pano, (0.45 * pano + 0.55 * np.array([255, 0, 255])).astype("uint8"))
    Image.fromarray(over.astype("uint8")).save(OUT / "vehicle_mask_overlay.jpg", quality=88)
    say(f"capture vehicle covers {frac:.3%} of each panorama")
    if frac < 0.002:
        return inputs
    dst = blank_vehicle(src, src.parent / (src.name + "-novehicle"), keep)
    inputs["pano_dir"] = dst
    return inputs


def gps_exif(lat, lon, alt: float = 35.0) -> bytes:
    """EXIF with GPS lat/lon/alt only. COLMAP stores a position prior only when all three exist."""
    import piexif

    if lat is None or lon is None:
        return b""

    def dms(x):
        x = abs(float(x))
        d = int(x); m = int((x - d) * 60); s_ = (x - d - m / 60) * 3600
        return ((d, 1), (m, 1), (int(round(s_ * 10000)), 10000))

    gps = {piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S", piexif.GPSIFD.GPSLatitude: dms(lat),
           piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W", piexif.GPSIFD.GPSLongitude: dms(lon),
           piexif.GPSIFD.GPSAltitudeRef: 0, piexif.GPSIFD.GPSAltitude: (int(round(alt * 100)), 100)}
    return piexif.dump({"GPS": gps})


def blank_vehicle(src: Path, dst: Path, keep, width: int | None = None) -> Path:
    """Write panoramas with the capture vehicle replaced by a featureless fill, keeping GPS EXIF."""
    import cv2
    import numpy as np
    from PIL import Image

    if (dst / ".ok").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.jpg")):
        pil = Image.open(p)
        exif = pil.info.get("exif", b"")
        if width:
            pil = pil.convert("RGB").resize((width, width // 2), Image.LANCZOS)
        img = np.asarray(pil.convert("RGB"))[..., ::-1].copy()
        if keep is not None:
            m = cv2.resize((~keep).astype("uint8"), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = cv2.dilate(m, np.ones((15, 15), np.uint8))
            small = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8), interpolation=cv2.INTER_AREA)
            ms = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            fill = cv2.resize(cv2.inpaint(small, ms, 7, cv2.INPAINT_TELEA), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            fill = cv2.GaussianBlur(fill, (31, 31), 0)
            img = np.where(m[..., None] > 0, fill, img)
        Image.fromarray(img[..., ::-1]).save(dst / p.name, quality=95, exif=exif)
    (dst / ".ok").touch()
    return dst


def hires_rig_views(exp, inputs, sdir: Path, cams: list) -> list:
    """Re-render the SfM rig views from higher-resolution panoramas for training (same rotations)."""
    import numpy as np
    import pycolmap.panorama as P

    tw = int(exp.get("train_pano_width") or 0)
    raw = inputs.get("raw_dir")
    if not tw or raw is None:
        return cams
    hr = blank_vehicle(Path(raw), WORK / f"{Path(raw).name}-{tw}-novehicle", inputs.get("vehicle_erp"), width=tw)
    names = sorted(p.name for p in hr.glob("*.jpg"))
    out = sdir / "images_hr"
    if not (out / ".ok").exists():
        P.render_perspective_images(names, hr, out, sdir / "masks_hr", P.PanoRenderOptions(**RIG), False)
        (out / ".ok").touch()
    first = cams[0]
    from PIL import Image

    w_new = Image.open(out / first["name"]).size[0]
    s = w_new / first["width"]
    for c in cams:
        c["path"] = str(out / c["name"])
        c["width"], c["height"] = int(round(c["width"] * s)), int(round(c["height"] * s))
        K = np.array(c["K"], dtype=np.float64)
        K[:2] *= s
        c["K"] = K
    say(f"training views re-rendered from {tw}-px panoramas ({w_new}px views, x{s:.1f})")
    return cams


def panoramax_inputs(cfg) -> dict:
    """Street-level 360 route from Panoramax (open data, no key), resized to pano_width."""
    from PIL import Image
    import fetch_panoramax

    raw = WORK / f"pmx-{cfg['collection'][:8]}-{cfg.get('start', 0)}-{cfg.get('count', 40)}"
    fetch_panoramax.fetch(cfg["collection"], raw, int(cfg.get("count", 40)), float(cfg.get("spacing_m", 10.0)),
                          int(cfg.get("start", 0)), cfg.get("asset", "hd"), log=say)
    width = int(cfg.get("pano_width", 4096))
    pano_dir = WORK / f"{raw.name}-{width}"
    if not (pano_dir / ".ok").exists():
        pano_dir.mkdir(parents=True, exist_ok=True)
        man = json.loads((raw / "manifest.json").read_text())
        gps = {e["image"]: (e.get("lat"), e.get("lon")) for e in man["panoramas"]}
        for p in sorted(raw.glob("*.jpg")):
            Image.open(p).convert("RGB").resize((width, width // 2), Image.LANCZOS).save(
                pano_dir / p.name, quality=95, exif=gps_exif(*gps.get(p.name, (None, None))))
        (pano_dir / ".ok").touch()
    shutil.copy2(raw / "manifest.json", OUT / "route_manifest.json")
    return {"pano_dir": pano_dir, "raw_dir": raw, "gt_poses": None, "gt_ply": None, "gt_refs": {},
            "manifest": json.loads((raw / "manifest.json").read_text())}


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
    # initial points = the openMVG structure points the chosen panoramas observed,
    # coloured from colorized.ply (same coordinates; it also holds camera centres)
    from scipy.spatial import cKDTree

    cloud = scene_io.read_ply_points(inputs["gt_ply"], max_points=10_000_000)
    seen = np.unique(np.concatenate([v for v in inputs["gt_refs"].values()]), axis=0)
    d, idx = cKDTree(cloud[:, :3]).query(seen)
    pts = np.concatenate([seen, cloud[idx, 3:6]], 1).astype(np.float32)[d < 1e-4]
    say(f"GT init points observed by the chosen panoramas: {len(pts)}")
    check = scene_io.colour_check(cams, pts)
    say("GT colour check", json.dumps(check))
    refs = dict(inputs["gt_refs"])
    if exp["views"] == "legacy":
        order = sorted(p.stem for p in inputs["pano_dir"].glob("*.jpg"))
        refs = {f"p{i:05d}": refs[n] for i, n in enumerate(order) if n in refs}
    return cams, pts, {"poses": "gt", "views": exp["views"], "colour_check": check}, refs


import threading

_SFM_LOCK = threading.Lock()


def scene_pano_sfm(exp, inputs, sdir: Path):
    """Rig-aware SfM, computed once per input/options and shared by experiments in this job."""
    import pycolmap
    import pycolmap.panorama as P
    import scene_io

    key = f"{inputs['pano_dir'].name}-{exp.get('matcher', 'sequential')}-{exp.get('mapper', 'global')}"
    shared = WORK / "panosfm" / key
    with _SFM_LOCK:
        if not (shared / ".ok").exists():
            opts = P.PanoramaReconstructionOptions(
                matcher=P.Matcher(exp.get("matcher", "sequential")),
                mapper=P.Mapper(exp.get("mapper", "global")),
                use_gpu=False, show_progress=False)
            shared.mkdir(parents=True, exist_ok=True)
            P.reconstruct(inputs["pano_dir"], shared, opts)
            (shared / ".ok").touch()
    sdir.mkdir(parents=True, exist_ok=True)
    if not (sdir / "images").exists():
        (sdir / "images").symlink_to(shared / "images")
    recs = {}
    for d in sorted((shared / "sparse").iterdir()):
        if (d / "images.bin").is_file() or (d / "images.txt").is_file():
            recs[int(d.name)] = pycolmap.Reconstruction(str(d))
    if not recs:
        raise RuntimeError("pycolmap.panorama produced no reconstruction")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    say("pano_sfm:", rec.summary().replace("\n", " | ")[:600])
    cams, pts = scene_io.from_pycolmap(rec, sdir / "images", scene_io.pano_of_name)
    cams = hires_rig_views(exp, inputs, sdir, cams)
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
def run_variants(exp, tr, sdir, out, env, log):
    results = {}
    for var in exp["variants"]:
        vt = {**tr, **var.get("train", {})}
        vname = var["name"]
        vout = out / vname
        vout.mkdir(parents=True, exist_ok=True)
        mdir = sdir / f"model_{vname}"
        args = [sys.executable, HERE / "train_splats.py", "--scene", sdir, "--out", mdir,
                "--steps", vt.get("steps", 4000), "--cap", vt.get("cap", 1_000_000),
                "--long-side", vt.get("long_side", 960), "--sh-degree", vt.get("sh_degree", 1),
                "--depth-weight", vt.get("depth_weight", 0.0), "--aniso-weight", vt.get("aniso_weight", 0.0),
                "--max-minutes", vt.get("max_minutes", 60), "--packed", int(vt.get("packed", 1)),
                "--depth-loss", vt.get("depth_loss", "rel"), "--crop", vt.get("crop", 1024),
                "--scale-cap-frac", vt.get("scale_cap_frac", 0.3),
                "--opacity-reg", vt.get("opacity_reg", 0.0), "--noise-lr", vt.get("noise_lr", 5e5),
                "--init-opacity", vt.get("init_opacity", 0.1), "--panos", vt.get("panos", "")]
        if vt.get("sky"):
            args.append("--sky")
        if vt.get("carve"):
            args.append("--carve")
        t0 = time.time()
        try:
            sh(args, env=env, log=vout / "train.log")
            shutil.copy2(mdir / "metrics.json", vout / "metrics.json")
            sh([sys.executable, HERE / "offpath.py", "--model", mdir, "--scene", sdir, "--out", vout],
               env=env, log=vout / "train.log")
            results[vname] = json.loads((mdir / "metrics.json").read_text())
            sc = vout / "scorecard.json"
            if sc.is_file():
                results[vname]["scorecard"] = json.loads(sc.read_text())
        except Exception as exc:  # noqa: BLE001 - one broken variant must not hide the others
            results[vname] = {"error": f"{type(exc).__name__}: {exc}"[-1500:]}
        results[vname]["variant_minutes"] = round((time.time() - t0) / 60, 1)
        shutil.rmtree(mdir, ignore_errors=True)
        (out / "variants.json").write_text(json.dumps(results, indent=2))
        say(exp["name"], vname, json.dumps({k: results[vname].get(k) for k in ("psnr", "ssim", "lpips", "error")})[:400])
    return results


def run_experiment(exp, inputs):
    import scene_io

    name = exp["name"]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    sdir = WORK / "scenes" / name
    t = time.time()
    export_error = None
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
        if pri.get("masks") and pri.get("vehicle", True) and exp["views"] == "rig" and inputs.get("vehicle_erp") is not None:
            import numpy as np

            np.save(sdir / "vehicle_erp.npy", inputs["vehicle_erp"])
        if pri.get("depth") or pri.get("masks"):
            # GPU priors in their own process (CUDA lazy init is not thread-safe across experiments)
            import numpy as np

            sdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sdir / "refs.npz", **{k: v for k, v in refs.items()})
            scene_io.write_scene(sdir, cams, pts, meta)
            penv = os.environ.copy()
            penv["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", 0))
            sh([sys.executable, HERE / "stage.py", "priors", "--scene", sdir, "--model", sdir, "--out", sdir,
                "--priors", json.dumps({**pri, "views": exp["views"], "pano_dir": str(inputs["pano_dir"])})],
               env=penv, log=log)
            cams, pts, meta = scene_io.read_scene(sdir)
            if (sdir / "vehicle_erp.npy").is_file():
                from PIL import Image

                Image.fromarray((np.load(sdir / "vehicle_erp.npy") * 255).astype("uint8")).save(out / "vehicle_mask_erp.png")
        scene_io.write_scene(sdir, cams, pts, meta)
        (out / "scene_meta.json").write_text(json.dumps(
            {**meta, "views": len(cams), "init_points": int(len(pts)),
             "within_pano_centre_spread": scene_io.within_pano_spread(cams)}, indent=2))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", 0))
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
        tr = exp.get("train", {})
        if exp.get("variants"):
            # ablation: several short trainings on the same scene and priors, one switch changed per variant
            status = {"name": name, "status": "ok", "variants": run_variants(exp, tr, sdir, out, env, log)}
            status["minutes"] = round((time.time() - t) / 60, 1)
            (out / "status.json").write_text(json.dumps(status, indent=2))
            say(name, "variants done")
            return status
        args = [sys.executable, HERE / "train_splats.py", "--scene", sdir, "--out", sdir / "model",
                "--steps", tr.get("steps", 7000), "--cap", tr.get("cap", 1_000_000),
                "--long-side", tr.get("long_side", 960), "--sh-degree", tr.get("sh_degree", 1),
                "--depth-weight", tr.get("depth_weight", 0.0), "--aniso-weight", tr.get("aniso_weight", 0.0),
                "--max-minutes", tr.get("max_minutes", 240),
                "--packed", int(tr.get("packed", 1)), "--depth-loss", tr.get("depth_loss", "rel"),
                "--crop", tr.get("crop", 1024), "--scale-cap-frac", tr.get("scale_cap_frac", 0.3),
                "--opacity-reg", tr.get("opacity_reg", 0.0), "--noise-lr", tr.get("noise_lr", 5e5),
                "--init-opacity", tr.get("init_opacity", 0.1), "--panos", tr.get("panos", "")]
        if tr.get("sky"):
            args.append("--sky")
        if tr.get("carve"):
            args.append("--carve")
        args += ["--snapshot", out / "snapshot.pt"]
        sh(args, env=env, log=log)
        (out / "snapshot.pt").unlink(missing_ok=True)
        model_dir, scene_now = sdir / "model", sdir
        shutil.copy2(model_dir / "metrics.json", out / "metrics_base.json")
        if (model_dir / "fit_sheet.jpg").is_file():
            shutil.copy2(model_dir / "fit_sheet.jpg", out / "fit_sheet.jpg")
        sh([sys.executable, HERE / "offpath.py", "--model", model_dir, "--scene", scene_now, "--out", out / "base"],
           env=env, log=log)
        # optional refinement rounds: hole-only generative fill, then masked Fixer
        rounds = [("fill", r) for r in range(int(exp.get("fill", {}).get("rounds", 0)))]
        rounds += [("fixer", r) for r in range(int(exp.get("fixer", {}).get("rounds", 0)))]
        rounds += [("densify", r) for r in range(int(exp.get("densify", {}).get("rounds", 0)))]
        round_errors = {}
        for kind, r in rounds:
            try:
                opts = exp.get(kind, {})
                rdir = sdir / f"{kind}{r}"
                st = [sys.executable, HERE / "stage.py", kind, "--scene", scene_now, "--model", model_dir,
                      "--out", rdir, "--max-views", opts.get("max_views", 160)]
                stage_min = float(opts.get("stage_minutes", 60))
                if kind == "densify":
                    st += ["--max-minutes", stage_min]
                if (OUT / "route_manifest.json").is_file():
                    st += ["--manifest", OUT / "route_manifest.json"]
                # backstop: H12 densify ran 10 h (21 min per clip) and ate the 12 h Kaggle limit before export
                sh(st, env=env, log=log, timeout=60 * (stage_min + 45))
                for rep_name in ("fill_report.json", "fixer_report.json", "densify_report.json"):
                    if (rdir / rep_name).is_file():
                        shutil.copy2(rdir / rep_name, out / f"{kind}{r}_{rep_name}")
                if (rdir / "sheet").is_dir():
                    shutil.copytree(rdir / "sheet", out / f"{kind}{r}_sheet", dirs_exist_ok=True)
                new_model = sdir / f"model_{kind}{r}"
                ft = [sys.executable, HERE / "train_splats.py", "--scene", rdir / "scene", "--out", new_model,
                      "--steps", opts.get("steps", 3000), "--cap", tr.get("cap", 1_000_000),
                      "--long-side", tr.get("long_side", 960), "--sh-degree", tr.get("sh_degree", 1),
                      "--depth-weight", tr.get("depth_weight", 0.0), "--aniso-weight", tr.get("aniso_weight", 0.0),
                      "--init-ckpt", model_dir / "splats.pt", "--lr-scale", opts.get("lr_scale", 0.5),
                      "--max-minutes", opts.get("max_minutes", 90),
                "--packed", int(tr.get("packed", 1)), "--depth-loss", tr.get("depth_loss", "rel"),
                "--crop", tr.get("crop", 1024), "--scale-cap-frac", tr.get("scale_cap_frac", 0.3),
                "--opacity-reg", tr.get("opacity_reg", 0.0), "--noise-lr", tr.get("noise_lr", 5e5),
                "--init-opacity", tr.get("init_opacity", 0.1), "--panos", tr.get("panos", "")]
                if tr.get("sky"):
                    ft.append("--sky")
                if tr.get("carve"):
                    ft.append("--carve")
                sh(ft, env=env, log=log)
                model_dir, scene_now = new_model, rdir / "scene"
                shutil.copy2(model_dir / "metrics.json", out / f"metrics_{kind}{r}.json")
            except Exception as exc:  # noqa: BLE001 - keep the last good model and still export
                msg = f"{type(exc).__name__}: {exc}"
                round_errors[f"{kind}{r}"] = msg if len(msg) < 1800 else msg[:400] + " ... " + msg[-1400:]
                say(name, f"{kind}{r} failed, keeping the previous model:", str(exc)[-300:])
        if round_errors:
            (out / "round_errors.json").write_text(json.dumps(round_errors, indent=2))
        sh([sys.executable, HERE / "offpath.py", "--model", model_dir, "--scene", scene_now, "--out", out],
           env=env, log=log)
        shutil.copy2(model_dir / "metrics.json", out / "metrics.json")
        if exp.get("keep_ply", True):
            shutil.copy2(model_dir / "model.ply", out / "model.ply")
        if (model_dir / "carve.pt").is_file() and exp.get("keep_ply", True):
            shutil.copy2(model_dir / "carve.pt", out / "carve.pt")
        if exp.get("keep_scene", False):
            shutil.copy2(scene_now / "scene.json", out / "scene.json")
        if exp.get("export"):
            args_e = [sys.executable, HERE / "stage.py", "export", "--scene", scene_now, "--model", model_dir,
                      "--out", out / "game", "--pano-dir", inputs["pano_dir"],
                      "--atlas", str(exp.get("atlas", 8192))]
            if (OUT / "route_manifest.json").is_file():
                args_e += ["--manifest", OUT / "route_manifest.json"]
            try:  # the trained model and metrics stand on their own; a stuck export must not eat the 12 h job
                sh(args_e, env=env, log=log, timeout=60 * exp.get("export_minutes", 60))
            except Exception as exc:  # noqa: BLE001
                export_error = f"{type(exc).__name__}: {exc}"[-1500:]
                say(name, "export failed:", export_error[:300])
        status = {"name": name, "status": "ok"}
        if export_error:
            status["export_error"] = export_error
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
    if any(e.get("priors", {}).get("depth") or e.get("fill") for e in cfg["experiments"]):
        os.environ["NAMMA_NEED_MOGE"] = "1"
    if any(e.get("fill") or e.get("densify") for e in cfg["experiments"]):
        os.environ["NAMMA_NEED_FILL"] = "1"
    install_env()
    data = cfg.get("data", {})
    inputs = ggps_inputs(data) if data.get("source", "ggps") == "ggps" else panoramax_inputs(data)
    inputs = remove_capture_vehicle(inputs, bool(data.get("remove_vehicle", True)))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda e: run_experiment(e, inputs), cfg["experiments"]))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    say("ALL DONE", json.dumps(results))


if __name__ == "__main__":
    main()
