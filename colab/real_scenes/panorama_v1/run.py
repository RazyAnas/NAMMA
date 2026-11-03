"""NAMMA Panorama v1 Kaggle GPU job.

This is a standalone panorama-to-Gaussian-Splat runner. It does not import, call,
patch, or write into colab/real_scenes/v3 or the video capture pipeline.
"""
import glob, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

JOB_CFG = json.loads(os.environ.get("JOB_CFG", "{}"))
ROOT = Path("/kaggle/temp/namma-panorama-v1")
OUT = Path("/kaggle/working/out")
INPUT = Path("/kaggle/input")
T0 = time.time()
ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent


def find_code_root() -> Path:
    required = ("prepare_panorama.py", "pairing.py", "sfm_hloc.py", "make_tour.py")
    def complete(candidate: Path) -> bool:
        return all((candidate / name).is_file() for name in required) and (
            candidate / "fused_ssim_shim" / "fused_ssim.py"
        ).is_file()

    matches = [p.parent for p in INPUT.glob("**/prepare_panorama.py")]
    for candidate in matches:
        if complete(candidate):
            return candidate
    if complete(HERE):
        return HERE
    raise RuntimeError(
        "Panorama pipeline helpers are incomplete in the Kaggle input dataset. "
        "Required files: " + ", ".join(required) + ", fused_ssim_shim/fused_ssim.py"
    )


def say(*args):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *args, flush=True)


def run(command, cwd=None, env=None, log=None):
    say("$", " ".join(map(str, command))[:240])
    if log:
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=f,
                               stderr=subprocess.STDOUT, text=True)
        tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        for line in tail:
            print(line, flush=True)
    else:
        p = subprocess.run(list(map(str, command)), cwd=cwd, env=env, text=True)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str, command))}")


def safe_extract(bundle: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(bundle) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in panorama bundle: {item.filename}")
        z.extractall(destination)
    if not (destination / "manifest.json").is_file():
        # Accept a single nested top-level directory from Finder's Compress action.
        candidates = list(destination.glob("*/manifest.json"))
        if len(candidates) == 1:
            nested = candidates[0].parent
            for item in nested.iterdir():
                target = destination / item.name
                if target.exists():
                    raise RuntimeError(f"Ambiguous nested bundle: {item.name}")
                shutil.move(str(item), str(target))
            nested.rmdir()
    if not (destination / "manifest.json").is_file():
        raise RuntimeError("The uploaded ZIP must contain manifest.json at its root.")


def find_bundle() -> Path:
    # Kaggle may expose an uploaded ZIP as a regular file or unpack its members.
    # Use a neutral extension for new datasets and accept legacy .zip files too.
    candidates = (
        list(INPUT.glob("**/bundle.namma"))
        + list(INPUT.glob("**/bundle.zip"))
        + list(INPUT.glob("**/*.zip"))
    )
    candidates = [p for p in candidates if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_size)

    # Fallback when Kaggle has unpacked the archive into a directory.
    for manifest_path in INPUT.glob("**/manifest.json"):
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            panoramas = manifest.get("panoramas", [])
            if not 8 <= len(panoramas) <= 200:
                continue
            archive_path = ROOT / "bundle-repacked-from-kaggle-input.namma"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("manifest.json", manifest_bytes)
                root = manifest_path.parent.resolve()
                for entry in panoramas:
                    relative = Path(str(entry.get("image", "")))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe panorama image path in Kaggle input manifest.")
                    image_path = (root / relative).resolve()
                    if root not in image_path.parents or not image_path.is_file():
                        raise RuntimeError(f"Missing panorama image in unpacked Kaggle input: {relative}")
                    archive.write(image_path, arcname=relative.as_posix())
            say("Repacked Kaggle-unpacked panorama data into", archive_path)
            return archive_path
        except Exception as exc:
            say("Ignoring incomplete unpacked panorama manifest", manifest_path, ":", str(exc)[:300])

    inventory = [
        str(path.relative_to(INPUT))
        for path in INPUT.rglob("*")
        if path.is_file()
    ][:80]
    raise RuntimeError(
        "No panorama bundle or unpacked manifest found in the attached Kaggle input dataset. "
        "Input inventory: " + json.dumps(inventory)
    )


def install_sfm_environment():
    sfm_root = ROOT / "sfm-env"
    sfm_python = sfm_root / "bin" / "python"
    if sfm_python.exists():
        return sfm_python
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "venv", "-q", "-p", "3.10", str(sfm_root)])
    run(["uv", "pip", "install", "--python", str(sfm_python),
         "torch==2.1.2", "torchvision==0.16.2", "--index-url",
         "https://download.pytorch.org/whl/cu118"])
    hloc = ROOT / "Hierarchical-Localization"
    if not hloc.exists():
        run(["git", "clone", "-q", "--recursive",
             "https://github.com/cvg/Hierarchical-Localization", str(hloc)])
    lightglue = "lightglue @ git+https://github.com/cvg/LightGlue"
    run(["uv", "pip", "install", "--python", str(sfm_python),
         "numpy<2", "opencv-python-headless", "Pillow", "pycolmap", "-e", str(hloc), lightglue])
    return sfm_python


def install_gsplat():
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("No NVIDIA GPU is available in this Kaggle session.")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    if gpu.returncode or not gpu.stdout.strip():
        raise RuntimeError("Kaggle assigned no GPU; stopping before CPU reconstruction/training.")
    say("GPU:", gpu.stdout.strip().replace("\n", " | "))
    gsplat_root = ROOT / "gsplat"
    if not gsplat_root.exists():
        run(["git", "clone", "-q", "--depth", "1", "-b", "v1.5.3",
             "https://github.com/nerfstudio-project/gsplat", str(gsplat_root)])
    (gsplat_root / "examples" / "datasets" / "__init__.py").touch()
    packages = [
        "ninja", "gsplat==1.5.3", "viser", "tyro", "torchmetrics[image]",
        "tensorboard", "tensorly", "scikit-learn", "imageio[ffmpeg]",
        "pyyaml", "matplotlib", "splines", "plyfile", "scikit-image",
        "trimesh", "opencv-python-headless", "Pillow", "scipy", "pycolmap",
        "fast-simplification", "tqdm", "typing_extensions"
    ]
    # Keep Kaggle's preinstalled CUDA PyTorch pinned, exactly as the established
    # NAMMA gsplat runner does; only install the trainer's other dependencies.
    pin = ROOT / "torch-pin.txt"
    torch_version = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    pin.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    run([sys.executable, "-m", "pip", "install", "-q", "-c", pin, *packages])
    run([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
    ])
    warm = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); import gsplat; print(torch.__version__, torch.cuda.get_device_name(0))"],
                          capture_output=True, text=True)
    if warm.returncode:
        raise RuntimeError("PyTorch/gsplat CUDA environment is not ready: " + warm.stderr[-1200:])
    say(warm.stdout.strip())
    return gsplat_root


def main():
    global HERE
    HERE = find_code_root()
    sys.path.insert(0, str(HERE))
    say("NAMMA Panorama v1 starting", json.dumps(JOB_CFG, sort_keys=True))
    # A Kaggle GPU is a hard requirement. Never silently fall back to CPU.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable GPU for the separate namma-panorama kernel.")
    bundle = find_bundle()
    source = ROOT / "source"
    safe_extract(bundle, source)
    say("input ZIP:", round(bundle.stat().st_size / 1e6, 1), "MB")

    sys.path.insert(0, str(HERE))
    import prepare_panorama
    data = ROOT / "data"
    prep = prepare_panorama.make_views(
        source, data,
        image_size=int(JOB_CFG.get("image_size", 1280)),
        view_count=int(JOB_CFG.get("view_count", 12)),
        hfov_deg=float(JOB_CFG.get("hfov_deg", 60.0)),
        max_panoramas=int(JOB_CFG.get("max_panoramas", 100))
    )
    say("prepared", prep["source_panoramas"], "panoramas ->", prep["perspective_views"], "views")

    sfm_python = install_sfm_environment()
    sfm_out = ROOT / "sfm"
    run([sfm_python, HERE / "sfm_hloc.py", "--images", data / "images",
         "--output", sfm_out, "--sequence-window", str(int(JOB_CFG.get("sequence_window", 18))),
         "--retrieval", str(int(JOB_CFG.get("retrieval_count", 10)))],
        log=ROOT / "sfm.log")
    model = sfm_out / "model"
    sparse = data / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = model / name
        if not src.is_file():
            raise RuntimeError(f"SfM did not write {src}")
        shutil.copy2(src, sparse / name)
    shutil.copy2(sfm_out / "sfm_report.json", OUT / "sfm_report.json")
    shutil.copy2(sfm_out / "camera_poses.json", OUT / "camera_poses.json")

    gsplat_root = install_gsplat()
    examples = gsplat_root / "examples"
    steps = int(JOB_CFG.get("steps", 30000))
    cap = int(JOB_CFG.get("max_splats", 1_500_000))
    if not 5000 <= steps <= 60000:
        raise ValueError("steps must be between 5,000 and 60,000.")
    if not 100_000 <= cap <= 3_000_000:
        raise ValueError("max_splats must be between 100,000 and 3,000,000.")
    result = ROOT / "train-full"
    train_log = OUT / "full" / "train.log"
    (OUT / "full").mkdir(parents=True, exist_ok=True)
    train_env = os.environ.copy()
    native_ssim = subprocess.run(
        [sys.executable, "-c", "import fused_ssim"],
        capture_output=True, text=True
    )
    if native_ssim.returncode != 0:
        shim = HERE / "fused_ssim_shim"
        if not (shim / "fused_ssim.py").is_file():
            raise RuntimeError("Fused SSIM is unavailable and the panorama fallback shim is missing.")
        train_env["PYTHONPATH"] = str(shim) + os.pathsep + train_env.get("PYTHONPATH", "")
        say("Using the bundled pure-PyTorch SSIM fallback.")
    cmd = [
        sys.executable, examples / "simple_trainer.py", "mcmc",
        "--data-dir", data, "--data-factor", "1",
        "--result-dir", result,
        "--max-steps", str(steps),
        "--eval-steps", str(steps),
        "--save-steps", str(steps),
        "--ply-steps", str(steps),
        "--save-ply", "--disable-viewer", "--disable-video",
        "--use-bilateral-grid",
        "--strategy.cap-max", str(cap)
    ]
    sfm_report = json.loads((OUT / "sfm_report.json").read_text(encoding="utf-8"))
    depth_loss_enabled = int(sfm_report.get("images_without_3d_points", 1)) == 0
    if depth_loss_enabled:
        cmd.insert(cmd.index("--strategy.cap-max"), "--depth-loss")
    say(
        "Video-parity training profile:",
        json.dumps({
            "image_size": prep["image_width"],
            "steps": steps,
            "max_splats": cap,
            "bilateral_grid": True,
            "depth_loss": depth_loss_enabled
        }),
        flush=True
    )
    run(cmd, cwd=examples, env=train_env, log=train_log)
    ply_files = sorted(
        result.glob("ply/point_cloud_*.ply"),
        key=lambda p: int(re.search(r"point_cloud_(\d+)\.ply$", p.name).group(1))
    )
    if not ply_files:
        raise RuntimeError("gsplat completed without exporting a PLY point cloud.")
    shutil.copy2(ply_files[-1], OUT / "full" / "model.ply")

    stats = sorted(result.glob("stats/val_step*.json"))
    if not stats:
        raise RuntimeError("gsplat completed without held-out validation metrics; refusing to publish an unverified model.")
    metrics = json.loads(stats[-1].read_text(encoding="utf-8"))
    metrics.update({
        "quality_profile": "namma-video-parity",
        "image_width": int(prep["image_width"]),
        "image_height": int(prep["image_height"]),
        "training_steps": steps,
        "splat_cap": cap,
        "bilateral_grid": True,
        "depth_loss": depth_loss_enabled
    })
    (OUT / "full" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(data / "views_manifest.json", OUT / "views_manifest.json")
    shutil.copy2(data / "manifest.json", OUT / "manifest.json")
    shutil.copytree(data / "images", OUT / "images", dirs_exist_ok=True)
    shutil.copytree(sparse, OUT / "sparse" / "0", dirs_exist_ok=True)

    # Prepare route cameras and a metric similarity transform for the static tour.
    sys.path.insert(0, str(examples))
    import make_tour as panorama_tour
    tour_dir = OUT / "tour"
    tour_info = panorama_tour.build_plan(data, OUT, tour_dir, examples,
                                         float(JOB_CFG.get("camera_height_m", 2.2)))
    summary = {
        "pipeline": "panorama_v1",
        "job_id": JOB_CFG.get("job_id"),
        "source_panoramas": prep["source_panoramas"],
        "perspective_views": prep["perspective_views"],
        "registered_views": sfm_report["registered_views"],
        "registered_fraction": sfm_report["registered_fraction"],
        "points3D": sfm_report["points3D"],
        "training_steps": steps,
        "max_splats": cap,
        "depth_loss_enabled": depth_loss_enabled,
        "quality_profile": "namma-video-parity",
        "validation_metrics": metrics,
        "exported_ply": str(OUT / "full" / "model.ply"),
        "tour": tour_info,
        "elapsed_minutes": round((time.time() - T0) / 60, 1)
    }
    (OUT / "panorama_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    say("PANORAMA JOB COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        say("PANORAMA JOB FAILED:", type(exc).__name__, str(exc))
        (OUT / "failure.json").write_text(json.dumps({
            "pipeline": "panorama_v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_minutes": round((time.time() - T0) / 60, 1)
        }, indent=2), encoding="utf-8")
        raise
