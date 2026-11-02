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
    matches = [p.parent for p in INPUT.glob('**/prepare_panorama.py')]
    for candidate in matches:
        if (candidate / 'sfm_hloc.py').is_file() and (candidate / 'make_tour.py').is_file():
            return candidate
    if (HERE / 'prepare_panorama.py').is_file():
        return HERE
    raise RuntimeError('Panorama pipeline helper modules are missing from the Kaggle input dataset.')


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
    candidates = list(INPUT.glob("**/bundle.zip")) + list(INPUT.glob("**/*.zip"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise RuntimeError("No panorama ZIP found in the attached Kaggle input dataset.")
    return max(candidates, key=lambda p: p.stat().st_size)


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
        "ninja", "tyro", "tensorboard", "tensorly", "scikit-learn",
        "imageio[ffmpeg]", "pyyaml", "matplotlib", "splines", "plyfile",
        "scikit-image", "trimesh", "opencv-python-headless", "Pillow",
        "scipy", "fast-simplification"
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *packages])
    # Use the source checkout for the trainer examples and the same pinned gsplat version.
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "gsplat==1.5.3"])
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
        image_size=int(JOB_CFG.get("image_size", 768)),
        view_count=int(JOB_CFG.get("view_count", 12)),
        hfov_deg=float(JOB_CFG.get("hfov_deg", 60.0)),
        max_panoramas=int(JOB_CFG.get("max_panoramas", 200))
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
    cmd = [
        sys.executable, examples / "simple_trainer.py", "mcmc",
        "--data-dir", data, "--data-factor", "1",
        "--result-dir", result,
        "--max-steps", str(steps),
        "--eval-steps", str(steps),
        "--save-steps", str(steps),
        "--ply-steps", str(steps),
        "--save-ply", "--disable-viewer",
        "--use-bilateral-grid",
        "--strategy.cap-max", str(cap)
    ]
    run(cmd, cwd=examples, log=train_log)
    ply_files = sorted(
        result.glob("ply/point_cloud_*.ply"),
        key=lambda p: int(re.search(r"point_cloud_(\d+)\.ply$", p.name).group(1))
    )
    if not ply_files:
        raise RuntimeError("gsplat completed without exporting a PLY point cloud.")
    shutil.copy2(ply_files[-1], OUT / "full" / "model.ply")

    stats = sorted(result.glob("stats/val_step*.json"))
    metrics = json.loads(stats[-1].read_text(encoding="utf-8")) if stats else {}
    (OUT / "full" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(data / "views_manifest.json", OUT / "views_manifest.json")
    shutil.copy2(data / "manifest.json", OUT / "manifest.json")
    shutil.copytree(data / "images", OUT / "images", dirs_exist_ok=True)
    shutil.copytree(sparse, OUT / "sparse" / "0", dirs_exist_ok=True)
    shutil.copy2(train_log, OUT / "full" / "train.log")

    # Prepare route cameras and a metric similarity transform for the static tour.
    sys.path.insert(0, str(examples))
    import make_tour as panorama_tour
    tour_dir = OUT / "tour"
    tour_info = panorama_tour.build_plan(data, OUT, tour_dir, examples,
                                         float(JOB_CFG.get("camera_height_m", 2.2)))
    sfm_report = json.loads((OUT / "sfm_report.json").read_text(encoding="utf-8"))
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
