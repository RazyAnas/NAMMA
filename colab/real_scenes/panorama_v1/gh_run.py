"""GitHub Actions orchestrator for an independent NAMMA panorama job.

Trigger: a new JSON request under panorama_v1/jobs/ on main or the feature branch.
Input images live in a private Hugging Face dataset; the GPU work runs in a dedicated
Kaggle kernel; the static tour is added under panorama/<job_id>/ in NAMMA's HF Space.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
T0 = time.time()
OUT = Path("out")
OUTPUT_DATASET = os.environ.get("PANORAMA_OUTPUT_DATASET", "anasrazy/namma-panorama-out")
TOUR_SPACE = os.environ.get("PANORAMA_TOUR_SPACE", "anasrazy/namma-tour")

KAGGLE_CODE_FILES = (
    "prepare_panorama.py",
    "pairing.py",
    "sfm_hloc.py",
    "make_tour.py",
    "gsplat_colmap_compat.py",
    "gsplat_trainer_compat.py",
    "gsplat_cuda_compat.py",
    "ssim_preflight.py",
)


def stage_kaggle_code(destination: Path) -> None:
    """Copy the complete, explicit panorama runtime dependency set into Kaggle input."""
    destination.mkdir(parents=True, exist_ok=True)
    missing = [name for name in KAGGLE_CODE_FILES if not (HERE / name).is_file()]
    shim = HERE / "fused_ssim_shim" / "fused_ssim.py"
    if not shim.is_file():
        missing.append("fused_ssim_shim/fused_ssim.py")
    if missing:
        raise RuntimeError("Panorama Kaggle package is incomplete; missing: " + ", ".join(missing))
    for name in KAGGLE_CODE_FILES:
        shutil.copy2(HERE / name, destination / name)
    shutil.copytree(HERE / "fused_ssim_shim", destination / "fused_ssim_shim", dirs_exist_ok=True)


def say(*args):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *args, flush=True)


def command(args, check=True, cwd=None):
    p = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode:
        raise RuntimeError(f"Command failed: {' '.join(map(str, args[:3]))}: {out[-1800:]}")
    return p.returncode, out


def kaggle(*args, check=True):
    return command(["kaggle", *args], check=check)


def wait_dataset(dataset_id, minutes=45):
    start = time.time()
    last = ""
    while time.time() - start < minutes * 60:
        time.sleep(20)
        rc, last = kaggle("datasets", "status", dataset_id, check=False)
        if rc == 0 and last.strip().lower().endswith("ready"):
            time.sleep(30)
            say("Kaggle dataset ready:", dataset_id)
            return
    raise RuntimeError(f"Kaggle input dataset did not become ready: {last[-500:]}")


def job_file_from_event() -> Path:
    explicit = os.environ.get("PANORAMA_JOB_FILE", "").strip()
    if explicit:
        p = (REPO_ROOT / explicit).resolve()
        if REPO_ROOT.resolve() not in p.parents or not p.is_file():
            raise RuntimeError(f"PANORAMA_JOB_FILE is not a repository job file: {explicit}")
        return p
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise RuntimeError("Set PANORAMA_JOB_FILE or run inside a GitHub Actions event.")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    files = []
    commit_sha = os.environ.get("GITHUB_SHA", "").strip()
    if commit_sha:
        changed = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=ACMR", commit_sha],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if changed.returncode == 0:
            files = [line.strip() for line in changed.stdout.splitlines() if line.strip()]
        else:
            say(
                "Could not inspect the exact GITHUB_SHA diff; falling back to push-event metadata:",
                changed.stderr[-300:]
            )
    if not files:
        head = event.get("head_commit") or {}
        files = list(head.get("added", [])) + list(head.get("modified", []))
    if not files:
        for commit in event.get("commits", []):
            files.extend(commit.get("added", []))
            files.extend(commit.get("modified", []))

    jobs = sorted({x for x in files if x.startswith("colab/real_scenes/panorama_v1/jobs/") and x.endswith(".json")})
    if len(jobs) != 1:
        raise RuntimeError(f"Expected exactly one new panorama job manifest in this commit; found {jobs}.")
    p = (REPO_ROOT / jobs[0]).resolve()
    if REPO_ROOT.resolve() not in p.parents or not p.is_file():
        raise RuntimeError(f"Job manifest is missing: {jobs[0]}")
    return p


def validate_job(job: dict) -> dict:
    job_id = str(job.get("job_id", ""))
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{2,50}", job_id):
        raise ValueError("job_id must contain 3-51 letters, digits or hyphens.")
    hf_dataset = str(job.get("hf_dataset", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hf_dataset):
        raise ValueError("hf_dataset must be an owner/dataset repository id.")

    bundle_mode = str(job.get("bundle_mode", "zip"))
    bundle_path = str(job.get("bundle_path", ""))
    manifest_path = str(job.get("manifest_path", ""))
    panorama_paths = job.get("panorama_paths", [])
    if bundle_mode == "zip":
        if not bundle_path or bundle_path.startswith("/") or ".." in Path(bundle_path).parts:
            raise ValueError("bundle_path must be a safe path inside the HF dataset.")
    elif bundle_mode == "loose":
        if not manifest_path or manifest_path.startswith("/") or ".." in Path(manifest_path).parts:
            raise ValueError("manifest_path must be a safe path inside the HF dataset.")
        if not isinstance(panorama_paths, list) or not 8 <= len(panorama_paths) <= 200:
            raise ValueError("loose jobs require between 8 and 200 panorama_paths.")
        for path in panorama_paths:
            if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("Every panorama path must be a safe relative HF dataset path.")
    else:
        raise ValueError("bundle_mode must be 'zip' or 'loose'.")

    params = job.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("parameters must be a JSON object.")
    allowed = {
        "image_size": (384, 1536), "view_count": (6, 24),
        "max_panoramas": (8, 200), "steps": (5000, 60000),
        "max_splats": (100000, 3000000),
        "sequence_window": (4, 40), "retrieval_count": (0, 30)
    }
    safe = {}
    for key, bounds in allowed.items():
        if key in params:
            value = int(params[key])
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{key} must be in {bounds[0]}..{bounds[1]}.")
            safe[key] = value
    for key, bounds in {"hfov_deg": (40.0, 90.0), "camera_height_m": (0.5, 5.0)}.items():
        if key in params:
            value = float(params[key])
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{key} must be in {bounds[0]}..{bounds[1]}.")
            safe[key] = value
    return {
        "job_id": job_id,
        "hf_dataset": hf_dataset,
        "bundle_mode": bundle_mode,
        "bundle_path": bundle_path or None,
        "manifest_path": manifest_path or None,
        "panorama_paths": panorama_paths,
        "parameters": safe,
        "output_dataset": job.get("output_dataset", OUTPUT_DATASET),
        "tour_space": job.get("tour_space", TOUR_SPACE)
    }


def hf_client(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def update_status(api, repo_id, job_id, status, **extra):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    status_path = f"{job_id}/status.json"
    try:
        existing_path = hf_hub_download(
            repo_id=repo_id,
            filename=status_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            force_download=True,
        )
    except EntryNotFoundError:
        previous = {}
    else:
        previous = json.loads(Path(existing_path).read_text(encoding="utf-8"))

    if status == "staging":
        for key in ("kaggle_dataset", "kaggle_kernel", "tour_url", "results", "error"):
            previous.pop(key, None)

    payload = {
        **previous,
        "job_id": job_id,
        "pipeline": "panorama_v1",\n        "status": status,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    api.upload_file(
        path_or_fileobj=json.dumps(payload, indent=2).encode("utf-8"),
        path_in_repo=status_path, repo_id=repo_id,
        repo_type="dataset", commit_message=f"panorama {job_id}: {status}"
    )
    say("status:", status)


def upload_input_bundle(api, job, token, dataset_id):
    from huggingface_hub import hf_hub_download
    import zipfile

    folder = Path("kaggle-input")
    folder.mkdir(exist_ok=True)
    bundle_file = folder / "bundle.namma"

    if job["bundle_mode"] == "zip":
        source = hf_hub_download(
            repo_id=job["hf_dataset"], filename=job["bundle_path"],
            repo_type="dataset", token=token, local_dir="hf-input"
        )
        shutil.copy2(source, bundle_file)
    else:
        manifest_file = hf_hub_download(
            repo_id=job["hf_dataset"], filename=job["manifest_path"],
            repo_type="dataset", token=token, local_dir="hf-input"
        )
        manifest_bytes = Path(manifest_file).read_bytes()
        manifest = json.loads(manifest_bytes)
        panoramas = manifest.get("panoramas", [])
        if len(panoramas) != len(job["panorama_paths"]) or not 8 <= len(panoramas) <= 200:
            raise RuntimeError("Loose inbox job manifest and panorama file count do not match.")
        seen_images = set()
        for entry in panoramas:
            image_path = str(entry.get("image", ""))
            if not image_path or image_path.startswith("/") or ".." in Path(image_path).parts:
                raise RuntimeError("A panorama manifest contains an unsafe relative image path.")
            if image_path in seen_images:
                raise RuntimeError("The panorama manifest contains duplicate image paths.")
            seen_images.add(image_path)

        with zipfile.ZipFile(bundle_file, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            for index, (remote_path, entry) in enumerate(zip(job["panorama_paths"], panoramas), start=1):
                local = hf_hub_download(
                    repo_id=job["hf_dataset"], filename=remote_path,
                    repo_type="dataset", token=token, local_dir="hf-input"
                )
                archive.write(local, arcname=entry["image"])
                say(f"Staged browser-uploaded panorama {index}/{len(panoramas)}")

    stage_kaggle_code(folder)
    metadata = {
        "title": f"NAMMA panorama job {job['job_id']}",
        "id": dataset_id,
        "licenses": [{"name": "other"}]
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rc, state = kaggle("datasets", "status", dataset_id, check=False)
    if rc == 0 and state.strip().lower().endswith("ready"):
        kaggle("datasets", "version", "-p", folder, "-m", f"NAMMA panorama {job['job_id']}", "-q")
    else:
        kaggle("datasets", "create", "-p", folder, "-q")
    wait_dataset(dataset_id)


def push_kernel(job, dataset_id, kaggle_user):
    attempt = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))[-8:]
    attempt_no = os.environ.get("GITHUB_RUN_ATTEMPT", "1")[-1:]
    slug = f"namma-pano-{job['job_id'][:20].rstrip('-')}-{attempt}a{attempt_no}".lower()
    kernel_id = f"{kaggle_user}/{slug}"
    folder = Path("kaggle-kernel")
    folder.mkdir(exist_ok=True)
    source = (HERE / "run.py").read_text(encoding="utf-8")
    marker = 'import glob, json, os, re, shutil, subprocess, sys, time, zipfile\\n'
    if source.count(marker) != 1:
        raise RuntimeError("Could not safely inject the job settings into panorama_v1/run.py.")
    injected = source.replace(
        marker,
        marker + f'os.environ["JOB_CFG"] = {json.dumps(json.dumps(dict(job["parameters"], job_id=job["job_id"])))}\\n',\n        1
    )
    (folder / "run.py").write_text(injected, encoding="utf-8")
    metadata = {
        "id": kernel_id,
        "title": " ".join(slug.split("-")),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_id],
        "kernel_sources": []
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    say("Pushing independent Kaggle kernel:", kernel_id, "input:", dataset_id)
    say(kaggle("kernels", "push", "-p", folder)[1][-500:])
    return kernel_id


def wait_kernel(kernel_id, minutes=330):
    start = time.time()
    seen_running = False
    last = ""

    def fetch_failure_logs():
        rc, logs = kaggle("kernels", "logs", kernel_id, check=False)
        Path("out").mkdir(exist_ok=True)
        log_path = Path("out") / "kaggle_kernel.log"
        log_path.write_text(logs or f"Kaggle logs command returned exit code {rc}.", encoding="utf-8")
        say("Kaggle execution logs saved to", log_path)
        return logs

    while time.time() - start < minutes * 60:
        time.sleep(60)
        rc, status = kaggle("kernels", "status", kernel_id, check=False)
        if rc:
            say("Kaggle status query failed:", status[-160:])
            continue
        low = status.lower()
        if low != last:
            say("Kaggle status:", low[-160:])
            last = low
        if "running" in low or "queued" in low:
            seen_running = True
        if "error" in low or "cancel" in low:
            logs = fetch_failure_logs()
            raise RuntimeError(
                f"Kaggle panorama kernel ended unsuccessfully: {status[-600:]}\\n"\n                f"Kaggle kernel log tail:\\n{(logs or '[no execution logs returned]')[-12000:]}"\n            )
        if "complete" in low and seen_running:
            return
    logs = fetch_failure_logs()
    raise RuntimeError(
        f"Kaggle panorama kernel did not complete within {minutes} minutes; last status: {last}\\n"\n        f"Kaggle kernel log tail:\\n{(logs or '[no execution logs returned]')[-12000:]}"\n    )


def ensure_viewer_toolchain():
    node_root = Path("panorama-tour-builder")
    node_root.mkdir(exist_ok=True)
    package_json = node_root / "package.json"
    if not package_json.is_file():
        command(["npm", "init", "-y"], cwd=node_root)

    public = node_root / "node_modules" / "@playcanvas" / "supersplat-viewer" / "public"
    transform = node_root / "node_modules" / ".bin" / "splat-transform"
    required_public = ("index.html", "index.css", "index.js")
    if not transform.is_file() or any(not (public / name).is_file() for name in required_public):
        command([
            "npm", "install", "--silent", "--no-audit", "--no-fund",
            "playcanvas@2.23.2", "@playcanvas/supersplat-viewer@1.37.0",
            "@playcanvas/splat-transform@3.10.1"
        ], cwd=node_root)

    missing = [name for name in required_public if not (public / name).is_file()]
    if missing or not transform.is_file():
        raise RuntimeError(
            "Static viewer toolchain preflight failed; missing viewer files: "
            + (", ".join(missing) if missing else "none")
            + ("; splat-transform executable is missing." if not transform.is_file() else "")
        )
    say("Static SuperSplat viewer and splat-transform preflight passed.")
    return public, transform


def preflight_static_exporter():
    import importlib.util
    import numpy, scipy, plyfile, trimesh  # noqa: F401

    final_dir = REPO_ROOT / "colab" / "real_scenes" / "final"
    exporter_path = final_dir / "make_tour.py"
    if not exporter_path.is_file():
        raise RuntimeError(f"The generic NAMMA static-tour exporter is missing: {exporter_path}")
    spec = importlib.util.spec_from_file_location("namma_static_tour_builder_preflight", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load NAMMA's generic static-tour exporter.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None))):
        raise RuntimeError("The generic NAMMA static-tour exporter has no callable build() function.")
    say("Generic NAMMA static-tour exporter import preflight passed.")
    return module


def ensure_lfs(api, space, token, repo_type="space"):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id=space, filename=".gitattributes", repo_type=repo_type, token=token)
        current = Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception:
        current = ""
    patterns = ("*.ply", "*.sog", "*.spz", "*.glb", "*.bin", "*.jpg", "*.png", "*.webp", "*.mp4")
    missing = [f"{x} filter=lfs diff=lfs merge=lfs -text" for x in patterns if f"{x} filter=lfs" not in current]
    if missing:
        updated = current.rstrip() + ("\\n" if current.strip() else "") + "\\n".join(missing) + "\\n"
        api.upload_file(
            path_or_fileobj=updated.encode("utf-8"), path_in_repo=".gitattributes",
            repo_id=space, repo_type=repo_type,
            commit_message="Configure LFS for NAMMA panorama assets"
        )


def publish_tour(api, space, job_id, output_dir, token):
    from huggingface_hub import CommitOperationAdd
    tour = output_dir / "tour"
    required = ("plan.json", "settings.json", "tour.json")
    for name in required:
        if not (tour / name).is_file():
            raise RuntimeError(f"Panorama tour is missing {name}.")
    api.create_repo(repo_id=space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    ensure_lfs(api, space, token)
    public, transform = ensure_viewer_toolchain()
    static_tour_builder = preflight_static_exporter()
    os.environ["TOUR_FORMAT"] = "cply"
    os.environ["TOUR_SH"] = "3"
    os.environ["TOUR_CROP"] = "0"
    os.environ["TOUR_SMEAR"] = "0"
    static_tour_builder.build(str(output_dir), str(tour), str(public), str(transform))
    dest = f"panorama/{job_id}"
    api.upload_folder(
        folder_path=str(tour), path_in_repo=dest,
        repo_id=space, repo_type="space",
        ignore_patterns=["plan.json", "clean.ply", "full_t.ply"],
        commit_message=f"Publish NAMMA Street View panorama tour {job_id}"
    )
    return f"https://{space.replace('/', '-')}.static.hf.space/{dest}/index.html"


def publish_results(api, repo_id, job_id, output_dir, tour_url):
    from huggingface_hub import HfApi
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    curated = Path("panorama-results")
    if curated.exists():
        shutil.rmtree(curated)
    curated.mkdir()
    names = ["panorama_summary.json", "manifest.json", "views_manifest.json",
             "sfm_report.json", "camera_poses.json"]
    for name in names:
        src = output_dir / name
        if src.is_file():
            shutil.copy2(src, curated / name)
    for name in ("model.ply", "metrics.json", "train.log"):
        src = output_dir / "full" / name
        if src.is_file():
            shutil.copy2(src, curated / name)
    for name in ("plan.json", "settings.json", "tour.json"):
        src = output_dir / "tour" / name
        if src.is_file():
            (curated / "tour").mkdir(exist_ok=True)
            shutil.copy2(src, curated / "tour" / name)
    (curated / "tour_url.txt").write_text(tour_url + "\\n", encoding="utf-8")
    (curated / "README.md").write_text(
        "# NAMMA Panorama v1 results\\n\\n"\n        f"Job: {job_id}\\n\\n"\n        f"Tour: {tour_url}\\n\\n"\n        "The reconstruction used perspective views generated from the source 360-degree panoramas. "\n        "Inspect sfm_report.json and panorama_summary.json before treating the scene as metrically accurate. "\n        "The viewer is an appearance model; it does not imply complete collision geometry.\\n",
        encoding="utf-8"
    )
    ensure_lfs(api, repo_id, os.environ.get("HF_TOKEN", ""), repo_type="dataset")
    api.upload_folder(
        folder_path=str(curated), path_in_repo=job_id,
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"Store NAMMA panorama results {job_id}"
    )


def main():
    from huggingface_hub import HfApi
    from kaggle.api.kaggle_api_extended import KaggleApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN repository secret is required.")
    if not os.environ.get("KAGGLE_API_TOKEN"):\n        raise RuntimeError("KAGGLE_API_TOKEN repository secret is required.")
    job_path = job_file_from_event()
    raw_job = json.loads(job_path.read_text(encoding="utf-8"))
    job = validate_job(raw_job)
    ensure_viewer_toolchain()
    preflight_static_exporter()
    api = HfApi(token=hf_token)
    output_dataset = job["output_dataset"]
    api.create_repo(repo_id=output_dataset, repo_type="dataset", private=True, exist_ok=True)
    update_status(api, output_dataset, job["job_id"], "staging",
                  source_dataset=job["hf_dataset"], source_bundle=job["bundle_path"], source_manifest=job["manifest_path"])

    reused_kernel = os.environ.get("PANORAMA_REUSE_KERNEL", "").strip()
    if reused_kernel:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[a-z0-9][a-z0-9-]{2,99}", reused_kernel):
            raise ValueError("PANORAMA_REUSE_KERNEL must be a safe Kaggle owner/slug.")
        kernel_id = reused_kernel
        say("Reusing completed Kaggle panorama kernel:", kernel_id)
    else:
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()
        kaggle_user = kaggle_api.config_values["username"]
        ds_slug = f"namma-panorama-{job['job_id'][:35]}".lower()
        dataset_id = f"{kaggle_user}/{ds_slug}"
        upload_input_bundle(api, job, hf_token, dataset_id)
        update_status(api, output_dataset, job["job_id"], "queued", kaggle_dataset=dataset_id)
        kernel_id = push_kernel(job, dataset_id, kaggle_user)

    update_status(api, output_dataset, job["job_id"], "training", kaggle_kernel=kernel_id)
    if not reused_kernel:
        wait_kernel(kernel_id)
    OUT.mkdir(exist_ok=True)
    kaggle("kernels", "output", kernel_id, "-p", OUT, "-o", "--page-size", "200")
    artifact_dir = OUT / "out" if (OUT / "out").is_dir() else OUT
    summary = artifact_dir / "panorama_summary.json"
    failure = artifact_dir / "failure.json"
    if failure.is_file():
        raise RuntimeError("Kaggle panorama pipeline failed: " + failure.read_text(encoding="utf-8")[-2000:])
    if not summary.is_file() or not (artifact_dir / "full" / "model.ply").is_file():
        raise RuntimeError("Kaggle finished without panorama_summary.json and full/model.ply.")
    update_status(api, output_dataset, job["job_id"], "publishing")
    tour_url = publish_tour(api, job["tour_space"], job["job_id"], artifact_dir, hf_token)
    publish_results(api, output_dataset, job["job_id"], artifact_dir, tour_url)
    update_status(api, output_dataset, job["job_id"], "complete",
                  tour_url=tour_url, results=f"https://huggingface.co/datasets/{output_dataset}/tree/main/{job['job_id']}")
    summary_obj = json.loads(summary.read_text(encoding="utf-8"))
    summary_obj["tour_url"] = tour_url
    (summary.parent / "panorama_summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    summary_md = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_md:
        with open(summary_md, "a", encoding="utf-8") as f:
            f.write(f"## NAMMA panorama job {job['job_id']}\\n\\n")
            f.write(f"**Tour:** {tour_url}\\n\\n")
            f.write(f"**Registered views:** {summary_obj['registered_views']} / {summary_obj['perspective_views']}\\n\\n")
            f.write(f"**Source panoramas:** {summary_obj['source_panoramas']}\\n\\n")
            f.write(f"**Private results:** https://huggingface.co/datasets/{output_dataset}/tree/main/{job['job_id']}\\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        say("PANORAMA ORCHESTRATION FAILED:", type(exc).__name__, str(exc))
        try:
            from huggingface_hub import HfApi
            token = os.environ.get("HF_TOKEN")
            if token:
                p = job_file_from_event()
                jid = json.loads(p.read_text(encoding="utf-8")).get("job_id", "unknown")
                repo = json.loads(p.read_text(encoding="utf-8")).get("output_dataset", OUTPUT_DATASET)
                api = HfApi(token=token)
                api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
                detail = f"{type(exc).__name__}: {exc}"
                if "Kaggle kernel log tail:" in detail:
                    summary, log_tail = detail.split("Kaggle kernel log tail:", 1)
                    detail = summary[:450] + "\\nKaggle kernel log tail:\\n" + log_tail[-950:]
                else:
                    detail = detail[-1400:]
                update_status(api, repo, jid, "failed", error=detail[:1500])
        except Exception as status_exc:
            say("Could not update failure status:", str(status_exc)[:300])
        raise
