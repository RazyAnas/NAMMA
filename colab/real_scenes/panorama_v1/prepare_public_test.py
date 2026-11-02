"""Prepare a non-commercial GGPS FTP benchmark bundle and launch metadata.

This is a one-off cloud test fixture. It does not touch the video pipeline.
The source archive is licensed CC BY-NC 4.0; the exported tour carries attribution.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_REPO = "Insta360-Research/GGPS"
SOURCE_FILE = "datasets/FTP.zip"
SOURCE_LICENSE = "CC-BY-NC-4.0"
INBOX_REPO = "anasrazy/namma-panorama-inbox"
OUTPUT_REPO = "anasrazy/namma-panorama-out"
TOUR_SPACE = "anasrazy/namma-tour"
PANORAMA_COUNT = 40


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Check the GitHub Actions secret name/fallback.")

    print(f"Downloading public benchmark {SOURCE_REPO}/{SOURCE_FILE}", flush=True)
    archive = Path(hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=SOURCE_FILE,
        repo_type="model",
        token=token
    ))
    print(f"Benchmark archive ready: {archive.stat().st_size / 1e9:.2f} GB", flush=True)

    with zipfile.ZipFile(archive) as source_zip:
        candidates = [
            info for info in source_zip.infolist()
            if not info.is_dir()
            and Path(info.filename).suffix.lower() in {".png", ".jpg", ".jpeg"}
            and "images" in [part.lower() for part in Path(info.filename).parts]
        ]
        candidates.sort(key=lambda info: natural_key(info.filename))
        selected = []
        for info in candidates:
            try:
                with source_zip.open(info) as handle, Image.open(handle) as image:
                    width, height = image.size
                    if width >= 2048 and height >= 1024 and 1.75 <= width / height <= 2.25:
                        selected.append((info, width, height))
            except Exception as exc:
                print(f"Skipping unreadable panorama {info.filename}: {exc}", flush=True)
            if len(selected) == PANORAMA_COUNT:
                break

        if len(selected) < 8:
            raise RuntimeError(
                f"Expected at least 8 2:1 equirectangular images in {SOURCE_FILE}; "
                f"found {len(selected)} valid panoramas from {len(candidates)} image files."
            )
        if len(selected) < PANORAMA_COUNT:
            print(f"Only {len(selected)} valid panoramas were available; continuing with that count.", flush=True)

        job_id = "ggps-ftp-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        work = ROOT / "panorama-public-test"
        if work.exists():
            shutil.rmtree(work)
        (work / "bundle").mkdir(parents=True)
        bundle_path = work / "bundle.zip"
        panoramas = []
        attribution = (
            "Test imagery source: Insta360-Research/GGPS, scene FTP.\n"
            "Source archive: https://huggingface.co/Insta360-Research/GGPS\n"
            "Source license: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).\n"
            "This derived test scene is for non-commercial evaluation only. Do not use it in commercial NAMMA deployments.\n"
        )
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_STORED) as bundle_zip:
            for sequence, (info, width, height) in enumerate(selected):
                original_name = Path(info.filename).name
                pano_id = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_name).stem).strip("-")
                if not pano_id:
                    pano_id = f"ftp-{sequence:04d}"
                suffix = Path(original_name).suffix.lower()
                rel = f"panoramas/{sequence:04d}-{pano_id}{suffix}"
                with source_zip.open(info) as src, bundle_zip.open(rel, "w") as dst:
                    shutil.copyfileobj(src, dst)
                panoramas.append({
                    "pano_id": pano_id,
                    "image": rel,
                    "sequence": sequence
                })
                print(
                    f"Selected {sequence + 1}/{len(selected)}: {original_name} "
                    f"({width}x{height})",
                    flush=True
                )
            manifest = {
                "schema_version": 1,
                "scene_name": "GGPS FTP — NAMMA panorama quality test",
                "source": {
                    "repository": SOURCE_REPO,
                    "file": SOURCE_FILE,
                    "license": SOURCE_LICENSE,
                    "attribution": "Insta360 Research Team — GGPS / PanoLOG"
                },
                "panoramas": panoramas
            }
            bundle_zip.writestr("manifest.json", json.dumps(manifest, indent=2))
            bundle_zip.writestr("SOURCE_ATTRIBUTION.txt", attribution)

    print(f"Created {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    api = HfApi(token=token)
    api.create_repo(repo_id=INBOX_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(repo_id=OUTPUT_REPO, repo_type="dataset", private=True, exist_ok=True)
    bundle_repo_path = f"{job_id}/bundle.zip"
    api.upload_file(
        path_or_fileobj=str(bundle_path),
        path_in_repo=bundle_repo_path,
        repo_id=INBOX_REPO,
        repo_type="dataset",
        commit_message=f"NAMMA panorama quality test input {job_id}"
    )

    job = {
        "job_id": job_id,
        "pipeline": "panorama_v1",
        "hf_dataset": INBOX_REPO,
        "bundle_path": bundle_repo_path,
        "output_dataset": OUTPUT_REPO,
        "tour_space": TOUR_SPACE,
        "parameters": {
            "steps": 30000,
            "max_splats": 1500000,
            "max_panoramas": PANORAMA_COUNT,
            "image_size": 1280,
            "view_count": 12,
            "hfov_deg": 60.0,
            "sequence_window": 18,
            "retrieval_count": 10
        }
    }
    job_path = HERE / "jobs" / f"{job_id}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"job_file={job_path.relative_to(ROOT).as_posix()}\n")
            handle.write(f"job_id={job_id}\\n")
            handle.write(f"panoramas={len(panoramas)}\\n")
    print(
        f"Prepared {len(panoramas)} contiguous benchmark panoramas; "
        f"job={job_id}; inbox={INBOX_REPO}/{bundle_repo_path}",
        flush=True
    )


if __name__ == "__main__":
    main()
