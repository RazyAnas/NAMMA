"""Upload a panorama bundle to HF and commit a job request to the isolated NAMMA branch.

Required environment variables:
  HF_TOKEN       write token for the private panorama inbox dataset
  GITHUB_TOKEN   fine-grained PAT with Contents: read/write for RazyAnas/NAMMA

A folder with manifest.json is uploaded as-is. If no manifest exists, the script
creates one from JPG/PNG filenames; in that case GPS alignment is unavailable.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

DEFAULT_REPO = "RazyAnas/NAMMA"
DEFAULT_BRANCH = "feature/streetview-panorama-3d"
DEFAULT_INBOX = "anasrazy/namma-panorama-inbox"


def build_bundle(source: Path, destination: Path, scene_name: str) -> int:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Input must be a directory: {source}")
    manifest = source / "manifest.json"
    temp_root = destination.parent / "bundle-build"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    if manifest.is_file():
        # Keep the supplied manifest's relative paths unchanged; only include images
        # and JSON files, not macOS metadata or unrelated downloads.
        entries = [p for p in source.rglob("*") if p.is_file()
                   and not any(part.startswith(".") for part in p.relative_to(source).parts)
                   and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".json")]
        for item in entries:
            rel = item.relative_to(source)
            target = temp_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        count = len(json.loads((temp_root / "manifest.json").read_text(encoding="utf-8")).get("panoramas", []))
    else:
        images = sorted(p for p in source.rglob("*") if p.is_file()
                        and p.suffix.lower() in (".jpg", ".jpeg", ".png")
                        and not any(part.startswith(".") for part in p.relative_to(source).parts))
        if len(images) < 8:
            raise ValueError("At least 8 panorama images are required.")
        if len(images) > 200:
            raise ValueError("The first panorama pipeline release supports at most 200 panoramas per job.")
        pano_dir = temp_root / "panoramas"
        pano_dir.mkdir()
        panoramas = []
        for i, item in enumerate(images):
            safe_name = f"pano_{i:05d}{item.suffix.lower()}"
            shutil.copy2(item, pano_dir / safe_name)
            panoramas.append({
                "pano_id": item.stem[:120] or f"pano-{i:05d}",
                "image": f"panoramas/{safe_name}",
                "sequence": i
            })
        (temp_root / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "scene_name": scene_name,
            "panoramas": panoramas
        }, indent=2), encoding="utf-8")
        count = len(panoramas)
    if count < 8:
        raise ValueError("manifest.json contains fewer than 8 panoramas.")
    if count > 200:
        raise ValueError("manifest.json contains more than 200 panoramas.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as z:
        for item in sorted(temp_root.rglob("*")):
            if item.is_file():
                z.write(item, item.relative_to(temp_root).as_posix())
    shutil.rmtree(temp_root)
    return count


def github_put(repo: str, branch: str, path: str, content: bytes, token: str, message: str) -> dict:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = json.dumps({
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": branch
    }).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="PUT", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "NAMMA-panorama-submitter"
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub rejected the panorama job commit ({exc.code}): {detail[:1200]}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit a NAMMA Street View panorama reconstruction job.")
    ap.add_argument("--input", type=Path, required=True, help="Folder of panorama JPEGs/PNGs and optional manifest.json")
    ap.add_argument("--scene-name", default="NAMMA Street View scene")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--inbox", default=DEFAULT_INBOX)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--max-panoramas", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--view-count", type=int, default=12)
    ap.add_argument("--hfov", type=float, default=60.0)
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    github_token = os.environ.get("GITHUB_TOKEN")
    if not hf_token or not github_token:
        raise SystemExit("Set HF_TOKEN and GITHUB_TOKEN in the environment. Never paste either token into chat.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.inbox):
        raise SystemExit("--inbox must be an owner/dataset repository id.")
    if not 5000 <= args.steps <= 60000:
        raise SystemExit("--steps must be between 5000 and 60000.")
    if not 8 <= args.max_panoramas <= 200:
        raise SystemExit("--max-panoramas must be between 8 and 200.")
    if not 384 <= args.image_size <= 1536 or not 6 <= args.view_count <= 24:
        raise SystemExit("image-size must be 384..1536 and view-count 6..24.")
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    temp = Path("namma-panorama-submit")
    temp.mkdir(exist_ok=True)
    bundle = temp / "bundle.zip"
    count = build_bundle(args.input, bundle, args.scene_name)

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.inbox, repo_type="dataset", private=True, exist_ok=True)
    bundle_path = f"{job_id}/bundle.zip"
    api.upload_file(
        path_or_fileobj=str(bundle), path_in_repo=bundle_path,
        repo_id=args.inbox, repo_type="dataset",
        commit_message=f"Upload panorama bundle {job_id}"
    )

    job = {
        "job_id": job_id,
        "pipeline": "panorama_v1",
        "hf_dataset": args.inbox,
        "bundle_path": bundle_path,
        "parameters": {
            "steps": args.steps,
            "max_panoramas": args.max_panoramas,
            "image_size": args.image_size,
            "view_count": args.view_count,
            "hfov_deg": args.hfov,
            "sequence_window": 18,
            "retrieval_count": 10
        }
    }
    job_path = f"colab/real_scenes/panorama_v1/jobs/{job_id}.json"
    result = github_put(
        args.repo, args.branch, job_path,
        json.dumps(job, indent=2).encode("utf-8"), github_token,
        f"Queue NAMMA panorama reconstruction {job_id}"
    )
    print(json.dumps({
        "job_id": job_id,
        "panoramas": count,
        "bundle_mb": round(bundle.stat().st_size / 1e6, 1),
        "hf_input": f"https://huggingface.co/datasets/{args.inbox}/tree/main/{job_id}",
        "branch": args.branch,
        "job_manifest": job_path,
        "github_commit": result.get("commit", {}).get("html_url"),
        "message": "The panorama-only GitHub Actions workflow should now start."
    }, indent=2))


if __name__ == "__main__":
    main()
