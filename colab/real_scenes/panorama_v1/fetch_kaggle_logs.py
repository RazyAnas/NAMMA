from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("out")
DEFAULT_OUTPUT_DATASET = "anasrazy/namma-panorama-out"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: fetch_kaggle_logs.py <diagnostic-manifest.json>")

    manifest_path = (ROOT / sys.argv[1]).resolve()
    if ROOT.resolve() not in manifest_path.parents or not manifest_path.is_file():
        raise RuntimeError("Diagnostic manifest must be a file inside panorama_v1.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_id = str(manifest.get("job_id", ""))
    kernel_id = str(manifest.get("kernel_id", ""))
    output_dataset = str(manifest.get("output_dataset", DEFAULT_OUTPUT_DATASET))

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{2,50}", job_id):
        raise ValueError("Invalid job_id in diagnostic manifest.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", kernel_id):
        raise ValueError("kernel_id must be an owner/kernel-slug identifier.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", output_dataset):
        raise ValueError("Invalid output dataset identifier.")

    hf_token = os.environ.get("HF_TOKEN", "")
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not hf_token or not kaggle_token:
        raise RuntimeError("HF_TOKEN and KAGGLE_API_TOKEN are both required.")

    OUT.mkdir(exist_ok=True)
    log_path = OUT / "kaggle_kernel.log"
    proc = subprocess.run(
        ["kaggle", "kernels", "logs", kernel_id],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if not log_text.strip():
        log_text = f"Kaggle logs returned no content (exit code {proc.returncode})."
    log_path.write_text(log_text, encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(log_path),
        path_in_repo=f"{job_id}/kaggle_kernel_full.log",
        repo_id=output_dataset,
        repo_type="dataset",
        commit_message=f"Store full Kaggle diagnostics for {job_id}",
    )

    print(f"Stored Kaggle diagnostics privately for {job_id}.")
    print(f"Log bytes: {log_path.stat().st_size}")
    print(f"Kaggle logs exit code: {proc.returncode}")
    if proc.returncode != 0:
        print("Kaggle logs command reported a non-zero exit; inspect the uploaded log.")
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
