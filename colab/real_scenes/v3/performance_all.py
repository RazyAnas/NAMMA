#!/usr/bin/env python3
"""Enable NAMMA Performance Mode for every currently live tour URL, in place.

The private results dataset holds the URL records; the public static Spaces hold
the pages. Stale URL records are skipped when their target page no longer exists.
Each unique live page is updated without changing its URL or scene assets.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "final"))
import make_tour  # noqa: E402


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_WRITE_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN / HUGGINGFACE_WRITE_TOKEN is required")

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    results_repo = f"{owner}/namma-v3-out"
    result_files = api.list_repo_files(results_repo, repo_type="dataset")

    # Gather every tour URL record, including named variants such as tour_v2 and tour2.
    url_records = [
        p for p in result_files
        if re.fullmatch(r"tour[^/]*_url\.txt", Path(p).name)
    ]
    if not url_records:
        raise SystemExit(f"No tour URL records found in {results_repo}")

    targets: dict[tuple[str, str], list[str]] = {}
    stale_refs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="namma-tour-perf-") as tmp:
        for ref_path in sorted(url_records):
            try:
                local_ref = hf_hub_download(
                    repo_id=results_repo,
                    filename=ref_path,
                    repo_type="dataset",
                    token=token,
                    local_dir=tmp,
                )
                url = Path(local_ref).read_text(encoding="utf-8").strip()
            except Exception as exc:
                raise RuntimeError(f"Could not read tour URL record {ref_path}: {exc}") from exc

            parsed = urlparse(url)
            host = parsed.hostname or ""
            prefix = owner + "-"
            suffix = ".static.hf.space"
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if not host.endswith(suffix) or not host[:-len(suffix)].startswith(prefix) or not parts:
                print(f"SKIP unsupported URL: {ref_path} -> {url}", flush=True)
                continue

            space_name = host[:-len(suffix)][len(prefix):]
            space_repo = f"{owner}/{space_name}"
            folder = parts[0]
            targets.setdefault((space_repo, folder), []).append(ref_path)

        # Enumerate each public Space once; folders without a page are stale and not active.
        space_files: dict[str, set[str]] = {}
        for space_repo, _ in targets:
            if space_repo not in space_files:
                space_files[space_repo] = set(
                    api.list_repo_files(space_repo, repo_type="space", token=token)
                )

        live = {
            key: refs for key, refs in targets.items()
            if f"{key[1]}/index.html" in space_files.get(key[0], set())
        }
        stale_refs = [
            ref
            for key, refs in targets.items()
            if key not in live
            for ref in refs
        ]
        if not live:
            raise SystemExit("No currently published tour pages were found.")

        updated: list[str] = []
        failures: list[str] = []
        for (space_repo, folder), refs in sorted(live.items()):
            page = f"{folder}/index.html"
            try:
                local_page = hf_hub_download(
                    repo_id=space_repo,
                    filename=page,
                    repo_type="space",
                    token=token,
                    local_dir=tmp,
                    force_download=True,
                )
                html = Path(local_page).read_text(encoding="utf-8")
                if 'id="sse-bootstrap"' not in html or "./index.js" not in html:
                    raise ValueError("page is not the expected NAMMA SuperSplat tour")

                content_match = re.search(r'"contentUrl"\\s*:\\s*"([^"]+)"', html)
                fmt = "lod" if content_match and "lod-meta.json" in content_match.group(1).lower() else "sog"
                budget_match = re.search(r'q\\.set\\("budget","(\\d+)"\\)', html)
                budget = int(budget_match.group(1)) if budget_match else 4

                patched = make_tour.head(html, fmt, budget)
                if 'localStorage.setItem("quality","performance")' not in patched:
                    raise ValueError("patched page is missing performance quality default")
                if 'localStorage.setItem("performanceMode","true")' not in patched:
                    raise ValueError("patched page is missing universal Performance Mode default")
                if 'localStorage.setItem("nammaQ","4")' not in patched:
                    raise ValueError("patched page is missing preference migration marker")

                Path(local_page).write_text(patched, encoding="utf-8")
                api.upload_file(
                    path_or_fileobj=local_page,
                    path_in_repo=page,
                    repo_id=space_repo,
                    repo_type="space",
                    commit_message="Enable Performance Mode by default on every device",
                )

                # Force a fresh Hub read so success means the live Space has the new settings.
                verify_page = hf_hub_download(
                    repo_id=space_repo,
                    filename=page,
                    repo_type="space",
                    token=token,
                    local_dir=tmp,
                    force_download=True,
                )
                verified = Path(verify_page).read_text(encoding="utf-8")
                assert 'localStorage.setItem("quality","performance")' in verified
                assert 'localStorage.setItem("performanceMode","true")' in verified
                assert 'localStorage.setItem("nammaQ","4")' in verified
                url = f"https://{space_repo.replace('/', '-')}.static.hf.space/{folder}/index.html"
                updated.append(url)
                print(f"VERIFIED {space_repo}/{page} ({len(refs)} URL record(s))", flush=True)
            except Exception as exc:
                failures.append(f"{space_repo}/{page}: {exc}")
                print(f"FAILED {space_repo}/{page}: {exc}", flush=True)

    print(
        f"SUMMARY url_records={len(url_records)} live_tours={len(live)} "
        f"updated={len(updated)} stale_url_records={len(stale_refs)} failed={len(failures)}",
        flush=True,
    )
    if stale_refs:
        print("STALE (target folder no longer exists): " + ", ".join(sorted(stale_refs)), flush=True)
    if failures:
        raise SystemExit("Some active tours were not updated: " + " | ".join(failures))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("# NAMMA tour performance defaults\n\n")
            summary.write(f"- URL records found: {len(url_records)}\n")
            summary.write(f"- Live tours updated and read-back verified: {len(updated)}\n")
            summary.write(f"- Stale URL records skipped: {len(stale_refs)}\n\n")
            summary.write("\n".join(f"- {url}" for url in updated) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
