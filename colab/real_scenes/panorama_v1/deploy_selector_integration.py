"""Deploy the browser-only panorama submitter into NAMMA's existing static Space.

Only these four frontend files are committed:
  streetview-selector/index.html
  streetview-selector/app.js
  streetview-selector/style.css
  streetview-selector/panorama-client.js

The Space already has Hugging Face OAuth enabled with read-repos/write-repos scopes.
No personal token is embedded in or persisted by the browser client.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

SPACE = "anasrazy/namma-tour"
REPO_ROOT = Path(__file__).resolve().parents[3]
CLIENT_SOURCE = Path(__file__).resolve().parent / "selector" / "panorama-client.js"
FILES = (
    "streetview-selector/index.html",
    "streetview-selector/app.js",
    "streetview-selector/style.css",
    "streetview-selector/panorama-client.js",
)

OLD_DOWNLOADER_COPY = (
    'Uses the public Hugging Face Gradio downloader for the selected panorama IDs. '
    'No Google Maps Platform Map Tiles API key is requested for downloading. '
    'Test a few panoramas first; high zoom can take a long time.'
)

def patch_html(text: str) -> str:
    if 'id="panoramaJobProgress"' in text and 'panorama-client.js' in text:
        return text
    if 'id="downloadSelectedBtn"' not in text:
        raise ValueError("The live selector download button was not found.")

    text = re.sub(
        r'(<button id="downloadSelectedBtn"[^>]*>).*?(</button>)',
        r'\1Start 3D Rendering online <span>↗</span>\2',
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        OLD_DOWNLOADER_COPY,
        "Sign in with the NAMMA Hugging Face account. Selected panoramas are downloaded and uploaded directly in this browser; no Terminal, local launcher or pasted token is required.",
        1,
    )
    marker = '        <p class="microcopy">Sign in with the NAMMA Hugging Face account. Selected panoramas are downloaded and uploaded directly in this browser; no Terminal, local launcher or pasted token is required.</p>'
    if marker not in text:
        raise ValueError("Could not replace the old downloader instructions in index.html.")
    status = """
        <div id="panoramaJobProgress" class="progress-wrap hidden">
          <div class="progress-track"><div id="panoramaJobBar"></div></div>
          <div><span id="panoramaJobText">Preparing online panorama reconstruction…</span><b id="panoramaJobCount"></b></div>
          <a id="panoramaJobLink" class="panorama-job-link hidden" href="#" rel="noopener noreferrer">Open 3D tour ↗</a>
        </div>"""
    text = text.replace(marker, marker + status, 1)
    if '<script type="module" src="./panorama-client.js"></script>' not in text:
        text = text.replace(
            '<script src="./app.js"></script>',
            '<script src="./app.js"></script>\n  <script type="module" src="./panorama-client.js"></script>',
            1,
        )
    if 'type="module" src="./panorama-client.js"' not in text:
        raise ValueError("Could not load panorama-client.js from index.html.")
    return text

def patch_app(text: str) -> str:
    pattern = re.compile(r"  function downloadSelectedWithHfSpace\(\) \{[\s\S]*?\n  \}\n")
    replacement = """  function downloadSelectedWithHfSpace() {
    const selected = state.results
      .filter((item) => item.selected)
      .sort((a, b) => a.routeOrder - b.routeOrder);
    if (selected.length < 8) {
      toast("Select at least 8 panorama positions before starting 3D reconstruction.");
      return;
    }
    if (selected.length > 200) {
      toast("The current panorama pipeline supports up to 200 positions per job.");
      return;
    }
    if (selected.some((item) => !/^[A-Za-z0-9_-]+$/.test(item.panoId))) {
      toast("A panorama ID contains unexpected characters. Nothing was uploaded.");
      return;
    }
    if (typeof window.NAMMA_START_PANORAMA !== "function") {
      toast("The online panorama module is still loading. Please try again.");
      return;
    }
    const settings = {
      zoom: Number($("downloadZoom").value),
      columns: Number($("downloadCols").value),
      rows: Number($("downloadRows").value)
    };
    const payload = selected.map((item, sequence) => ({
      panoId: item.panoId,
      lat: item.lat,
      lng: item.lng,
      sequence: sequence
    }));
    $("downloadSelectedBtn").disabled = true;
    window.NAMMA_START_PANORAMA(payload, settings)
      .catch((error) => {
        const message = error && error.message ? error.message : String(error);
        toast("Online 3D submission failed: " + message);
      })
      .finally(() => {
        $("downloadSelectedBtn").disabled =
          state.results.filter((item) => item.selected).length < 8;
      });
  }
"""
    updated, count = pattern.subn(lambda _: replacement, text, count=1)
    if count != 1:
        raise ValueError("Could not safely replace the old Terminal launcher function in app.js.")
    if "document.createElement(\"a\")" in replacement or ".sh" in replacement:
        raise ValueError("Local launcher code unexpectedly remained in the replacement.")
    return updated

def patch_css(text: str) -> str:
    if "Browser-only NAMMA panorama job progress" in text:
        return text
    return text.rstrip() + """

/* Browser-only NAMMA panorama job progress */
.panorama-job-link{display:inline-block;margin-top:9px;color:#caff70;font:600 10px "DM Sans";text-decoration:none}
.panorama-job-link:hover{text-decoration:underline}
#panoramaJobBar.is-indeterminate{width:35%;animation:namma-panorama-progress 1.25s ease-in-out infinite}
@keyframes namma-panorama-progress{0%{transform:translateX(-110%)}100%{transform:translateX(310%)}}
"""

def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN / HUGGINGFACE_WRITE_TOKEN GitHub Action secret is missing.")
    if not CLIENT_SOURCE.is_file():
        raise SystemExit(f"Browser module is missing: {CLIENT_SOURCE}")

    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

    api = HfApi(token=token)
    identity = api.whoami()
    username = identity.get("name") or identity.get("user", {}).get("name")
    if username != "anasrazy":
        raise SystemExit(f"Refusing to deploy with Hugging Face account {username!r}; expected 'anasrazy'.")

    with tempfile.TemporaryDirectory(prefix="namma-panorama-selector-") as tmp:
        root = Path(snapshot_download(
            repo_id=SPACE,
            repo_type="space",
            token=token,
            allow_patterns=list(FILES[:3]),
            local_dir=tmp,
        ))
        html_path, app_path, css_path = (root / name for name in FILES[:3])
        for path in (html_path, app_path, css_path):
            if not path.is_file():
                raise SystemExit(f"Live selector file was not downloaded: {path}")

        old = {
            "html": html_path.read_text(encoding="utf-8"),
            "app": app_path.read_text(encoding="utf-8"),
            "css": css_path.read_text(encoding="utf-8"),
        }
        new = {
            "html": patch_html(old["html"]),
            "app": patch_app(old["app"]),
            "css": patch_css(old["css"]),
        }
        client_text = CLIENT_SOURCE.read_text(encoding="utf-8")
        if "oauthHandleRedirectIfPresent" not in client_text or "uploadFilesWithProgress" not in client_text:
            raise SystemExit("The browser client is missing the expected OAuth or direct-upload flow.")

        for path, key in zip((html_path, app_path, css_path), ("html", "app", "css")):
            path.write_text(new[key], encoding="utf-8")
        client_path = root / FILES[3]
        client_path.parent.mkdir(parents=True, exist_ok=True)
        client_path.write_text(client_text, encoding="utf-8")

        operations = [
            CommitOperationAdd(path_in_repo=FILES[0], path_or_fileobj=str(html_path)),
            CommitOperationAdd(path_in_repo=FILES[1], path_or_fileobj=str(app_path)),
            CommitOperationAdd(path_in_repo=FILES[2], path_or_fileobj=str(css_path)),
            CommitOperationAdd(path_in_repo=FILES[3], path_or_fileobj=str(client_path)),
        ]
        commit = api.create_commit(
            repo_id=SPACE,
            repo_type="space",
            operations=operations,
            commit_message="Replace terminal launcher with browser-only panorama submission",
        )
        print("Updated only the Street View selector frontend:")
        for name in FILES:
            print(" -", name)
        print("Space commit:", getattr(commit, "oid", str(commit)))
        print("Open: https://anasrazy-namma-tour.static.hf.space/streetview-selector/")

if __name__ == "__main__":
    main()
