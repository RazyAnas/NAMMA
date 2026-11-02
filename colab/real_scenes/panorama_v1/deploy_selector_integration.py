"""Deploy the secure panorama rendering launcher to the existing NAMMA static Space.

Only streetview-selector/index.html, app.js and style.css are changed.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

SPACE = "anasrazy/namma-tour"
FILES = (
    "streetview-selector/index.html",
    "streetview-selector/app.js",
    "streetview-selector/style.css",
)

HTML_ANCHOR = (
    '<p class="microcopy">Uses the public Hugging Face Gradio downloader for the selected panorama IDs. '
    "No Google Maps Platform Map Tiles API key is requested for downloading. Test a few panoramas first; "
    "high zoom can take a long time.</p>"
)
HTML_PANEL = """
        <div class="panorama-render-panel">
          <div class="panel-heading"><span>05</span><div><b>Reconstruct a 3D tour</b><small>PANORAMAS → GAUSSIAN SPLATTING</small></div></div>
          <p class="microcopy">Select at least 8 panorama positions and download their images above. For a meaningful street scene, 20–100+ well-spaced panoramas are recommended.</p>
          <button id="start3DRenderBtn" class="button button-lime button-wide" disabled>Start 3D Rendering <span>↗</span></button>
          <p id="panoramaRenderStatus" class="microcopy">This creates a local Terminal launcher. It keeps write tokens out of the browser and submits the selected images to NAMMA's separate panorama pipeline.</p>
        </div>
"""

EXPORT_OLD = '''  function exportSelected() {
    const ids = state.results.filter((item) => item.selected).map((item) => item.panoId);
    if (!ids.length) return;
    const payload = {
      format: "namma-streetview-route-v1",
      created_at: new Date().toISOString(),
      route: {
        type: "LineString",
        coordinates: state.routePoints.map((point) => [point.lng(), point.lat()])
      },
      panorama_ids: ids
    };
'''
EXPORT_NEW = '''  function exportSelected() {
    const selected = state.results
      .filter((item) => item.selected)
      .sort((a, b) => a.routeOrder - b.routeOrder);
    const ids = selected.map((item) => item.panoId);
    if (!ids.length) return;
    const payload = {
      format: "namma-streetview-route-v1",
      created_at: new Date().toISOString(),
      route: {
        type: "LineString",
        coordinates: state.routePoints.map((point) => [point.lng(), point.lat()])
      },
      panorama_ids: ids,
      panoramas: selected.map((item, sequence) => ({
        pano_id: item.panoId,
        lat: item.lat,
        lon: item.lng,
        sequence: sequence
      }))
    };
'''
START_FUNCTION = r'''  function start3DRendering() {
    const selected = state.results
      .filter((item) => item.selected)
      .sort((a, b) => a.routeOrder - b.routeOrder);
    if (selected.length < 8) {
      toast("Select at least 8 panorama positions before starting 3D reconstruction.");
      return;
    }
    if (selected.length > 200) {
      toast("The first panorama pipeline supports up to 200 panorama positions per job.");
      return;
    }
    if (selected.some((item) => !/^[A-Za-z0-9_-]+$/.test(item.panoId))) {
      toast("A panorama ID contains unexpected characters. Nothing was generated.");
      return;
    }
    const payload = {
      format: "namma-streetview-route-v1",
      created_at: new Date().toISOString(),
      route: {
        type: "LineString",
        coordinates: state.routePoints.map((point) => [point.lng(), point.lat()])
      },
      panorama_ids: selected.map((item) => item.panoId),
      panoramas: selected.map((item, sequence) => ({
        pano_id: item.panoId, lat: item.lat, lon: item.lng, sequence: sequence
      }))
    };
    const selectionBase64 = btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
    const submitUrl = "https://raw.githubusercontent.com/RazyAnas/NAMMA/feature/streetview-panorama-3d/colab/real_scenes/panorama_v1/submit_job.py";
    const shell = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "echo 'NAMMA · Street View panorama to 3D'",
      "echo 'Selected panorama positions: " + selected.length + "'",
      "echo 'Pipeline: panorama crops → camera reconstruction → Gaussian splat training → NAMMA tour'",
      "echo ''",
      "if ! command -v python3 >/dev/null 2>&1; then echo 'Python 3 is required.'; exit 1; fi",
      "WORK_DIR=\"$(mktemp -d -t namma-panorama-XXXXXX)\"",
      "trap 'rm -rf \"$WORK_DIR\"' EXIT",
      "OUT_DIR=\"$HOME/Downloads/NAMMA-StreetView-HFSPACE\"",
      "read -r -p \"Panorama image folder [$OUT_DIR]: \" INPUT_DIR",
      "if [ -z \"$INPUT_DIR\" ]; then INPUT_DIR=\"$OUT_DIR\"; fi",
      "if [ ! -d \"$INPUT_DIR\" ]; then echo \"Image folder not found: $INPUT_DIR\"; exit 1; fi",
      "python3 -c \"import pathlib,sys; p=pathlib.Path(sys.argv[1]); n=sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.png')); print('Image files found:',n); sys.exit(0 if n >= 8 else 2)\" \"$INPUT_DIR\" || { echo 'Download at least 8 panorama images first.'; exit 1; }",
      "echo ''",
      "read -r -s -p 'Hugging Face write token: ' HF_TOKEN; printf '\\n'",
      "read -r -s -p 'GitHub token (Contents: read/write on RazyAnas/NAMMA): ' GITHUB_TOKEN; printf '\\n'",
      "if [ -z \"$HF_TOKEN\" ] || [ -z \"$GITHUB_TOKEN\" ]; then echo 'Both tokens are required.'; exit 1; fi",
      "export HF_TOKEN GITHUB_TOKEN",
      "python3 -m venv \"$WORK_DIR/venv\"",
      "source \"$WORK_DIR/venv/bin/activate\"",
      "python -m pip install --upgrade pip",
      "python -m pip install 'huggingface_hub>=0.28'",
      "curl -fsSL '" + submitUrl + "' -o \"$WORK_DIR/submit_job.py\"",
      "printf '%s' '" + selectionBase64 + "' | python3 -c \"import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))\" > \"$WORK_DIR/vit-streetview-route-selection.json\"",
      "python \"$WORK_DIR/submit_job.py\" --input \"$INPUT_DIR\" --selection-json \"$WORK_DIR/vit-streetview-route-selection.json\" --scene-name 'NAMMA Street View route' --steps 30000 --max-panoramas 200 --image-size 768 --view-count 12 --branch 'feature/streetview-panorama-3d'",
      "unset HF_TOKEN GITHUB_TOKEN",
      "echo ''",
      "echo 'The panorama job has been submitted. Follow the NAMMA Panorama workflow in GitHub Actions.'",
      "echo 'The final tour link and private results will be published when the GPU job completes.'",
      ""
    ].join("\n");
    const blob = new Blob([shell], { type: "text/x-shellscript;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    anchor.href = url;
    anchor.download = "namma-start-panorama-3d-" + stamp + ".sh";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    $("panoramaRenderStatus").textContent =
      "Launcher prepared for " + selected.length + " panoramas. Download their JPEGs first, then run the .sh file in Terminal; tokens are entered privately there.";
    toast("Prepared the panorama 3D launcher for " + selected.length + " selected positions.");
  }
'''
RENDER_DISABLED_ANCHOR = '''    $("exportBtn").disabled = !state.results.some((item) => item.selected);
    if (!state.results.length) {'''
RENDER_DISABLED_REPLACEMENT = '''    $("exportBtn").disabled = !state.results.some((item) => item.selected);
    $("start3DRenderBtn").disabled = state.results.filter((item) => item.selected).length < 8;
    if (!state.results.length) {'''
EVENT_ANCHOR = '''  $("downloadSelectedBtn").addEventListener("click", downloadSelectedWithHfSpace);
'''
EVENT_REPLACEMENT = '''  $("downloadSelectedBtn").addEventListener("click", downloadSelectedWithHfSpace);
  $("start3DRenderBtn").addEventListener("click", start3DRendering);
'''
CSS_APPEND = """
/* Independent panorama-to-3D launch panel */
.panorama-render-panel{border-top:1px solid #303a2d;padding:17px 0 18px}
.panorama-render-panel .panel-heading{margin-bottom:10px}
.panorama-render-panel #start3DRenderBtn{margin-top:12px}
.panorama-render-panel #panoramaRenderStatus{color:#b8c99f}
"""


def patch_html(text: str) -> str:
    if 'id="start3DRenderBtn"' in text:
        return text
    if HTML_ANCHOR not in text:
        raise ValueError("Could not find the Street View downloader section in index.html.")
    return text.replace(HTML_ANCHOR, HTML_ANCHOR + HTML_PANEL, 1)


def patch_app(text: str) -> str:
    if EXPORT_OLD in text:
        text = text.replace(EXPORT_OLD, EXPORT_NEW, 1)
    elif 'panoramas: selected.map((item, sequence)' not in text:
        raise ValueError("Could not identify the current exportSelected() function in app.js.")

    if "function start3DRendering()" not in text:
        marker = "  const downloadPresets ="
        if text.count(marker) != 1:
            raise ValueError("Could not find the downloader setup section in app.js.")
        text = text.replace(marker, START_FUNCTION + "\n" + marker, 1)

    if RENDER_DISABLED_REPLACEMENT not in text:
        if text.count(RENDER_DISABLED_ANCHOR) != 1:
            raise ValueError("Could not find the renderResults() enable/disable logic in app.js.")
        text = text.replace(RENDER_DISABLED_ANCHOR, RENDER_DISABLED_REPLACEMENT, 1)

    if 'addEventListener("click", start3DRendering)' not in text:
        if text.count(EVENT_ANCHOR) != 1:
            raise ValueError("Could not find the downloader button event binding in app.js.")
        text = text.replace(EVENT_ANCHOR, EVENT_REPLACEMENT, 1)
    return text


def patch_css(text: str) -> str:
    if "Independent panorama-to-3D launch panel" in text:
        return text
    return text.rstrip() + "\n" + CSS_APPEND


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: python3 -m pip install huggingface_hub") from exc

    api = HfApi(token=token) if token else HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("user", {}).get("name")
    if username != "anasrazy":
        raise SystemExit(f"Log in as anasrazy before deploying; current account is {username!r}.")

    with tempfile.TemporaryDirectory(prefix="namma-selector-patch-") as tmp:
        local = Path(snapshot_download(
            repo_id=SPACE, repo_type="space", token=token,
            allow_patterns=list(FILES), local_dir=tmp
        ))
        html_path, app_path, css_path = (local / name for name in FILES)
        for path in (html_path, app_path, css_path):
            if not path.is_file():
                raise SystemExit(f"Space file was not downloaded: {path}")
        html_old = html_path.read_text(encoding="utf-8")
        app_old = app_path.read_text(encoding="utf-8")
        css_old = css_path.read_text(encoding="utf-8")
        html_new, app_new, css_new = patch_html(html_old), patch_app(app_old), patch_css(css_old)
        if (html_new, app_new, css_new) == (html_old, app_old, css_old):
            print("The panorama rendering integration is already deployed; no files changed.")
            return
        html_path.write_text(html_new, encoding="utf-8")
        app_path.write_text(app_new, encoding="utf-8")
        css_path.write_text(css_new, encoding="utf-8")
        operations = [
            CommitOperationAdd(path_in_repo=FILES[0], path_or_fileobj=str(html_path)),
            CommitOperationAdd(path_in_repo=FILES[1], path_or_fileobj=str(app_path)),
            CommitOperationAdd(path_in_repo=FILES[2], path_or_fileobj=str(css_path)),
        ]
        commit = api.create_commit(
            repo_id=SPACE, repo_type="space", operations=operations,
            commit_message="Add secure Start 3D Rendering action to Street View Selector"
        )
        print("Updated only:")
        for name in FILES:
            print(" -", name)
        print("Space commit:", getattr(commit, "oid", str(commit)))
        print("Open: https://anasrazy-namma-tour.static.hf.space/streetview-selector/")


if __name__ == "__main__":
    main()
