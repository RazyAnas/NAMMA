from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from deploy_selector_integration import patch_app, patch_css, patch_html


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "colab" / "real_scenes" / "panorama_v1" / "selector" / "panorama-client.js"


def selector_html() -> str:
    return """<!doctype html>
<button id="downloadSelectedBtn" class="button">Download selected</button>
<p id="downloadQualityHint" class="microcopy">Current grid.</p>
<select id="downloadZoom">
  <option value="2" selected>Balanced</option>
  <option value="4">Very high</option>
  <option value="5">Maximum</option>
</select>
<input id="downloadCols" type="number" value="16">
<input id="downloadRows" type="number" value="8">
<p class="microcopy">Uses the public Hugging Face Gradio downloader for the selected panorama IDs. No Google Maps Platform Map Tiles API key is requested for downloading. Test a few panoramas first; high zoom can take a long time.</p>
<script src="./app.js"></script>
"""


def test_html_patch_adds_online_progress_and_high_resolution_defaults():
    patched = patch_html(selector_html())
    assert "Start 3D Rendering online" in patched
    assert 'id="panoramaJobProgress"' in patched
    assert 'type="module" src="./panorama-client.js"' in patched
    assert 'value="44"' in patched
    assert 'value="22"' in patched
    assert '<option value="4" selected>' in patched
    assert "968 tiles per panorama" in patched
    assert "no Terminal" in patched
    assert "local launcher" in patched
    assert patch_html(patched) == patched


def test_app_patch_calls_browser_submitter_without_local_launcher():
    source = '''  function downloadSelectedWithHfSpace() {
    // old local/terminal implementation
    const shell = ["#!/usr/bin/env bash", "read -r -s -p 'HF token' HF_TOKEN"].join("\\n");
    return shell;
  }
'''
    patched = patch_app(source)
    assert "window.NAMMA_START_PANORAMA(payload, settings)" in patched
    assert "panoramas" not in patched
    assert "python3 -m venv" not in patched
    assert ".sh" not in patched
    assert "read -r -s -p" not in patched


def test_css_patch_is_idempotent():
    patched = patch_css("")
    assert "Browser-only NAMMA panorama job progress" in patched
    assert patch_css(patched) == patched


def test_browser_module_uses_oauth_and_direct_private_uploads():
    source = CLIENT.read_text(encoding="utf-8")
    assert "oauthHandleRedirectIfPresent" in source
    assert "oauthLoginUrl" in source
    assert "uploadFilesWithProgress" in source
    assert "namma-panorama-inbox" in source
    assert 'image_size: 1280' in source
    assert 'steps: 30000' in source
    assert 'max_splats: 1500000' in source
    assert 'window.NAMMA_START_PANORAMA' in source
    assert "read -r -s -p" not in source
    assert "python3 -m venv" not in source
    assert "#!/usr/bin/env bash" not in source


def test_browser_module_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "panorama-client.mjs"
        module.write_text(CLIENT.read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run([node, "--check", str(module)], check=True, capture_output=True, text=True)
