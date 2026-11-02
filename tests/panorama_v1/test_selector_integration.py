from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from deploy_selector_integration import (
    CSS_APPEND,
    EVENT_ANCHOR,
    EXPORT_OLD,
    HTML_ANCHOR,
    HTML_PANEL,
    RENDER_DISABLED_ANCHOR,
    START_FUNCTION,
    patch_app,
    patch_css,
    patch_html,
)


def test_html_patch_adds_start_render_panel_once():
    original = "<section>" + HTML_ANCHOR + "</section>"
    patched = patch_html(original)
    assert 'id="start3DRenderBtn"' in patched
    assert HTML_PANEL in patched
    assert patch_html(patched) == patched


def test_app_patch_exports_gps_and_adds_secure_launcher():
    source = (
        EXPORT_OLD
        + RENDER_DISABLED_ANCHOR
        + "\n"
        + EVENT_ANCHOR
        + "  const downloadPresets = {};"
    )
    patched = patch_app(source, "feature/streetview-panorama-3d")
    assert "panoramas: selected.map((item, sequence)" in patched
    assert "start3DRendering()" in patched
    assert 'addEventListener("click", start3DRendering)' in patched
    assert "--branch 'feature/streetview-panorama-3d'" in patched

    # Reapplying for main should safely retarget both the downloader URL and job branch.
    main_version = patch_app(patched, "main")
    assert "raw.githubusercontent.com/RazyAnas/NAMMA/main/colab/" in main_version
    assert "--branch 'main'" in main_version
    assert patch_app(main_version, "main") == main_version


def test_css_patch_is_idempotent():
    patched = patch_css("")
    assert CSS_APPEND.strip() in patched
    assert patch_css(patched) == patched


def test_render_launcher_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "launcher.js"
        js.write_text(START_FUNCTION, encoding="utf-8")
        subprocess.run([node, "--check", str(js)], check=True, capture_output=True, text=True)
