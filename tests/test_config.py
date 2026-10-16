from pathlib import Path

from app.config import Settings


def test_prefer_native_colmap_executable_for_windows_bundle(tmp_path: Path) -> None:
    wrapper = tmp_path / "COLMAP.bat"
    executable = tmp_path / "bin" / "colmap.exe"
    executable.parent.mkdir()
    wrapper.write_text("@echo off\n", encoding="utf-8")
    executable.write_bytes(b"")

    assert Settings._prefer_colmap_executable(wrapper) == str(executable.resolve())


def test_keep_standalone_colmap_wrapper(tmp_path: Path) -> None:
    wrapper = tmp_path / "custom-colmap.cmd"
    wrapper.write_text("@echo off\n", encoding="utf-8")

    assert Settings._prefer_colmap_executable(wrapper) == str(wrapper.resolve())
