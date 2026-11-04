from pathlib import Path

from gsplat_cuda_compat import patch_gsplat_cuda_jit_cache


LEGACY_BACKEND = (
    "import shutil\n"
    "\n"
    "def clean(build_dir):\n"
    "    try:\n"
    "        if build_dir:\n"
    "            shutil.rmtree(build_dir)\n"
    "    except OSError:\n"
    "        pass\n"
)
PATCHED_BACKEND = (
    "import shutil\n"
    "\n"
    "def clean(build_dir):\n"
    "    try:\n"
    "        if build_dir:\n"
    "            shutil.rmtree(build_dir, ignore_errors=True)\n"
    "    except OSError:\n"
    "        pass\n"
)


def test_patches_only_the_legacy_gsplat_jit_cleanup(tmp_path: Path):
    backend = tmp_path / "gsplat" / "cuda" / "_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(LEGACY_BACKEND, encoding="utf-8")

    patch_gsplat_cuda_jit_cache(tmp_path)
    assert backend.read_text(encoding="utf-8") == PATCHED_BACKEND
    patch_gsplat_cuda_jit_cache(tmp_path)


def test_patches_installed_gsplat_package_root(tmp_path: Path):
    package_root = tmp_path / "site-packages" / "gsplat"
    backend = package_root / "cuda" / "_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(LEGACY_BACKEND, encoding="utf-8")

    patch_gsplat_cuda_jit_cache(package_root)
    assert backend.read_text(encoding="utf-8") == PATCHED_BACKEND


def test_rejects_unrecognized_gsplat_backend(tmp_path: Path):
    backend = tmp_path / "gsplat" / "cuda" / "_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("different cleanup\n", encoding="utf-8")

    try:
        patch_gsplat_cuda_jit_cache(tmp_path)
    except RuntimeError as exc:
        assert "Unrecognized" in str(exc)
    else:
        raise AssertionError("expected guarded patch to reject unknown source")
