from pathlib import Path

from colab.real_scenes.panorama_v1.gsplat_cuda_compat import patch_gsplat_cuda_jit_cache


def test_patches_only_the_legacy_gsplat_jit_cleanup(tmp_path: Path):
    backend = tmp_path / "gsplat" / "cuda" / "_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("before\n            shutil.rmtree(build_dir)\nafter\n", encoding="utf-8")

    patch_gsplat_cuda_jit_cache(tmp_path)
    assert backend.read_text(encoding="utf-8") == (
        "before\n            shutil.rmtree(build_dir, ignore_errors=True)\nafter\n"
    )
    patch_gsplat_cuda_jit_cache(tmp_path)


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
