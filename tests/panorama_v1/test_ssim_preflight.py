import os
import subprocess
import sys
from pathlib import Path

PANORAMA_DIR = (
    Path(__file__).resolve().parents[2]
    / "colab"
    / "real_scenes"
    / "panorama_v1"
)
sys.path.insert(0, str(PANORAMA_DIR))

from ssim_preflight import trainer_preflight_env


REQUIRED_TRAINER_FLAGS = (
    "--data-dir",
    "--data-factor",
    "--result-dir",
    "--max-steps",
    "--eval-steps",
    "--save-steps",
    "--ply-steps",
    "--save-ply",
    "--disable-viewer",
    "--disable-video",
    "--use-bilateral-grid",
    "--strategy.cap-max",
    "--depth-loss",
)


def test_gsplat_trainer_help_can_import_embedded_ssim_fallback(tmp_path: Path):
    shim_dir = tmp_path / "embedded-fused-ssim"
    trainer = tmp_path / "gsplat" / "examples" / "simple_trainer.py"
    shim_dir.mkdir()
    trainer.parent.mkdir(parents=True)
    (shim_dir / "fused_ssim.py").write_text(
        "def fused_ssim(img1, img2, **kwargs):\n    return 1.0\n",
        encoding="utf-8",
    )
    trainer.write_text(
        "from fused_ssim import fused_ssim\n"
        "if __name__ == '__main__':\n"
        f"    print({(' '.join(REQUIRED_TRAINER_FLAGS))!r})\n",
        encoding="utf-8",
    )

    command = [sys.executable, str(trainer), "mcmc", "--help"]

    # This reproduces the Kaggle failure: the trainer cannot import the fallback
    # when the subprocess inherits no fallback module path.
    before = subprocess.run(
        command,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert before.returncode != 0
    assert "No module named 'fused_ssim'" in before.stderr

    env = trainer_preflight_env(shim_dir, base_env={})
    after = subprocess.run(command, env=env, capture_output=True, text=True)
    assert after.returncode == 0, after.stderr
    assert all(flag in after.stdout for flag in REQUIRED_TRAINER_FLAGS)
    assert env["PYTHONPATH"] == str(shim_dir.resolve())


def test_trainer_preflight_rejects_missing_fallback(tmp_path: Path):
    try:
        trainer_preflight_env(tmp_path, base_env={})
    except FileNotFoundError as exc:
        assert "fused_ssim.py" in str(exc)
    else:
        raise AssertionError("Expected missing fused_ssim.py to fail before trainer launch")
