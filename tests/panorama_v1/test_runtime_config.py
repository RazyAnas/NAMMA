import ast
from pathlib import Path


def test_gsplat_torch_constraint_ends_with_a_real_newline():
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "colab" / "real_scenes" / "panorama_v1" / "run.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pin"
    ]
    assert len(calls) == 1, "Expected exactly one write to torch-pin.txt"

    requirement = calls[0].args[0]
    assert isinstance(requirement, ast.JoinedStr)
    final_part = requirement.values[-1]
    assert isinstance(final_part, ast.Constant)
    assert final_part.value == "\n", (
        "torch-pin.txt must end with one real newline, not a literal backslash-n"
    )


def test_embedded_ssim_fallback_is_self_contained_and_matches_reference():
    root = Path(__file__).resolve().parents[2]
    runner = root / "colab/real_scenes/panorama_v1/run.py"
    reference = root / "colab/real_scenes/panorama_v1/fused_ssim_shim/fused_ssim.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))

    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SSIM_SHIM_SOURCE"
                for target in node.targets)
    ]
    assert len(values) == 1, "Runner must contain exactly one embedded SSIM source"
    embedded = ast.literal_eval(values[0])
    assert embedded == reference.read_text(encoding="utf-8")
    shim_tree = ast.parse(embedded)
    functions = {
        node.name for node in shim_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {"_window", "fused_ssim"} <= functions
    assert "torch.nn.functional as F" in embedded
    assert "groups=channels" in embedded
    assert "BCHW image tensors" in embedded
    compile(embedded, "embedded_fused_ssim.py", "exec")


def test_runner_materializes_embedded_ssim_and_smoke_tests_cuda_before_training():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "colab/real_scenes/panorama_v1/run.py").read_text(encoding="utf-8")
    assert "def ensure_ssim_shim()" in runner
    assert "shim_file.write_text(SSIM_SHIM_SOURCE" in runner
    assert "Embedded fused_ssim CUDA preflight failed before reconstruction" in runner
    assert "torch.rand((1,3,16,16), device='cuda')" in runner
    assert "Fused SSIM is unavailable and the panorama fallback shim is missing" not in runner


def test_runner_isolates_the_torch_extension_cache_for_kaggle():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "colab/real_scenes/panorama_v1/run.py").read_text(encoding="utf-8")
    assert 'os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "torch-extensions")' in runner
    assert 'train_env["TORCH_EXTENSIONS_DIR"] = str(ROOT / "torch-extensions")' in runner


def test_route_planner_is_loaded_by_explicit_panorama_file_path():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "colab/real_scenes/panorama_v1/run.py").read_text(encoding="utf-8")
    assert 'panorama_tour_path = HERE / "make_tour.py"' in runner
    assert 'spec_from_file_location(\n        "namma_panorama_route_plan"' in runner
    assert "import make_tour as panorama_tour" not in runner


def test_gpu_and_trainer_preflights_run_before_expensive_sfm():
    root = Path(__file__).resolve().parents[2]
    runner = root / "colab/real_scenes/panorama_v1/run.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")

    def statement_index(call_name):
        for index, statement in enumerate(main.body):
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == call_name
                for node in ast.walk(statement)
            ):
                return index
        raise AssertionError(f"main() does not call {call_name}")

    assert statement_index("find_bundle") < statement_index("preflight_ssim_fallback")
    assert statement_index("preflight_ssim_fallback") < statement_index("install_gsplat")
    assert statement_index("install_gsplat") < statement_index("preflight_gsplat_trainer")
    assert statement_index("preflight_gsplat_trainer") < statement_index("preflight_gsplat_cuda_backend")
    assert statement_index("preflight_gsplat_cuda_backend") < statement_index("preflight_native_ssim")
    assert statement_index("preflight_native_ssim") < statement_index("install_sfm_environment")
    assert statement_index("install_sfm_environment") < statement_index("run")

    assert "def preflight_native_ssim()" in runner.read_text(encoding="utf-8")
    assert "v=fused_ssim.fused_ssim(x,y)" in runner.read_text(encoding="utf-8")
    assert "def preflight_gsplat_cuda_backend()" in runner.read_text(encoding="utf-8")
    assert "gsplat CUDA backend preflight failed before SfM" in runner.read_text(encoding="utf-8")


def test_gsplat_trainer_preflight_checks_every_flag_used_by_training():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "colab/real_scenes/panorama_v1/run.py").read_text(encoding="utf-8")
    required = (
        "--data-dir", "--data-factor", "--result-dir", "--max-steps",
        "--eval-steps", "--save-steps", "--ply-steps", "--save-ply",
        "--disable-viewer", "--disable-video", "--use-bilateral-grid",
        "--strategy.cap-max", "--depth-loss",
    )
    for flag in required:
        assert f'"{flag}"' in runner


def test_runner_patches_the_installed_gsplat_cuda_backend_after_pip_install():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "colab/real_scenes/panorama_v1/run.py").read_text(encoding="utf-8")
    install = runner.split("def install_gsplat():", 1)[1].split("\ndef main():", 1)[0]

    install_packages = 'run([sys.executable, "-m", "pip", "install", "-q", "-c", pin, *packages])'
    patch_installed = "patch_gsplat_cuda_jit_cache(installed_gsplat_root)"
    assert install_packages in install
    assert "installed_gsplat_root = Path(gsplat.__file__).resolve().parent" in install
    assert patch_installed in install
    assert install.index(install_packages) < install.index(patch_installed)
    assert "gsplat CUDA JIT cleanup patch applied to installed package." in install
