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
