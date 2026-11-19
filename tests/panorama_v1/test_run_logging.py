import ast
import time
from pathlib import Path


RUN_PATH = (
    Path(__file__).resolve().parents[2]
    / "colab"
    / "real_scenes"
    / "panorama_v1"
    / "run.py"
)


def test_say_accepts_flush_keyword_and_preserves_default_flush():
    tree = ast.parse(RUN_PATH.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "say"
    ]
    assert len(functions) == 1

    print_calls = []

    def capture_print(*args, **kwargs):
        print_calls.append(kwargs.get("flush"))

    namespace = {"time": time, "T0": time.time(), "print": capture_print}
    module = ast.Module(body=[functions[0]], type_ignores=[])
    exec(compile(module, str(RUN_PATH), "exec"), namespace)

    namespace["say"]("default flush")
    namespace["say"]("explicit flush", flush=True)

    assert print_calls == [True, True]
