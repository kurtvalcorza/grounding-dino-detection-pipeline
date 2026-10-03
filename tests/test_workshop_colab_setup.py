"""Colab's already imported NumPy is never replaced: the setup installs nothing into the kernel.

Before 2026-10-03 this test simulated the in-kernel pip install and checked that it kept Colab's
preloaded NumPy. The models now run in a uv isolated environment, so the setup cell must leave the
kernel's packages alone entirely: every subprocess it starts is the pinned uv binary, and the only
install targets the isolated interpreter.
"""
import ast
import json
from pathlib import Path


def runtime_cell():
    root = Path(__file__).resolve().parents[1]
    path = root / "tutorials" / "DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "326fd12e")


def test_colab_preloaded_numpy_survives_setup():
    source = runtime_cell()
    tree = ast.parse(source)
    assert "PINS" not in source and "importlib.metadata" not in source and "sys.executable" not in source
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
    ]
    installs = []
    for call in calls:
        if call.func.attr == "Popen":  # stage runs use the isolated interpreter
            continue
        assert call.func.attr == "run"
        command = ast.unparse(call.args[0])
        assert command.startswith("[str(UV), "), command
        if "'install'" in command:
            installs.append(command)
    assert len(installs) == 1
    assert "'--python', str(PYTHON)" in installs[0]
    # The kernel imports the NumPy it already has, after (and independent of) the environment build.
    assert any(isinstance(n, ast.Import) and any(a.name == "numpy" for a in n.names) for n in tree.body)
