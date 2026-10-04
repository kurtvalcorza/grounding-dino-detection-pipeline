"""The open-vocabulary workshop runs its models in a uv isolated environment (2026-10-03)."""

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELL_BY_ID = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"]}
CODE = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"] if cell["cell_type"] == "code"}
RUNNER = ROOT / "tools" / "ovd_workshop.py"
LOCK = ROOT / "tools" / "ovd-workshop-requirements.lock"
REQUIREMENTS_IN = ROOT / "tools" / "ovd-workshop-requirements.in"
RUNTIME_CELL = CELL_BY_ID["326fd12e"]


def load_runner():
    spec = importlib.util.spec_from_file_location("ovd_workshop_uv_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def carrier_literals():
    tree = ast.parse(CELL_BY_ID["uvcarrier"])
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"CARRIED_FILES", "CARRIED_HASHES"}):
            values[node.targets[0].id] = node.value
    return values


def test_no_kernel_install_and_no_restart_guard():
    # The carrier cell only writes carried text (the lock's header names the uv install command).
    for cell_id, text in CODE.items():
        if cell_id == "uvcarrier":
            continue
        assert not re.search(r"pip\s+install|-m\s+pip|^\s*[!%]pip", text, re.M), cell_id
        assert "Restart session" not in text and "restart" not in text.lower(), cell_id
        assert "importlib.metadata" not in text, cell_id


def test_kernel_imports_only_stdlib_and_hosted_display_libraries():
    allowed_third_party = {"numpy", "PIL", "IPython"}
    for cell_id, text in CODE.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.partition(".")[0]
                assert top in sys.stdlib_module_names or top in allowed_third_party, (cell_id, name)


def test_lock_is_fully_hashed_and_keeps_the_former_pins():
    text = LOCK.read_text(encoding="utf-8")
    assert "\r" not in text
    requirements = re.findall(r"^([A-Za-z0-9_.-]+)==(\S+) \\$", text, re.M)
    assert len(requirements) > 20
    blocks = re.split(r"^(?=[A-Za-z0-9_.-]+==)", text, flags=re.M)[1:]
    assert len(blocks) == len(requirements)
    for block in blocks:
        assert "--hash=sha256:" in block, block.splitlines()[0]
    locked = {name.lower(): version for name, version in requirements}
    former = {"torch": "2.14.0", "torchvision": "0.29.0", "torchaudio": "2.11.0", "transformers": "4.57.6",
              "safetensors": "0.8.0", "numpy": "2.1.3", "pillow": "11.3.0", "huggingface-hub": "0.36.2"}
    assert {k: locked[k] for k in former} == former
    added = {"scipy": "1.18.1"}  # 2026-10-04: needed by the OWLv2 image processor
    assert {k: locked[k] for k in added} == added
    direct = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)$",
                             REQUIREMENTS_IN.read_text(encoding="utf-8"), re.M))
    assert direct == {**former, **added} == load_runner().PINS


def test_environment_is_built_hash_locked_wheels_only_and_stages_use_its_interpreter():
    for needle in ['"--require-hashes"', '"--only-binary", ":all:"', '"--managed-python"', '"3.12.12"',
                   'UV_SHA256 = "', 'PYTHON = ENV_ROOT / "bin" / "python"', 'command = [str(PYTHON), "-u"',
                   'ENV["MPLBACKEND"] = "Agg"', 'platform.machine() != "x86_64"']:
        assert needle in RUNTIME_CELL, needle
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "HF_TOKEN"):
        assert f'"{name}"' in RUNTIME_CELL.split("ENV.pop")[0].rsplit("for name in", 1)[1]
    assert "subprocess.Popen(command, env=ENV" in RUNTIME_CELL
    # Every model stage goes through run_stage; no model code is left in the kernel.
    for cell_id, stage in [("48be2c06", "weights"), ("53f2f300", "gdino"), ("2f66f39b", "owlv2"),
                           ("3f9f36da", "byod"), ("326fd12e", "runtime")]:
        assert f'run_stage("{stage}"' in CODE[cell_id] or f"run_stage('{stage}'" in CODE[cell_id], cell_id
    kernel = "\n".join(CODE.values())
    for model_call in ("from_pretrained", "hf_hub_download", "torch.", "inference_mode"):
        assert model_call not in kernel.replace(CELL_BY_ID["uvcarrier"], ""), model_call


def test_no_cell_line_over_2000_characters_and_carried_pieces_at_most_1000():
    for cell in NB["cells"]:
        for line in "".join(cell["source"]).split("\n"):
            assert len(line) <= 2000, (cell["id"], len(line))
    literals = carrier_literals()
    files = ast.literal_eval(literals["CARRIED_FILES"])
    hashes = ast.literal_eval(literals["CARRIED_HASHES"])
    # Each carried file is a parenthesised run of one-line string pieces of at most 1,000 characters.
    # Implicit concatenation folds them into one constant, so the pieces are read from the source lines.
    source_lines = CELL_BY_ID["uvcarrier"].split("\n")
    for value in literals["CARRIED_FILES"].values:
        pieces = [ast.literal_eval(line.strip()) for line in source_lines[value.lineno - 1:value.end_lineno]]
        assert len(pieces) > 1
        assert all(len(piece) <= 1000 for piece in pieces), max(map(len, pieces))
        assert "".join(pieces) == value.value
    sources = {"ovd_workshop.py": RUNNER, "requirements.lock.txt": LOCK}
    for name, path in sources.items():
        assert files[name] == path.read_text(encoding="utf-8")
        assert hashes[name] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert NB["metadata"]["dimer"]["carried_files"][name]["sha256"] == hashes[name]


def test_carrier_tool_check_passes():
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "build_ovd_workshop_carrier.py"), "--check"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_revision_logged():
    revisions = NB["metadata"]["dimer"]["revisions"]
    assert any(r["date"] == "2026-10-03" and "uv isolated environment" in r["change"] for r in revisions)


def test_runner_imports_without_model_libraries(forbid_model_imports):
    runner = load_runner()
    assert runner.torch is None and set(runner.STAGES) == {"runtime", "weights", "gdino", "owlv2", "byod"}


def test_stage_refuses_pixels_that_differ_from_the_kernel_digest(tmp_path):
    runner = load_runner()
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    good = runner.image_digest(Image.open(path))
    assert runner.load_images([{"id": "a", "path": str(path), "pixel_sha256": good}])[0]["id"] == "a"
    with pytest.raises(RuntimeError, match="decoded pixels differ"):
        runner.load_images([{"id": "a", "path": str(path), "pixel_sha256": "0" * 64}])


def kernel_namespace(records, prompt_ids):
    ns = {"np": __import__("numpy"), "Path": Path, "MAX_PROMPTS": 16, "MAX_PROMPT_CHARS": 48,
          "IOU_THRESHOLDS": (0.5,)}
    for cell_id in ("7fff2d1b", "ff6dbcd8", "53f2f300"):
        module = ast.parse(CELL_BY_ID[cell_id])
        module.body = [n for n in module.body if isinstance(n, ast.FunctionDef)]
        exec(compile(module, cell_id, "exec"), ns)
    ns.update(test_records=records, canonical_prompts=["a", "b"],
              prompt_experiment_records=[r for r in records if r["id"] in prompt_ids])
    return ns


def test_stage_json_round_trips_to_the_kernel_objects_exactly(tmp_path):
    """run_model's output, through strict JSON, rebuilds the kernel's former in-process objects."""
    runner = load_runner()
    runner.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    runner.configure({"settings": {
        "run_prompt_experiment": True, "run_threshold_sweep": True, "reordered_prompts": ["b", "a"],
        "prompt_variants": {"canonical": ["a", "b"], "upper": ["A", "B"]},
        "prompt_variant_to_canonical": {"canonical": {"a": "a", "b": "b"}, "upper": {"a": "a", "b": "b"}},
    }})
    records = [
        {"id": f"r{i}", "image": Image.new("RGB", (32, 32)), "boxes": [[0, 0, 16, 16]], "labels": ["a"]}
        for i in range(3)
    ]

    def predict(image, prompts, floor=None):
        vocab = runner.normalize_prompt_list(prompts)
        return [(0.1 + 1 / 3, vocab[0], [0.5, 1 / 7, 16.25, 31.0]), (0.05, vocab[-1], [2.0, 2.0, 3.0, 3.0])]

    result = runner.run_model("fake", predict, records, records[:2], ["a", "b"], predict)
    payload = json.loads(json.dumps({"stage": "gdino", **result}, allow_nan=False))
    ns = kernel_namespace(records, {"r0", "r1"})
    predictions, visual, prompt_results, sweep = ns["unpack_model_result"](payload, "fake")
    assert predictions == [predict(None, ["a", "b"])] * 3
    assert visual == {"r0": predict(None, ["a", "b"]), "r1": predict(None, ["a", "b"])}
    assert [(v, i) for v, i, _p in prompt_results] == [
        ("canonical", "r0"), ("canonical", "r1"), ("upper", "r0"), ("upper", "r1"),
        ("order_canonical", "r0"), ("order_reordered", "r0")]
    assert prompt_results[-1][2] == predict(None, ["b", "a"])
    assert [floor for floor, _p in sweep] == [0.03, 0.05, 0.10, 0.20]
    assert len(payload["times"]) == 3 and payload["peak_gpu_memory_bytes"] is None


def test_scipy_is_pinned_for_the_owlv2_image_processor():
    # transformers' slow Owlv2ImageProcessor imports SciPy. The 2026-10-04 Colab T4 run of 873e998 failed
    # without it, because the old in-kernel install got SciPy from the Colab image and never pinned it.
    assert re.search(r"^scipy==\d", REQUIREMENTS_IN.read_text(encoding="utf-8"), re.M)
    lock = LOCK.read_text(encoding="utf-8")
    assert re.search(r"^scipy==\S+ \\\n\s+--hash=sha256:[0-9a-f]{64}", lock, re.M)
    assert '"scipy":' in RUNNER.read_text(encoding="utf-8")
