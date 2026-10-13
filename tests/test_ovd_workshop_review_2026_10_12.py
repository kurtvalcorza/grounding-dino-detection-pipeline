"""Regression tests for the 2026-10-12 review round of the open-vocabulary detection workshop (GDN2-M1..M3,
GDN2-m1..m8). They exercise the notebook's own cells and the carried stage file with NumPy/Pillow stand-ins;
no model weights are loaded."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from test_open_vocabulary_workshop import CELL_BY_ID, CELLS, fake_models, fixture, helpers, load_runner

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads(
    (ROOT / "tutorials" / "DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb").read_text(encoding="utf-8")
)
MARKDOWN = "\n".join("".join(c["source"]) for c in CELLS if c["cell_type"] == "markdown")
CANONICAL = ["red blood cell", "white blood cell", "platelet"]


def cell_functions(ns, cell_id, names):
    module = ast.parse(CELL_BY_ID[cell_id])
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(module, cell_id, "exec"), ns)
    return ns


# ------------------------------------------------------------------- GDN2-m1: one copy of the diagnostics

FAKE_STAGE = """
import sys
stage = sys.argv[sys.argv.index("--stage") + 1]
if stage == "byod":
    print("Refused: boxes.csv labels: too long. Fix your dataset, then run the BYOD cell again.")
    sys.exit(65)
print("Traceback (most recent call last):")
print("ValueError: child failure marker")
sys.exit(65 if stage == "weights" else 1)
"""


def run_stage_ns(tmp_path):
    script = tmp_path / "fake_stage.py"
    script.write_text(FAKE_STAGE, encoding="utf-8")
    ns = {"json": json, "os": os, "subprocess": subprocess, "WORK_ROOT": tmp_path / "work",
          "PYTHON": sys.executable, "STAGE_SCRIPT": script, "ENV": dict(os.environ)}
    return cell_functions(ns, "326fd12e", {"run_stage"})


def test_stage_failure_prints_child_traceback_once(tmp_path, capsys):
    ns = run_stage_ns(tmp_path)
    with pytest.raises(RuntimeError, match="gdino failed with exit 1"):
        ns["run_stage"]("gdino", {})
    out = capsys.readouterr().out
    assert out.count("child failure marker") == 1


def test_byod_refusal_returns_none_and_other_exit_65_still_raises(tmp_path, capsys):
    ns = run_stage_ns(tmp_path)
    assert ns["run_stage"]("byod", {}) is None
    assert capsys.readouterr().out.count("Refused:") == 1
    with pytest.raises(RuntimeError, match="weights failed with exit 65"):
        ns["run_stage"]("weights", {})


# --------------------------------------------------------------------------- GDN2-m1/m2: carried stage file

def test_stage_main_turns_byod_refusal_into_one_line_and_exit_65(tmp_path, capsys):
    runner = load_runner()
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")

    def refused(_job):
        raise runner.InputRefused("img-1: decoded pixels differ")

    runner.STAGES["byod"] = refused
    with pytest.raises(SystemExit) as stop:
        runner.main(["--stage", "byod", "--job", str(job), "--out", str(tmp_path / "out.json")])
    assert stop.value.code == runner.REFUSED_EXIT == 65
    out = capsys.readouterr().out
    assert out.count("Refused:") == 1 and "img-1" in out and "Traceback" not in out
    runner.STAGES["gdino"] = refused  # any other stage keeps its traceback
    with pytest.raises(runner.InputRefused):
        runner.main(["--stage", "gdino", "--job", str(job), "--out", str(tmp_path / "out.json")])


def test_stage_byod_refuses_changed_pixels(tmp_path):
    runner = load_runner()
    runner.load_libraries = lambda: None
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(runner.InputRefused, match="a: decoded pixels differ"):
        image = {"id": "a", "path": str(path), "pixel_sha256": "0" * 64}
        runner.stage_byod({"images": [image], "prompts": ["x"]})


def test_byod_token_limit_is_an_input_refusal():
    ns = vars(load_runner())
    _refs, events = fake_models(ns, bad_tokens=True)
    with pytest.raises(ns["InputRefused"], match="boxes.csv labels"):
        ns["predict_byod"]([{"image": None}], ["a cell"])
    assert events == []


# --------------------------------------------------------------------------- GDN2-m2: kernel BYOD refusals


@pytest.mark.parametrize(
    "change, expected",
    [
        (lambda r, p: r[0].update(x0="abc"), "boxes.csv line 2: box coordinates"),
        (lambda r, p: r[2].update(label=""), "boxes.csv line 4: label ''"),
        (lambda r, p: r[1].update(filename="missing.png"), "boxes.csv line 3: 'missing.png' is missing"),
        (lambda r, p: r[0].update(x1=100), "boxes.csv line 2: box"),
        (lambda r, p: r.append(dict(r[0])), "boxes.csv line 10: duplicate annotation"),
        (lambda r, p: Image.new("RGB", (32, 32), "white").save(p / "extra.png"), "extra.png"),
        (lambda r, p: (p / "7.png").write_bytes((p / "0.png").read_bytes()), "7.png: same pixels as '0.png'"),
        (lambda r, p: (p / "3.png").write_bytes(b"not an image"), "3.png: cannot be read as an image"),
    ],
)
def test_byod_refusal_names_file_or_line(tmp_path, change, expected):
    ns = helpers(tmp_path)
    with pytest.raises(ns["BYODInputRefused"]) as refused:
        ns["load_byod_dataset"](fixture(tmp_path, change))
    assert expected in str(refused.value)
    assert isinstance(refused.value, ValueError)


def byod_ns(tmp_path, dataset):
    ns = helpers(tmp_path)
    ns.update(USE_BYOD=True, BYOD_DATASET_PATH=str(dataset), GDINO_DIR=Path("gd"), OWLV2_DIR=Path("owl"),
              byod_result={"stale": True})
    return ns


def test_byod_cell_prints_one_refusal_without_raising(tmp_path, capsys):
    ns = byod_ns(tmp_path, fixture(tmp_path, lambda r, p: r[0].update(x0="abc")))
    capsys.readouterr()  # drop what the helper's own exec of the cell printed
    exec(CELL_BY_ID["3f9f36da"], ns)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("Refused: boxes.csv line 2")
    assert ns["byod_result"] is None and not (Path(ns["OUTPUT_DIR"]) / "byod").exists()


def test_byod_cell_stage_refusal_exports_nothing(tmp_path, capsys):
    ns = byod_ns(tmp_path, fixture(tmp_path))
    calls = []
    ns["run_stage"] = lambda stage, job: calls.append(stage)  # returns None: the stage refused the input
    exec(CELL_BY_ID["3f9f36da"], ns)
    assert calls == ["byod"] and ns["byod_result"] is None
    assert not (Path(ns["OUTPUT_DIR"]) / "byod").exists()
    assert "Traceback" not in capsys.readouterr().out


# --------------------------------------------------------------------------- GDN2-M1/M2: where the boxes land


def record(boxes, labels):
    return {"id": "r", "image": Image.new("RGB", (64, 64)), "boxes": boxes, "labels": labels}


def confusion_ns(tmp_path):
    ns = helpers(tmp_path)
    ns["canonical_prompts"] = list(CANONICAL)
    return cell_functions(ns, "dd94f31a", {"phrase_confusion", "localized_by_class"})


def test_phrase_confusion_shows_naming_and_localization(tmp_path):
    ns = confusion_ns(tmp_path)
    red, white = [0, 0, 10, 10], [30, 30, 50, 50]
    rec = record([red, white], ["red blood cell", "white blood cell"])
    preds = [[(0.9, "white blood cell", red), (0.8, "white blood cell", white),
              (0.3, "platelet", [55, 55, 60, 60])]]
    table = ns["phrase_confusion"](preds, [rec])
    assert table["white blood cell"]["red blood cell"] == 1  # a red cell named "white blood cell"
    assert table["white blood cell"]["white blood cell"] == 1
    assert table["platelet"]["no reference"] == 1
    assert sum(sum(row.values()) for row in table.values()) == 3
    found = ns["localized_by_class"](preds, [rec])
    assert found == {"red blood cell": (1, 1), "white blood cell": (1, 1), "platelet": (0, 0)}
    # The evaluator itself still scores the red cell as missed: its AP measures naming.
    assert ns["detection_metrics"](preds, [rec], CANONICAL)["per_phrase"]["red blood cell"]["recall50"] == 0.0


# --------------------------------------------------------------------------- GDN2-M3: prompt experiment


def test_prompt_table_sums_every_prompt_image_with_one_to_one_assignment(tmp_path):
    ns = confusion_ns(tmp_path)
    a = dict(record([[0, 0, 10, 10]], ["red blood cell"]), id="a")
    b = dict(record([[0, 0, 10, 10], [20, 20, 30, 30]], ["red blood cell", "red blood cell"]), id="b")
    ns["prompt_experiment_records"] = [a, b]
    cell_functions(ns, "1c25de4f", {"prompt_match_table"})
    rows = [
        ("canonical", "a", [(0.9, "red blood cell", [0, 0, 10, 10]),
                            (0.8, "red blood cell", [0, 0, 10, 10])]),
        ("canonical", "b", [(0.9, "white blood cell", [0, 0, 10, 10])]),
        ("order_canonical", "a", [(0.9, "red blood cell", [0, 0, 10, 10])]),
    ]
    table = ns["prompt_match_table"](rows)
    red = table[("canonical", "red blood cell")]
    assert red == {"images": 2, "gt": 3, "boxes": 2, "true_positives": 1, "duplicates": 1, "wrong_phrase": 0}
    assert table[("canonical", "white blood cell")]["wrong_phrase"] == 1
    assert not any(k[0].startswith("order_") for k in table)


def test_prompt_cell_no_longer_prints_only_the_first_image():
    src = CELL_BY_ID["1c25de4f"]
    assert "prompt_experiment_records[0]" not in src and "prompt_match_table(rows)" in src
    # The exported per-image rows are still built by the unchanged summary function.
    assert 'prompt_sensitivity_rows += prompt_result_summary("OWLv2 Base/16 Ensemble"' in src


# --------------------------------------------------------------------------- declarations and learner text


def test_metadata_declares_profile_kind_and_one_spec_version():
    dimer = NB["metadata"]["dimer"]
    assert dimer["notebook_profile"] == "TASK-INFERENCE" and "profile" not in dimer
    assert dimer["notebook_mode"] == dimer["notebook_kind"] == "WORKSHOP"
    spec = dimer["notebook_spec"]
    assert f"**DIMER Notebook Specification:** `{spec}`" in CELL_BY_ID["cffd6d72"]
    assert "**Notebook kind:** `WORKSHOP`" in CELL_BY_ID["cffd6d72"]
    assert f'"notebook_spec": "{spec}"' in CELL_BY_ID["1b4955b0"]
    assert any(r["date"] == "2026-10-13" and "GDN2" in r["change"] for r in dimer["revisions"])


def test_rerun_guidance_and_output_dir_field():
    assert re.search(r'^OUTPUT_DIR = ".+"  # @param \{type:"string"\}$', CELL_BY_ID["f024b78f"], re.M)
    guided = CELL_BY_ID["guided-01"]
    assert "not a valid continuation" not in guided and "OUTPUT_DIR" in guided and "Section 11" in guided


def test_opening_states_download_size_and_labelled_runtime():
    opening = CELL_BY_ID["cffd6d72"]
    assert "1.3 GB" in opening and "292 s" in opening and "Tesla T4" in opening
    assert "not re-run for this revision" in opening


def test_learner_text_names_naming_floor_labels_and_baseline():
    md = MARKDOWN.lower()
    assert "labelled `white blood cell`" in md and "naming, not finding" in md
    assert "grounding dino evaluator's score floor" in md and "fixed recipe" in md
    assert "reference boxes are incomplete" in md and "not *no object*" in md
    assert "grid prior's recall50" in md
    assert "not re-run in this notebook" in md  # reviewer measurements are labelled (REL13)
    assert "different object sets" in CELL_BY_ID["6c2fade4"] and "phrase" in CELL_BY_ID["96b3745f"]
    assert "unzip" in CELL_BY_ID["24bffb83"] and "Refused:" in CELL_BY_ID["24bffb83"]
