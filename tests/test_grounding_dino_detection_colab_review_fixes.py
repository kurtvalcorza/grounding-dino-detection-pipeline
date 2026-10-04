"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/grounding_dino_detection_colab.ipynb`
(review PR #11: GDD-M1..M4, GDD-m1..m4).

The notebook's own cells are executed from the committed JSON in a namespace of the package's public API and inert
stand-ins (a stub pipeline, a fake `google.colab`, tiny PIL images). Nothing here loads the pinned checkpoint, so the
whole file runs under CI's install line.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before any NumPy linear algebra in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from PIL import Image, ImageDraw  # noqa: E402

import grounding_dino_detection_pipeline as gd  # noqa: E402
from grounding_dino_detection_pipeline import (  # noqa: E402
    MAX_BYOD_BYTES,
    GroundingDINOPipeline,
    byod_record_limits,
    detection_metrics,
    drop_duplicate_images,
    load_byod_dataset,
    split_dataset,
    split_minimums,
)
from grounding_dino_detection_pipeline import samples as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "grounding_dino_detection_colab.ipynb"
VOCAB = ["red disc", "blue square"]


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _scene(k: int) -> Image.Image:
    image = Image.new("RGB", (64, 48), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.ellipse([4, 4, 24, 24], fill=(220, 30, 30))
    draw.rectangle([34, 20, 58, 44], fill=(30, 30, 220))
    image.putpixel((k % 64, 0), (k % 256, (k // 64) % 256, 0))
    return image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(tmp_path: Path, name: str, n: int, *, duplicates: int = 0, clash: bool = False) -> Path:
    rows = []
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n):
            archive.writestr(f"photos/p{i:03d}.png", _png(_scene(i)))
            rows += [(f"p{i:03d}.png", 4, 4, 24, 24, "Red Disc"), (f"p{i:03d}.png", 34, 20, 58, 44, "blue square.")]
        for j in range(duplicates):
            archive.writestr(f"photos/dup{j:03d}.png", _png(_scene(j)))
            rows.append((f"dup{j:03d}.png", 4, 4, 24, 24, "red disc"))
        if clash:
            archive.writestr("other/p000.png", _png(_scene(n + 1)))
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(["file", "x0", "y0", "x1", "y1", "phrase"])
        writer.writerows(rows)
        archive.writestr("boxes.csv", text.getvalue())
    return path


def _fake_colab(monkeypatch, uploads: list[dict]):
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _namespace() -> dict:
    ns: dict = {name: getattr(gd, name) for name in gd.__all__}
    ns.update({"os": __import__("os"), "Path": Path, "json": json})
    return ns


def _section4(monkeypatch, tmp_path, *, byod_path: str = "") -> dict:
    """Execute Section 4 verbatim with USE_BYOD = True (form literals substituted) in a namespace of the package API."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. BCCD photographs, the phrase vocabulary and the split")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


class _StubPipe:
    """Stands in for a loaded pipeline: fixed per-record predictions, `evaluate` through the real metrics."""

    def __init__(self, predict, adapter=None):
        self._predict = predict
        self.adapter = adapter
        self.device = "cpu"

    def predict_boxes(self, image, prompts):
        return self._predict(image, list(prompts))

    def evaluate(self, records, prompts):
        checked = gd.validate_dataset(records, prompts, min_records=1)
        preds = [self._predict(r["image"], checked["prompts"]) for r in checked["records"]]
        return {**detection_metrics(preds, checked["records"], checked["prompts"]), "adapted": self.adapter is not None, "verdict": "measured-small-sample", "n_predictions": sum(map(len, preds))}


def _records(n: int, offset: int = 0) -> list[dict]:
    return [{"id": f"r{offset + i:03d}", "image": _scene(offset + i), "boxes": [[4.0, 4.0, 24.0, 24.0], [34.0, 20.0, 58.0, 44.0]], "labels": list(VOCAB)} for i in range(n)]


# --- GDD-M3: the BYOD minimum, refusals that name the split, duplicates -------------------------------------------


def test_byod_minimum_is_twelve_distinct_images_and_matches_the_prose():
    assert byod_record_limits() == (12, 5000) and split_minimums() == {"train": 8, "validation": 2, "test": 2}
    assert "**at least 12 distinct images**" in _markdown()
    assert "at least eight images" not in _markdown() and "a dataset needs 8..5,000 records" not in _markdown()


def test_twelve_images_pass_section4_and_eleven_are_refused_naming_the_split(monkeypatch, tmp_path, capsys):
    ns = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "twelve", 12)))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 2, "validation": 2, "train": 8}
    assert ns["byod"]["distinct_images"] == 12 and ns["data_source"] == "BYOD (twelve.zip)" and ns["vocabulary"] == VOCAB
    assert "'byod_minimum_distinct_images': 12" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"the train split would hold 7 records \(at least 8 are required\): 11 records, 11 distinct images.*at least 12 distinct images"):
        _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "eleven", 11)))


def test_split_dataset_refusals_name_split_and_counts():
    with pytest.raises(ValueError, match=r"the train split would hold 2 records \(at least 8 are required\): 3 records, 3 distinct images, split into train/validation/test as 2/0/1; a dataset needs at least 12 distinct images"):
        split_dataset(_records(3), VOCAB)
    records = _records(11) + [{**_records(1)[0], "id": "copy"}]
    with pytest.raises(ValueError, match=r"11 distinct images \(1 duplicate image\(s\) dropped\)"):
        split_dataset(records, VOCAB)
    assert sum(len(v) for v in split_dataset(_records(12), VOCAB).values()) == 12


def test_pixel_duplicates_are_reported_before_the_split(monkeypatch, tmp_path, capsys):
    ns = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "dups", 14, duplicates=3)))
    out = capsys.readouterr().out
    assert "'pixel_duplicates_dropped': 3" in out and "dup000" in out
    assert ns["byod"]["duplicates_dropped"] == ["dup000", "dup001", "dup002"]
    assert sum(len(v) for v in ns["splits"].values()) == 14
    kept, dropped = drop_duplicate_images(_records(2) + [{**_records(1)[0], "id": "again"}])
    assert [r["id"] for r in kept] == ["r000", "r001"] and dropped == ["again"]


# --- GDD-m1: BYOD input handling --------------------------------------------------------------------------------


def test_duplicate_file_names_in_two_folders_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"two files are named 'p000.png' \(photos/p000.png and other/p000.png\).*without its folders"):
        load_byod_dataset(_zip(tmp_path, "clash", 12, clash=True))


def test_oversized_archive_is_refused_before_decompression(monkeypatch, tmp_path):
    path = _zip(tmp_path, "big", 12)
    reads: list = []
    real_read = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, *a, **k: reads.append(a) or real_read(self, *a, **k))
    monkeypatch.setattr(sm, "MAX_BYOD_BYTES", 100)
    with pytest.raises(ValueError, match=r"expands to .* bytes; at most MAX_BYOD_BYTES \(100\)"):
        load_byod_dataset(path)
    assert reads == []
    monkeypatch.setattr(sm, "MAX_BYOD_BYTES", MAX_BYOD_BYTES)
    monkeypatch.setattr(sm, "MAX_BYOD_FILES", 5)
    with pytest.raises(ValueError, match=r"holds 13 files; at most MAX_BYOD_FILES \(5\)"):
        load_byod_dataset(path)
    assert reads == []


def test_cancelled_upload_no_colab_and_missing_path_are_actionable(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}])
    with pytest.raises(ValueError, match=r"Upload exactly one .zip file \(received 0\).*BYOD_PATH"):
        _section4(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` raises ImportError
    with pytest.raises(RuntimeError, match="needs Google Colab.*BYOD_PATH"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist in this runtime"):
        _section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.zip"))


def test_an_uploaded_zip_goes_through_section4(monkeypatch, tmp_path):
    payload = _zip(tmp_path, "up", 13).read_bytes()
    _fake_colab(monkeypatch, [{"up.zip": payload}])
    ns = _section4(monkeypatch, tmp_path)
    assert ns["byod"]["zip_sha256"] and sum(len(v) for v in ns["splits"].values()) == 13


def test_byod_path_reads_a_folder_without_colab(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder = tmp_path / "folder"
    with zipfile.ZipFile(_zip(tmp_path, "f", 12)) as archive:
        for info in archive.infolist():
            target = folder / Path(info.filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    ns = _section4(monkeypatch, tmp_path, byod_path=str(folder))
    assert ns["byod"]["zip_sha256"] is None and sum(len(v) for v in ns["splits"].values()) == 12


# --- GDD-M2: every pass starts from the pretrained model --------------------------------------------------------


def test_adapt_refuses_an_already_adapted_pipeline():
    pipe = GroundingDINOPipeline(lambda *a: [], "cpu", _model=object(), _processor=object(), adapter={"best_epoch": 1})
    with pytest.raises(ValueError, match="already adapted.*from_pretrained"):
        pipe.adapt(_records(8), None, VOCAB)


def test_sections_5_to_7_reset_and_section_6_refuses_an_adapted_model():
    for heading in ("## 5. Detect a synthetic scene", "## 6. Baselines and the frozen model", "## 7. Bounded fine-tuning"):
        assert "reset_to_pretrained()" in _code_after(heading), heading
    section6 = _code_after("## 6. Baselines and the frozen model")
    assert section6.index("reset_to_pretrained()") < section6.index("pipe.predict_boxes(") < section6.index("pipe.evaluate(")
    assert "if frozen_test['adapted']:" in section6
    section7 = _code_after("## 7. Bounded fine-tuning")
    assert section7.index("reset_to_pretrained()") < section7.index("pipe.adapt(")
    parity = _code_after("## 9. Look at the boxes")
    assert "raise RuntimeError(f'Reload parity failed: {parity}." in parity and "re-run from Section 7" in parity


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline():
    source = _code_after("## 5. Detect a synthetic scene")
    block = source[source.index("def reset_to_pretrained():") : source.index("reset_to_pretrained()\n")]
    loads: list = []

    class _Loader:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return GroundingDINOPipeline(lambda *a: [], "cpu")

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns = {"gc": __import__("gc"), "torch": fake_torch, "GroundingDINOPipeline": _Loader, "WEIGHTS_DIR": "w", "pipe": GroundingDINOPipeline(lambda *a: [], "cpu")}
    exec(compile(block, "<reset>", "exec"), ns)
    ns["reset_to_pretrained"]()
    assert loads == []
    ns["pipe"] = GroundingDINOPipeline(lambda *a: [], "cpu", adapter={"best_epoch": 2})
    ns["reset_to_pretrained"]()
    assert loads == ["w"] and ns["pipe"].adapter is None


def test_experiments_name_their_cell_and_run_after_scope():
    markdown = _markdown()
    experiments = markdown[markdown.index("**Optional experiments") : markdown.index("## Troubleshooting")]
    for line in [ln for ln in experiments.splitlines() if ln.startswith("- **") and "Section 10" not in ln and "Synonyms" not in ln]:
        assert re.search(r"Section \d", line) and ("Run after" in line or "Section 4, then" in line), line
    assert "run only that cell" in experiments
    assert "**Predict → Change one thing → Run → Observe → Explain**" in markdown
    assert "select the Section 7 cell and choose **Runtime → Run after**" in markdown


# --- GDD-m2: the generic word is a field; a negative result is reported, not asserted away ---------------------


def _section6_namespace(monkeypatch, tmp_path, *, use_byod: bool, frozen_predict) -> dict:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    prompts_seen: list = []

    def predict(image, prompts):
        prompts_seen.append(prompts)
        return frozen_predict(image, prompts)

    ns = _namespace()
    ns.update({"pipe": _StubPipe(predict), "USE_BYOD": use_byod, "train_records": _records(8), "test_records": _records(4, 50), "vocabulary": list(VOCAB), "time": __import__("time"), "reset_to_pretrained": lambda: None, "prompts_seen": prompts_seen})
    exec(compile(_code_after("## 6. Baselines and the frozen model"), "<section 6>", "exec"), ns)
    return ns


def test_section6_generic_word_follows_the_field_and_records_a_frozen_model_below_the_grid(monkeypatch, tmp_path, capsys):
    nothing = lambda image, prompts: []  # noqa: E731 -- a frozen model that finds nothing
    ns = _section6_namespace(monkeypatch, tmp_path, use_byod=True, frozen_predict=nothing)
    assert ns["generic_prompt"] == "object" and ["object"] in ns["prompts_seen"]
    assert ns["frozen_beats_grid_prior"] is False and "'frozen_beats_grid_prior': False" in capsys.readouterr().out
    ns = _section6_namespace(monkeypatch, tmp_path, use_byod=False, frozen_predict=nothing)
    assert ns["generic_prompt"] == "cell"
    source = _code_after("## 6. Baselines and the frozen model").replace("GENERIC_PROMPT = ''", "GENERIC_PROMPT = 'shape'", 1)
    ns.update({"pipe": _StubPipe(nothing)})
    exec(compile(source, "<section 6>", "exec"), ns)
    assert ns["generic_prompt"] == "shape"


def _section8(monkeypatch, tmp_path, frozen_predict, adapted_predict) -> dict:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    test_records, val_records, train_records = _records(4, 50), _records(2, 60), _records(8)
    frozen = _StubPipe(frozen_predict)
    adapted = _StubPipe(adapted_predict, adapter={"best_epoch": 3})
    ns = _namespace()
    ns.update({"pipe": adapted, "METRICS": ("map50", "map75", "recall50"), "USE_BYOD": True, "byod": {"file": "x.zip"}, "EPOCHS": 6, "LEARNING_RATE": 5e-5, "data_source": "BYOD (x.zip)", "vocabulary": list(VOCAB), "generic_prompt": "object", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {"test": 4}, "adapt_seconds": 1.0, "test_records": test_records, "val_records": val_records})
    ns["baseline_grid"] = gd.grid_prior_baseline(train_records, test_records, VOCAB)
    ns["baseline_generic"] = gd.majority_relabel_baseline([[] for _ in test_records], train_records, test_records, VOCAB)
    ns["frozen_test"] = frozen.evaluate(test_records, VOCAB)
    ns["frozen_fields"] = {c: {"n": v["n"], "ap50": round(v["ap50"], 3), "ap75": round(v["ap75"], 3)} for c, v in ns["frozen_test"]["per_category"].items()}
    ns["frozen_beats_grid_prior"] = ns["frozen_beats_generic_prompt"] = True
    ns["adapt_result"] = {"best_epoch": 3, "history": [], "trainable_names": [], "seed": 0}
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    return ns


def test_section8_counts_a_change_below_the_printed_precision_as_no_gain(monkeypatch, tmp_path):
    # Harness finding (real weights, CPU): a 1e-5 run moved mAP50 by < 0.0005 and the reading said "both rose".
    exact = lambda image, prompts: [(0.9, "red disc", [4.0, 4.0, 24.0, 24.0]), (0.8, "blue square", [34.0, 20.0, 58.0, 44.0])]  # noqa: E731
    ns = _section8(monkeypatch, tmp_path, exact, exact)
    assert ns["comparison"]["delta_vs_frozen"]["map50"] == 0.0 and ns["adapted_beats_frozen"] is False
    assert "not above the frozen mAP50" in ns["reading"] and "(delta +0.000)" in ns["reading"]


def test_section8_reports_a_negative_result_and_writes_the_report(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    test_records, val_records, train_records = _records(4, 50), _records(2, 60), _records(8)
    exact = lambda image, prompts: [(0.9, "red disc", [4.0, 4.0, 24.0, 24.0]), (0.8, "blue square", [34.0, 20.0, 58.0, 44.0])]  # noqa: E731
    nothing = lambda image, prompts: []  # noqa: E731
    frozen = _StubPipe(exact)
    adapted = _StubPipe(nothing, adapter={"best_epoch": 3})
    ns = _namespace()
    ns.update({"pipe": adapted, "METRICS": ("map50", "map75", "recall50"), "USE_BYOD": True, "byod": {"file": "x.zip"}, "EPOCHS": 6, "LEARNING_RATE": 5e-5, "data_source": "BYOD (x.zip)", "vocabulary": list(VOCAB), "generic_prompt": "object", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {"test": 4}, "adapt_seconds": 1.0, "test_records": test_records, "val_records": val_records})
    ns["baseline_grid"] = gd.grid_prior_baseline(train_records, test_records, VOCAB)
    ns["baseline_generic"] = gd.majority_relabel_baseline([[] for _ in test_records], train_records, test_records, VOCAB)
    ns["frozen_test"] = frozen.evaluate(test_records, VOCAB)
    ns["frozen_fields"] = {c: {"n": v["n"], "ap50": round(v["ap50"], 3), "ap75": round(v["ap75"], 3)} for c, v in ns["frozen_test"]["per_category"].items()}
    ns["frozen_beats_grid_prior"] = ns["frozen_beats_generic_prompt"] = True
    ns["adapt_result"] = {"best_epoch": 3, "history": [], "trainable_names": [], "seed": 0}
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    out = capsys.readouterr().out
    assert ns["adapted_beats_frozen"] is False and "did not help here" in out
    report = json.loads((tmp_path / "outputs" / "grounding_dino_detection_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["outcomes"]["adapted_beats_frozen"] is False and report["reading"].startswith("The adapted mAP50 is not above")
    assert report["generic_prompt"] == "object" and len(report["run_history"]) == 1


def test_no_learner_cell_asserts_a_result():
    for cell in _cells():
        source = _source(cell)
        if cell["cell_type"] != "code" or "dimer" in cell.get("metadata", {}) or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:120]
    assert "['cell']" not in "".join(_source(c) for c in _cells() if c["cell_type"] == "code")


# --- GDD-m3 / GDD-m4: recorded numbers with their environment, run-to-run spread, the Python statement ---------


def test_worked_answers_quote_the_recorded_t4_run_and_the_spread():
    markdown = _markdown()
    assert "**Run-to-run spread.**" in markdown and "**0.570, 0.608 and 0.632**" in markdown
    assert "0.109 → 0.632 and mAP75 0.052 → 0.450" in markdown and "In the recorded T4 run" in markdown
    assert "recorded Kaggle Tesla T4 run" in markdown
    assert "to about 0.64" not in markdown and "0.109 → 0.570" not in markdown


def test_runtime_statement_names_the_isolated_python():
    markdown = _markdown()
    assert "Section 1 builds its own **Python 3.12.12** environment" in markdown and "Python 3.13" in markdown
    assert "Google Colab or Kaggle GPU, Python 3.12)" not in markdown


# --- GDD-M1: isolated runtime -----------------------------------------------------------------------------------


def test_exactly_two_kernel_cells_and_no_restart_text():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 2
    install = _source(kernel[0])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install and "MANAGED_PYTHON = '3.12.12'" in install
    assert "scipy==1.18.1" in install  # the Hungarian matcher's import, no longer left to the hosted image
    markdown = _markdown()
    assert "Restart the runtime" not in markdown and "its restart" not in markdown
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status and "manual restart" in status
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "**Completed in two passes, not promotion evidence**" in verification and "**PASSED** — 11/11 code cells ok (1 restart" not in verification


def test_lock_pins_every_direct_dependency_with_hashes():
    build = _load_build()
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)
    assert len(build.lock_packages(lock)) == 48 and "timm" not in build.lock_packages(lock)


def _load_build():
    spec = importlib.util.spec_from_file_location("build_notebook_under_test", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on a spec-less stub."""
    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


def test_generator_refuses_a_top_level_name_bound_differently_in_two_modules():
    build = _load_build()
    build.check_name_collisions({"a.py": "X = 1\n", "b.py": "X = 1\n"})
    with pytest.raises(SystemExit):
        build.check_name_collisions({"a.py": "X = 1\n", "b.py": "X = 2\n"})


# --- GDD-M4: guided layer and infrastructure labels ---------------------------------------------------------------


def test_guided_layer_and_infrastructure_labels():
    markdown = _markdown()
    for marker, least in (("**Predict before running:**", 6), ("**What to notice:**", 6), ("<summary>Check your reasoning</summary>", 7), ("> **Infrastructure.**", 3)):
        assert markdown.count(marker) >= least, marker
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## 10. Your turn — change one thing", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert marker in markdown, marker
    code = [c for c in _cells() if c["cell_type"] == "code"]
    setup = code[:7]  # install, router, runtime record, three carried modules, model staging
    assert all(c["metadata"].get("cellView") == "form" for c in setup)
    assert all(_source(c).startswith("# @title Infrastructure: ") for c in setup)
    assert "cellView" not in code[7]["metadata"]  # the learning path starts in Section 4
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    for item in ("GDL1", "GDL2", "GDL3", "GDL6", "GDL7", "GDL9", "GDL10", "GDL11", "GDL13", "GDL14"):
        assert f"| {item} " in registry, item
