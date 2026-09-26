import ast
import csv
import gc
import hashlib
import json
import types
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "tutorials/DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb"
)
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def helpers(tmp_path):
    ns = {
        "Path": Path,
        "np": np,
        "Image": Image,
        "hashlib": hashlib,
        "json": json,
        "gc": gc,
        "USE_BYOD": False,
        "OUTPUT_DIR": str(tmp_path / "outputs"),
        "RUNTIME": {"device": "cpu"},
        "EVAL_SCORE_FLOOR": 0.05,
        "IOU_THRESHOLDS": (0.5, 0.75),
        "GDINO_MANIFEST": {"modelId": "gd", "revision": "pinned"},
        "OWLV2_MANIFEST": {"modelId": "owl", "revision": "pinned"},
    }
    for i in [13, 15, 17]:
        module = ast.parse("".join(CELLS[i]["source"]))
        module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        exec(compile(module, "helper", "exec"), ns)
    ns.update(MAX_PROMPTS=16, MAX_PROMPT_CHARS=48)
    ns["sha256_file"] = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    exec("".join(CELLS[47]["source"]), ns)
    return ns


def fixture(tmp_path, change=None):
    root = tmp_path / "dataset"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(8):
        Image.new("RGB", (32, 32), (i * 20, 1, 2)).save(images / f"{i}.png")
        rows.append(
            {
                "image_id": str(i),
                "filename": f"{i}.png",
                "label": "A Cell.",
                "x0": 0,
                "y0": 0,
                "x1": 16,
                "y1": 16,
            }
        )
    if change:
        change(rows, images)
    with (root / "boxes.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_valid_load_and_separate_exports(tmp_path):
    ns = helpers(tmp_path)
    records, prompts, digest = ns["load_byod_dataset"](fixture(tmp_path))
    assert prompts == ["a cell"] and len(records) == 8
    # Exercise real evaluator and export, not a mocked report writer.
    preds = [[(0.9, "a cell", [0, 0, 16, 16])] for r in records]
    result, out = ns["export_byod"](records, prompts, digest, preds, preds)
    assert len(list(out.iterdir())) == 4 and result["images"] == 8
    manifest = json.loads((out / "input_provenance.json").read_text())
    assert manifest["annotation_sha256"] == digest
    _, second = ns["export_byod"](records, prompts, digest, preds, preds)
    assert out != second and out.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda r, p: r[0].update(filename="../outside.png"),
        lambda r, p: r[0].update(filename="C:/outside.png"),
        lambda r, p: r[0].update(filename="folder\\image.png"),
        lambda r, p: r[0].update(image_id="1"),
        lambda r, p: r[0].update(filename="1.png"),
        lambda r, p: r[0].update(label=""),
        lambda r, p: r[0].update(x0="nan"),
        lambda r, p: r[0].update(x1=100),
        lambda r, p: r.append(dict(r[0])),
        lambda r, p: Image.new("RGB", (32, 32), "white").save(p / "extra.png"),
        lambda r, p: (p / "7.png").write_bytes((p / "0.png").read_bytes()),
    ],
)
def test_reject_bad_byod(tmp_path, change):
    ns = helpers(tmp_path)
    with pytest.raises(ValueError):
        ns["load_byod_dataset"](fixture(tmp_path, change))


def test_runtime_public_versions():
    from packaging.version import Version

    module = ast.parse("".join(CELLS[7]["source"]))
    module.body = [
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "public_version_matches"
    ]
    ns = {"Version": Version}
    exec(compile(module, "guard", "exec"), ns)
    assert ns["public_version_matches"]("2.14.0+cu130", "2.14.0")
    assert not ns["public_version_matches"]("2.14.0rc1", "2.14.0")


def fake_models(ns, fail=False, bad_tokens=False):
    refs = []
    events = []

    class Processor:
        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k["local_files_only"] and not k["trust_remote_code"]
            return cls()

        def tokenizer(self, *a, **k):
            return {"input_ids": [1] * (300 if bad_tokens else 3)}

    class Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            gc.collect()
            assert all(r() is None for r in refs)
            obj = cls()
            refs.append(weakref.ref(obj))
            events.append("load")
            return obj

        def to(self, device):
            return self

        def eval(self):
            return self

    def infer(image, prompts):
        held = ns["gdino_model"]  # Retained by the failing frame unless traceback cleanup works.
        assert held is not None
        if fail:
            raise RuntimeError("model failed")
        return []

    ns.update(
        AutoProcessor=Processor,
        Owlv2Processor=Processor,
        AutoModelForZeroShotObjectDetection=Model,
        Owlv2ForObjectDetection=Model,
        GDINO_DIR=Path("gd"),
        OWLV2_DIR=Path("owl"),
        GDINO_REVISION="pin",
        OWLV2_REVISION="pin",
        DEVICE="cpu",
        GDINO_MAX_TEXT_TOKENS=256,
        gdino_prompt_text=lambda p: ("a cell.", p),
        phrase_token_groups=lambda ids: [ids],
        owlv2_validate_tokens=lambda p: p,
        gdino_predict=infer,
        owlv2_predict=lambda image, p: [],
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    return refs, events


def test_inference_sequential_release_and_retry(tmp_path):
    ns = helpers(tmp_path)
    refs, events = fake_models(ns)
    for _ in range(2):
        assert ns["predict_byod"]([{"image": None}], ["a cell"]) == ([[]], [[]])
    gc.collect()
    assert len(events) == 4 and all(r() is None for r in refs)


def test_failure_retained_traceback_does_not_keep_model(tmp_path):
    ns = helpers(tmp_path)
    refs, events = fake_models(ns, fail=True)
    errors = []
    for _ in range(2):
        try:
            ns["predict_byod"]([{"image": None}], ["a cell"])
        except RuntimeError as exc:
            errors.append(exc)
    gc.collect()
    assert len(errors) == 2 and all(r() is None for r in refs)


def test_bad_tokens_rejected_before_model_load(tmp_path):
    ns = helpers(tmp_path)
    refs, events = fake_models(ns, bad_tokens=True)
    with pytest.raises(ValueError):
        ns["predict_byod"]([{"image": None}], ["a cell"])
    assert events == []


def test_notebook_cells_parse_and_outputs_not_fabricated():
    for cell in CELLS:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell", "exec")


def bccd_xml(path, boxes):
    objects = "".join(
        f"<object><name>RBC</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
        for x0, y0, x1, y1 in boxes
    )
    path.write_text(
        f"<annotation><size><width>32</width><height>32</height></size>{objects}</annotation>",
        encoding="utf-8",
    )


def test_bccd_parse_drops_zero_area_points_and_rejects_other_bad_boxes(tmp_path):
    ns = helpers(tmp_path)
    ns.update(ET=ET, CLASS_PHRASES={"RBC": "red blood cell"}, zero_area_dropped=[])
    image = tmp_path / "BloodImage_00338.jpg"
    Image.new("RGB", (32, 32)).save(image)
    xml = tmp_path / "BloodImage_00338.xml"
    bccd_xml(xml, [(1, 1, 9, 9), (5, 5, 5, 5)])
    record = ns["parse_record"](image, xml)
    assert record["boxes"] == [[1.0, 1.0, 9.0, 9.0]]
    assert ns["zero_area_dropped"] == [("BloodImage_00338", "RBC", [5.0, 5.0, 5.0, 5.0])]
    for bad in [(9, 1, 1, 9), (1, 5, 9, 5), (0, 0, 40, 9)]:
        bccd_xml(xml, [bad])
        with pytest.raises(ValueError, match="invalid box"):
            ns["parse_record"](image, xml)


def test_bccd_pinned_counts_account_for_the_zero_area_points():
    constants = "".join(CELLS[11]["source"])
    loader = "".join(CELLS[13]["source"])
    assert 'BCCD_ZERO_AREA_BOXES = ["BloodImage_00338", "BloodImage_00343"]' in constants
    assert "BCCD_EXPECTED_BOXES = 4_886" in constants
    assert "!= BCCD_ZERO_AREA_BOXES" in loader
