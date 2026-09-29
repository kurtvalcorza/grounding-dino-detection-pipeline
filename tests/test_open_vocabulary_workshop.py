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


# ---------------------------------------------------------------------------
# 2026-09-30 review: output validation, one-to-one diagnostics, visual handoff.

CELL_BY_ID = {cell["id"]: "".join(cell["source"]) for cell in CELLS}


def functions_from(ns, cell_id, names=None):
    module = ast.parse(CELL_BY_ID[cell_id])
    module.body = [
        n for n in module.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)
    ]
    exec(compile(module, cell_id, "exec"), ns)
    return ns


def record(boxes, labels, size=(32, 32)):
    return {"id": "r", "image": Image.new("RGB", size), "boxes": boxes, "labels": labels}


@pytest.mark.parametrize(
    "det",
    [
        (float("nan"), "a cell", [0, 0, 16, 16]),
        (float("inf"), "a cell", [0, 0, 16, 16]),
        (-1.0, "a cell", [0, 0, 16, 16]),
        (2.0, "a cell", [0, 0, 16, 16]),
        (0.9, "a cell", [float("nan")] * 4),
        (0.9, "a cell", [10, 0, 5, 16]),
        (0.9, "a cell", [0, 0, 40, 16]),
        (0.9, "not a phrase", [0, 0, 16, 16]),
        (0.9, "a cell", [0, 0, 16]),
        (0.9, "a cell"),
    ],
)
def test_malformed_detections_fail_before_scoring(tmp_path, det):
    ns = helpers(tmp_path)
    with pytest.raises(ValueError):
        ns["validate_detections"]([det], (32, 32), ["a cell"], "probe")
    # The review's fixture: a NaN-scored but well-placed box used to score AP50 = 1.0.
    with pytest.raises(ValueError):
        ns["detection_metrics"]([[det]], [record([[0, 0, 16, 16]], ["a cell"])], ["a cell"])


def test_empty_and_low_confidence_outputs_stay_valid(tmp_path):
    ns = helpers(tmp_path)
    rec = record([[0, 0, 16, 16]], ["a cell"])
    assert ns["detection_metrics"]([[]], [rec], ["a cell"])["recall50"] == 0.0
    low = ns["detection_metrics"]([[(0.01, "a cell", [0, 0, 16, 16])]], [rec], ["a cell"])
    assert low["map50"] == 1.0


class _Inputs(dict):
    def to(self, device):
        return self


def gdino_namespace(tmp_path, logits, boxes):
    torch = pytest.importorskip("torch")
    ns = helpers(tmp_path)
    ns.update(torch=torch, GDINO_SPECIAL_TOKEN_IDS=(101, 102, 1012, 1029, 0), GDINO_MAX_TEXT_TOKENS=256,
              DEVICE="cpu")
    functions_from(ns, "53f2f300", {"gdino_prompt_text", "phrase_token_groups", "gdino_predict"})
    ids = torch.tensor([[101, 7, 1012, 8, 1012, 102]])  # "a. b." -> phrase tokens at 1 and 3
    ns["gdino_processor"] = lambda **k: _Inputs(input_ids=ids)
    ns["gdino_model"] = lambda **k: types.SimpleNamespace(logits=logits, pred_boxes=boxes)
    return ns


def raw_gdino(active=((4.0, -9.0), (-9.0, 4.0))):
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 2, 256), float("-inf"))  # masked positions are legitimately -inf
    logits[0, :, :6] = -9.0
    for q, (a, b) in enumerate(active):
        logits[0, q, 1], logits[0, q, 3] = a, b
    boxes = torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]])
    return logits, boxes


def test_gdino_adapter_accepts_masked_padding_and_rejects_invalid_active_outputs(tmp_path):
    image = Image.new("RGB", (100, 100))
    out = gdino_namespace(tmp_path, *raw_gdino())["gdino_predict"](image, ["a", "b"])
    assert sorted(label for _s, label, _b in out) == ["a", "b"] and all(0 <= s <= 1 for s, _l, _b in out)
    negative = gdino_namespace(tmp_path, *raw_gdino(((-9.0, -9.0), (-9.0, -9.0))))
    assert negative["gdino_predict"](image, ["a", "b"]) == []  # a legitimate empty result
    nan = float("nan")
    for active in [((nan, -9.0), (-9.0, 4.0)), ((nan, nan), (nan, nan)), ((float("inf"), -9.0), (-9.0, 4.0))]:
        with pytest.raises(FloatingPointError):
            gdino_namespace(tmp_path, *raw_gdino(active))["gdino_predict"](image, ["a", "b"])
    logits, boxes = raw_gdino()
    bad_boxes = boxes.clone()
    bad_boxes[0, 0] = nan
    with pytest.raises(FloatingPointError):
        gdino_namespace(tmp_path, logits, bad_boxes)["gdino_predict"](image, ["a", "b"])
    with pytest.raises(RuntimeError):
        gdino_namespace(tmp_path, logits[:, :, :2], boxes)["gdino_predict"](image, ["a", "b"])


def owlv2_namespace(tmp_path, logits, boxes):
    torch = pytest.importorskip("torch")
    owlv2 = pytest.importorskip("transformers.models.owlv2.image_processing_owlv2")
    upstream = owlv2.Owlv2ImageProcessor()
    ns = helpers(tmp_path)
    ns.update(torch=torch, DEVICE="cpu")
    functions_from(ns, "2f66f39b", {"owlv2_predict"})

    class Processor:
        def __call__(self, **k):
            return _Inputs()

        def post_process_grounded_object_detection(self, outputs, threshold, target_sizes, text_labels):
            # The real upstream box scaling; only the text-label lookup is reproduced here.
            result = upstream.post_process_object_detection(outputs, threshold, target_sizes)
            for item, labels in zip(result, text_labels, strict=True):
                item["text_labels"] = [labels[int(i)] for i in item["labels"]]
            return result

    ns["owlv2_processor"] = Processor()
    ns["owlv2_validate_tokens"] = lambda prompts: list(prompts)
    ns["owlv2_model"] = lambda **k: types.SimpleNamespace(logits=logits, pred_boxes=boxes)
    return ns


def test_owlv2_boxes_map_to_image_pixels_clip_and_invalid_outputs_fail(tmp_path):
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[5.0], [5.0]]])
    # Normalized to the padded 640x640 square: a cell centred at (320, 320) and one reaching past y=480.
    boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1], [0.5, 0.7, 0.1, 0.1]]])
    out = owlv2_namespace(tmp_path, logits, boxes)["owlv2_predict"](Image.new("RGB", (640, 480)), ["a cell"])
    got = sorted(b for _s, _l, b in out)
    assert got[0] == pytest.approx([288.0, 288.0, 352.0, 352.0])  # upstream padded-square rescaling
    assert got[1] == pytest.approx([288.0, 416.0, 352.0, 480.0])  # clipped to the image
    for bad_logits, bad_boxes in [(logits * float("nan"), boxes), (logits, boxes * float("nan"))]:
        with pytest.raises(FloatingPointError):
            owlv2_namespace(tmp_path, bad_logits, bad_boxes)["owlv2_predict"](
                Image.new("RGB", (640, 480)), ["a cell"]
            )


A, B = [0, 0, 10, 10], [2, 0, 12, 10]


@pytest.mark.parametrize(
    ("preds", "expected"),
    [
        # One merged box over two overlapping cells: one match and one miss (was "2 matched").
        ([(0.9, "c", [0, 0, 12, 10])], dict(true_positives=1, missed_objects=1, duplicate_predictions=0)),
        # Two exact boxes: two true positives and no duplicates (was "2 redundant").
        ([(0.9, "c", A), (0.8, "c", B)], dict(true_positives=2, missed_objects=0, duplicate_predictions=0)),
        # A second box on A still overlaps B at IoU 0.67, so greedy matching gives it B: not a duplicate.
        ([(0.9, "c", A), (0.8, "c", A)], dict(true_positives=2, missed_objects=0, duplicate_predictions=0)),
        ([(0.9, "d", A)], {"true_positives": 0, "missed_objects": 2, "wrong_phrase_predictions": 1}),
        ([(0.9, "c", [20, 20, 30, 30])], {"true_positives": 0, "spurious_predictions": 1}),
        ([], {"true_positives": 0, "missed_objects": 2, "unmatched_predictions": 0}),
    ],
)
def test_one_to_one_diagnostics(tmp_path, preds, expected):
    ns = helpers(tmp_path)
    rec = record([A, B], ["c", "c"])
    summary = ns["assignment_summary"]([preds], [rec])
    assert {k: summary[k] for k in expected} == expected
    assert summary["unmatched_predictions"] == summary["predictions"] - summary["true_positives"]
    metrics = ns["detection_metrics"]([preds], [rec], ["c", "d"])
    assert summary["true_positives"] == round(metrics["per_phrase"]["c"]["recall50"] * 2)
    functions_from(ns, "db4f6823", {"threshold_diagnostics"})
    row = ns["threshold_diagnostics"]([preds], [rec])
    assert row["true_positives"] == summary["true_positives"] and row["objects"] == 2


def test_duplicate_on_an_isolated_object(tmp_path):
    ns = helpers(tmp_path)
    rec = record([A, [20, 20, 30, 30]], ["c", "c"])
    summary = ns["assignment_summary"]([[(0.9, "c", A), (0.8, "c", A)]], [rec])
    counts = (summary["true_positives"], summary["duplicate_predictions"], summary["missed_objects"])
    assert counts == (1, 1, 1)


def test_overlap_coverage_is_reported_separately(tmp_path):
    ns = helpers(tmp_path)
    summary = ns["assignment_summary"]([[(0.9, "c", [0, 0, 12, 10])]], [record([A, B], ["c", "c"])])
    assert summary["overlap_coverage_objects"] == 2 and summary["true_positives"] == 1


def test_byod_rejects_ambiguous_headers_short_rows_and_multiframe(tmp_path):
    ns = helpers(tmp_path)
    root = fixture(tmp_path)
    table = root / "boxes.csv"
    lines = table.read_text(encoding="utf-8").splitlines()
    table.write_text("\n".join([lines[0] + ",x0"] + [r + ",1" for r in lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column headers"):
        ns["load_byod_dataset"](root)
    table.write_text("\n".join(lines[:-1] + [lines[-1].rsplit(",", 1)[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one value per column"):
        ns["load_byod_dataset"](root)

    def multiframe(rows, images):
        (images / "0.png").unlink()
        frames = [Image.new("RGB", (32, 32), c) for c in ("red", "blue")]
        frames[0].save(images / "0.tiff", save_all=True, append_images=frames[1:])
        rows[0]["filename"] = "0.tiff"

    with pytest.raises(ValueError, match="multi-frame"):
        ns["load_byod_dataset"](fixture(tmp_path / "mf", multiframe))


def test_byod_export_refuses_nan_before_writing(tmp_path):
    ns = helpers(tmp_path)
    records, prompts, digest = ns["load_byod_dataset"](fixture(tmp_path))
    good = [[(0.9, "a cell", [0, 0, 16, 16])] for _ in records]
    bad = [list(x) for x in good]
    bad[3] = [(float("nan"), "a cell", [0, 0, 16, 16])]
    with pytest.raises(ValueError):
        ns["export_byod"](records, prompts, digest, bad, good)
    assert not (Path(ns["OUTPUT_DIR"]) / "byod").exists()
    _, out = ns["export_byod"](records, prompts, digest, good, good)
    assert "NaN" not in (out / "predictions.json").read_text(encoding="utf-8")


def test_visual_handoff_and_export_contracts_in_source():
    order = [cell["id"] for cell in CELLS]
    assert order.index("5750ceab") < order.index("53f2f300")  # reference preview before inference
    assert "show_image(" in CELL_BY_ID["5750ceab"]
    assert "show_image(panel" in CELL_BY_ID["a10c45f5"] and "evaluator_view(" in CELL_BY_ID["a10c45f5"]
    assert "show_image(concat_horizontal" in CELL_BY_ID["3f9f36da"]  # bounded BYOD preview
    for cell_id in ("1b4955b0", "3f9f36da"):
        text = CELL_BY_ID[cell_id]
        assert text.count("json.dumps(") == text.count("allow_nan=False"), cell_id
    assert "unlink(missing_ok=True)" in CELL_BY_ID["1b4955b0"]  # no stale threshold_sweep.csv
    assert "output_inventory.json" in CELL_BY_ID["5dbbe2be"]
    assert "rounded_detection_sets_identical" in CELL_BY_ID["80f45dd6"]
    assert "min(max(y1, 0.0), height)" in CELL_BY_ID["2f66f39b"]  # OWLv2 boxes clipped to the image


def test_title_strip_does_not_cover_boxes_at_the_top_edge():
    from PIL import ImageDraw

    ns = {"Image": Image, "ImageDraw": ImageDraw, "PHRASE_COLORS": {"platelet": "gold"}}
    functions_from(ns, "5750ceab", {"title_bar", "draw_boxes"})
    image = Image.new("RGB", (200, 100), "black")
    out = ns["draw_boxes"](image, [(0.9, "platelet", [0, 0, 200, 100])], "title")
    assert out.size == (200, 122)  # 22-px strip added above, image not overwritten
    pixels = np.asarray(out)
    strip, top = pixels[:22], pixels[22:34]
    assert (strip[:, 100:] == 255).all()  # the strip holds only the title text on the left
    gold = np.all(top == np.array([255, 215, 0]), axis=-1)
    assert gold[0].all()  # the box's top edge is visible
    label = top[2:12, 2:60].astype(int)  # anti-aliased text: look for yellowish ink, not exact gold
    assert ((label[..., 0] > 100) & (label[..., 1] > 80) & (label[..., 2] < 80)).sum() > 10


def test_show_image_falls_back_without_ipython(capsys, monkeypatch):
    ns = {"Image": Image}
    functions_from(ns, "5750ceab", {"show_image"})
    monkeypatch.setitem(__import__("sys").modules, "IPython.display", None)
    ns["show_image"](Image.new("RGB", (2400, 100)), "caption", path="x.png")
    out = capsys.readouterr().out
    assert "full size: x.png" in out and "inline display is unavailable" in out
