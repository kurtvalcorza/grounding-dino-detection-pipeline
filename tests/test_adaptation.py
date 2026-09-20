"""Offline checks of the adaptation contract: the detection record contract and its refusals, the prompt vocabulary,
the pinned-corpus refusals and the draw, splitting, the BYOD loader, the metrics and baselines, the phrase-token
grouping, the artifact-manifest checks, and the model-free refusals of `evaluate` / `adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image, ImageDraw

from grounding_dino_detection_pipeline import (
    ARTIFACT_FORMAT,
    CLASS_PHRASES,
    MODEL_ID,
    MODEL_REVISION,
    PROMPTS,
    SAMPLE_IMAGES,
    SAMPLE_SPLIT,
    GroundingDINOPipeline,
    build_sample_dataset,
    check_prompts,
    check_split_disjoint,
    dataset_digest,
    detection_metrics,
    fetch_corpus,
    grid_prior_baseline,
    image_digest,
    load_byod_dataset,
    majority_relabel_baseline,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from grounding_dino_detection_pipeline import pipeline as pl
from grounding_dino_detection_pipeline import samples as sm

VOCAB = ["red disc", "blue square"]
COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _scene(i, size=(160, 120)):
    image = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 60, 60], fill=(220, 30, 30))
    draw.rectangle([90, 50, 140, 100], fill=(30, 30, 220))
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    return image


def _record(i, boxes=None, labels=None):
    return {"id": f"r{i:03d}", "image": _scene(i), "boxes": boxes or [[20.0, 20.0, 60.0, 60.0], [90.0, 50.0, 140.0, 100.0]], "labels": labels or ["red disc", "blue square"]}


def _records(n=12):
    return [_record(i) for i in range(n)]


def _no_runner(image, text, box_t, text_t):
    return []


# --- record contract -----------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    manifest = validate_dataset(_records(), VOCAB)
    assert manifest["n_records"] == 12 and manifest["n_boxes"] == 24 and manifest["label_counts"] == {"red disc": 12, "blue square": 12}
    assert manifest["boxes_per_image"] == {"min": 2, "max": 2} and manifest["prompts"] == VOCAB and manifest["image_side"] == {"min": 160, "max": 160}
    assert len(manifest["digest"]) == 64 and manifest["model_id"] == MODEL_ID
    upper = validate_dataset([{**r, "labels": ["Red Disc.", "BLUE SQUARE"]} for r in _records()], ["Red disc", "blue square."])
    assert upper["records"][0]["labels"] == ["red disc", "blue square"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.__setitem__("id", "bad id"), "id must match"),
        (lambda r: r.__setitem__("boxes", [[10.0, 10.0, 5.0, 50.0], [90.0, 50.0, 140.0, 100.0]]), "not a non-empty box"),
        (lambda r: r.__setitem__("boxes", [[20.0, 20.0, 60.0, 60.0]]), "boxes for 2 labels"),
        (lambda r: r.__setitem__("labels", ["red disc", "green triangle"]), "not in the prompt vocabulary"),
        (lambda r: r.__setitem__("labels", ["red disc", 3]), "must be a phrase"),
        (lambda r: r.__setitem__("boxes", []) or r.__setitem__("labels", []), "1..MAX_BOXES"),
        (lambda r: r.__setitem__("boxes", "x"), "must be sequences"),
        (lambda r: r.__setitem__("image", Image.new("RGB", (8, 8))), "MIN_IMAGE_SIDE"),
        (lambda r: r.pop("labels"), "missing 'labels'"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[0])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records, VOCAB)


def test_validate_dataset_enforces_bounds_unique_ids_and_the_vocabulary():
    with pytest.raises(ValueError, match="8..5000"):
        validate_dataset(_records(4), VOCAB)
    records = _records()
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records, VOCAB)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"}, VOCAB)
    with pytest.raises(ValueError, match="distinct"):
        check_prompts(["cell", "Cell."])
    with pytest.raises(TypeError):
        check_prompts("cell")
    assert check_prompts(["Red disc.", " blue square "]) == VOCAB


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset(_records(3), VOCAB)
    validate_dataset(_records(), VOCAB)


def test_digests_and_split_disjointness():
    records = _records(24)
    assert dataset_digest(records) == dataset_digest(list(reversed(records)))
    assert image_digest(records[0]["image"]) != image_digest(records[1]["image"])
    splits = split_dataset(records, VOCAB, seed=1)
    assert sum(len(v) for v in splits.values()) == 24 and all(splits.values())
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    leaked = {**splits, "test": [*splits["test"], splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)
    duplicated = [*records, {**records[0], "id": "copy"}]
    assert sum(len(v) for v in split_dataset(duplicated, VOCAB, seed=1).values()) == 24
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, VOCAB, val_fraction=0.9)


# --- pinned corpus ----------------------------------------------------------------------------------------------


def test_pins_vocabulary_and_split_sizes():
    assert len(SAMPLE_IMAGES) == sm.CORPUS_IMAGES == 364 and sum(e[1] for e in SAMPLE_IMAGES) == sm.CORPUS_BYTES
    assert sum(len(e[5].split(";")) for e in SAMPLE_IMAGES) == sm.CORPUS_BOXES == 4886
    assert all(len(e[2]) == 64 and (e[3], e[4]) == (640, 480) for e in SAMPLE_IMAGES)
    assert PROMPTS == ("red blood cell", "white blood cell", "platelet") and set(CLASS_PHRASES) == {"RBC", "WBC", "Platelets"}
    assert sum(SAMPLE_SPLIT.values()) == 364 and check_prompts(PROMPTS) == list(PROMPTS)


def _fake_photo(entry):
    image = _scene(int(entry[0][-3:]), (entry[3], entry[4]))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_fetch_corpus_refuses_an_image_that_does_not_match_its_pin(tmp_path):
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: b"not-the-image")
    good = _fake_photo(SAMPLE_IMAGES[0])
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: good)
    assert not list(tmp_path.iterdir())


def test_read_corpus_builds_records_from_verified_bytes():
    entries = SAMPLE_IMAGES[:10]
    files = {e[0]: _fake_photo(e) for e in entries}
    records = read_corpus(files)
    assert len(records) == 10 and records[0]["id"] == f"bccd-{entries[0][0]}" and records[0]["source"].startswith("Shenggan/BCCD_Dataset@")
    assert all(len(r["boxes"]) == len(r["labels"]) and set(r["labels"]) <= set(PROMPTS) for r in records)
    manifest = validate_dataset(records, PROMPTS)
    assert manifest["n_boxes"] == sum(len(e[5].split(";")) for e in entries)
    splits = build_sample_dataset(records, sizes={"train": 6, "validation": 2, "test": 2})
    assert {k: len(v) for k, v in splits.items()} == {"train": 6, "validation": 2, "test": 2}
    assert build_sample_dataset(records, sizes={"train": 6, "validation": 2, "test": 2}) == splits
    with pytest.raises(ValueError, match="need"):
        build_sample_dataset(records)
    with pytest.raises(ValueError, match="decoded size"):
        read_corpus({entries[0][0]: _fake_photo((entries[0][0], 0, "", 50, 50, ""))})


def test_default_draw_matches_the_pinned_digest_when_the_images_are_cached():
    cached = sm.DEFAULT_CACHE_DIR
    if not all((cached / f"{e[0]}.jpg").is_file() for e in SAMPLE_IMAGES):
        pytest.skip("BCCD images not cached")
    splits = build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cached)))
    assert {k: len(v) for k, v in splits.items()} == SAMPLE_SPLIT
    assert check_split_disjoint(splits)
    assert dataset_digest([r for part in splits.values() for r in part]) == sm.SAMPLE_DIGEST


# --- BYOD ---------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_images_with_a_boxes_csv_from_a_zip_or_directory(tmp_path):
    rows = ["file,x0,y0,x1,y1,phrase"]
    for i in range(9):
        rows.append(f"photo{i}.png,20,20,60,60,Red Disc")
        rows.append(f"photo{i}.png,90,50,140,100,blue square.")
    csv_text = "\n".join(rows) + "\n"
    zip_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for i in range(9):
            buffer = io.BytesIO()
            _scene(i).save(buffer, format="PNG")
            archive.writestr(f"photo{i}.png", buffer.getvalue())
        archive.writestr("boxes.csv", csv_text)
    records, vocabulary = load_byod_dataset(zip_path)
    assert len(records) == 9 and vocabulary == VOCAB and all(len(r["boxes"]) == 2 for r in records)
    assert validate_dataset(records, vocabulary)["n_records"] == 9 and records[0]["id"] == "photo0"
    folder = tmp_path / "folder"
    folder.mkdir()
    for i in range(9):
        _scene(i).save(folder / f"photo{i}.png")
    (folder / "boxes.csv").write_text(csv_text, encoding="utf-8")
    assert len(load_byod_dataset(folder)[0]) == 9
    (folder / "boxes.csv").write_text(csv_text.replace("photo8.png", "missing.png"), encoding="utf-8")
    with pytest.raises(ValueError, match="not among the uploaded files"):
        load_byod_dataset(folder)
    (folder / "boxes.csv").write_text("file,x0,y0,x1,y1,phrase\nphoto0.png,1,1,2,2,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no phrase"):
        load_byod_dataset(folder)
    (folder / "boxes.csv").unlink()
    with pytest.raises(ValueError, match="needs a boxes.csv"):
        load_byod_dataset(folder)
    with pytest.raises(ValueError, match="neither"):
        load_byod_dataset(tmp_path / "nothing")
    csv_path = write_dataset_csv(records, tmp_path / "train.csv")
    assert csv_path.read_text(encoding="utf-8").startswith("id,width,height,n_boxes,blue square,red disc")


# --- metrics, baselines, evaluate ----------------------------------------------------------------------------------


def test_detection_metrics_and_baselines():
    records = _records()
    exact = [[(0.9, label, list(box)) for box, label in zip(r["boxes"], r["labels"], strict=True)] for r in records]
    perfect = detection_metrics(exact, records, VOCAB)
    assert perfect["map50"] == 1.0 and perfect["map75"] == 1.0 and perfect["recall50"] == 1.0 and set(perfect["per_category"]) == set(VOCAB)
    swapped = [[(0.9, "blue square" if label == "red disc" else "red disc", list(box)) for box, label in zip(r["boxes"], r["labels"], strict=True)] for r in records]
    assert detection_metrics(swapped, records, VOCAB)["map50"] == 0.0
    shifted = [[(0.9, label, [box[0] + 8, box[1], box[2] + 8, box[3]]) for box, label in zip(r["boxes"], r["labels"], strict=True)] for r in records]
    loose = detection_metrics(shifted, records, VOCAB)
    assert loose["map50"] == 1.0 and loose["map75"] < 1.0  # 8 px of 40 px: IoU 0.67
    halved = [preds[:1] for preds in exact]
    half = detection_metrics(halved, records, VOCAB)
    assert half["per_category"]["red disc"]["ap50"] == 1.0 and half["per_category"]["blue square"]["ap50"] == 0.0 and half["recall50"] == 0.5
    noisy = [[*preds, (0.1, "red disc", [0.0, 100.0, 20.0, 119.0])] for preds in exact]  # a low-scored false positive ranks last
    assert detection_metrics(noisy, records, VOCAB)["map50"] == 1.0
    grid = grid_prior_baseline(records, records, VOCAB)
    assert grid["baseline"].endswith("'red disc'") and 0.0 <= grid["map50"] < 0.5
    relabel = majority_relabel_baseline(exact, records, records, VOCAB)
    assert relabel["per_category"]["red disc"]["ap50"] < 1.0 and relabel["per_category"]["blue square"]["ap50"] == 0.0
    with pytest.raises(ValueError, match="same length"):
        detection_metrics(exact[:1], records, VOCAB)
    with pytest.raises(ValueError, match="each prediction"):
        detection_metrics([[(0.9, "red disc")]] * 12, records, VOCAB)
    with pytest.raises(ValueError, match="at least one labelled box"):
        grid_prior_baseline([], records, VOCAB)


def test_phrase_token_groups_follow_the_delimiters():
    # [CLS] red blood cell . white blood cell . platelet . [SEP]
    ids = [101, 2417, 2668, 3526, 1012, 2317, 2668, 3526, 1012, 20228, 1012, 102, 0, 0]
    assert pl._phrase_token_groups(ids) == [[1, 2, 3], [5, 6, 7], [9]]
    assert pl._phrase_token_groups([101, 102]) == []


def test_evaluate_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = GroundingDINOPipeline(_no_runner, "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.predict_boxes(_scene(0), VOCAB)
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.evaluate(_records(), VOCAB)
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records(), None, VOCAB)
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"prompts": list(PROMPTS)},
        "tensors": ["model.decoder.layers.0.self_attn.out_proj.weight", "bbox_embed.0.layers.0.weight"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="decoder or the box"):
        pl._check_artifact_manifest(_manifest(tmp_path, tensors=["model.backbone.conv_encoder.model.embeddings.weight"]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="prompts"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"prompts": "cell"}), tmp_path, "base-digest")


def test_trainable_names_selects_the_decoder_and_the_heads():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["model.backbone.conv_encoder.x", "model.text_backbone.encoder.x", "model.encoder.layers.0.x", "model.decoder.layers.0.x", "model.decoder.reference_points_head.x", "bbox_embed.0.x", "class_embed.0.x", "model.enc_output.weight"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model()) == ["model.decoder.layers.0.x", "model.decoder.reference_points_head.x", "bbox_embed.0.x", "class_embed.0.x"]


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 11400.5, "val": {"map50": 0.34, "map75": 0.1, "recall50": 0.5, "n": 50}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["map50"] == 0.34
