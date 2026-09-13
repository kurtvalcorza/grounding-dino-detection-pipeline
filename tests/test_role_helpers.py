"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from grounding_dino_detection_pipeline import (
    BOX_THRESHOLD,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_PROMPT_CHARS,
    MAX_PROMPTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    TEXT_THRESHOLD,
    evaluation_report,
    validate_inputs,
)

DRAWN = {"rectangle": [40.0, 60.0, 140.0, 180.0], "red circle": [200.0, 80.0, 280.0, 160.0]}


def _image(width: int = 320, height: int = 240) -> Image.Image:
    return Image.new("RGB", (width, height), (128, 128, 128))


def _result(detections: list[dict]) -> dict:
    return {
        "detections": detections,
        "prompt_text": "rectangle. red circle.",
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "width": 320,
        "height": 240,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), ["Rectangle", "red circle"], names=["scene.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["prompts"] == [1, MAX_PROMPTS]
    assert manifest["schema"]["prompt_chars"] == [1, MAX_PROMPT_CHARS]
    assert manifest["inputs"] == [{"id": "scene.png", "mode": "RGB", "size": [320, 240], "n_prompts": 2}]
    assert manifest["prompt_text"] == "rectangle. red circle."
    assert (manifest["box_threshold"], manifest["text_threshold"]) == (BOX_THRESHOLD, TEXT_THRESHOLD)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_thresholds() -> None:
    manifest = validate_inputs(_image(), ["rectangle"], box_threshold=0.2, text_threshold=0.15)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert (manifest["box_threshold"], manifest["text_threshold"]) == (0.2, 0.15)


def test_validate_inputs_rejects_like_detect() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64), ["rectangle"])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8), ["rectangle"])
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image", ["rectangle"])
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(_image(), "rectangle")
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        validate_inputs(_image(), ["rectangle"] * (MAX_PROMPTS + 1))
    with pytest.raises(ValueError, match="box_threshold"):
        validate_inputs(_image(), ["rectangle"], box_threshold=1.5)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), ["rectangle"], names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference_boxes() -> None:
    detection = {"box": [40.0, 60.0, 140.0, 180.0], "label": "rectangle", "score": 0.7}
    report = evaluation_report(_result([detection]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_detections"] == 1
    assert "box_iou" in report["needs"]
    assert report["baselines"] == []
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference_boxes() -> None:
    detections = [
        {"box": [200.0, 80.0, 280.0, 160.0], "label": "red circle", "score": 0.93},
        {"box": [40.0, 60.0, 140.0, 180.0], "label": "rectangle", "score": 0.70},
    ]
    report = evaluation_report(_result(detections), DRAWN, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "synthetic"
    assert [metric["id"] for metric in report["metrics"]] == ["box_iou", "box_iou"]
    by_reference = {metric["reference"]: metric for metric in report["metrics"]}
    assert by_reference["rectangle"]["value"] == pytest.approx(1.0)
    assert by_reference["rectangle"]["matched_label"] == "rectangle"
    assert by_reference["red circle"]["label_matches_reference"] is True
    assert all(metric["estimation"] for metric in report["metrics"])


def test_evaluation_report_records_a_mismatched_label_without_inventing_a_metric() -> None:
    detections = [{"box": [200.0, 80.0, 280.0, 160.0], "label": "red circle", "score": 0.93}]
    report = evaluation_report(_result(detections), DRAWN)
    by_reference = {metric["reference"]: metric for metric in report["metrics"]}
    assert by_reference["rectangle"]["value"] == 0.0
    assert by_reference["rectangle"]["label_matches_reference"] is False
    assert {metric["id"] for metric in report["metrics"]} == {"box_iou"}


def test_evaluation_report_handles_zero_detections() -> None:
    report = evaluation_report(_result([]), DRAWN)
    assert report["n_detections"] == 0
    assert [metric["value"] for metric in report["metrics"]] == [0.0, 0.0]
    assert [metric["matched_label"] for metric in report["metrics"]] == [None, None]
