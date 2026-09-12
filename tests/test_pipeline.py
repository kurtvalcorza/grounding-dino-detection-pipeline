import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grounding_dino_detection_pipeline import (
    BOX_THRESHOLD,
    DEFAULT_WEIGHTS_DIR,
    MAX_IMAGE_SIDE,
    MAX_PROMPT_CHARS,
    MAX_PROMPTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    TEXT_THRESHOLD,
    GroundingDINOPipeline,
    box_iou,
    format_prompts,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "IDEA-Research/grounding-dino-tiny"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert 0 < TEXT_THRESHOLD < BOX_THRESHOLD < 1
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "grounding-dino"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "grounding-dino"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_format_prompts_lowercases_and_terminates():
    assert format_prompts(["A Cat", "remote control."]) == "a cat. remote control."
    with pytest.raises(TypeError):
        format_prompts("a cat")
    with pytest.raises(TypeError):
        format_prompts(["a cat", 3])
    with pytest.raises(ValueError, match="empty"):
        format_prompts(["a cat", " . "])
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        format_prompts(["x"] * (MAX_PROMPTS + 1))
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        format_prompts([])
    with pytest.raises(ValueError, match="MAX_PROMPT_CHARS"):
        format_prompts(["a" * (MAX_PROMPT_CHARS + 1)])


def _fake_pipeline(calls: list | None = None) -> GroundingDINOPipeline:
    def runner(image: Image.Image, text: str, box_threshold: float, text_threshold: float) -> list[dict]:
        if calls is not None:
            calls.append((image.mode, text, box_threshold, text_threshold))
        return [
            {"box": [1.0, 2.0, 10.0, 20.0], "label": "cat", "score": 0.5},
            {"box": [0.0, 0.0, 5.0, 5.0], "label": "dog", "score": 0.9},
        ]

    return GroundingDINOPipeline(runner, "cpu")


def test_detect_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.detect(Image.new("L", (40, 30)), ["Cat", "dog"])
    assert [d["label"] for d in result["detections"]] == ["dog", "cat"]  # sorted by score desc
    assert result["prompt_text"] == "cat. dog."
    assert result["box_threshold"] == BOX_THRESHOLD and result["text_threshold"] == TEXT_THRESHOLD
    assert (result["width"], result["height"]) == (40, 30)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert calls == [("RGB", "cat. dog.", BOX_THRESHOLD, TEXT_THRESHOLD)]


def test_detect_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.detect(np.zeros((30, 40, 3), dtype=np.uint8), ["cat"])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.detect(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)), ["cat"])
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.detect(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)), ["cat"])
    with pytest.raises(ValueError, match="box_threshold"):
        pipe.detect(Image.new("RGB", (64, 64)), ["cat"], box_threshold=1.5)
    with pytest.raises(ValueError, match="text_threshold"):
        pipe.detect(Image.new("RGB", (64, 64)), ["cat"], text_threshold=-0.1)


def test_detect_rejects_malformed_backend_output():
    pipe = GroundingDINOPipeline(lambda *_: [{"box": [0, 0, 1], "label": "x", "score": 0.1}], "cpu")
    with pytest.raises(RuntimeError):
        pipe.detect(Image.new("RGB", (64, 64)), ["x"])


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    with pytest.raises(ValueError):
        box_iou([10, 0, 0, 10], [0, 0, 1, 1])
