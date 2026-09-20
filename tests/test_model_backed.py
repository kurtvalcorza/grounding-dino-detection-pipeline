"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): phrase-scored boxes and
evaluation with the per-phrase breakdown, a one-epoch adaptation of the decoder on a dozen drawn scenes, the artifact
round trip, the loader's scope check, the transactional guarantee and — where CUDA is visible — the same path on the
accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from PIL import Image, ImageDraw

from grounding_dino_detection_pipeline import (
    DECODER_PARAMETERS,
    DEFAULT_WEIGHTS_DIR,
    PARAMETER_COUNT,
    WEIGHT_FILE,
    GroundingDINOPipeline,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

VOCAB = ["red ball", "blue box"]


def _record(i, size=(240, 180)):
    image = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    dx = (i % 3) * 10
    draw.ellipse([30 + dx, 30, 90 + dx, 90], fill=(220, 30, 30))
    draw.rectangle([140, 80, 210, 150], fill=(30, 30, 220))
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    return {"id": f"r{i:03d}", "image": image, "boxes": [[30.0 + dx, 30.0, 90.0 + dx, 90.0], [140.0, 80.0, 210.0, 150.0]], "labels": VOCAB}


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(16)]


@pytest.fixture(scope="module")
def pipe():
    return GroundingDINOPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_identity_predict_boxes_and_frozen_evaluation_with_the_breakdown(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith(("model.decoder.", "bbox_embed.", "class_embed."))) == DECODER_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    boxes = pipe.predict_boxes(records[0]["image"], VOCAB)
    assert boxes and all(0.0 <= s <= 1.0 and label in VOCAB and len(box) == 4 for s, label, box in boxes)
    assert boxes == sorted(boxes, key=lambda item: -item[0])
    assert len(pipe.predict_boxes(records[0]["image"], VOCAB, score_floor=0.9)) <= len(boxes)
    with pytest.raises(ValueError, match="score_floor"):
        pipe.predict_boxes(records[0]["image"], VOCAB, score_floor=2.0)
    metrics = pipe.evaluate(records[:8], VOCAB)
    assert metrics["n"] == 8 and 0.0 <= metrics["map50"] <= 1.0 and metrics["adapted"] is False and metrics["prompts"] == VOCAB
    assert set(metrics["per_category"]) == set(VOCAB) and metrics["verdict"] == "measured-small-sample"


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], VOCAB, epochs=1, batch_size=4)
    assert result["n_trainable"] == DECODER_PARAMETERS and result["n_total"] == PARAMETER_COUNT and result["prompts"] == VOCAB
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"map50", "map75", "recall50", "n"}
    assert all(n.startswith(("model.decoder.", "bbox_embed.", "class_embed.")) for n in result["trainable_names"])
    assert not any(n.startswith(("model.backbone", "model.text_backbone", "model.encoder.")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["adapter"]["prompts"] == VOCAB and manifest["metadata"] == {"note": "test"}
    reloaded = GroundingDINOPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert all(pipe.predict_boxes(r["image"], VOCAB) == reloaded.predict_boxes(r["image"], VOCAB) for r in records[:3])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[:4], VOCAB)["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, VOCAB, epochs=2, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = GroundingDINOPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])


def test_adapt_refuses_bad_hyperparameters_and_labels(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:12], None, VOCAB, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:12], None, VOCAB, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records[:12], None, VOCAB, epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="8..5000"):
        pipe.adapt(records[:4], None, VOCAB, epochs=1)
    with pytest.raises(ValueError, match="not in the prompt vocabulary"):
        pipe.adapt(records[:12], None, ["red ball"], epochs=1)
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, VOCAB, epochs=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        GroundingDINOPipeline.from_artifact(fewer, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["bbox_embed.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        GroundingDINOPipeline.from_artifact(extra, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    encoder = tmp_path / "encoder"
    shutil.copytree(artifact, encoder)
    (encoder / "manifest.json").write_text(json.dumps({**manifest, "tensors": [*manifest["tensors"], "model.encoder.layers.0.x"]}))
    with pytest.raises(ValueError, match="decoder or the box"):
        GroundingDINOPipeline.from_artifact(encoder, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, VOCAB, epochs=2, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_evaluate_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    cuda = GroundingDINOPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert cuda.device == "cuda:0"
    metrics = cuda.evaluate(records[:8], VOCAB)
    assert 0.0 <= metrics["map50"] <= 1.0
    result = cuda.adapt(records[:12], records[12:], VOCAB, epochs=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = GroundingDINOPipeline.from_artifact(artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert all(cuda.predict_boxes(r["image"], VOCAB) == reloaded.predict_boxes(r["image"], VOCAB) for r in records[:3])
