# Grounding DINO tiny zero-shot detection pipeline

DIMER inference wrapper for **Grounding DINO tiny** (`IDEA-Research/grounding-dino-tiny`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline detects whatever short English phrases the caller supplies and returns pixel-space boxes with an uncalibrated score; there is no fixed class list and no calibrated confidence.

## Upstream alignment

- Model: `IDEA-Research/grounding-dino-tiny`
- Revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`
- Upstream weight license: Apache-2.0
- Upstream task: zero-shot (open-vocabulary, text-prompted) object detection
- Repository adaptation: a bounded fine-tuning contract (`evaluate`, `adapt`, `save_artifact`, `from_artifact`) over the cross-modality decoder and the box / contrastive heads with the upstream Hungarian criterion, against labelled boxes and a phrase vocabulary; the image backbone, the text encoder and the feature enhancer stay frozen and the base weights are never modified on disk

## Quick start

```python
from PIL import Image
from grounding_dino_detection_pipeline import GroundingDINOPipeline, box_iou

pipe = GroundingDINOPipeline.from_pretrained()       # stages + verifies weights/grounding-dino-tiny first
result = pipe.detect(Image.open("photo.jpg"), ["a cat", "remote control"])
for det in result["detections"]:                      # sorted by score, boxes are [x0, y0, x1, y1] pixels
    print(det["label"], det["box"], round(det["score"], 3))
print(result["prompt_text"])                          # "a cat. remote control."

# thresholds are the upstream card's example values; override per call
result = pipe.detect(Image.open("photo.jpg"), ["a cat"], box_threshold=0.3, text_threshold=0.25)
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the test suite (`tests/test_model_backed.py` runs only where the snapshot is staged and skips otherwise). On a fresh clone the manifest is committed but the weights are not: `GroundingDINOPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/grounding-dino-tiny/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (9 files)
  config.json
  preprocessor_config.json
  tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json, added_tokens.json
  model.safetensors          # git-ignored, 689,359,096 bytes
  README.md
```

## Adaptation contract

```python
from grounding_dino_detection_pipeline import (
    PROMPTS, GroundingDINOPipeline, build_sample_dataset, fetch_corpus, grid_prior_baseline, load_byod_dataset, read_corpus, split_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 364 BCCD blood smears, 220 / 50 / 94
# or: records, prompts = load_byod_dataset("my_boxes.zip"); splits = split_dataset(records, prompts, seed=42)

pipe = GroundingDINOPipeline.from_pretrained()                        # cuda:0 if available, else cpu
grid = grid_prior_baseline(splits["train"], splits["test"], PROMPTS)  # map50, map75, recall50, per_category
frozen = pipe.evaluate(splits["test"], PROMPTS)
result = pipe.adapt(splits["train"], splits["validation"], PROMPTS, epochs=6, lr=5e-5, batch_size=4)
adapted = pipe.evaluate(splits["test"], PROMPTS)
pipe.save_artifact("outputs/adapter")                                 # adapter.safetensors + manifest.json
again = GroundingDINOPipeline.from_artifact("outputs/adapter")
```

- Records are `{id, image, boxes, labels}`: `image` a PIL image within the ceilings, `boxes` one or more `[x0, y0, x1, y1]` pixel boxes inside it (at most `MAX_BOXES` = 300), `labels` one phrase per box drawn from the prompt vocabulary passed beside the records (1..16 distinct phrases that `format_prompts` accepts); `validate_dataset(records, prompts)` checks the shape (8..5,000 records, unique ids, boxes inside the image, labels in the vocabulary) before any model import, and `split_dataset` de-duplicates by decoded pixels; `check_split_disjoint` asserts no image is shared.
- The default sample (`samples.py`) is the BCCD blood-cell dataset (`Shenggan/BCCD_Dataset` at commit `d272fb14cdff6e473fafeeeba32aba5f560e9e43`, **MIT**): 364 microscope photographs of blood smears pinned by byte size and SHA-256 (`SAMPLE_IMAGES`, with their 4,886 Pascal-VOC boxes inline), fetched from GitHub's raw-content host at run time and cached git-ignored under `weights/bccd/`; the vocabulary `PROMPTS` spells the three classes as `red blood cell`, `white blood cell`, `platelet`. `build_sample_dataset` draws a seeded 220 / 50 / 94 image-level split (`SAMPLE_DIGEST` pins the draw).
- `predict_boxes(image, prompts, *, score_floor=0.05)` scores every one of the 900 queries per phrase — the maximum over that phrase's tokens of the query's sigmoid token logits — and keeps each with its best phrase when the score reaches the floor: the single-phrase-per-box rule the corpus measures use (`detect` keeps the processor's text-threshold labelling). `evaluate(records, prompts)` runs it on every record and returns `detection_metrics` (`metrics.py`): AP50 and AP75 per phrase (Pascal-VOC all-point interpolation, greedy one-to-one matching), their means `map50` / `map75`, the recall at IoU 0.5, plus `verdict` (`measured` / `measured-small-sample`) and `adapted`. `grid_prior_baseline` (mean-sized boxes tiled over the image, majority phrase) and `majority_relabel_baseline` (any predictor's boxes relabelled to the majority phrase — the tutorial feeds it the frozen model's boxes for the word `cell`) are the two non-adapted references the tutorial scores beside the model.
- `adapt(train, val=None, prompts, *, epochs=6, lr=5e-5, batch_size=4, seed=0, progress=None)` trains only the decoder (its layers and reference-point head) and the box / contrastive heads (11,187,460 of 172,249,090 parameters): every image is paired with the formatted vocabulary and each box with its phrase index, and the upstream Grounding DINO criterion — Hungarian matching, sigmoid-focal query-to-phrase alignment, L1 and generalised-IoU box terms — scores the queries; AdamW without weight decay, gradient clipping at 0.1, seeded shuffling, no augmentation; epoch 0 records the frozen model and the epoch with the highest validation mAP50 is kept. The update is transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.grounding-dino-tiny.adapter.v1`: base id, revision and weight digest, tensor names, file size and SHA-256, the prompt vocabulary, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the decoder and the heads, and overlays the tensors onto a freshly loaded base.

## Input ceilings and thresholds

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 256`; `BOX_THRESHOLD = 0.4`, `TEXT_THRESHOLD = 0.3`. One image per call. See `MODEL_CARD.md` for who owns tuning the thresholds and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/tutorials/grounding_dino_detection_colab.ipynb)

`tutorials/grounding_dino_detection_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's three modules (`pipeline.py`, `metrics.py`, `samples.py`), the model identity, the 9-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`; see `tutorials/README.md`). It stages and digest-verifies the snapshot, fetches the 364 digest-pinned BCCD photographs with their boxes and the three-phrase vocabulary, splits them by image without leakage, detects a synthetic scene through the inference contract with an input manifest and a rejection probe, scores the frozen model's per-phrase AP50 and AP75 on the 94 held-out images beside the grid-prior and generic-prompt baselines (the frozen model grounds `platelet` not at all), runs a bounded fine-tuning of the decoder and the heads with the upstream criterion and validation-mAP50 epoch selection, re-scores the held-out split per phrase, re-detects the scene and four images as side-by-side panels, exports the adapter and reloads it with verified box parity, and writes `grounding_dino_detection_train.csv`, `grounding_dino_detection_input_manifest.json`, `grounding_dino_detection_scene_frozen.png`, `grounding_dino_detection_scene_adapted.png`, `grounding_dino_detection_evaluation_report.json`, `grounding_dino_detection_examples/`, `grounding_dino_detection_adapter/` and `grounding_dino_detection_result.json` under `outputs/`.

The default path needs a CUDA runtime in practice (about thirteen minutes on an RTX 5070 Ti after the downloads, of which ten are the six epochs; a Tesla T4 takes roughly twice that; a CPU runtime would take hours). The metrics it prints are one seeded split of one 364-image sample with one three-phrase vocabulary — evidence that the adaptation contract works, not a detection benchmark or production-fitness evidence.

## Release status

**Release-grade** — the `E2E` notebook blob `dc1218be` (committed at `832d5d4`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 1503.7 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, adaptation contract and build record, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
