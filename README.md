# Grounding DINO tiny zero-shot detection pipeline

DIMER inference wrapper for **Grounding DINO tiny** (`IDEA-Research/grounding-dino-tiny`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline detects whatever short English phrases the caller supplies and returns pixel-space boxes with an uncalibrated score; there is no fixed class list and no calibrated confidence.

## Upstream alignment

- Model: `IDEA-Research/grounding-dino-tiny`
- Revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`
- Upstream weight license: Apache-2.0
- Upstream task: zero-shot (open-vocabulary, text-prompted) object detection
- Repository adaptation: **none**; inference only

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

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `GroundingDINOPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

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

## Input ceilings and thresholds

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 256`; `BOX_THRESHOLD = 0.4`, `TEXT_THRESHOLD = 0.3`. One image per call. See `MODEL_CARD.md` for who owns tuning the thresholds and the measured CPU timings.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
