---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: zero-shot-object-detection
base_model: IDEA-Research/grounding-dino-tiny
date_published: "2023-09-25"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny)"
---

# Grounding DINO tiny (DIMER package v0.1.0) — Zero-Shot Object Detection (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IDEA--Research%2Fgrounding--dino--tiny-ffcc4d?style=flat)](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-IDEA--Research%2FGroundingDINO-181717?style=flat&logo=github&logoColor=white)](https://github.com/IDEA-Research/GroundingDINO)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2303.05499-b31b1b.svg)](https://arxiv.org/abs/2303.05499)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/tutorials/grounding_dino_detection_colab.ipynb) [`grounding_dino_detection_colab.ipynb`](https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/tutorials/grounding_dino_detection_colab.ipynb)  
  *Zero-shot text-prompted object detection with the pinned `IDEA-Research/grounding-dino-tiny` weights: score-ordered xyxy boxes with grounded phrases and uncalibrated sigmoid scores under caller-owned `box_threshold`/`text_threshold`; `box_iou` against drawn boxes as sanity evidence only, no mAP.*

---

#### Description

`IDEA-Research/grounding-dino-tiny` is the Transformers-format release of the Grounding DINO tiny model (Liu et al., arXiv:2303.05499), pinned here to revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. The snapshot `config.json` declares `GroundingDinoForObjectDetection`: a Swin-T image backbone (`depths` 2/2/6/2) and a BERT text encoder feed a DETR-style encoder–decoder (`d_model` 256, 6 encoder and 6 decoder layers, 8 heads, 4 deformable-attention points, 4 feature levels, `two_stage` query selection, `num_queries` 900) whose cross-modality fusion lets each query attend to the prompt tokens. At inference the model reads one image and one text string and emits, per query, a box and a per-token similarity; the processor keeps queries whose maximum token similarity clears a box threshold and labels each box with the tokens above a text threshold. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `GroundingDINOPipeline.from_pretrained` (verified local loading, `trust_remote_code=False`), `format_prompts` (the lowercase-and-period prompt grammar the upstream card requires), `detect` (input validation, threshold checks, sorted pixel-space output), and `box_iou`.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is text-prompted (open-vocabulary) object detection: input one RGB still (`PIL.Image.Image`, any mode, converted to RGB) and a list of 1–16 phrases; output a list of detections, each an xyxy pixel box, the matched phrase text as `label`, and an uncalibrated sigmoid `score`, sorted by score. Envisioned applications are ad-hoc object localisation where no detector has been trained for the vocabulary — finding "forklift" or "safety vest" in warehouse photos, "solar panel" in aerial imagery, "logo" in product shots — bootstrapping labels for a conventional detector, and cropping regions for a downstream classifier or segmenter (the SAM 2 sibling pipeline accepts these boxes as prompts). Within DIMER the pipeline is an inference component and a zero-configuration baseline, not a certified detector for any specific class.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, and application developers integrating open-vocabulary detection into research prototypes, internal enterprise tooling, or the DIMER workbench. A user is expected to understand that the score is a sigmoid similarity and not a probability, that the two thresholds trade recall against false positives and must be tuned per vocabulary and per domain, that phrasing changes results ("red circle" and "circle" are different prompts), that boxes for the same object may be returned under several phrases, and that mean average precision can only be measured on a labelled set they supply. Users who need tracking, instance masks, counting under occlusion, or a fixed closed vocabulary with calibrated confidences are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no segmentation masks (use `sam2-segmentation-pipeline` with these boxes), no tracking across frames, no attribute or relationship reasoning beyond what a short noun phrase expresses, no OCR, no depth. Phrases are matched by token similarity, so negation ("not a dog") and counting ("three cats") are not understood.
2. **Input boundary:** `detect` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a single string instead of a list, empty phrases, more than `MAX_PROMPTS = 16` phrases, any phrase over `MAX_PROMPT_CHARS = 48` characters, prompt text over `MAX_TEXT_TOKENS = 256` BERT tokens (config `max_text_len`), and thresholds outside `[0, 1]` (`ValueError`). One image per call. Every image is resized to shortest edge 800 / longest edge 1333, so objects smaller than a few pixels at that scale are unlikely to be found.
3. **Input boundary:** non-photographic imagery (line art, medical scans, thermal, radar) and non-English prompts fall outside the grounding pretraining; the pipeline does not detect them and results on them are undefined.
4. **Decision boundary:** not for autonomous safety or security decisions — intrusion alarms, weapon detection, collision avoidance, medical screening — without a human reviewing each detection and a locally measured precision/recall.

#### Factors

###### Groups

This pipeline is not human-centric by design: it localises whatever the prompt names and does not classify or identify people. It will, however, detect "person", "man", "woman", "child", or any demographic noun a caller types, and the grounding corpora the paper names (Objects365, GoldG, Cap4M — the snapshot README itself lists no training data) are web and detection collections that neither the upstream authors nor this repository have audited for balance across skin tone, age, gender, disability, or dress. Any difference in recall or box quality across such groups, or in how readily a demographic phrase attaches to a box, is therefore unknown, not known to be absent. A downstream operator who runs person-related prompts is responsible for a fairness audit on their own data: stratify a labelled sample by the relevant groups and compare per-group recall, precision, and `box_iou` before relying on the output.

###### Instrumentation

The upstream training data comes from detection datasets (Objects365), phrase-grounding datasets (GoldG) and image–caption pairs (Cap4M), all consumer and web photographs of unspecified camera provenance with human-drawn or caption-derived boxes; the text side is tokenised by a BERT WordPiece vocabulary (`vocab.txt`, 231,508 bytes in the snapshot). Inference images arrive from whatever camera the operator uses; sensor resolution, lens distortion, compression, motion blur, and unusual viewpoints (top-down, fisheye) all change the visual evidence, and the 800/1333 resize (`preprocessor_config.json`, bilinear, ImageNet mean/std) discards detail below that scale. Prompt text is a second instrument: capitalisation is normalised by `format_prompts`, but typos, plurals, and synonyms reach the text encoder unchanged. The pipeline validates type, size, count, and length only; it cannot detect a miscalibrated camera or an ambiguous phrase.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`): load 6.18 s (689 MB checkpoint), a 320x240 synthetic scene with two prompts 4.37 s, a 4096x4096 noise image with one prompt 3.28 s — cost is dominated by the fixed 800/1333 working resolution and the 900 queries, not by the caller's pixel count. Data environment: the model assumes an ordinary photograph in which the prompted objects are visually distinct and nameable by a short English noun phrase; crowded scenes, heavy occlusion, tiny objects, and abstract or ambiguous phrases lower recall and raise spurious matches, and the pipeline reports no signal when they do.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each detection carries `score`, the maximum sigmoid similarity between that query and the prompt tokens — a ranking signal, not a probability and not a measure of correctness. The repository ships `box_iou(a, b)`, the intersection-over-union of two xyxy boxes, because it is the primitive every detection metric is built from; mean average precision itself is not implemented, since it needs a labelled set with one convention for matching, class assignment, and IoU thresholds that the caller must choose. To evaluate, the caller supplies ground-truth boxes per phrase and computes precision/recall or mAP at their chosen IoU with `box_iou`. The public `evaluation_report(result, ground_truth_boxes=None)` stage returns that report in machine-readable form: one `box_iou` entry per supplied reference box with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled box set that would be required when no reference is supplied. The upstream README's COCO zero-shot AP is an upstream-reported number that this pipeline does not reproduce or claim.

###### Decision thresholds

Two thresholds are applied, both exposed as module constants: `BOX_THRESHOLD = 0.4` keeps a query only if its best token similarity is at least 0.4, and `TEXT_THRESHOLD = 0.3` decides which tokens form the returned `label`. These are the values the upstream model card's usage example passes; they were not tuned by this repository and are not calibrated for any domain. Both can be overridden per call (`detect(..., box_threshold=, text_threshold=)`), and the smoke run shows the effect is domain-dependent (a clean synthetic scene returned the same two boxes at 0.4/0.3 and 0.2/0.2). A deployment owns tuning them on its own labelled data: lower the box threshold when a missed object costs more than a false alarm (a reviewer will discard extras), raise it when a false alarm triggers an action, and re-tune whenever the prompt vocabulary or the image source changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, box coordinates, and scores on one synthetic scene, not accuracy. Run-to-run variability comes from floating-point kernel selection across CPU builds and accelerators and from the deformable-attention sampling implementation (`disable_custom_kernels` false in the config selects a custom kernel when available); there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines. The `score` is an uncalibrated sigmoid: a 0.9 is not a 90 % chance the box is right. A caller who needs calibrated confidences must fit a calibration map on their own labelled detections; a caller who needs an uncertainty estimate for a metric must supply labelled data and compute it over many images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README discloses no training data; the Grounding DINO paper reports pretraining on Objects365, GoldG (Flickr30k entities and Visual Genome grounding), and Cap4M web captions, collections whose per-image consent and licence status the paper does not enumerate and which are known to contain photographs of people, so personal data in the corpus is not ruled out — it is expected. This repository distributes code, tests, and documentation; it does not distribute the 689,359,096-byte `model.safetensors`, which is staged locally under `weights/grounding-dino-tiny/` and git-ignored, and it ships no sample images. The operator must audit the images and prompts they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will localise whatever it is asked to.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — weapon or contraband screening, locating people in search-and-rescue imagery, detecting findings in medical images, monitoring workers for compliance — would be admissible only with human review of every detection before action, a locally measured precision/recall on the deployment's own labelled data, a documented threshold policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 9 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True` and always passes `trust_remote_code=False`. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused.
- **Input integrity:** the public `validate_inputs(image, prompts, *, box_threshold, text_threshold)` stage applies exactly the checks `detect` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; `format_prompts` rejects a bare string, empty phrases, more than 16 phrases, or phrases over 48 characters, and normalises case and terminators; the runner rejects prompt text over 256 tokens before the forward pass; thresholds outside `[0, 1]` are rejected; `detect` raises on a malformed backend detection.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; every result carries `model_id`, `model_revision`, the exact `prompt_text` sent, and the thresholds used.
- **Refusals:** no batching, no download without the explicit flag, no threshold defaults hidden inside the runner.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Spurious matches:** a phrase that names nothing in the image can still clear the thresholds on a look-alike region; the operator and any downstream action bear the harm; likely for vague prompts; magnitude depends on what the detection triggers.
- **Missed objects:** small, occluded, or unusually rendered instances fall below the box threshold; whoever relies on completeness (safety monitoring, inventory) bears the harm.
- **Prompt-driven profiling:** the model will localise "person wearing hijab" or "wheelchair" as readily as "car"; the data subject bears the harm when such prompts drive decisions; the pipeline cannot distinguish a legitimate from an abusive phrase.
- **Automation bias:** tight, confident-looking boxes invite trust that a 0.4 sigmoid threshold has not earned.
- **Privacy exposure:** images of people or private spaces are processed without any content check.
- **Bias amplification:** any imbalance in the web-derived grounding corpora is reproduced as uneven recall across appearance groups, undetected because no per-group evaluation exists.
- **Resource use:** ~4 s per image on the reference CPU and a 689 MB checkpoint; a request stream can saturate a shared host.

###### Use cases

Prohibited even where the model would work: covert surveillance or tracking of individuals, biometric or demographic profiling (prompting for protected characteristics to locate, count, or sort people), social scoring, and any use that discriminates unlawfully in employment, housing, credit, insurance, education, or healthcare access. Also prohibited are targeting of people or property for harm, deceptive uses that present detections as verified facts or evidence, and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by an unreviewed detection are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `IDEA-Research/grounding-dino-tiny`
- Revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`
- Snapshot manifest: `weights/grounding-dino-tiny/dimer-base-manifest.json`, 9 files, `totalBytes` 690308125
- `model.safetensors` SHA-256: `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` (689,359,096 bytes)
- `config.json` SHA-256: `eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a` (1,644 bytes)
- Weight format: SafeTensors; loader `AutoModelForZeroShotObjectDetection.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)` with `AutoProcessor` (image processor + BERT tokenizer) from the same directory

## Input/output contract

- `GroundingDINOPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `detect(image, prompts, *, box_threshold=0.4, text_threshold=0.3) -> dict` with keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": str, "score": float}` in input-pixel coordinates, sorted by descending score), `prompt_text` (the normalised string sent to the model), `box_threshold`, `text_threshold`, `width`, `height`, `model_id`, `model_revision`.
- `format_prompts(prompts) -> str` — lowercases, strips, and period-terminates each phrase: `["A Cat", "remote control."]` → `"a cat. remote control."`.
- Ceilings: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 256`.
- `box_iou(a, b) -> float` on xyxy boxes; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32; preprocessing resize to shortest edge 800 / longest edge 1333, bilinear, ImageNet mean/std, padding (`GroundingDinoImageProcessor` from the snapshot); text through the snapshot BERT tokenizer.
- Measured 2026-09-12 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=""`, device `cpu`: `verify_snapshot` 0.46 s (9 files, 690 MB); load 6.18 s; `detect` on a synthetic 320x240 scene (grey background, dark rectangle at [40, 60, 140, 180], red disc at [200, 80, 280, 160]) with prompts `["rectangle", "red circle"]` → 2 detections in 4.37 s: `red circle` [199.4, 79.1, 281.7, 161.4] score 0.932, `rectangle` [39.6, 59.3, 141.4, 181.4] score 0.698; same call at thresholds 0.2/0.2 → identical 2 boxes in 4.51 s; 4096x4096 uniform-noise image with prompt `["rectangle"]` → 1 detection in 3.28 s. Process wall 21 s.
- Tests: `pytest -q -o addopts= tests` — 11 passed, offline, no weights required; `ruff check src tests` clean.
- Not executed: CUDA path, half precision, any precision/recall measurement against labelled boxes.

## References

- Liu, Zeng, Ren, Li, Zhang, Yang, Li, Yang, Su, Zhu, Zhang. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV 2024. https://arxiv.org/abs/2303.05499
- Zhang et al. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. ICLR 2023. https://arxiv.org/abs/2203.03605
- Upstream code: https://github.com/IDEA-Research/GroundingDINO
- Upstream card: https://huggingface.co/IDEA-Research/grounding-dino-tiny
- Transformers `GroundingDINO` documentation: https://huggingface.co/docs/transformers/model_doc/grounding-dino
