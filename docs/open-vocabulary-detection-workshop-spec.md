# DIMER Open-Vocabulary Object Detection Workshop

## Grounding DINO Tiny vs OWLv2 Base/16 Ensemble

**Proposed filename:** `DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`  
**Notebook Specification:** DIMER `NOTEBOOK_SPEC.md` **v2.1**  
**Profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Comparison scope:** `MULTI-MODEL`  
**Standalone:** `true`  
**Canonical workflow:** Frozen zero-shot inference and common evaluation; no fine-tuning  
**Recommended runtime:** CUDA GPU / Tesla T4 or equivalent  
**CPU support:** Technically possible but not recommended for the full evaluation  
**Task:** Text-prompted / open-vocabulary object detection  
**Primary evaluation dataset:** BCCD blood-cell detection dataset  
**Default evaluation split:** 94 held-out images  
**Prompt vocabulary:** `red blood cell`, `white blood cell`, `platelet`

---

# 1. Purpose

This notebook compares two live DIMER open-vocabulary object detection models:

1. **Grounding DINO Tiny**
2. **OWLv2 Base Patch16 Ensemble**

Both accept an image plus user-supplied text phrases and return localized objects:

`image + text phrases → phrase-labelled bounding boxes + scores`

Unlike the closed-set object-detection workshop, the class vocabulary is not fixed in the model's detection head. The requested concepts are supplied at inference time through natural-language prompts.

The notebook focuses on the mechanics and consequences of text-conditioned detection:

- phrase formulation;
- query representation;
- candidate generation;
- score thresholds;
- duplicate detections;
- localization;
- prompt sensitivity;
- prompt-set interactions;
- domain shift; and
- the relationship between language choice and detector behavior.

The notebook MUST NOT fine-tune either model.

Grounding DINO's existing E2E notebook already demonstrates bounded supervised adaptation on BCCD. This workshop instead compares the **frozen pretrained models** under the same labelled evaluation contract.

---

# 2. Learning objectives

By the end of the notebook, the learner should be able to:

1. explain what makes object detection **open-vocabulary**;
2. distinguish open-vocabulary detection from closed-set classification-head detection;
3. describe the high-level architectural difference between Grounding DINO and OWLv2;
4. provide a text vocabulary and obtain phrase-conditioned bounding boxes;
5. understand why prompt wording can change detections;
6. explain why Grounding DINO and OWLv2 do not interpret the prompt list identically;
7. normalize both detectors into one common prediction representation;
8. measure phrase-level AP50, AP75, recall, and localization quality;
9. distinguish a detection score from a calibrated probability;
10. recognize duplicate detections created when no NMS is applied;
11. analyze cases where models disagree on phrase identity or localization;
12. evaluate prompt sensitivity without using the held-out set to tune the canonical prompt vocabulary; and
13. apply the same workflow to a labelled user dataset with a custom vocabulary.

---

# 3. Models

## 3.1 Grounding DINO Tiny

**DIMER profile:** Grounding DINO Tiny Zero-Shot Object Detection

| Field | Value |
|---|---|
| Upstream model | `IDEA-Research/grounding-dino-tiny` |
| Immutable revision | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| License | Apache-2.0 |
| Weight format | SafeTensors |
| `model.safetensors` size | 689,359,096 bytes |
| `model.safetensors` SHA-256 | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |
| Parameters | 172,249,090 |
| Image backbone | Swin-T |
| Text encoder | BERT |
| Encoder/decoder | DETR-style multimodal transformer |
| Object queries | 900 |
| Maximum prompts | 16 |
| Maximum phrase length | 48 characters |
| Maximum combined text length | 256 tokens |
| Native box threshold | 0.40 |
| Native text threshold | 0.30 |
| Evaluation score floor | 0.05 |
| NMS | None |

### Image preprocessing

Grounding DINO's processor:

- converts input to RGB;
- resizes with shortest edge approximately 800 px;
- constrains the longest edge to approximately 1333 px;
- applies the checkpoint's native image normalization.

The original aspect ratio is preserved.

---

## 3.2 OWLv2 Base Patch16 Ensemble

**DIMER profile:** OWLv2 Base P16 Ensemble Open-Vocabulary Detection

| Field | Value |
|---|---|
| Upstream model | `google/owlv2-base-patch16-ensemble` |
| Immutable revision | `cfd3195ba4ea9592eec887ded089f4c08eff231d` |
| License | Apache-2.0 |
| Weight format | SafeTensors |
| `model.safetensors` size | 619,918,824 bytes |
| `model.safetensors` SHA-256 | `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7` |
| Image tower | CLIP ViT-B/16 |
| Text tower | CLIP text encoder |
| Projection dimension | 512 |
| Working image size | 960 × 960 |
| Patch candidates | 3,600 |
| Maximum prompts | 16 |
| Maximum phrase length | 48 characters |
| Maximum tokens per phrase | 16 |
| Native detection threshold | 0.10 |
| NMS | None |

Parameter count SHOULD be computed from the loaded checkpoint at runtime rather than hard-coded into the notebook.

### Image preprocessing

OWLv2:

- converts input to RGB;
- pads the image to a square;
- resizes to `960 × 960`;
- normalizes using CLIP image statistics.

The padding/resizing policy therefore differs substantially from Grounding DINO.

---

# 4. Architectural comparison

The notebook SHOULD introduce the models before execution.

## Grounding DINO

Conceptually:

```text
image
↓
Swin-T image backbone
↓
multi-scale visual features
       ↕
cross-modal feature fusion
       ↕
BERT-encoded phrase sequence
↓
DETR-style object queries
↓
900 candidate boxes
↓
phrase-token grounding scores
```

The user prompt list is normalized and joined into one textual sequence:

```text
red blood cell. white blood cell. platelet.
```

Grounding DINO therefore performs cross-modal grounding between image features and token groups within a shared prompt sequence.

---

## OWLv2

Conceptually:

```text
image
↓
CLIP ViT-B/16
↓
60 × 60 image patches
↓
3,600 patch representations
         ↕
independent CLIP text-query embeddings
↓
box + image/text similarity per patch
```

Each phrase is represented as an individual text query.

The detector chooses the best-matching query for each patch candidate.

---

# 5. Important prompt-semantics difference

This distinction is a core teaching objective.

## Grounding DINO

The prompt vocabulary becomes a **single concatenated BERT sequence** separated by periods.

The phrases therefore share one text context.

Changing:

- wording;
- tokenization;
- phrase order; or
- the surrounding phrase set

can potentially affect grounding behavior.

---

## OWLv2

Each phrase becomes a separate CLIP text query.

The image patches are scored against those query embeddings.

The phrase list is therefore conceptually closer to:

```text
image patch × independent text queries
```

although changing wording still changes the text representation.

---

# 6. Why BCCD is the primary comparison dataset

Use the same digest-pinned **BCCD blood-cell dataset** already carried by the Grounding DINO DIMER pipeline.

**Dataset:** BCCD blood-cell detection  
**Repository:** `Shenggan/BCCD_Dataset`  
**Immutable commit:** `d272fb14cdff6e473fafeeeba32aba5f560e9e43`  
**License:** MIT  
**Images:** 364  
**Bounding boxes:** 4,886  
**Total image bytes:** approximately 7.6 MB

The DIMER carrier already records per-image:

- file size;
- SHA-256;
- width;
- height;
- bounding boxes; and
- original BCCD class.

The three classes map to natural-language prompts:

```text
RBC        → red blood cell
WBC        → white blood cell
Platelets  → platelet
```

Canonical prompt vocabulary:

```python
PROMPTS = [
    "red blood cell",
    "white blood cell",
    "platelet",
]
```

---

# 7. Why this is a useful open-vocabulary workshop

BCCD is intentionally not a generic COCO-style everyday-image task.

It provides:

- real labelled photographs;
- many object instances;
- three concepts expressible in ordinary language;
- small-object detection;
- dense scenes;
- strong class imbalance;
- a fixed held-out set; and
- a substantial domain shift from typical web/everyday imagery.

The workshop MUST explicitly state that BCCD is a **domain-shift stress test**, not a general open-vocabulary benchmark.

The notebook must not infer from BCCD performance that one detector is globally superior.

---

# 8. Dataset split

Reuse the existing deterministic DIMER split:

```text
train       220 images
validation   50 images
test         94 images
```

Seed:

`42`

Pinned complete-split digest:

`af9390b9803ac37416f0fcc5b59cc1ef0926b9ea42a0c27510eef3f399e32c08`

The comparison notebook uses:

**test split only for model evaluation.**

The training and validation portions are loaded only as necessary to reproduce and verify the exact existing split.

No fine-tuning or model selection occurs.

---

# 9. Split integrity

The notebook MUST verify:

- all 364 pinned images;
- every image byte size;
- every image SHA-256;
- decoded image dimensions;
- all 4,886 recorded boxes;
- deterministic split construction;
- full sample digest;
- 220 / 50 / 94 split sizes;
- no duplicate decoded image across splits.

The split identity must be established before model execution.

---

# 10. Common input contract

For the canonical workflow:

```text
image:
    PIL RGB image
    sides 16..4096 px

prompt vocabulary:
    1..16 distinct phrases
    each <= 48 characters
```

Because OWLv2 has the tighter text limit, the workshop's common vocabulary must additionally fit within:

`16 CLIP tokens per phrase`

Grounding DINO's combined BERT prompt must remain within:

`256 tokens total`

The notebook MUST validate both constraints.

---

# 11. Common output schema

Every model prediction becomes:

```text
model
image_id
prompt
score
x0
y0
x1
y1
```

Requirements:

- prompt belongs to the supplied vocabulary;
- coordinates are mapped to original image pixels;
- coordinates finite;
- positive box area;
- boxes clipped to image bounds;
- score finite;
- score in `[0, 1]`;
- detections sorted by descending score per image.

This normalized representation is the boundary between model-specific inference and common evaluation.

---

# 12. Single-phrase-per-box comparison contract

The main quantitative evaluation MUST use one phrase label per candidate box.

This already matches:

### Grounding DINO evaluation path

For every one of the model's 900 object queries:

1. calculate token-level sigmoid grounding scores;
2. reduce the tokens belonging to each phrase;
3. obtain one score per phrase;
4. choose the highest-scoring phrase;
5. assign that phrase to the box;
6. retain the candidate when its score reaches the evaluation floor.

### OWLv2

For each of the 3,600 image patches:

1. score the patch against each independent text query;
2. select the best query;
3. assign its phrase to the predicted box;
4. retain the candidate when its score reaches the evaluation floor.

This gives both systems the same externally meaningful evaluation object:

```text
(score, phrase, box)
```

without pretending their internal scoring mechanisms are identical.

---

# 13. Evaluation score floor

Default:

```python
EVAL_SCORE_FLOOR = 0.05
```

This matches the Grounding DINO DIMER evaluator's current candidate floor.

OWLv2 should be evaluated at the same numeric floor to retain low-score candidates for AP calculation.

The notebook MUST state:

> A common numeric floor does **not** imply that Grounding DINO and OWLv2 scores are calibrated to the same scale.

The floor exists to retain candidate detections before ranking-based evaluation, not to define an equivalent deployment operating point.

---

# 14. Native visualization thresholds

Quantitative evaluation and visual examples should be separated.

## Grounding DINO native demonstration thresholds

```text
box threshold  = 0.40
text threshold = 0.30
```

## OWLv2 native demonstration threshold

```text
threshold = 0.10
```

These are upstream/example defaults and are appropriate for readable visualizations.

They MUST NOT replace the lower common score floor used for AP evaluation.

---

# 15. Common evaluator

Use one notebook-local evaluator for both models.

Predictions:

```text
(score, phrase, [x0, y0, x1, y1])
```

Ground truth:

```text
boxes
labels
```

Matching must be:

- within the same image;
- within the same phrase;
- descending prediction-score order;
- one-to-one;
- highest-IoU unclaimed ground truth;
- independently repeated at each IoU threshold.

---

# 16. Principal metrics

To preserve compatibility with the current Grounding DINO evaluator, report:

### Required

- **mAP50**
- **mAP75**
- **recall50**

### Per phrase

For:

- `red blood cell`
- `white blood cell`
- `platelet`

report:

- number of ground-truth boxes;
- AP50;
- AP75;
- recall50;
- number of predictions.

These metrics use Pascal-VOC all-point AP semantics, matching the existing Grounding DINO carrier.

---

# 17. Supplemental AP@[.50:.95]

The workshop MAY additionally calculate:

**mAP@[.50:.95]**

over:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

This is a workshop-level common metric.

If implemented, it MUST be described separately from the carrier's existing AP50/AP75 evaluator semantics.

It MUST NOT be represented as the official COCO evaluator.

---

# 18. Trivial baseline

Retain the existing Grounding DINO tutorial's model-free baseline for context:

**grid-prior baseline**

Using the training portion only:

1. compute typical box dimensions;
2. tile fixed-size boxes over test images;
3. assign the training split's majority phrase;
4. evaluate using the same common metric implementation.

The baseline is not intended to be competitive.

Its purpose is to demonstrate that:

> localization and phrase grounding must both contribute information beyond dataset geometry and class frequency.

No test-set information may be used to construct the baseline.

---

# 19. Runtime

Both carriers use the same runtime family.

Required principal pins:

```text
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
transformers==4.57.6
safetensors==0.8.0
numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2
```

Python:

`3.12`

The notebook MUST report:

- Python;
- PyTorch;
- TorchVision;
- Transformers;
- Hugging Face Hub;
- NumPy;
- Pillow;
- CUDA availability;
- CUDA device;
- dtype;
- GPU memory when available.

---

# 20. Default form parameters

```python
USE_BYOD = False
BYOD_DATASET_PATH = ""

EVAL_SCORE_FLOOR = 0.05

GDINO_BOX_THRESHOLD = 0.40
GDINO_TEXT_THRESHOLD = 0.30

OWLV2_THRESHOLD = 0.10

RUN_PROMPT_EXPERIMENT = True
PROMPT_EXPERIMENT_IMAGES = 4

RUN_THRESHOLD_SWEEP = False

OUTPUT_DIR = "outputs/open_vocabulary_detection"
```

The canonical default path MUST require no user edits.

---

# 21. Model supply-chain verification

## 21.1 Grounding DINO

Embed the full nine-file DIMER snapshot manifest.

Model:

`IDEA-Research/grounding-dino-tiny`

Revision:

`a2bb814dd30d776dcf7e30523b00659f4f141c71`

Expected snapshot:

9 files

Total bytes:

`690,308,125`

Model weight digest:

`1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`

---

## 21.2 OWLv2

Model:

`google/owlv2-base-patch16-ensemble`

Revision:

`cfd3195ba4ea9592eec887ded089f4c08eff231d`

Expected snapshot:

9 files

Total bytes:

`621,510,370`

Weight digest:

`e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`

---

## 21.3 Loading rules

For both:

- fetch only manifest-listed files;
- use immutable revision;
- verify byte size;
- verify SHA-256;
- fail closed on mismatch;
- load only from local verified snapshot;
- use `trust_remote_code=False`;
- do not use upstream `.bin` alternatives when SafeTensors are available.

---

# 22. Standalone implementation

The notebook MUST NOT:

- clone either repository;
- import either DIMER pipeline package;
- fetch DIMER Python source at runtime;
- call DIMER workers;
- call DIMER APIs.

Notebook-local/reference code must implement:

- snapshot verification;
- prompt validation;
- image validation;
- Grounding DINO frozen inference;
- Grounding DINO single-phrase candidate extraction;
- OWLv2 frozen inference;
- prediction normalization;
- common evaluator;
- BCCD acquisition and split verification;
- prompt experiments;
- exports.

---

# 23. Canonical workflow

```text
Purpose
→ Runtime
→ Architecture comparison
→ Immutable model provenance
→ Dataset provenance
→ Fetch/verify BCCD
→ Reproduce split
→ Validate vocabulary
→ Trivial baseline
→ Grounding DINO load
→ Grounding DINO inference on 94 images
→ Unload model
→ OWLv2 load
→ OWLv2 inference on 94 images
→ Common evaluation
→ Per-phrase comparison
→ Object-level disagreement analysis
→ Prompt sensitivity experiment
→ Optional threshold experiment
→ Qualitative panels
→ Resource/timing comparison
→ Export
→ Optional BYOD
→ Interpretation
```

---

# 24. Grounding DINO inference stage

For every held-out BCCD image:

1. encode the canonical prompt sequence;
2. run the frozen model;
3. obtain 900 predicted boxes;
4. derive one score per phrase for each query;
5. choose the best phrase;
6. retain candidates with score ≥ `EVAL_SCORE_FLOOR`;
7. convert boxes to original pixel coordinates;
8. normalize into the common schema.

The stage MUST retain:

```text
image_id
prompt
score
box
query_index
```

where practical.

No Grounding DINO adaptation tensors may be loaded.

---

# 25. OWLv2 inference stage

For every held-out image:

1. independently encode the three text queries;
2. preprocess image with OWLv2's native square-padding pipeline;
3. process the 3,600 patch candidates;
4. choose the best text query for every patch;
5. retain candidates with score ≥ `EVAL_SCORE_FLOOR`;
6. map boxes to original pixel coordinates;
7. normalize into the common schema.

The stage SHOULD retain:

```text
image_id
prompt
score
box
patch_index
```

where accessible.

---

# 26. Sequential model execution

The two large checkpoints should not remain resident simultaneously.

Canonical sequence:

```text
load Grounding DINO
→ evaluate
→ retain normalized results only
→ delete model
→ garbage collect
→ empty CUDA cache
→ load OWLv2
→ evaluate
```

This reduces GPU-memory requirements and improves notebook portability.

---

# 27. Timing protocol

For each model:

1. load model and record load time separately;
2. perform one warm-up inference excluded from latency statistics;
3. synchronize CUDA when applicable;
4. time all 94 evaluation images;
5. record individual image inference times.

Report:

- mean;
- median;
- minimum;
- maximum;
- total evaluation inference time;
- model-load time;
- peak GPU memory where available.

Do not include:

- installation;
- model download;
- dataset download

inside inference latency.

All timing claims must identify the execution environment.

---

# 28. Main comparison table

Generate:

| Model | mAP50 | mAP75 | Recall50 | Predictions | Mean latency | Weight size | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grid prior | measured | measured | measured | measured | — | — | — |
| Grounding DINO Tiny | measured | measured | measured | measured | measured | 689 MB | measured |
| OWLv2 Base/16 Ensemble | measured | measured | measured | measured | measured | 620 MB | measured |

If supplemental mAP@[.50:.95] is implemented, include it as a separate column.

The notebook MUST NOT hard-code an expected winner.

---

# 29. Per-phrase comparison

Produce:

| Phrase | GT boxes | GDINO AP50 | GDINO AP75 | GDINO Recall50 | OWLv2 AP50 | OWLv2 AP75 | OWLv2 Recall50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| red blood cell | measured | ... | ... | ... | ... | ... | ... |
| white blood cell | measured | ... | ... | ... | ... | ... | ... |
| platelet | measured | ... | ... | ... | ... | ... | ... |

This section is essential because aggregate mAP can hide concept-specific failure.

---

# 30. Object-level disagreement analysis

For every ground-truth box, calculate each model's best **same-phrase** prediction.

Produce:

```text
image_id
gt_object_id
prompt
gt_box

grounding_dino_detected
grounding_dino_score
grounding_dino_iou

owlv2_detected
owlv2_score
owlv2_iou
```

A detection counts as successful when:

`same phrase AND IoU >= 0.50`

---

# 31. Disagreement categories

Automatically categorize ground-truth objects.

### Both detected

Both models locate the correct phrase at IoU ≥ 0.50.

### Grounding-DINO-only

Grounding DINO succeeds and OWLv2 does not.

### OWLv2-only

OWLv2 succeeds and Grounding DINO does not.

### Both miss

Neither model succeeds.

### Localization disagreement

Both have the correct phrase, but their best IoUs differ materially, for example:

`|IoU_GDINO - IoU_OWLV2| >= 0.20`

These thresholds are analysis aids, not model-quality thresholds.

---

# 32. Phrase-confusion analysis

For ground-truth objects where the best-overlapping high-score prediction has the wrong prompt, record:

```text
expected phrase
predicted phrase
score
IoU
```

This separates:

- localization failure

from:

- language/grounding failure.
For example:

```text
correct box region
+
wrong phrase
```

is qualitatively different from:

```text
no candidate overlaps the object
```

---

# 33. Duplicate-detection analysis

Neither canonical pipeline applies NMS.

The notebook should therefore quantify redundant detections.

For each ground-truth object:

1. find same-phrase predictions with IoU ≥ 0.50;
2. the highest-scoring prediction counts as the primary match;
3. additional qualifying boxes count as redundant detections.

Report:

```text
model
primary_matches
redundant_same-object_boxes
mean_redundant_boxes_per_matched_object
```

This is a diagnostic statistic, not a standard benchmark metric.

The notebook MUST explain that AP already penalizes high-ranking false-positive duplicates.

---

# 34. Prompt sensitivity experiment

This is a defining section of the workshop.

Do not change the canonical evaluation vocabulary.

Instead, select a small deterministic set of:

`4 held-out images`

before looking at model predictions.

Run three prompt formulations.

## A. Canonical labels

```text
red blood cell
white blood cell
platelet
```

## B. Natural article phrasing

```text
a red blood cell
a white blood cell
a platelet
```

## C. Domain abbreviations

```text
RBC
WBC
platelet
```

After each model's own normalization, compare:

- number of detections;
- top score per concept;
- best same-class IoU;
- whether each ground-truth concept is recovered.

The experiment demonstrates:

> An open-vocabulary detector's effective class definition includes the wording supplied by the operator.

---

# 35. Prompt experiment selection rule

The four images MUST be selected without examining model predictions.

Recommended deterministic rule:

- take the first four test records by stable sorted `image_id`.

Alternative selection based solely on ground-truth composition is permitted, but the rule must be predetermined and documented.

Do not cherry-pick images because one model behaves interestingly.

---

# 36. Prompt-order experiment

A small second experiment SHOULD demonstrate the architectural difference in prompt representation.

Canonical:

```text
red blood cell
white blood cell
platelet
```

Reordered:

```text
platelet
red blood cell
white blood cell
```

Run on one or more predetermined images.

Compare whether normalized detections remain identical or change.

Interpretation:

- OWLv2 text queries are encoded independently, so ordering should ordinarily have no semantic role beyond query indexing;
- Grounding DINO receives one concatenated prompt sequence, so ordering changes token position/context and may affect numerical grounding behavior.

The notebook MUST report the observed result rather than assert that an effect must occur.

---

# 37. Optional threshold sweep

Default:

`RUN_THRESHOLD_SWEEP = False`

Suggested values:

```text
0.03
0.05
0.10
0.20
```

Run on a small predetermined image subset.

For each model/floor, record:

- total candidate boxes;
- correctly matched ground-truth objects;
- redundant detections;
- unmatched predictions.

The notebook MUST state:

> Equal numeric score thresholds are not equivalent confidence operating points across the two models.

The experiment is about each model's sensitivity to filtering, not calibration between models.

---

# 38. Qualitative comparison panels

Export side-by-side examples:

```text
Ground Truth | Grounding DINO | OWLv2
```

Use native readable visualization thresholds:

- Grounding DINO: box 0.40 / text 0.30
- OWLv2: 0.10

Recommended examples should be chosen by deterministic ground-truth criteria or by predefined indices.

Do not choose only model successes.

Each box label should display:

```text
phrase
score
```

---

# 39. Workshop exercise

Before revealing the model outputs for one selected BCCD image, ask the learner:

> Both detectors receive exactly the same three concepts.
>
> Predict:
>
> 1. Which class will be easiest to localize?
> 2. Which class may be hardest because objects are small or sparse?
> 3. Will the two models return the same number of boxes?
> 4. Could two boxes overlap the same cell even though neither detector uses NMS?
> 5. Would replacing `white blood cell` with `WBC` necessarily preserve the result?

Then reveal:

- model predictions;
- phrase-level scores;
- object-level IoUs;
- prompt-variant outputs.

The exercise must not interrupt `Run all`.

---

# 40. Confidence versus localization

For matched predictions, collect:

```text
model
image_id
prompt
score
iou
```

Provide a compact scatterplot or table.

Interpretation MUST state:

- high score does not imply tight localization;
- score distributions differ between models;
- scores are uncalibrated;
- scores must not be compared as though `0.7` means the same confidence for both models.

---

# 41. Score semantics

## Grounding DINO

Evaluation score:

- token-level sigmoid grounding scores;
- reduced within each phrase;
- best phrase selected for each query.

Native `detect()` additionally separates:

- box threshold; and
- text threshold.

---

## OWLv2

Score:

- sigmoid of the best image–text logit for one patch candidate.

---

Both are ranking signals.

Neither is a calibrated probability that:

> "this object truly belongs to this phrase."

No universal acceptance threshold is shipped.

---

# 42. BYOD — labelled comparison dataset

The optional comparison dataset should reuse a simple detection contract.

Recommended structure:

```text
dataset/
  images/
    image_001.jpg
    image_002.jpg
    ...

  boxes.csv
```

`boxes.csv`:

```text
image_id,filename,label,x0,y0,x1,y1
```

The unique `label` values become the prompt vocabulary.

---

# 43. BYOD limits

For the workshop path:

```text
8..200 images
1..16 distinct phrases
<=300 boxes per image
image sides 16..4096 px
phrase length <=48 characters
```

Additionally:

- each OWLv2 phrase must fit the 16-token query limit;
- the Grounding DINO concatenated vocabulary must fit the 256-token limit.

All boxes must:

- be finite;
- have positive area;
- lie within image bounds.

---

# 44. BYOD split semantics

Because this notebook performs no adaptation, a labelled BYOD dataset does not require a train/validation/test split.

All supplied labelled images may be treated as:

`evaluation set`

provided the notebook clearly labels the result as evaluation of the user's supplied sample.

The notebook must not call it an independent test set unless the user supplied it as such.

---

# 45. Optional unlabelled BYOD

A second lightweight branch MAY accept:

```text
one image
+
comma-separated prompts
```

and run both detectors without metrics.

When no ground truth is supplied:

```text
evaluation verdict = not-measurable
```

This is suitable for interactive exploration but must remain distinct from the labelled comparison path.

---

# 46. BYOD privacy warning

Before BYOD:

> Images and text prompts can contain personal, confidential, restricted, or sensitive information. Do not upload data to a hosted notebook environment unless you are authorized to process it there.

The notebook should state that the standalone inference path processes user images locally within the selected notebook runtime and does not transmit them to DIMER services.

Model-weight downloads do not require sending user images to Hugging Face.

---

# 47. Required machine-readable outputs

Write under:

```text
outputs/open_vocabulary_detection/
```

---

## `ground_truth.csv`

```text
image_id
object_id
prompt
x0
y0
x1
y1
```

---

## `detections.csv`

```text
model
image_id
prompt
score
x0
y0
x1
y1
```

---

## `model_metrics.csv`

```text
model
map50
map75
map50_95
recall50
n_predictions
mean_latency_s
median_latency_s
load_seconds
weight_bytes
parameter_count
peak_gpu_memory_bytes
```

`map50_95` may be blank/omitted if the supplemental metric is not implemented.

---

## `phrase_metrics.csv`

```text
model
prompt
n_ground_truth
ap50
ap75
recall50
n_predictions
```

---

## `object_analysis.csv`

```text
image_id
object_id
prompt
gt_box

grounding_dino_detected
grounding_dino_score
grounding_dino_iou

owlv2_detected
owlv2_score
owlv2_iou

comparison_category
```

---

## `prompt_sensitivity.csv`

```text
model
image_id
prompt_set
prompt
n_detections
best_score
best_same_class_iou
```

---

## `threshold_sweep.csv`

Only when enabled:

```text
model
score_floor
image_count
detections
matched_objects
redundant_detections
unmatched_detections
```

---

## `metrics.json`

Contains complete nested metric structures.

---

## `provenance.json`

Must contain:

```text
notebook_spec
profile
pedagogical_mode

Grounding DINO:
  model ID
  immutable revision
  model digest
  parameter count
  preprocessing description

OWLv2:
  model ID
  immutable revision
  model digest
  runtime-measured parameter count
  preprocessing description

BCCD:
  repository
  commit
  license
  image count
  box count
  split seed
  split sizes
  sample digest

canonical prompts
evaluation score floor
prompt experiment definitions

runtime versions
device
dtype
timing measurements
```

---

# 48. Annotated outputs

Recommended:

```text
examples/
  <image_id>_ground_truth.png
  <image_id>_grounding_dino.png
  <image_id>_owlv2.png
  <image_id>_comparison.png
```

Also create prompt-variation panels for the prompt-sensitivity examples.

---

# 49. Resource comparison

Produce:

| Attribute | Grounding DINO Tiny | OWLv2 Base/16 Ensemble |
|---|---|---|
| Visual backbone | Swin-T | CLIP ViT-B/16 |
| Text encoder | BERT | CLIP text tower |
| Candidate boxes | 900 queries | 3,600 patches |
| Text representation | concatenated prompt sequence | independent queries |
| Image preprocessing | resize preserving aspect | square pad + 960 resize |
| NMS | none | none |
| Max phrases | 16 | 16 |
| Text ceiling | 256 combined tokens | 16 tokens / phrase |
| Weight size | ~689 MB | ~620 MB |
| Parameters | 172.25M | measured at runtime |
| mAP50 | measured | measured |
| mAP75 | measured | measured |
| Recall50 | measured | measured |
| Mean latency | measured | measured |
| Peak GPU memory | measured | measured |

---

# 50. Terminal summary

Final output should resemble:

```text
DIMER Open-Vocabulary Detection Workshop
-----------------------------------------

Dataset:
  BCCD test images: 94
  Prompt vocabulary: 3
  Ground-truth objects: ...

Grounding DINO Tiny
  mAP50: ...
  mAP75: ...
  recall50: ...
  predictions: ...
  mean inference time: ...

OWLv2 Base/16 Ensemble
  mAP50: ...
  mAP75: ...
  recall50: ...
  predictions: ...
  mean inference time: ...

Prompt sensitivity examples: 4
Output directory:
  outputs/open_vocabulary_detection/
```

Do not automatically declare a "winner."

---

# 51. Required interpretation

The conclusion MUST explain the following.

## Open vocabulary does not mean unlimited language understanding

Both models are conditioned by text, but they do not understand arbitrary language in the same sense as a general-purpose language model.

Short noun phrases are the intended interface.

Negation, counting instructions, relationships, and complex reasoning are outside this detection contract.

---

## The prompt is part of the model configuration

Changing:

`white blood cell`

to:

`WBC`

is not merely cosmetic.

It changes the text representation the detector receives.

Open-vocabulary performance is therefore jointly determined by:

```text
image
+
model
+
prompt formulation
```

---

## Grounding DINO and OWLv2 ground text differently

Grounding DINO performs cross-modal fusion between visual features and a shared BERT phrase sequence.

OWLv2 uses CLIP-like image/text alignment between patch features and independent text queries.

Similar external APIs do not imply identical internal semantics.

---

## Domain shift matters

BCCD microscope imagery differs substantially from common web photographs.

Weak performance may reflect:

- visual-domain mismatch;
- phrase mismatch;
- small-object difficulty;
- class imbalance;
- localization limitations; or
- some combination of them.

The notebook cannot isolate all of these causes.

---

## No NMS means duplicates are visible

Both workshop paths deliberately preserve their canonical no-NMS behavior.

A single physical object may therefore yield multiple overlapping candidates.

This matters especially for:

- counting;
- cropping;
- pseudo-label generation.

---

## Scores are not cross-model probabilities

Grounding DINO and OWLv2 scores arise from different mechanisms.

A Grounding DINO score of `0.5` and an OWLv2 score of `0.5` must not be interpreted as equal certainty.

---

## Prompt experiments are exploratory

The canonical prompt vocabulary is fixed before evaluation.

Prompt-variant examples illustrate sensitivity; they are not used to tune prompts against the held-out test set.

---

# 52. Explicit non-goals

This notebook MUST NOT include:

- Grounding DINO fine-tuning;
- OWLv2 fine-tuning;
- adapter export;
- model selection using BCCD test results;
- automatic prompt optimization;
- LLM-generated prompt search;
- NMS added purely to improve metrics;
- segmentation;
- SAM integration;
- counting-specific logic;
- tracking;
- video inference;
- closed-set RT-DETR/YOLOX comparison;
- production DIMER service calls.

---

# 53. Notebook cell plan

| # | Type | Section | Default |
|---:|---|---|---|
| 0 | Markdown | Title, profile, learning objectives | Yes |
| 1 | Markdown | What open-vocabulary detection means | Yes |
| 2 | Markdown | Grounding DINO vs OWLv2 architecture | Yes |
| 3 | Code | Form parameters | Yes |
| 4 | Markdown | Runtime contract | Yes |
| 5 | Code | Install/check pinned runtime | Yes |
| 6 | Markdown | Immutable model provenance | Yes |
| 7 | Code | Embedded manifests + snapshot helpers | Yes |
| 8 | Markdown | BCCD provenance and domain-shift warning | Yes |
| 9 | Code | Fetch/digest-check BCCD | Yes |
| 10 | Markdown | Split and dataset validation | Yes |
| 11 | Code | Rebuild 220/50/94 split + verify digest | Yes |
| 12 | Markdown | Prompt vocabulary contract | Yes |
| 13 | Code | Validate canonical phrases | Yes |
| 14 | Markdown | Common evaluator | Yes |
| 15 | Code | IoU/AP50/AP75/recall helpers | Yes |
| 16 | Markdown | Grid-prior baseline | Yes |
| 17 | Code | Compute model-free baseline | Yes |
| 18 | Markdown | Grounding DINO inference semantics | Yes |
| 19 | Code | Stage + verify + load Grounding DINO | Yes |
| 20 | Code | Warm-up + 94-image frozen evaluation | Yes |
| 21 | Code | Unload Grounding DINO | Yes |
| 22 | Markdown | OWLv2 inference semantics | Yes |
| 23 | Code | Stage + verify + load OWLv2 | Yes |
| 24 | Code | Warm-up + 94-image frozen evaluation | Yes |
| 25 | Markdown | Common quantitative comparison | Yes |
| 26 | Code | Model + per-phrase metric tables | Yes |
| 27 | Markdown | Object-level disagreement analysis | Yes |
| 28 | Code | Build object analysis | Yes |
| 29 | Markdown | Workshop prediction exercise | Yes |
| 30 | Code | Selected-image qualitative results | Yes |
| 31 | Markdown | Prompt sensitivity | Yes |
| 32 | Code | Canonical/article/abbreviation experiment | Yes |
| 33 | Markdown | Prompt-order experiment | Yes |
| 34 | Code | Reordered-prompt comparison | Yes |
| 35 | Markdown | Threshold and duplicate behavior | Yes |
| 36 | Code | Optional threshold sweep | Default no-op |
| 37 | Markdown | Confidence versus localization | Yes |
| 38 | Code | Score/IoU analysis | Yes |
| 39 | Markdown | Resource comparison | Yes |
| 40 | Code | Timing/size/parameter table | Yes |
| 41 | Markdown | Visual panels | Yes |
| 42 | Code | Export selected panels | Yes |
| 43 | Markdown | Machine-readable exports | Yes |
| 44 | Code | CSV/JSON/provenance exports | Yes |
| 45 | Markdown | BYOD | Yes |
| 46 | Code | Optional BYOD branch | Default no-op |
| 47 | Markdown | Interpretation and limitations | Yes |
| 48 | Code | Terminal summary + output assertions | Yes |

---

# 54. Required assertions

The canonical path should validate at least:

```text
both model manifests identify the expected immutable models
all manifest-listed files exist after staging
all model file sizes match
all SHA-256 digests match

364 BCCD images verified
4,886 source boxes represented
split sizes = 220 / 50 / 94
dataset digest matches pinned value
split images are pixel-disjoint

prompt count = 3
prompts distinct
phrases <= 48 chars
phrases fit OWLv2 token limit
combined Grounding DINO text fits token limit

all GT boxes finite and valid
all prediction boxes finite and valid
all prediction labels belong to vocabulary
all scores finite and in [0,1]

Grounding DINO output count <= 900 per image
OWLv2 output count <= 3600 per image

all required metrics exist
all required exports exist
```

The notebook MUST NOT assert:

```text
one model must outperform the other
a specific AP value must be reproduced
canonical prompts must outperform abbreviations
prompt order must change Grounding DINO
prompt order must not change OWLv2
```

Those are empirical observations, not release invariants.

---

# 55. Release verification

Preferred qualification:

**Kaggle Tesla T4, Python 3.12**

The committed notebook must execute from a fresh environment with:

- no repository checkout;
- no model cache;
- no BCCD image cache;
- no secrets;
- no DIMER infrastructure.

Record:

```text
notebook commit
notebook blob SHA
date
runtime

Grounding DINO:
  ID
  revision
  digest
  staged bytes
  parameter count
  load time
  peak GPU memory
  inference timing

OWLv2:
  ID
  revision
  digest
  staged bytes
  parameter count
  load time
  peak GPU memory
  inference timing

BCCD:
  commit
  image count
  box count
  split sizes
  dataset digest

canonical vocabulary

grid baseline metrics
Grounding DINO metrics
OWLv2 metrics

prompt-experiment observations
cell success count
wall time
output files
```

Observed AP values belong in the release record, not the specification.

---

# 56. Recommended repository placement

This is another cross-model workshop, so keep one canonical copy:

```text
ml-worker/
  integrations/
    dimer/
      workshops/
        open-vocabulary-detection/
          DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb
          README.md
          docs/
            release-verification.md
          tools/
            build_notebook.py
            validate_notebook.py
```

The Grounding DINO and OWLv2 pipeline repositories can link to it.

---

# 57. Relationship to the other detection workshop

The two workshops should form a pair.

## Closed-set detection

```text
RT-DETR
YOLOX-S
YOLOX-X
```

Question:

> Given a detector with a fixed COCO vocabulary, how do architectures differ in accuracy, localization, latency, and resource use?

---

## Open-vocabulary detection

```text
Grounding DINO
OWLv2
```

Question:

> Given concepts supplied in natural language at inference time, how do visual grounding architecture, prompt formulation, score filtering, and domain shift affect what is detected?

Together they teach two fundamentally different ways of defining an object detector's label space.

---

# 58. Workshop learning arc

The intended sequence is:

**Closed-set assumption removed**  
↓  
*The learner supplies the vocabulary in language.*

**Same external task**  
↓  
*Both models return phrase-labelled boxes.*

**Different text-conditioning mechanisms**  
↓  
*Grounding DINO uses multimodal grounding over a phrase sequence; OWLv2 aligns image patches with independent CLIP queries.*

**Common measurable dataset**  
↓  
*Both models receive the same 94 held-out BCCD images and the same canonical three-phrase vocabulary.*

**Common evaluator**  
↓  
*Phrase-level AP and recall can now be compared.*
**Prompt variation**  
↓  
*The same concept can produce different behavior when worded differently.*

**Duplicate and threshold behavior**  
↓  
*Open-vocabulary scores are not calibrated and neither canonical path applies NMS.*

**Domain shift**  
↓  
*Language flexibility does not remove the visual-distribution problem.*

**Engineering lesson**  
↓  
*An open-vocabulary detector is a system comprising the visual model, text representation, prompt policy, filtering policy, and deployment domain—not merely a checkpoint.*