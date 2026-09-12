# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/grounding-dino-detection-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/tutorials/grounding_dino_detection_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IDEA--Research%2Fgrounding--dino--tiny-ffcc4d?style=flat)](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
[![Upstream](https://img.shields.io/badge/Upstream-IDEA--Research%2FGroundingDINO-181717?style=flat&logo=github&logoColor=white)](https://github.com/IDEA-Research/GroundingDINO)
[![arXiv](https://img.shields.io/badge/arXiv-2303.05499-b31b1b.svg)](https://arxiv.org/abs/2303.05499)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `grounding_dino_detection_colab.ipynb` | `TASK-INFERENCE` | zero-shot text-prompted object detection with `IDEA-Research/grounding-dino-tiny`; score-ordered xyxy boxes with grounded phrase and uncalibrated sigmoid score; caller-owned `box_threshold`/`text_threshold` exposed as form parameters; `box_iou` against the drawn boxes as sanity evidence, no mAP | CPU (CUDA used automatically when available) | single image file (16–4096 px per side) plus comma-separated prompts, gated off by default; no reference boxes, so no IoU | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `GroundingDINOPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks, at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (`local_files_only=True`, `trust_remote_code=False`; the notebook never calls `huggingface_hub`).
- The default sample is a synthetic 320×240 scene drawn in code (dark rectangle at [40, 60, 140, 180], red disc at [200, 80, 280, 160]) with prompts `rectangle` and `red circle` — the same scene the card-pass smoke used; `box_iou` of each detection against the drawn box of the same phrase is sanity evidence for the input contract, prompt formatting and coordinate mapping, not a detection metric (mAP needs a labelled box set and is not computed).
- Score semantics: every `score` is an uncalibrated sigmoid grounding score; the thresholds are caller-owned request parameters (Kurt's decision), exposed as `# @param` values with the package defaults 0.4 / 0.3 (upstream usage-example values, not a calibration) and passed explicitly on every `detect` call.
- Ceilings `MIN_IMAGE_SIDE` (16), `MAX_IMAGE_SIDE` (4096), `MAX_PROMPTS` (16), `MAX_PROMPT_CHARS` (48), `MAX_TEXT_TOKENS` (256) are printed before the model runs and the prompts are canonicalised through the package's `format_prompts`.
- CPU is documented as adequate for one small image (card-measured 6.2 s load, 4.4 s per `detect` on 320×240).
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
