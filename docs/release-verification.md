# Release verification

`tutorials/grounding_dino_detection_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `GroundingDINOPipeline.from_pretrained(weights_dir=...)`, `detect(image, prompts, box_threshold=..., text_threshold=...)`, `format_prompts`, `box_iou`), the ceiling print
  (`MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, `MAX_PROMPTS`, `MAX_PROMPT_CHARS`, `MAX_TEXT_TOKENS`), the exports, the learner-facing statements (caller-owned thresholds, uncalibrated sigmoid, score ordering, no mAP, IoU as sanity check, capability exclusions) and the
  gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, direct
  `from transformers import` / `AutoModelForZeroShotObjectDetection` / `AutoProcessor` / `from huggingface_hub import` use that bypasses the pipeline,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheels plus `transformers`, `safetensors`, `numpy` and `pillow`, runs `ruff` and the
offline unit suite (`tests/test_pipeline.py`, injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab`, sets `DIMER_TUTORIAL_REF`, and chdirs to a scratch directory so the bootstrap clones the candidate |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `box_threshold = 0.4`, `text_threshold = 0.3`);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins;
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - synthetic 320×240 scene drawn in code with its RGB SHA-256 printed, prompt text `rectangle. red circle.` from `format_prompts`, the thresholds echoed, and the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_PROMPTS` 16, `MAX_PROMPT_CHARS` 48, `MAX_TEXT_TOKENS` 256) surfaced;
   - pinned `IDEA-Research/grounding-dino-tiny` acquisition at the immutable revision through the package:
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports the git-ignored weight file(s) on a
     clean runtime, `verify_snapshot` returns its dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loads from the verified directory;
   - `detect` returning score-ordered detections; record the labels, scores and the `box_iou` sanity values against the drawn boxes (the card-pass CPU smoke returned `red circle` 0.932 and `rectangle` 0.698 with boxes within ~2 px; a materially different result is a finding to record, not a failure by itself, because no metric is asserted);
   - `outputs/grounding_dino_detection_result.json` and `outputs/grounding_dino_detection_detections.csv` and `outputs/grounding_dino_detection_annotated.png` written with repository SHA, model revision,
     runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/grounding_dino_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/grounding_dino_detection_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the notebook has been recorded yet; the run is **pending** and queued
to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real
`hf_hub_download` fetch into the snapshot directory has not been executed), and the card pass executed `detect` only on CPU in the Windows venv (returns/L2: CUDA path not executed), so the clean run will be the first real execution of the staging path; the CPU inference path against real weights was executed once locally (card Runtime: 4.4 s on 320×240).
