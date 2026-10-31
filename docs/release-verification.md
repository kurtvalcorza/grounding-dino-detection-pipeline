# Release verification

`tutorials/grounding_dino_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 9-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the isolated-environment
  install (pinned `uv` wheel checked by size and SHA-256, `--managed-python` CPython 3.12.12, the carried hash lock
  installed with `--require-hashes --only-binary :all:`) and the router to the isolated worker, whose `google.colab`
  stubs carry a module spec; the guided-layer markers, the Infrastructure titles and collapsed setup cells, no bare
  `assert` in learner cells and no stale learner-facing text; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the BCCD_Dataset
  commit `d272fb14cdff6e473fafeeeba32aba5f560e9e43` is the one other 40-hex revision the documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `GroundingDINOPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split against the vocabulary, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the over-long-prompt refusal probe, `detect` with the sanity checks and the
  per-image `evaluation_report` on the synthetic scene, `grid_prior_baseline`, `predict_boxes` with the generic
  prompt + `majority_relabel_baseline`, `pipe.evaluate` on the frozen model with the recorded baseline outcomes, `pipe.adapt`
  with its explicit hyperparameters, `pipe.evaluate` on the validation and test splits after adaptation with the
  recorded outcomes, reading and run history, `detect` + `evaluation_report` on the scene after adaptation, the example panels against a
  freshly loaded frozen base, `pipe.save_artifact`, `GroundingDINOPipeline.from_artifact` and the box-parity
  check, and the result fields `weight_file` / `weight_format` / `weight_sha256` and the `corpus` block), the
  eight expected `outputs/` paths, the learner-facing statements (Apache-2.0 weights, the sigmoid grounding score,
  adaptation with a labelled vocabulary, the frozen model at 0.109, the two non-adapted baselines, AP50 and AP75,
  the grid prior and the generic prompt, the upstream criterion, no dispersion estimate, the vocabulary and leakage
  guidance, the excluded tasks, the snapshot note) and the gated-off BYOD default; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `AutoModelForZeroShotObjectDetection` / `AutoProcessor` /
  `post_process_grounded_object_detection(` / `from torchvision import` / `torch.inference_mode(` / `from
  huggingface_hub import` / `urllib.request` / `safetensors` imports / `torch.optim` / `.backward(` /
  `requires_grad` / `pipe._model` / `extractall(` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `huggingface-hub`, `numpy` and
`pillow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`, `test_role_helpers.py`,
`test_notebook_parity.py`; injected runner and image fetcher, no weights — `tests/test_model_backed.py` is skipped
without the snapshot). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (CUDA; a CPU runtime is not practical for the default path) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA runtime (Colab GPU, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/grounding-dino-tiny/` or the photograph cache `weights/bccd/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and fetches the pinned photographs from GitHub's raw-content
   host, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2` and `scipy==1.18.1`, in the isolated CPython 3.12.12 environment; the run must complete in **one pass with no
   manual restart and no error output** (NOTEBOOK_SPEC 2.2 §5) — a run that needed a restart is recorded as such and is
   not promotion evidence;
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock (48 packages) with no GitHub access, and every later
     cell routed to it;
   - the three carried module cells execute (defining `GroundingDINOPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `box_iou`, `format_prompts`,
     `detection_metrics`, `grid_prior_baseline`, `majority_relabel_baseline`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_prompts`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `PROMPTS` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 9 manifest entries fetched from `IDEA-Research/grounding-dino-tiny` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (9 files, the 689 MB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory on `cuda:0`;
   - Section 4: `fetch_corpus` fetching the 364 pinned photographs with every byte count and SHA-256 matching; 4,886
     boxes over the three phrases; the seeded split into 220 / 50 / 94 with `check_split_disjoint` reporting no shared
     image and the three dataset digests printed; `outputs/…_train.csv` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_PROMPTS` 16, `MAX_PROMPT_CHARS` 48,
     `MAX_TEXT_TOKENS` 256, both thresholds, `SCORE_FLOOR` 0.05, `MAX_BOXES` 300, `MIN_RECORDS` 8, `MAX_RECORDS`
     5000) surfaced; the synthetic scene drawn; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict
     `accepted`, one recorded rejection finding from the over-long-prompt probe); `detect` on the 320×240 scene with
     every sanity check `True`, `outputs/…_scene_frozen.png` written and the per-image `evaluation_report` verdict
     `sample-sanity` (the inference-only card recorded `red circle` at 0.932 and `rectangle` at 0.698 — observations,
     not assertions);
   - Section 6: the grid prior (≈ 0.01 mAP50), the generic-prompt baseline (≈ 0.08) and the frozen model's test score
     (≈ 0.11 mAP50 / 0.05 mAP75 in the RTX 5070 Ti build record; `platelet` at 0.00) with the per-phrase breakdown,
     and the recorded `frozen_beats_grid_prior` / `frozen_beats_generic_prompt` outcomes;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 11,187,460 trainable of 172,249,090 parameters, and
     a six-epoch history with the validation mAP50 rising (build record: 0.162 → 0.352 / 0.423 / 0.516 / 0.573 /
     0.614 / 0.636, `best_epoch` 6; the training loss stays near 11,000 by construction of the upstream criterion);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on the three measures,
     the per-phrase breakdown and `outputs/…_evaluation_report.json` written (the cell records whether the adapted test mAP50
     exceeds the frozen one — 0.570 versus 0.109 in the build record, mAP75 0.052 → 0.430, recall 0.29 → 0.96,
     `red blood cell` 0.04 → 0.81, `white blood cell` 0.28 → 0.61, `platelet` 0.00 → 0.29; the adapted model also
     clears both baselines, reported, not asserted);
   - Section 9: the scene re-detected by the adapted model with the `sample-sanity` report,
     `outputs/…_scene_adapted.png` and four example panels under `outputs/…_examples/` written; `pipe.save_artifact`
     writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (228 tensors, 44,776,600 bytes) and
     `GroundingDINOPipeline.from_artifact` reloading it with 8/8 identical scored box lists on eight test images (the
     cell stops with an explanation otherwise); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block with the vocabulary, the
     inference-contract reports, the comparison, the artifact digest, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the photograph cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `grounding_dino_detection_colab.ipynb` (`E2E`, `GUIDED`, spec 2.5) | `2fb41e2` / `81eb95a4` | 2026-10-25 | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells, 1277.0 s wall; default settings only; test mAP50 frozen 0.109 → adapted 0.61, epoch 6 kept; see Recorded executions; evidence in `docs/execution-evidence/2026-10-25/` |
| `grounding_dino_detection_colab.ipynb` (`E2E`) | `832d5d4` / `dc1218be` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **Completed in two passes, not promotion evidence** — pass 1 stopped at the in-kernel install guard (`numpy: loaded=2.0.2, installed=2.5.3`; `cuda-bindings` likewise); after a manual restart pass 2 ran 11/11 code cells ok. NOTEBOOK_SPEC 2.2 §5 requires Run all with no manual restart (review GDD-M1); 384 files, 698 MB staged from the Hub into a clean cache; comparison {map50: {grid_prior: 0.009, generic_prompt: 0.075, frozen: 0.109, adapted: 0.632}, map75: {grid_prior: 0, generic_prompt: 0.027, frozen: 0.052, adapted: 0.45}, recall50: {grid_prior: 0.199, generic_prompt: 0.514, frozen: 0.293, adapted: 0.951}, delta_vs_frozen: {map50: 0.523, map75: 0.397, recall50: 0.659}, delta_vs_generic_prompt: {map50: 0.557, map75: 0.422, recall50: 0.437}, by_phrase: {red blood cell: {n: 1104, frozen_ap50: 0.041, adapted_ap50: 0.809, frozen_ap75: 0.014, adapted_ap75: 0.597}, white blood cell: {n: 93, frozen_ap50: 0.285, adapted_ap50: 0.744, frozen_ap75: 0.143, adapted_ap75: 0.653}, platelet: {n: 98, frozen_ap50: 0, adapted_ap50: 0.343, frozen_ap75: 0, adapted_ap75: 0.099}}}; reload parity {identical_images: 8, of: 8, boxes_per_image: [136, 136, 136, 146, 165, 99…]}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-grounding-dino-detection/v2/evidence/` in the workspace |
| `grounding_dino_detection_colab.ipynb` (`TASK-INFERENCE`, superseded) | `9bf9e00` / `b1e416c68b76` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v1) | PASS — 8/8 code cells, 271.8 s, 20 files, 690 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/grounding_dino_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/grounding_dino_detection_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-25 | `2fb41e2` / `81eb95a4` (PR #14 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `2fb41e2a0a5195f0925fadbb90380b9d963b5823`, git blob SHA-1 `81eb95a419f50af4fa2854cb748aef88cd42581e` checked, input unchanged; captured executed-notebook SHA-256 `cd3f742a555fc1ce46e6d9c447544273b379b74de71a2dc7433a13af69159816` equals `run_summary.json` `executed_sha256` | 1277.0 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); synthetic scene frozen `red circle` 0.931 / `rectangle` 0.695, `box_iou` 0.944 / 0.964; test (94 images, 1,295 boxes) grid prior mAP50 0.009, generic prompt `cell` 0.075 (recall 0.514), frozen 0.109 (mAP75 0.052, recall 0.293; `red blood cell` 0.041, `white blood cell` 0.285, `platelet` 0.0); validation mAP50 0.161 → 0.324 → 0.425 → 0.519 → 0.574 → 0.623 → 0.672, training loss 11,411.0 → 11,312.8, epoch 6 kept, adaptation 861.7 s; adapted mAP50 0.61, mAP75 0.453, recall 0.952 (`delta_vs_frozen` mAP50 0.502); per phrase AP50 0.812 / 0.731 / 0.289; adapted scene `red circle` 0.843 / `rectangle` 0.524, `box_iou` 0.893 / 0.922; reload parity 8 of 8 identical images. Not exercised: BYOD (REL12), the learning-rate activity and a browser `Run all` (REL2). Evidence in `docs/execution-evidence/2026-10-25/` |
| 2026-10-25 | `03f4a1a` / `a04b58a2` (PR #14 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `03f4a1a3fbbba0f85924db10aae772815d6da7ca`, git blob SHA-1 `a04b58a2aa1c965f0fac80fa551cced2f68078e9` checked, input unchanged; captured executed-notebook SHA-256 `341131950436006cd589b7359bd7850a622216dcc1f59c863db81ad3ffd1abe3` equals `run_summary.json` `executed_sha256` | 1333.1 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); test mAP50 grid prior 0.009, generic prompt `cell` 0.075, frozen 0.109 → adapted 0.614; mAP75 0.000 / 0.027 / 0.052 → 0.455; recall50 0.199 / 0.514 / 0.293 → 0.964; per phrase AP50 `red blood cell` 0.041 → 0.816, `white blood cell` 0.285 → 0.695, `platelet` 0.000 → 0.332; validation mAP50 0.161 → 0.338 / 0.416 / 0.525 / 0.569 / 0.644 / 0.685, epoch 6 kept; six epochs 952.8 s; adapter 228 tensors, 44,776,600 bytes, saved; reload parity 8/8. Evidence in `docs/execution-evidence/2026-10-25/` |
| 2026-10-26 | `361acff` / `4cd2580a` (PR #14 head; final prose pass, second follow-up, regenerated from `18a2a28`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `361acff007adb35ed2fd2c767c6b63b492fcda30`, git blob SHA-1 `4cd2580a90f9a48aa55988b795a1db352f6dfe3a` checked, input unchanged; captured executed-notebook SHA-256 `05b09fc69eecb8b881ceece56c80c3557d76a19205cb35173386fc87268c3229` equals `run_summary.json` `executed_sha256` | 1226.1 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); isolated environment 52 s; test (94 images, 1,295 boxes): grid prior 0.009, generic prompt `cell` 0.075, frozen 0.109 mAP50 (mAP75 0.052, recall 0.293) → adapted 0.636 mAP50, mAP75 0.47, recall 0.954 (Δ mAP50 0.527 vs frozen); per phrase AP50 red blood cell 0.041 → 0.811, white blood cell 0.285 → 0.748, platelet 0.0 → 0.348; validation mAP50 0.161 → 0.329 / 0.42 / 0.513 / 0.589 / 0.634 / 0.701, epoch 6 kept, training 881.7 s; adapter 228 tensors, 44,776,600 bytes, reload parity 8/8. Drift against the prose's recorded values (recorded, not fixed): session 1226.1 s; test mAP50 0.636, mAP75 0.47, white blood cell AP50 0.748; per-epoch validation mAP50; adapted scene values (see Current status). Evidence in `docs/execution-evidence/2026-10-26/` |
| 2026-10-25 | `c831c71` / `21e405bf` (PR #14 head; historical wording, code `2e7806a`, notebook `f577416`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `c831c7119bf1e3ea4e9c38571c961edf28689c29`, git blob SHA-1 `21e405bf382ebeaff2d6c230847e8fc61db71d83` checked, input unchanged; captured executed-notebook SHA-256 `2abc3a2cd1e886d1906f8fac1157f69ffec5b1b787a5ea521fd46ffa0aa9ae33` equals `run_summary.json` `executed_sha256` | 1304.8 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); isolated environment 51 s (48 locked packages); 364 photographs, 4,886 boxes, split 220 / 50 / 94; frozen scene `red circle` 0.931 (`box_iou` 0.944), `rectangle` 0.695 (0.964); test mAP50 grid prior 0.009, generic `cell` 0.075, frozen 0.109 (mAP75 0.052, recall 0.293; `red blood cell` 0.041, `white blood cell` 0.285, `platelet` 0.0); six epochs in 901.1 s, validation mAP50 0.161 → 0.345 / 0.415 / 0.513 / 0.587 / 0.626 / 0.68, training loss 11,411.0 → 11,312.8, epoch 6 kept; adapted mAP50 0.605, mAP75 0.452, recall 0.954; per phrase AP50 `red blood cell` 0.813, `white blood cell` 0.668, `platelet` 0.334; adapted scene `red circle` 0.857 (`box_iou` 0.888), `rectangle` 0.488 (0.922); adapter 228 tensors, 44,776,600 bytes; reload parity 8/8. Drift against the prose's recorded values (recorded, not fixed, per the 2026-10-26 ruling): mAP50 0.605 against the recorded hosted 0.610–0.632; `white blood cell` 0.668; adapted-scene `rectangle` 0.488; a new validation-mAP50 sequence. Evidence in `docs/execution-evidence/2026-10-25/` |
| 2026-10-14 | `b49d22a` / `cb532ab5` (`DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`, GDN2 + CX follow-up) | Colab CLI 0.7.4, fresh Colab Tesla T4 (`colab exec -f`, session released afterwards); kernel Python 3.13.15, isolated CPython 3.12.12 with torch 2.14.0+cu130 / transformers 4.57.6 | Default path (no `# @param` changed); CLI sequential execution, not a browser `Run all` | 402.7 s (session wall time) | **PASSED** — 25/25 code cells in order, 0 errors; 0.05 operating recipe identical to the `f5097fd` run (Grounding DINO 0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,173; OWLv2 0.0537 / 0.0331 / 0.0295 / 0.0680 / 1,506), the floor-sensitivity cell's equality check passed; Grounding DINO mAP50 / recall50 0.1088 / 0.293 → 0.1224 / 0.761 at floor 0.001, OWLv2 0.0537 / 0.068 → 0.0550 / 0.082 |
| 2026-10-04 | `a2bdaaf` / `b539602b` (`E2E`, uv isolated environment) | Colab CLI 0.7.4, fresh Colab Tesla T4 (`colab exec -f`, session released afterwards); kernel Python 3.13.15, isolated CPython 3.12.12 with torch 2.14.0+cu130 / transformers 4.57.6, `cuda:0` | Default sample path (no `# @param` changed); CLI sequential execution in one kernel, not a browser `Run all`; no execution counts (order from `exec.log`) | 1301.6 s (session wall time) | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells in order; grid prior 0.009, generic `cell` 0.075, frozen 0.109 / 0.052 / 0.293 (identical to the 2026-09-20 T4 run); six epochs 879.3 s, validation mAP50 0.161 → 0.341 / 0.422 / 0.525 / 0.568 / 0.637 / 0.695, epoch 6 kept; adapted 0.630 / 0.461 / 0.957; reload parity 8/8 |
| 2026-10-04 | `f5097fd` / `40ff6fdd` (`DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`, uv isolated environment with SciPy) | Colab CLI 0.7.4, fresh Colab Tesla T4 (`colab exec -f`); kernel Python 3.13.15, isolated CPython 3.12.12 with torch 2.14.0+cu130 / transformers 4.57.6 | Default path (no `# @param` changed); CLI sequential execution, not a browser `Run all` | 292.2 s (session wall time) | **PASSED** — 24/24 code cells in order, 0 errors; Grounding DINO 0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,173 and OWLv2 0.0537 / 0.0331 / 0.0295 / 0.0680 / 1,506, identical to the 2026-09-30 T4 run of `563b0f6` |
| 2026-10-04 | `873e998` / `924cef66` (`DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`, uv isolated environment) | Google Colab (browser, maintainer-run), Tesla T4; kernel Python 3.13.15, isolated CPython 3.12.12 with torch 2.14.0+cu130 / transformers 4.57.6 | Default path (no `# @param` changed), `Run all` | not recorded | **FAILED** — 12/24 code cells ran, counts 1–12 in order; the OWLv2 stage (cell `2f66f39b`) exited 1: `Owlv2ImageProcessor requires the scipy library`, which the hash lock does not contain. Grounding DINO printed metrics identical to the 2026-09-30 T4 run (0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,173) |
| 2026-09-20 | `832d5d4` / `dc1218be` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 1503.7 s | **Completed in two passes, not promotion evidence** — pass 1 stopped at the in-kernel install guard (`numpy: loaded=2.0.2, installed=2.5.3`; `cuda-bindings` likewise); after a manual restart pass 2 ran 11/11 code cells ok. NOTEBOOK_SPEC 2.2 §5 requires Run all with no manual restart (review GDD-M1); 384 files, 698 MB staged from the Hub into a clean cache; comparison {map50: {grid_prior: 0.009, generic_prompt: 0.075, frozen: 0.109, adapted: 0.632}, map75: {grid_prior: 0, generic_prompt: 0.027, frozen: 0.052, adapted: 0.45}, recall50: {grid_prior: 0.199, generic_prompt: 0.514, frozen: 0.293, adapted: 0.951}, delta_vs_frozen: {map50: 0.523, map75: 0.397, recall50: 0.659}, delta_vs_generic_prompt: {map50: 0.557, map75: 0.422, recall50: 0.437}, by_phrase: {red blood cell: {n: 1104, frozen_ap50: 0.041, adapted_ap50: 0.809, frozen_ap75: 0.014, adapted_ap75: 0.597}, white blood cell: {n: 93, frozen_ap50: 0.285, adapted_ap50: 0.744, frozen_ap75: 0.143, adapted_ap75: 0.653}, platelet: {n: 98, frozen_ap50: 0, adapted_ap50: 0.343, frozen_ap75: 0, adapted_ap75: 0.099}}}; reload parity {identical_images: 8, of: 8, boxes_per_image: [136, 136, 136, 146, 165, 99…]}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-grounding-dino-detection/v2/evidence/` in the workspace |
| 2026-09-20 | committed template at the candidate revision (blob differs from the committed one in the recorded generating revision only) | Local WSL harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and photographs pre-staged, the CPU unit suite running alongside | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (9 files), 364 photographs re-hashed from the cache, 4,886 boxes split 220 / 50 / 94, four refusal probes raised, the synthetic scene detected (`box_iou` 0.964 / 0.944), grid prior 0.009 and generic `cell` relabelled 0.076, frozen 0.109 / 0.052 / 0.294 in 53.2 s, six epochs 897.8 s (validation mAP50 0.162 → 0.348 / 0.429 / 0.520 / 0.591 / 0.648 / 0.677, epoch 6 kept), adapted 0.608 / 0.450 / 0.952 (`red blood cell` 0.042 → 0.812, `white blood cell` 0.284 → 0.715, `platelet` 0.000 → 0.297), the scene re-detected with both boxes at `box_iou` 0.916 / 0.9, four panels written, adapter 44,776,600 B / 228 tensors, reload parity 8/8, 8 outputs written | 1166.8 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-14 | `9bf9e00` / `b1e416c68b76` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v1) | Default sample path | 271.8 s | PASS — 8/8 ok code cells, 20 files, 690 MB staged; not evidence for the `E2E` blob |

### Hosted Colab T4 runs of blob `3d248f84` (2026-10-29)

These runs executed the PR head `2040785777d77441f8cbd20bee4253b00c41b799`, notebook `tutorials/grounding_dino_detection_colab.ipynb` blob `3d248f84054adf124557c51072e501e7a7fa2ea6`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 12:53:38 UTC start).** Default settings, no overrides; session `suite-grounding-2040785-d705`; wall 1356.5 s (`wall_s`); `code_cells` 13, `executed_cells` 10, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29/`](execution-evidence/2026-10-29/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 9, 'revision': 'a2bb814dd30d776dcf7e30523b00659f4f141c71'}
{'grid_prior_baseline': {'map50': 0.009, 'map75': 0.0, 'recall50': 0.199}, 'n': 94, 'note': "a grid of 107x101 px boxes labelled 'red blood cell'"}
{'generic_prompt_baseline': {'map50': 0.075, 'map75': 0.027, 'recall50': 0.514}, 'generic_prompt': 'cell', 'note': "every predicted box relabelled 'red blood cell'", 'seconds': 46.2}
{'frozen_model_test': {'map50': 0.109, 'map75': 0.052, 'recall50': 0.293}, 'n': 94, 'boxes': 1295, 'predictions': 5173, 'adapted': False, 'verdict': 'measured', 'seconds': 47.7}
{'map50': {'grid_prior': 0.009, 'generic_prompt': 0.075, 'frozen': 0.109, 'adapted': 0.62}}
{'map75': {'grid_prior': 0.0, 'generic_prompt': 0.027, 'frozen': 0.052, 'adapted': 0.447}}
{'recall50': {'grid_prior': 0.199, 'generic_prompt': 0.514, 'frozen': 0.293, 'adapted': 0.947}}
{'delta_vs_frozen': {'map50': 0.511, 'map75': 0.395, 'recall50': 0.655}}
{'frozen_beats_grid_prior': True, 'frozen_beats_generic_prompt': True}
{'reload_parity': {'identical_images': 8, 'of': 8, 'boxes_per_image': [130, 150, 123, 143, 158, 112, 123, 128]}, 'reloaded_best_epoch': 6}
```

**DAT20 refusal record (2026-10-29, blob `3d248f84`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD` `False` → `True` and `BYOD_PATH` `''` → `'/etc/hostname'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `grounding_dino_detection-17`, dropping 11 cells (5 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `8a02aa21a8231dd437eaf27cf347167d36dc6fa483027a84a555c10c413fd0e7` (Git blob `76ef1a9f`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: hostname: ValueError: /etc/hostname is neither a directory nor a zip file; fix the file, then run this cell and every cell after it again (the earlier results of this session and their outputs/ files were removed).
```

This run says nothing about the 11 cut cells, which did not run in it. Session `suite-grounding-2040785-d705`, wall 111.8 s. Retained evidence: [`docs/execution-evidence/2026-10-29/`](execution-evidence/2026-10-29/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `grounding_dino_detection_colab_2040785_colab-cli-t4_default.ipynb` | `3a93a9e9a6a9f5cf87331398c64fbf09c26d580090f8185617d7f7a98205710a` | `3a93a9e9a6a9f5cf87331398c64fbf09c26d580090f8185617d7f7a98205710a` (unchanged, no local path) |
| `grounding_dino_detection_colab_2040785_colab-cli-t4_default_exec.log` | `8bf9ec91d168779c04872f5a7a9e62bb79a5ce3d61c07223771cb85093a889b8` | `884736b83a529384e40ca12adf870273a44a1ffc3cc879ee860a1d6be789640b` (2 local-path replacements) |
| `grounding_dino_detection_colab_2040785_colab-cli-t4_default_run_summary.json` | `bb7bbfa073edc346464dfd1d685e96b13218d893b82fd652c721c19fa91f7aa9` | `bb7bbfa073edc346464dfd1d685e96b13218d893b82fd652c721c19fa91f7aa9` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `grounding_dino_detection_colab_2040785_colab-cli-t4_refusal.ipynb` | `73a6ea5b2b15dde6538b7d510b2f7fabdd477aefa81078fe50a13105c089c045` | `73a6ea5b2b15dde6538b7d510b2f7fabdd477aefa81078fe50a13105c089c045` (unchanged, no local path) |
| `grounding_dino_detection_colab_2040785_colab-cli-t4_refusal_exec.log` | `3c2b747f0746eee915c018e8b5bd0f22776b3efbd009b1c5a0cf736c4f5c2ce7` | `87870e7a23bc245756a95ff97bab930acefa55d0b98204819211bc9f09f5d149` (2 local-path replacements) |
| `grounding_dino_detection_colab_2040785_colab-cli-t4_refusal_run_summary.json` | `4e92b91d0815590e72b305403d99e2d5dad0fd0085bf039ae0cd2c83d02fa776` | `4e92b91d0815590e72b305403d99e2d5dad0fd0085bf039ae0cd2c83d02fa776` (unchanged, no local path) |
| `grounding_dino_detection_colab_2040785_colab-cli-t4_refusal_run_copy.ipynb` | `8a02aa21a8231dd437eaf27cf347167d36dc6fa483027a84a555c10c413fd0e7` | `8a02aa21a8231dd437eaf27cf347167d36dc6fa483027a84a555c10c413fd0e7` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12) and a browser `Run all` of the interactive journeys (REL2).

## Current status

**Candidate (2026-10-26, final prose pass, second follow-up).** The notebook blob of `tutorials/grounding_dino_detection_colab.ipynb` at `b327d91` is `4cd2580a90f9a48aa55988b795a1db352f6dfe3a` (commit `b327d91`), regenerated from `18a2a28` on the coordinator's request: the Section 8 dispersion paragraph's 'the training itself varies by a few hundredths between runs (Section 7)' now reads 'the recorded training runs differed from each other (Section 7, **Run-to-run spread**)'. A grep of the notebook for 'hundredth', 'decimal', 'expect' and 'typically' finds no other generalisation (the remaining 'expect' hits are a Predict question to the learner, one run-labelled sentence and code identifiers). Markdown only. This exact blob (PR #14 head `361acff`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-26, 13/13 code cells in one pass, no restart, 0 errors, 1226.1 s wall; test mAP50 0.109 → 0.636, mAP75 0.052 → 0.47, epoch 6 kept, reload parity 8/8; evidence in `docs/execution-evidence/2026-10-26/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). Blobs `ff3ad503` and `63fb86c1` were never run; the confirming run of `21e405bf` and the earlier records below stay as history.

The current blob `3d248f84` (code commit `d50e742`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); on 2026-10-29 its default path (13 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (`USE_BYOD=True;BYOD_PATH=/etc/hostname`: exactly one `Refused:` line at cell `grounding_dino_detection-17`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`). The remaining gates are REL12 (a hosted BYOD run with an accepted input) and REL2 (a browser run of the interactive journeys); earlier blobs' runs are history (REL14).

**Drift recorded, not fixed (2026-10-26, final prose pass; Kurt's ruling: drift is recorded here, never fixed).** The notebook prose of blob `4cd2580a` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Header: 'the whole session between 21.3 and 25.1 minutes including the downloads (1,301.6 s, 1,277.0 s and 1,333.1 s on Colab T4, 1,503.7 s on Kaggle T4)'; this run took 1226.1 s.
- Sections 7, 8 and Interpretation: test mAP50 '0.570, 0.608, 0.632, 0.630, 0.610 and 0.614' in the six named GPU runs (0.610 to 0.632 across the named hosted runs); this run printed 0.636.
- Section 8 and Interpretation: mAP75 '0.450 to 0.461' across the named hosted runs; this run printed 0.47.
- Section 8: white blood cell AP50 '0.695 to 0.744' across the named hosted runs; this run printed 0.748.
- Section 7: validation mAP50 per epoch in the three named Colab runs (epoch 3 0.519–0.525, epoch 4 0.568–0.574, epoch 6 0.672–0.695); this run printed 0.513 at epoch 3, 0.589 at epoch 4 and 0.701 at epoch 6.
- Section 9: adapted scene `rectangle` 0.517, 0.524 and 0.542 with `box_iou` 0.924, 0.922 and 0.922, `red circle` `box_iou` 0.883, 0.893 and 0.892 in the named runs; this run printed `rectangle` 0.511 (`box_iou` 0.913) and `red circle` 0.887 (`box_iou` 0.878).

Evidence files for the 2026-10-26 default run of blob `4cd2580a`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-26/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `grounding_dino_detection_colab_361acff_colab-cli-t4_default.ipynb` | `05b09fc69eecb8b881ceece56c80c3557d76a19205cb35173386fc87268c3229` | `05b09fc69eecb8b881ceece56c80c3557d76a19205cb35173386fc87268c3229` (unchanged, no local path) |
| `grounding_dino_detection_colab_361acff_colab-cli-t4_default_exec.log` | `8b7009975a9be33347f56963b0e8e2fd195950487b4da0b70ef204f47b4b3ba7` | `408a759ea960dd2f16c64dda1f0fef8dffccb2950a7627edced7815911196c96` (2 local-path replacements) |
| `grounding_dino_detection_colab_361acff_colab-cli-t4_default_run_summary.json` | `f1f79348a7c961dd02cde354ae63ac571502125b05c9f0a0c058f9b5216f89bb` | `f1f79348a7c961dd02cde354ae63ac571502125b05c9f0a0c058f9b5216f89bb` (unchanged, no local path) |

**Superseded, unrun (2026-10-26, final prose pass).** The current notebook blob of `tutorials/grounding_dino_detection_colab.ipynb` is `63fb86c1c78220c43fcf24fb81b6c2edc794a51c` (commit `b0cb11f`), regenerated from `91cd979` under Kurt's 2026-10-26 ruling as a follow-up to the unrun blob `ff3ad503` (which fixed the items flagged on the confirming run of `21e405bf`), on the coordinator's request: the intro's 'training results by a few hundredths' and Troubleshooting's 'frozen numbers move only in the third decimal between GPUs' (with 'training results vary by a few hundredths between runs') are recast as what the recorded runs showed, with the six recorded test mAP50s 0.570 to 0.632; yours may differ. Markdown only; no code or output changed, and the drift recorded for the `21e405bf` run below is kept, not fixed. **It was never run (superseded by `4cd2580a`, REL14);** blob `ff3ad503` was never run either; the confirming run of `21e405bf` and the earlier records below stay as history.

**Previous candidate (2026-10-26, final prose pass).** The current notebook blob of `tutorials/grounding_dino_detection_colab.ipynb` is `ff3ad503b27ac7973f151f87566d611a5ff78cd5` (commit `5ec18d4`), regenerated from `cda8c49` under Kurt's 2026-10-26 ruling to fix only the prose-rule items flagged on the confirming run of `21e405bf`: the Section 7 'Run-to-run spread' paragraph no longer generalises a few-hundredths spread and third-decimal frozen numbers to future runs; the Section 10 experiment names the six recorded runs' 0.570 to 0.632 instead of an unlabelled 'about 0.6'; and both CPU timings are labelled as estimates. Markdown only; no code or output changed, and the drift recorded for that run below is kept, not fixed. **It was never run (superseded by `63fb86c1`, then `4cd2580a`, REL14);** the confirming run of `21e405bf` and the earlier records below stay as history.

**Historical wording (2026-10-26, code commit `2e7806a`, notebook commit `f577416`).** The previous notebook blob of `tutorials/grounding_dino_detection_colab.ipynb` is `21e405bf382ebeaff2d6c230847e8fc61db71d83`, the previous candidate, regenerated under Kurt's 2026-10-26 ruling: every run-derived number in the prose names the hosted runs that printed it (Kaggle T4 of `dc1218be` 2026-09-20; Colab T4 of `b539602b` 2026-10-04, `81eb95a4` and `a04b58a2` 2026-10-25), with no expectation ranges or timing promises; it resolves the four prose mismatches recorded for the `a04b58a2` run (recall 0.964; `red blood cell` 0.816; `white blood cell` 0.695; Section 7 952.8 s). Markdown only. This exact blob (PR #14 head `c831c71`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-25, 13/13 code cells in one pass, no restart, 0 errors, 1304.8 s wall; test mAP50 0.109 → 0.605, mAP75 0.052 → 0.452, recall 0.293 → 0.954, reload parity 8/8; evidence in `docs/execution-evidence/2026-10-25/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The records below stay as history.

**Drift recorded, not fixed (2026-10-26, Kurt's 2026-10-26 "historical wording, then stop" ruling).** The notebook prose of blob `21e405bf` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Section 8 worked answer and Interpretation: adapted test mAP50 0.605, below the 0.610–0.632 the prose records for the four named hosted runs (inside the 0.570–0.632 it records for six GPU runs).
- Section 8 worked answer and Interpretation: `white blood cell` AP50 0.668, below the recorded 0.695–0.744.
- Section 9 worked answer: adapted scene `rectangle` score 0.488, below the recorded 0.517, 0.524 and 0.542 (`red circle` 0.857 lies between the recorded 0.843 and 0.910).
- Section 7 worked answer: validation mAP50 0.345 / 0.415 / 0.513 / 0.587 / 0.626 / 0.68, a sequence none of the three named Colab runs printed (it rose every epoch and epoch 6 was kept, as in each of them).

Evidence files for the 2026-10-25 default run of blob `21e405bf`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `grounding_dino_detection_colab_c831c71_colab-cli-t4_default.ipynb` | `2abc3a2cd1e886d1906f8fac1157f69ffec5b1b787a5ea521fd46ffa0aa9ae33` | `2abc3a2cd1e886d1906f8fac1157f69ffec5b1b787a5ea521fd46ffa0aa9ae33` (unchanged, no local path) |
| `grounding_dino_detection_colab_c831c71_colab-cli-t4_default_exec.log` | `b17810d5d640121873fe4a94ca785e5d6b9cb1388e182bd9cc952d04f5de0e8d` | `3f02fecbcb97819f640d78f591b9b7eac45d3f93b7d7150c02f6d9064f12f883` (2 local-path replacements) |
| `grounding_dino_detection_colab_c831c71_colab-cli-t4_default_run_summary.json` | `d02bea34d4b48e3616ef8c6f3ffeded1ffc18abda2085d2d8a13b236af18100c` | `d02bea34d4b48e3616ef8c6f3ffeded1ffc18abda2085d2d8a13b236af18100c` (unchanged, no local path) |

**Prose kept true across runs (2026-10-25, code commit `a745158`, notebook commit `4c28b48`).** The previous notebook blob of `tutorials/grounding_dino_detection_colab.ipynb` is `a04b58a2aa1c965f0fac80fa551cced2f68078e9`, **Candidate**: that exact blob (commit `03f4a1a`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 13/13 code cells, 1333.1 s wall; test mAP50 0.109 → 0.614, mAP75 0.052 → 0.455, epoch 6 kept; reload parity 8/8; evidence in `docs/execution-evidence/2026-10-25/`). The paths this blob's runs cover: the default path of this tutorial only. Remaining gates: REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). Markdown only: every run-derived number in the prose is equal in all recorded hosted runs (Kaggle T4 of `dc1218be`, 2026-09-20; Colab T4 of `b539602b`, 2026-10-04, and of `81eb95a4`, 2026-10-25) and labelled with them, or stated as the range over those runs with a pointer to the printed output: test mAP50 0.610–0.632, mAP75 0.450–0.461, recall 0.951–0.957, `platelet` AP50 0.289–0.366, both Colab validation curves quoted, adapted scene scores 0.910 / 0.843 and 0.517 / 0.524, six-epoch time 862–931 s. The Section 5 worked answer now pairs `box_iou` 0.944 with `red circle` and 0.964 with `rectangle`, as printed; the Section 7 curve 0.319 / … / 0.692, which matched no recorded run, is gone. The records below are of earlier blobs and stay as history.

Evidence files for the 2026-10-25 default run of blob `a04b58a2`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `grounding_dino_detection_colab_03f4a1a_colab-cli-t4_default.ipynb` | `341131950436006cd589b7359bd7850a622216dcc1f59c863db81ad3ffd1abe3` | `341131950436006cd589b7359bd7850a622216dcc1f59c863db81ad3ffd1abe3` (unchanged, no local path) |
| `grounding_dino_detection_colab_03f4a1a_colab-cli-t4_default_exec.log` | `a1106cad418b3223a4f0e15ac30fae67da271feacff6cee94d9ccd0146bfec35` | `e4d97d4019a5e51695123a1ccd6d3e03d0da38cbaae6aed3dd0b0389361b155c` (2 local-path replacements) |
| `grounding_dino_detection_colab_03f4a1a_colab-cli-t4_default_run_summary.json` | `f69cb220041910cf27bd0a6575a70f292ab19f73e59fd14c93f9df110039c622` | `f69cb220041910cf27bd0a6575a70f292ab19f73e59fd14c93f9df110039c622` (unchanged, no local path) |

**Prose confirmation (2026-10-26).** This run is the confirming run for the "prose kept true across runs" revision. Every run-derived number in the notebook prose was checked against what this run printed: each number given as agreed across runs equals this run's printed value, and each stated range contains it. These still fail (the notebook is unchanged in this evidence commit):

- Section 8 worked answer (md26): recall "0.293 → 0.951 to 0.957" across the recorded hosted runs; this run's adapted recall50 was 0.964, outside the range.
- Section 8 worked answer (md26): `red blood cell` AP50 "0.041 → 0.809 to 0.812"; this run 0.816, outside the range (the Interpretation's "0.04 → 0.81" rounds to 0.82 here).
- Section 8 worked answer (md26) and Interpretation (md28): `white blood cell` AP50 "0.285 → 0.715 to 0.744" / "0.29 → 0.72–0.74"; this run 0.695, below the range.
- Activity (md28): "Section 7 alone took 862 s to 931 s across the recorded hosted T4 runs"; this run's fine-tune took 952.8 s, above the range (the header's "about 15 minutes" and the session's "about 21 to 25 minutes", 1,333.1 s here, hold).

- Holding: test mAP50 0.614 (range 0.610–0.632 and 0.570–0.632), mAP75 0.455 (0.450–0.461), `platelet` AP50 0.332 (0.289–0.366, still the smallest gain); frozen 0.109 / 0.052 / 0.293 with per-phrase 0.041 / 0.285 / 0.000, grid prior 0.009, generic `cell` 0.075 (recall 0.514); scene `red circle` 0.931 (IoU 0.944) and `rectangle` 0.695 (IoU 0.964); split 220 / 50 / 94 with 1,295 test boxes (1,104 / 93 / 98) and 2,942 training boxes; validation mAP50 rose every epoch (0.161 → 0.338 / 0.416 / 0.525 / 0.569 / 0.644 / 0.685), epoch 6 kept, training loss 11,411 → 11,313; artifact 228 tensors, 44,776,600 bytes, reload parity 8/8. The adapted scene scores (`red circle` 0.864, IoU 0.892; `rectangle` 0.542, IoU 0.922) are new values beside the per-run values the prose attributes to the two earlier Colab runs.

**Tier C wording revision (2026-10-25, branch `fix/grounding-dino-detection-tier-c-2026-10-25`, code commit `85f11b4`).** Markdown, header and metadata only for `tutorials/grounding_dino_detection_colab.ipynb`: worked answers re-quoted from the 2026-10-04 run of `a2bdaaf` (REL13), pretraining-overlap statement (DAT9), the learning-rate activity wording, use notice, `notebook_kind`, IEEE references and the DIMER Notebook Specification 2.5 declaration. The regenerated notebook blob `81eb95a419f50af4fa2854cb748aef88cd42581e` (commit `2fb41e2`) is **Candidate**: that exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 13/13 code cells, 1277.0 s wall; test mAP50 frozen 0.109 → adapted 0.61, mAP75 0.052 → 0.453, epoch 6 kept; reload parity 8/8; evidence in `docs/execution-evidence/2026-10-25/`). The paths this blob's runs cover: the default path of this tutorial only (the workshop notebook is tracked separately). The remaining gates are REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). The 2026-10-04 run is history for blob `b539602b` (REL14).

Evidence files for the 2026-10-25 default run of blob `81eb95a4`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `grounding_dino_detection_colab_2fb41e2_colab-cli-t4_default.ipynb` | `cd3f742a555fc1ce46e6d9c447544273b379b74de71a2dc7433a13af69159816` | `cd3f742a555fc1ce46e6d9c447544273b379b74de71a2dc7433a13af69159816` (unchanged, no local path) |
| `grounding_dino_detection_colab_2fb41e2_colab-cli-t4_default_exec.log` | `b02777967f9b75cbcec0680ec51c113e87cdf32c2d9862f865fee67962178d99` | `fef6476ae8220558ab9bb7f14d22549091dcbbc1c692e07caf4f57eb0a0dd984` (2 local-path replacements) |
| `grounding_dino_detection_colab_2fb41e2_colab-cli-t4_default_run_summary.json` | `b87093e4f034fb2b36aca7f1b110491a225d79eed45540836e5e75a203226ea6` | `b87093e4f034fb2b36aca7f1b110491a225d79eed45540836e5e75a203226ea6` (unchanged, no local path) |

**Prose check (2026-10-25, not fixed in this evidence commit).** The notebook prose was compared number by number with what this run printed. These quoted figures differ from this run's output (the notebook is unchanged here; a later generator change can re-quote them):

- Section 5 worked answer ('the recorded T4 run'): `red circle` 0.931 and `rectangle` 0.695 'with `box_iou` 0.964 and 0.944'; this run printed the same scores but `box_iou` red circle 0.944 and rectangle 0.964 (the prose pairs them the other way; Section 9's worked answer pairs them as printed).
- Section 7 worked answer ('the recorded T4 run (931 s)'): validation mAP50 0.161 → 0.319 / 0.426 / 0.516 / 0.577 / 0.637 / 0.692; this run printed 0.161 → 0.324 / 0.425 / 0.519 / 0.574 / 0.623 / 0.672 (epoch 6 kept in both; training loss 11,411.0 → 11,312.8 as quoted) and adapted in 861.7 s.
- Section 8 worked answer (labelled as the run of `a2bdaaf`): mAP50 0.109 → 0.630, mAP75 0.052 → 0.461, recall 0.293 → 0.957; per phrase `red blood cell` 0.041 → 0.809, `white blood cell` 0.285 → 0.715, `platelet` 0.000 → 0.366; this run printed 0.109 → 0.61, 0.052 → 0.453, 0.293 → 0.952; 0.041 → 0.812, 0.285 → 0.731, 0.0 → 0.289 (0.61 lies inside the quoted 0.570–0.632 range).
- Section 9 worked answer (labelled as the run of `a2bdaaf`): adapted `red circle` 0.931 → 0.910 and `rectangle` 0.695 → 0.517 with `box_iou` 0.883 and 0.924; this run printed 0.843 and 0.524 with `box_iou` 0.893 and 0.922.

**Candidate.** The `E2E` notebook was regenerated on 2026-10-04 for the notebook-review findings (see the review fixes section below). Its blob `b539602b` (commit `a2bdaaf`) ran once on a fresh Colab Tesla T4 through the Colab CLI on 2026-10-04: 13/13 code cells in order in one pass, no restart, 0 errors (record below). That is a CLI sequential execution, not a browser `Run all`; the BYOD run at the stated minimum with one refused input and the documented experiment re-run (epoch 0 at the frozen score) were not exercised and remain open, so the status is not promoted. The only hosted record of the previous blob (`dc1218be`, committed at `832d5d4`) is the 2026-09-20 Kaggle Tesla T4 run above, which completed only after a manual restart following the in-kernel install (pass 1 stopped at the install guard, pass 2 ran 11/11 code cells, 1503.7 s). NOTEBOOK_SPEC 2.2 §5 requires Run all with no manual restart, so that run is history, not promotion evidence, and the earlier `Release-grade` status that rested on it is withdrawn. The local pre-flight rows above remain history. The status can move to **Release-grade** only after a one-pass hosted GPU Run all of the current blob is recorded here.

Facts a reviewer should weigh: the vocabulary is three microscope phrases the checkpoint's training captions barely
name, which is why the frozen model sits near the non-adapted baselines (0.109 against 0.076 for its own `cell` boxes
relabelled) and why the gain (to 0.570 mAP50) is large — it is a repair of a vocabulary gap, not evidence about other
vocabularies; the corpus measures use `predict_boxes` (one phrase per query, the token maximum) rather than `detect`'s
text-threshold labelling, and the notebook says so; the rarest phrase (`platelet`, 212 training boxes) gained least; the
50-image validation split selects the epoch and the best epoch was the last of six, so the recipe is bounded by budget
rather than convergence; the loss the upstream criterion reports stays near 11,000 because it sums the focal alignment
over 900 queries and 256 text logits — its level is not a signal, its validation curve is; and the grounding scores
remain uncalibrated after adaptation. Two GPU runs of the recipe on the same split landed at 0.570 (package-API probe)
and 0.608 (notebook pre-flight) mAP50, so a Kaggle number a few hundredths off either is the expected spread of
non-deterministic Hungarian training, not a finding — and the T4 run landed there (0.632 mAP50, 0.450 mAP75, recall
0.951; `red blood cell` 0.809, `white blood cell` 0.744, `platelet` 0.343; epoch 6 kept; the six epochs took 931.6 s
on the T4 against 619 s on the build GPU).


## Supplemental open-vocabulary notebook remediation — 2026-09-26

Applies only to `DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`. No fresh hosted execution or real-model BYOD run was performed in this remediation. The primary notebook's previous execution record cannot establish qualification for this distinct frozen two-model comparison. **Candidate** status is retained.

Confirmed gaps fixed: distribution version checks rejected valid local build suffixes; BYOD paths/IDs could alias or escape the intended directory; declared box caps were unenforced; BYOD results were only printed; surviving model aliases defeated sequential memory release. The optional path now validates bounded labelled inputs, checks both token contracts before model loading, performs the same frozen local inference, clears model references on success/failure/retry, and exports isolated JSON/CSV evidence with source/model/runtime digests. Labels are normalized identically to query phrases. The controlled activity explicitly requires a separate copy/fresh runtime and cannot rewrite the canonical comparison.

Local baseline: 5 primary parity tests passed. The release validator already failed because its exact `STATUS.md` Current status regex does not accept the existing status line. This unrelated primary status contract is unchanged.

Measured remediation checks: 17 focused tests pass using the actual notebook loader, evaluator, export and lifecycle functions with model doubles. They cover valid/repeated exports, malformed labels/boxes, traversal, inconsistent IDs/files, extra/unlabelled images, duplicate pixels/annotations, public-version comparison, token rejection before model load, sequential model release, and failure/retry with retained exception tracebacks. These tests demonstrate local control flow; they are not real detector or GPU memory measurements.

Remaining evidence procedure: run the supplemental notebook from a fresh T4 runtime with defaults, record commit/blob, outputs, versions, wall/VRAM and restart count. For BYOD, supply a valid 8–200-image labelled local directory within the documented bounds, enable `USE_BYOD` and set `BYOD_DATASET_PATH`; execute through both local models and verify all four files in the new `byod/run-*` directory. Then repeat with an invalid box or inconsistent image mapping and retain the clear rejection before BYOD model loading. Keep each run's provenance separately. REL12 real-model BYOD and uninterrupted hosted qualification remain open; no release promotion follows from local tests.

The shared Colab setup failure observed in the other four curriculum notebooks also applied here: replacing preloaded NumPy2.1.3 with2.5.3. This supplemental notebook now pins2.1.3 and has an actual setup-prefix regression for the preloaded-host scenario. No hosted execution of this revision is claimed.


### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb) and authorized merging PR #8 (merge commit `94d6579`). The file is archived byte-for-byte, SHA-256 `5434bf1737dd6fe7d4e73d93329ec617b45d044510fea25118ff01a3f5ddc1d7`. All 23 code cells have execution counts, 42 saved outputs and zero saved errors. Code-cell sources match commit `8a518154b075a3ca8dfdac8a774993b6fdcd53ed`, tutorial blob `44ddb2707e43415bb8899c0d2511f01314f6f629`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`a70b55f` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: Default path: pinned BCCD commit `d272fb14`, 364 images and 4,886 boxes after the two zero-area annotations are dropped (pinned sample digest `af9390b9…` reproduced), 94 test images with 1,295 objects and a three-phrase vocabulary; Grounding DINO Tiny against the OWLv2 Base/16 ensemble. BYOD was not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, huggingface_hub 0.36.2, NumPy 2.1.3, CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 23; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Grounding DINO Tiny / OWLv2: mAP50 0.1088 / 0.0537, mAP75 0.0523 / 0.0331, mAP50-95 0.0603 / 0.0294, recall50 0.2927 / 0.0680, 5,173 / 1,506 predictions, mean inference 0.475 / 0.783 s per image. The low scores are the expected domain shift of web-trained detectors on blood-smear images.

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).


## Supplemental open-vocabulary notebook review fixes — 2026-09-30

Applies only to `DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb` and answers the 2026-09-30 notebook review of commit `a70b55f` (blob `db5026ed`; unchanged on `main` at `e5d53e0`). The notebook has no generator; every edit was applied as an anchor-checked replacement keyed by cell id, so cell ids, order, metadata and the declared specification (2.1) are unchanged. **Candidate** status is retained.

| Finding | Change |
|---|---|
| M1 invalid numerical outputs | Both adapters validate raw output before thresholding: shapes, finite Grounding DINO logits at the active phrase-token positions (masked `-inf` padding is allowed), finite OWLv2 logits for the supplied queries, and finite boxes. Normalized predictions are validated again by the evaluator and before every export: finite score in [0, 1], four finite ordered coordinates inside the image, vocabulary phrase. OWLv2 boxes are clipped to the image, matching the Grounding DINO adapter. JSON is written with `allow_nan=False`; BYOD scores before creating its run directory. |
| M2 crowded-object diagnostics | New `assign_detections` / `assignment_summary` use the AP evaluator's greedy rule, so each prediction and each object is counted once (true positive, duplicate, wrong phrase, spurious; matched or missed). Sections 15, 16 and 19 use it; overlap coverage is reported separately and named as such. Section 15 asserts that unique matches equal the AP evaluator's recall50 matches. |
| M3 visual handoff | Reference-box previews (Section 10) before inference; the first two comparison panels and an evaluator view (matched / duplicate / missed) shown inline in Section 22; a two-image BYOD preview; captions give the saved file path. A kernel without IPython prints the path instead. |
| 4.1, 4.2 | BYOD rejects duplicate CSV headers, rows with the wrong number of fields and multi-frame images. |
| 4.3 | Prompt order reports both ordered-list identity and set identity. |
| 4.4 | Prompt summaries add `gt_support`; an absent class shows `n/a`, not `0.0`. Glossary adds AP (with a worked example), true positive/duplicate, confidence floor vs IoU threshold, and NMS. |
| 4.5 | Per-session `RUN_ID`, `output_inventory.json` with the size and SHA-256 of this run's files, verified by the terminal summary; a stale `threshold_sweep.csv` is removed when the sweep is off. |

User-visible contract changes: `metrics.json` `duplicates` now holds the assignment summary (keys `true_positives`, `duplicate_predictions`, …) instead of `primary_matches` / `redundant_same_object_boxes`; `object_analysis.csv` `*_detected` / `*_score` / `*_iou` describe the one-to-one match and gain `*_best_overlap_iou`; `prompt_sensitivity.csv` gains `gt_support`; `threshold_sweep.csv` columns follow the assignment; a new `output_inventory.json`; OWLv2 boxes are clipped to the image. The workshop specification document was updated to match.

Checked and not changed: OWLv2's padded-square box mapping. transformers 4.57.6 `_scale_boxes` already rescales by `max(height, width)`, so the notebook's `target_sizes=[image.size[::-1]]` was correct.

Local checks (Windows, Python 3.12.14, torch 2.13.0+cpu, transformers 4.57.6): `ruff check src tests tools` clean; `pytest` 99 passed / 2 skipped (baseline `e5d53e0`: 74 passed / 2 skipped); the 24 new workshop tests all fail against the unfixed notebook; `validate_release_assets.py` PASS; `build_notebook.py --check` OK. The Grounding DINO and OWLv2 adapter tests use substitute models; the OWLv2 test calls the real upstream `Owlv2ImageProcessor.post_process_object_detection`.

CPU real-model execution (not hosted, not clean-runtime evidence): the notebook's own cells ran in order in one interpreter with the real pinned Grounding DINO and OWLv2 snapshots and the pinned BCCD archive, all digest-verified by the notebook (sample digest `af9390b9…` reproduced; 94 test images, 1,295 objects). Deviations from the supported runtime: CPU instead of T4; local torch 2.13 (the setup cell's pip-install prefix was skipped); torchvision absent (only its version string is read); `IPython.display` captured to files. The fixed revision was run with the threshold sweep on; the unfixed revision (`e5d53e0`) with defaults. The fixed run used an intermediate revision that differs from the committed notebook only in comments, markdown, one `json.dumps(..., allow_nan=False)` keyword and passing `(side, side)` rather than `(height, width)` as the OWLv2 target size, which upstream reduces to the same `max` side.

| Same host, CPU | Unfixed `e5d53e0` | Fixed |
|---|---|---|
| Grounding DINO mAP50 / mAP75 / mAP50-95 / recall50 / predictions | 0.1089 / 0.0523 / 0.0607 / 0.2927 / 5,172 | identical |
| OWLv2 mAP50 / mAP75 / recall50 / predictions | 0.0537 / 0.0331 / 0.0680 / 1,506 | identical |
| OWLv2 mAP50-95 | 0.02941 | 0.02949 (image clipping) |
| Grounding DINO duplicates diagnostic | 387 primary matches, 336 redundant boxes | 379 true positives (= AP recall50 matches), 916 missed, 325 duplicate, 1,055 wrong phrase, 3,413 spurious; 387 objects with overlap coverage |
| OWLv2 duplicates diagnostic | 88 primary, 0 redundant | 88 true positives, 1,207 missed, 0 duplicate, 797 wrong phrase, 621 spurious |
| Object categories (both / GDINO-only / OWLv2-only / both miss / loc. disagreement) | 77 / 304 / 5 / 903 / 6 | 79 / 296 / 5 / 911 / 4 |
| Validation errors on real outputs | — | none (all 23 code cells ok) |

These CPU numbers agree with the recorded 2026-09-26 T4 run to within GPU/CPU numerical noise (T4: Grounding DINO 0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,173; OWLv2 0.0537 / 0.0331 / 0.0294 / 0.0680 / 1,506). All 5 inline displays rendered (2 reference previews, 2 comparison panels, 1 evaluator view); the 15-file inventory was verified by the terminal cell. BYOD was not exercised with real models.

Follow-up in the same PR: panel titles now sit in a 22-px strip above the image instead of being painted over its top 22 rows, so boxes and labels at the top edge stay visible (panels are 22 px taller), and each box label has a filled backing in its phrase colour with black or white text chosen for contrast. The CPU run above predates these display-only changes.

Remaining before promotion: a fresh hosted T4 `Run all` of the committed blob (defaults, then a separate copy with `RUN_THRESHOLD_SWEEP=True`), and real-model labelled BYOD positive and invalid-input runs (REL12). Learner walkthrough remains unperformed.

### Maintainer-supplied Colab execution of revision `563b0f6` — 2026-09-30

- **File:** [executed notebook](execution-evidence/2026-09-30/DIMER_Open_Vocabulary_Object_Detection_Workshop_563b0f6.ipynb), archived byte-for-byte, SHA-256 `b22a233f56e11cb5620f76379a408db950d2475c7f33b7a0227b3b2c1d421636`.
- **Source match:** all 56 cells have the same ids and order as PR-head commit `563b0f6` (notebook blob `07b6a321`). Code-cell sources are identical; no `# @param` value was changed, so this is the default path (`USE_BYOD=False`, `RUN_PROMPT_EXPERIMENT=True`, `RUN_THRESHOLD_SWEEP=False`).
- **Runtime:** Colab Tesla T4, Python 3.13.15. The setup cell installed torch 2.14.0 / torchvision 0.29.0 / transformers 4.57.6 / huggingface_hub 0.36.2 and continued without a restart request; the saved RUNTIME reports torch 2.14.0+cu130, NumPy 2.1.3, Pillow 11.3.0.
- **Executed cells:** 23/23 code cells, execution counts 1–23 in order, 0 saved errors. Five inline images rendered: 2 reference previews, 2 comparison panels, 1 evaluator view.
- **Sample:** digest `af9390b9…` reproduced; 364 images, 4,886 boxes; 94 test images with 1,295 objects.
- **Peak GPU memory:** Grounding DINO 1.85 GB; OWLv2 1.95 GB.
- **Mean latency:** Grounding DINO 0.460 s/image; OWLv2 0.694 s/image.

| Metric | Grounding DINO Tiny | OWLv2 Base/16 | Previous T4 run (2026-09-26, pre-fix code) |
|---|---|---|---|
| mAP50 | 0.1088 | 0.0537 | 0.1088 / 0.0537 |
| mAP75 | 0.0523 | 0.0331 | 0.0523 / 0.0331 |
| mAP50-95 | 0.0603 | 0.0295 | 0.0603 / 0.0294 |
| recall50 | 0.2927 | 0.0680 | 0.2927 / 0.0680 |
| predictions | 5,173 | 1,506 | 5,173 / 1,506 |
| unique matches @0.50 | 379 | 88 | — |
| duplicate / wrong phrase / spurious | 325 / 1,056 / 3,413 | 0 / 797 / 621 | — |

The primary metrics equal the pre-fix T4 run except OWLv2 mAP50-95 (0.0294 → 0.0295), the expected effect of clipping OWLv2 boxes to the image. Unique matches equal the AP evaluator's recall50 matches (the notebook's own assertion passed). The Grounding DINO prompt-order probe returned 69 vs 70 boxes; this differs from the CPU run (69 vs 69) because of GPU numerics near the score floor, and the notebook reports it as observed.

| Journey | Verdict |
|---|---|
| Default Run all (M1 validation active on real outputs, M2 diagnostics, M3 inline visuals, 4.5 inventory) | **Pass** |
| Threshold sweep (separate copy, `RUN_THRESHOLD_SWEEP=True`) | Not assessed in this run |
| BYOD, real models (valid and invalid inputs; REL12) | Not assessed in this run |
| Export re-run | Not assessed in this run |

**Evidence boundary:** the saved outputs were inspected; execution was not independently repeated. The exported files themselves were not supplied, so only their printed SHA-256 prefixes and the inventory count (14 files) were inspected. Freshness of the runtime and absence of manual reruns are supported by the sequential execution counts but not otherwise established.

**Status:** remains **Candidate**. Open before promotion: the threshold-sweep copy, real-model BYOD runs (REL12), and a learner walkthrough.

### 2026-10-03 uv isolated environment — hosted re-run pending

Applies only to `DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`. The notebook no longer pip-installs into its kernel. Section 4 downloads a pinned `uv` 0.12.15 wheel (size and SHA-256 checked), builds a managed CPython 3.12.12 environment and installs the carried `tools/ovd-workshop-requirements.lock` (compiled with `uv pip compile --generate-hashes` from the notebook's previous pins in `tools/ovd-workshop-requirements.in`; the pins are unchanged) with `--require-hashes --only-binary :all:`. Snapshot staging, model loading, inference, the prompt experiments, the optional sweep and BYOD inference now run as separate processes of the carried `tools/ovd_workshop.py` in that environment (the former cell code, moved unchanged) and return their detections as strict JSON; the kernel keeps the data checks, the common evaluator, the diagnostics, the panels and the exports, and validates every returned detection again. The carrier cell is written by `tools/build_ovd_workshop_carrier.py` (`--check` in CI via the tests); no notebook line exceeds 2,000 characters. The restart guard is gone. The notebook runs on **Linux x86_64 only** (Colab, Kaggle, Linux Jupyter). Learner cells are unchanged.

Blob `07b6a321ed4f158c988e747ca2c4e2ddf1207cde` → `924cef66efaaa07e014376023545f39dd922d104`. A local CPU Run all with real weights (Windows, anaconda torch 2.9.1, not the pinned environment; the Linux guard and the environment build were bypassed and the stages ran with the host interpreter) completed every code cell, including the 14-file inventory check. OWLv2 equals the 2026-09-30 T4 run: mAP50 0.0537, mAP75 0.0331, mAP50-95 0.0295, recall50 0.0680, 1,506 predictions. Grounding DINO gave 0.1086 / 0.0524 / 0.0606 / 0.2934 / 5,149 (T4: 0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,173). That gap is a harness artifact, not the move: with CUDA hidden from the process, the former cell code and the carried stage give bit-identical Grounding DINO detections (6 of 6 images checked), while making CUDA visible but unused shifts both on all 15 images checked; a same-host CPU run of the previous notebook with CUDA visible gave 0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,171. The stage-to-kernel JSON hand-off is exact (`tests/test_ovd_workshop_uv.py`). `uv pip compile` re-run on 2026-10-03 reproduces the lock exactly. Verification so far is not clean-runtime evidence. A hosted T4 Run all of the new blob is pending; its default results should match the 2026-09-30 Colab run of `563b0f6` above. Status stays **Candidate**.

### Maintainer-supplied Colab execution of revision `873e998` — 2026-10-04 (failed)

- **File:** [executed notebook](execution-evidence/2026-10-04/DIMER_Open_Vocabulary_Object_Detection_Workshop_873e998_colab-browser-t4.ipynb), archived byte-for-byte, SHA-256 `aabfcfac8f40939e9feaa8c4e1accd4c368400a838af3c0801e08d55b5d87faf`.
- **Source match:** all 57 cells have the same ids and order as PR-head commit `873e998` (notebook blob `924cef66`). Code-cell sources are identical; no `# @param` value was changed, so this is the default path (`USE_BYOD=False`, `RUN_PROMPT_EXPERIMENT=True`, `RUN_THRESHOLD_SWEEP=False`).
- **Runtime:** Google Colab (browser, maintainer-run), Tesla T4, kernel Python 3.13.15. Section 4 built the isolated CPython 3.12.12 environment with uv 0.12.15 from the hash lock (`lock_sha256` `0675bba3…`): torch 2.14.0+cu130, torchvision 0.29.0+cu130, transformers 4.57.6, huggingface_hub 0.36.2, NumPy 2.1.3, Pillow 11.3.0, `cuda:0`. No restart was requested. Wall time is not recorded in the saved file.
- **Executed cells:** 12 of 24 code cells, execution counts 1–12 in order. Cell 12 (`2f66f39b`, OWLv2 stage) raised `RuntimeError: owlv2 failed with exit 1`; the remaining 12 code cells did not run.
- **Cause:** the stage log shows `ImportError: Owlv2ImageProcessor requires the scipy library` from `transformers/models/owlv2/image_processing_owlv2.py` (`resize`). The slow OWLv2 image processor needs SciPy. The previous in-kernel install got SciPy from the Colab image; the isolated environment has only the hash lock, and `tools/ovd-workshop-requirements.in` / `.lock` do not list `scipy`. The local CPU check missed it because it ran the stages with the host interpreter, which has SciPy.
- **Results before the failure:** sample digest `af9390b9…`, 364 images, 4,886 boxes, 94 test images with 1,295 objects, and the grid prior equal the 2026-09-30 run. Grounding DINO printed metrics are identical to the 2026-09-30 T4 run of `563b0f6`: mAP50 0.1088, mAP75 0.0523, mAP50-95 0.0603, recall50 0.2927, 5,173 predictions (per-phrase values identical too); peak GPU memory 1.85 GB. The 0.1086 local CPU figure above was therefore a host effect, as stated. Differences in the other ran cells are the new environment-build lines, the absent HF_TOKEN warning, the `show_source` listings and timings.

| Journey | Verdict |
|---|---|
| Default Run all | **Fail** (OWLv2 stage: SciPy missing from the isolated environment) |
| Threshold sweep, BYOD, export re-run | Not assessed in this run |

**Evidence boundary:** the saved outputs were inspected; execution was not repeated. **Fixed in:** not yet; the lock needs SciPy (add `scipy` to `tools/ovd-workshop-requirements.in`, recompile the hashed lock, regenerate the carrier) and a test that the OWLv2 processor imports its backends in the locked set. Then a new hosted T4 Run all of the fixed head is needed. Status stays **Candidate**.

**2026-10-04 SciPy added to the isolated environment.** The maintainer's Colab T4 run of `873e998` stopped at the OWLv2 stage (`Owlv2ImageProcessor requires the scipy library`). SciPy was never pinned because the earlier in-kernel install got it from the Colab image. `scipy==1.18.1` is now in `tools/ovd-workshop-requirements.in`, the hash lock and the runner's pins, and the carrier cell is regenerated. A CPU-only import audit in a Linux venv built from the lock's exact versions (CUDA wheels replaced by CPU torch 2.14.0) imported every stage module and ran both the Grounding DINO and OWLv2 processors. A hosted re-run of the new blob is pending. Status stays **Candidate**.

### Colab CLI execution of revision `f5097fd` — 2026-10-04

- **File:** [executed notebook](execution-evidence/2026-10-03/DIMER_Open_Vocabulary_Object_Detection_Workshop_f5097fd_colab-cli-t4.ipynb), archived byte-for-byte, SHA-256 `d75cf7bf6b4df5a82452d874400e13f9b7cd9b3d7acc7524c085c4029d18e946`.
- **Source match:** the notebook was fetched from GitHub at full commit `f5097fd473c0ca32a95fc0d0a08c2181374bc2e7` and its blob `40ff6fdd0dbd1e1db5adf79cab9064afdeece549` checked before a VM was allocated. All 57 cells of the executed file have the same ids, order and sources as that commit; no `# @param` value was changed, so this is the default path (`USE_BYOD=False`, `RUN_PROMPT_EXPERIMENT=True`, `RUN_THRESHOLD_SWEEP=False`).
- **Runtime:** Colab CLI 0.7.4, fresh Colab Tesla T4 (session released afterwards), kernel Python 3.13.15. Section 4 built the isolated CPython 3.12.12 environment with uv 0.12.15 from the hash lock (`lock_sha256` `dced88888fbb…`, now including `scipy==1.18.1`): torch 2.14.0+cu130, torchvision 0.29.0+cu130, transformers 4.57.6, huggingface_hub 0.36.2, NumPy 2.1.3, Pillow 11.3.0, `cuda:0`. Session wall time 292.2 s.
- **Executed cells:** 24/24 code cells, run in order (the CLI log shows `Executing cell 1/24` … `24/24`; the CLI does not set execution counts), 0 error outputs. The OWLv2 stage (`2f66f39b`), which failed in the 2026-10-04 browser run of `873e998`, completed. Five inline images rendered.
- **Comparison with the 2026-09-30 Colab T4 run of `563b0f6`** (`difflib` over normalised output lines): every printed metric is identical at full float precision — sample digest `af9390b9…`, 364 images / 4,886 boxes / 94 test images with 1,295 objects, the grid prior, both models' metric dictionaries including per-phrase values, the error decomposition (Grounding DINO 379 matched, 325 / 1,056 / 3,413; OWLv2 88 matched, 0 / 797 / 621) and the summary table. `ground_truth.csv` and the other unchanged-content exports have the same SHA-256. Remaining differences: the environment-build lines replace the in-kernel install, download progress bars and the HF_TOKEN warning are absent, the `show_source` listings now quote `ovd_workshop.py`, load and latency timings, the run id, OWLv2 peak GPU memory (1,952,829,440 → 1,952,298,496 B; Grounding DINO 1,848,653,824 B both runs), and the SHA-256 of `model_metrics.csv`, `metrics.json` and `provenance.json` (each stores timings, the run id or the environment).
- **Open observation:** `detections.csv` also has a different SHA-256 (`e74e33cb8b3e` → `275909d9b4f7`), although its writer is unchanged and stores no timing or run id, and every metric computed from those detections is identical. The file was not retrievable after the VM was released, so the changed rows were not inspected. Inferred cause: float-level score or box differences that do not change any match — most plausibly OWLv2 preprocessing under the pinned SciPy 1.18.1 instead of the Colab image's SciPy, consistent with OWLv2's peak GPU memory changing while Grounding DINO's did not. Not confirmed.

| Journey | Verdict |
|---|---|
| Default path (CLI sequential execution) | **Pass** |
| Browser `Run all`, threshold sweep, BYOD, export re-run | Not assessed (the CLI cannot answer upload dialogs or re-run cells) |

**Evidence boundary:** Colab CLI sequential execution of the code cells in one kernel on a fresh T4; not a browser `Run all`, default path only. Status stays **Candidate**.

## Supplemental open-vocabulary notebook review fixes — 2026-10-13 (GDN2)

Applies only to `DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb` and answers the review round 2026-10-12
review (row W9, findings GDN2-M1..M3 and GDN2-m1..m8) of commit `f8c4f0f`, blob `40ff6fdd`. The notebook body has no
generator: every edit was an anchor-checked replacement keyed by cell id, applied by a script gated on that blob
and on the stage file's SHA-256; the carrier cell was regenerated by `tools/build_ovd_workshop_carrier.py`. Blob
`40ff6fdd` → `8cd471bb`. Cell ids and order are unchanged; the declared specification stays 2.1. **Candidate**
status is retained.

- **GDN2-M1:** Section 16 now prints, for each model, where its boxes landed (phrase given × phrase of the
  best-overlapping reference at IoU 0.50) and how many objects of each class received a box when the phrase is
  ignored (display only, supplementary). Sections 13, 14 and 25 state that, under this prompt set, winning-label
  policy and score floor, some OWLv2 boxes labelled `white blood cell` overlap annotated red blood cells and do not
  count as red-blood-cell detections, and that phrase-level AP measures localization, phrase assignment and
  ranking jointly; `EVAL_SCORE_FLOOR` is labelled as the Grounding DINO evaluator's floor reused as a fixed recipe.
  The floor dependence is now measured in the notebook (follow-up below); the reviewer's earlier CPU figures are
  kept only as an externally reported observation.
- **GDN2-M2:** Sections 6, 16, 22 and 25 name BCCD's incomplete red-cell annotations: correct detections of
  unannotated objects can be counted as false positives, recall describes annotated objects only, neither metric
  establishes performance over all visible cells, and omissions may affect the models differently. *Spurious* is
  defined as "no reference box". Section 16 prints how many images have no annotated red-cell support.
- **GDN2-M3:** the prompt-wording cell sums every prompt-experiment image with the one-to-one assignment (boxes,
  true positives, recall50, duplicates, wrong-phrase boxes per model, prompt set and phrase) instead of printing
  only the first image; Sections 17, 18 and 25 describe it as a small paired case study of prompt-set sensitivity
  (four images; one for the order test) and explain why OWLv2's rows are nearly empty. `prompt_sensitivity.csv`
  is unchanged.
- **Minors:** `run_stage` no longer reprints the child log after streaming it (m1); BYOD input refusals are one
  `Refused:` line naming the file or `boxes.csv` line, without a traceback, from the kernel (`BYODInputRefused`) and
  from the byod stage (`InputRefused`, exit 65, token limits only; `run_stage` returns `None`), while genuine
  failures, including a decoded-pixel integrity mismatch, still raise (m2); `metadata.dimer.notebook_profile` (renamed from `profile`) and `notebook_kind` are declared (m3);
  `OUTPUT_DIR` is a form field and the rerun guidance matches the per-process design (m4); the opening states
  download size and the measured T4 time of the previous revision, labelled (m5); the confidence-versus-
  localization cell prints per-phrase summaries and warns that the pair sets differ (m6); the baseline is
  interpreted, including that the grid prior out-recalls OWLv2 (m7, wording corrected in the follow-up); Section 24 gives the Colab/Kaggle route for
  a BYOD folder (m8).

Default path: every code cell is AST-identical except the five declared ones (the runtime cell's failure path, the
two display-only appends, the prompt cell's printed view with identical exported rows, and the BYOD-only cell);
every function of `tools/ovd_workshop.py` is AST-identical except `predict_byod`, `stage_byod` and `main`
(refusal handling). Default metrics and every default export except the timing/run-id/environment-dependent ones
are therefore expected to equal the 2026-10-04 Colab CLI run of `f5097fd`. A harness ran the changed display cells
on the pinned BCCD sample with the reviewer's stored CPU predictions of the pinned weights: the old printed lines
are an unchanged prefix of the new ones and the exported prompt rows are identical. That is not clean-runtime
evidence. A hosted default run of the new blob, a browser run with `RUN_THRESHOLD_SWEEP=True` and a real-model
BYOD run (one valid, one refused input) are pending. Status stays **Candidate**.

### Peer-review follow-up — 2026-10-13 (GDN2-CX1 = A, CX2..CX10)

A Codex peer review of PR #13, verified by the orchestrator, and the maintainer's decision GDN2-CX1 = A. Blob
`8cd471bb` → `cb532ab5`; cells `floor-sensitivity-md` and `floor-sensitivity` added after the Section 14 table.

- **CX1 (decision A):** each model stage runs one extra, untimed pass over the 94 test images that keeps every
  candidate down to `CANDIDATE_FLOOR` (0.001; the lowest of `FLOOR_SENSITIVITY_FLOORS`), after all other passes.
  The new cell filters the cache with each adapter's own rule in float32 (Grounding DINO `>=`, OWLv2 `>` as in the
  upstream post-processing), raises if the `EVAL_SCORE_FLOOR` filter differs from the canonical predictions or
  per-phrase metrics, prints AP50/recall50 per model, floor and phrase, and exports `floor_sensitivity.csv`. The
  0.05 table stays the labelled operating recipe.
- **CX2–CX7:** wording corrected (annotation-relative measurements; "no annotated support"; AP measures
  localization, phrase assignment and ranking jointly; the mAP gap over the grid prior does not isolate ranking,
  and the grid's equal scores make its AP depend on tie order; Grounding DINO's max-over-tokens phrase reducer
  explained; the prompt experiment is a paired case study with paired differences `dTP`/`dboxes`; the
  confidence/localization pairs' construction stated and localization compared on common reference support).
- **CX8:** the order experiment runs an identical-prompt repeat control; a changed set is necessary but not
  sufficient.
- **CX9:** `prompt_summary.csv`, `prompt_variant_detections.csv` and `prompt_experiment_config.json` (image ids,
  variants, alias mappings, reordered prompts, order control, sweep floors, floor-sensitivity settings) exported.
- **CX10:** `stage_byod` is restored to the reviewed code: a decoded-pixel mismatch is an integrity error naming
  the image, not an input refusal.

Default path: every stage function is AST-identical to the PR head except `configure` and `run_model`, which equal
it once the candidate pass, the repeat control and their keys are removed; `stage_byod` equals the reviewed
`f8c4f0f` version. Every code cell is AST-identical, or identical after removing the declared additions, except the
prompt cell (printed view; exported rows unchanged) and the display/control appends.

CPU check with the pinned weights (fixer machine, torch 2.14.0+cpu, transformers 4.57.6, SciPy 1.18.1; the new
`run_model` executed in-process on the 94 test images; not hosted evidence): the cached candidates filtered at 0.05 reproduced the canonical predictions and per-phrase metrics exactly (the cell's assertions passed; Grounding DINO 0.1088 / 0.2919 / 5,171 and OWLv2 0.0537 / 0.0680 / 1,506 at 0.05, equal to the reviewer's CPU run). mAP50 / recall50 at floors 0.05, 0.02, 0.01, 0.005, 0.001: Grounding DINO 0.1088 / 0.292, 0.1171 / 0.478, 0.1198 / 0.575, 0.1211 / 0.653, 0.1222 / 0.756 (81,901 candidates at 0.001); OWLv2 0.0537 / 0.068, 0.0545 / 0.071, 0.0546 / 0.072, 0.0546 / 0.073, 0.0550 / 0.082 (16,981 candidates). AP50 levels off for both models while Grounding DINO's recall keeps rising, mostly on red blood cells (recall50 0.263 to 0.784); OWLv2's red-blood-cell AP50 stays 0.000. The identical-prompt repeat control gave identical sets for both models. CPU wall time of the whole new run_model: 1,108 s (Grounding DINO) and 1,132 s (OWLv2).

A hosted default run of the new blob is pending; its 0.05 metrics should equal the 2026-10-04 Colab CLI run of
`f5097fd`, and the notebook itself now asserts that the cached candidates reproduce them. Status stays
**Candidate**.

### Colab CLI execution of revision `b49d22a` — 2026-10-14

- **File:** [executed notebook](execution-evidence/2026-10-14/DIMER_Open_Vocabulary_Object_Detection_Workshop_b49d22a_colab-cli-t4.ipynb), archived byte-for-byte, SHA-256 `993637d70c3b81530a8299f4d849495f5b9f17e474b5c5b14a10a2976facf1da`.
- **Source match:** the notebook was fetched from GitHub at full commit `b49d22ac68676d441a057996d2a90d62369e0834` and its blob `cb532ab5ba0811de3ef66103df36f097a5c43e6b` checked before a VM was allocated. All 59 cells of the executed file have the same ids, order and sources as that commit; no `# @param` value was changed, so this is the default path (`USE_BYOD=False`, `RUN_PROMPT_EXPERIMENT=True`, `RUN_THRESHOLD_SWEEP=False`).
- **Runtime:** Colab CLI 0.7.4, fresh Colab Tesla T4 (session released afterwards), kernel Python 3.13.15; isolated CPython 3.12.12 built with uv 0.12.15 from the hash lock (`lock_sha256` `dced88888fbb…`): torch 2.14.0+cu130, torchvision 0.29.0+cu130, transformers 4.57.6, `cuda:0`. Session wall time 402.7 s (292.2 s for `f5097fd`; the difference includes the extra untimed candidate pass per model).
- **Executed cells:** 25/25 code cells, run in order (CLI log `Executing cell 1/25` … `25/25`; the CLI does not set execution counts), 0 error outputs. On the default path no integrity or decoded-pixel error was raised (BYOD is off).
- **Notebook gates:** the floor-sensitivity cell printed that the 0.05 rows are "checked equal to the table above", so its equality assertion against the canonical predictions and per-phrase metrics passed on the T4. The identical-prompt repeat control gave identical sets for both models.
- **Comparison with the 2026-10-04 Colab CLI run of `f5097fd`** (`difflib` over normalised output lines): every 0.05 metric is identical at full float precision — both models' metric dictionaries including per-phrase values, the summary table, the error decomposition (Grounding DINO 379 matched, 325 / 1,056 / 3,413; OWLv2 88 matched, 0 / 797 / 621), the confidence/localization means, the order test and peak GPU memory. `ground_truth.csv`, `detections.csv` (`275909d9b4f7`, the same as `f5097fd`, so the earlier open observation does not recur between these two T4 runs), `phrase_metrics.csv`, `object_analysis.csv`, `prompt_sensitivity.csv` and `score_iou.csv` have the same SHA-256. Remaining differences are the PR's declared additions (floor-sensitivity table, phrase-confusion breakdown and missing-support line, the four-image prompt summary replacing the one-image view, the repeat-control line, per-phrase and common-support confidence/localization lines, four new exports), load and latency timings, the run id, and the SHA-256 of `model_metrics.csv`, `metrics.json` and `provenance.json`.
- **Floor sensitivity (T4)**, mAP50 / recall50 / predictions at floors 0.05, 0.02, 0.01, 0.005, 0.001: Grounding DINO 0.1088 / 0.293 / 5,173, 0.1172 / 0.479 / 12,692, 0.1198 / 0.575 / 23,089, 0.1212 / 0.653 / 40,039, 0.1224 / 0.761 / 81,898 (red blood cell recall50 0.264 → 0.789, platelet AP50 0.000 throughout); OWLv2 0.0537 / 0.068 / 1,506, 0.0545 / 0.071 / 2,715, 0.0546 / 0.072 / 4,066, 0.0546 / 0.073 / 6,346, 0.0550 / 0.082 / 16,981 (red blood cell AP50 0.000 throughout). Against the CPU check above, OWLv2 matches at every floor; Grounding DINO differs by at most 0.0002 mAP50 and 0.005 recall50 (81,898 against 81,901 candidates), the same GPU/CPU float spread already seen at 0.05 (0.2927 / 5,173 hosted against 0.2919 / 5,171 on CPU).

| Journey | Verdict |
|---|---|
| Default path (CLI sequential execution) | **Pass** |
| Browser `Run all`, threshold sweep, BYOD (valid and refused input), export re-run | Not assessed (the CLI cannot answer upload dialogs or re-run cells) |

**Evidence boundary:** Colab CLI sequential execution of the code cells in one kernel on a fresh T4; not a browser `Run all`, default path only; forms and uploads not exercised. Status stays **Candidate**.

## 2026-10-04 review fixes (GDD-M1..M4, GDD-m1..m4)

Applies to `grounding_dino_detection_colab.ipynb` and answers the Notebook Review Framework v1 review of blob
`dc1218be` (review PR #11, `docs/reviews/2026-10-02-notebook-review/`). Everything below is generator or package
source; the notebook is regenerated, not hand-edited.

- **GDD-M1 (restart after install):** the generator is the fleet `build_notebook.py/2.1` (uv isolated environment,
  managed CPython 3.12.12, the hash lock `tutorials/requirements-colab.lock.txt`, `google.colab` stubs with a module
  spec, Infrastructure titles). The lock is compiled from `pyproject.toml` constrained to
  detr-detection-pipeline's lock at `c3c81cb` (Colab T4 PASS) and equals it without `timm` (48 packages, same
  versions and hashes). `scipy==1.18.1` joins the pins: the upstream criterion's Hungarian matching imports it and
  the hosted images used to supply it (the open-vocabulary workshop hit the same gap on 2026-10-04). The notebook now
  supports **Linux x86_64 runtimes only**. Status returned to **Candidate** in `STATUS.md`, `README.md`,
  `tutorials/README.md` and this file; the 2026-09-20 Kaggle row is relabelled as a two-pass run.
- **GDD-M2 (re-runs read the adapted model as frozen):** Sections 5, 6 and 7 call `reset_to_pretrained()`, which
  reloads the verified snapshot when the model in memory is adapted; Section 6 refuses an adapted model; `adapt`
  refuses an already adapted pipeline (`ValueError`); every experiment names its re-run scope.
- **GDD-M3 (BYOD minimum):** `split_dataset` requires 8 training, 2 validation and 2 test images and its refusal
  names the split, the counts and the dataset size that works; `byod_record_limits()` computes the minimum (12
  distinct images), Section 4 prints it, and the prose states it. Per-split validation uses `split_minimums()`.
- **GDD-M4 (guided layer):** audience, Input → Model → Output, how to use, roadmap, objectives, six predictions,
  six *What to notice* notes, seven worked answers, a Section 10 change-one-thing activity, troubleshooting, glossary,
  conclusion template; Sections 1–3 titled Infrastructure and collapsed. The validator checks the markers.
- **GDD-m1 (BYOD input handling):** `BYOD_PATH` field (zip or folder, any runtime); a cancelled upload, a missing path
  or a runtime without Colab stops with a message naming the next step; duplicate base names are refused naming
  both members; an archive over 10,000 files or 1 GB expanded is refused before decompression; pixel duplicates are
  reported.
- **GDD-m2 (asserts and the hard-coded `cell`):** the generic-prompt word is the `GENERIC_PROMPT` field (`cell` for
  the sample, `object` under BYOD unless set); the comparisons are recorded (`outcomes`, a printed reading, a run
  history in the evaluation report and `result.json`), and reload parity raises an explained `RuntimeError`.
- **GDD-m3 (quoted numbers):** the worked answers quote the recorded Kaggle Tesla T4 run (0.632 mAP50 / 0.450 mAP75,
  validation 0.161 → 0.692), labelled with that environment, and Section 7 states the run-to-run spread (0.570 /
  0.608 / 0.632 mAP50 across the three recorded GPU runs).
- **GDD-m4 (Python version):** the Prerequisites say the notebook builds its own CPython 3.12.12 environment whatever
  the kernel runs (Colab's kernel is Python 3.13).

Verification so far (local, CPU, not clean-runtime evidence): unit suite, `ruff`, generator `--check` and this
validator; offline harness runs of the notebook's own cells with the real pinned weights on reduced subsets (see the
review PR). Still needed: a hosted GPU Run all of the new blob in one pass, a BYOD run at the stated minimum with one
refused input, and one documented experiment re-run with epoch 0 at the frozen score. Status stays **Candidate**.

### Colab CLI execution of revision `a2bdaaf` — 2026-10-04

- **File:** [executed notebook](execution-evidence/2026-10-04/grounding_dino_detection_colab_a2bdaaf_colab-cli-t4.ipynb), archived byte-for-byte, SHA-256 `0aa70920173e68643ba34dba3fffa328cf948d7cf3d922288284e9cb3bdd7093`; [run summary](execution-evidence/2026-10-04/grounding_dino_detection_colab_a2bdaaf_colab-cli-t4_run_summary.json) SHA-256 `d08ccd0eb764bbeea8b463d62de871040c24dbd67e9d4e469eb1701e98d8b834`; [CLI log](execution-evidence/2026-10-04/grounding_dino_detection_colab_a2bdaaf_colab-cli-t4_exec.log) SHA-256 `5a1acd295fe62ab05261847129aea56643b7e548ee5a82a4119ab6e01ec8dfd0`.
- **Identity:** commit `a2bdaaf2b3fa777f55aa1f5cdc11b09be37d7e2c`, notebook blob `b539602bf726b43a4446a92497bbff02c77ad230`, fetched from `raw.githubusercontent.com` at the full SHA and blob-checked before the VM was allocated. The executed file's 13 code-cell sources equal the committed ones; Section 1 reports `repository_revision` `18f9a171eaa4`, the revision recorded in the notebook.
- **Runtime:** Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4, session stopped afterwards; kernel Python 3.13.15. Section 1 built the isolated CPython 3.12.12 environment from the 48-package hash lock (including `scipy==1.18.1`) in 61 s: torch 2.14.0+cu130, transformers 4.57.6, `cuda: True`. Session wall time 1301.6 s.
- **Path:** default sample path, no `# @param` changed. Not a browser `Run all`: forms are not rendered and the notebook is not opened from GitHub; the CLI saves no execution counts, so order is evidenced by `exec.log` (`Executing cell 1/13` … `13/13`).
- **Outcome:** **one pass, no restart, 0 errors**; 13/13 code cells. The three carried-module cells (Section 2) define modules and print nothing; every other cell has output.
- **Results:** snapshot 9 files / 690,308,125 B verified at `a2bb814d`; 364 photographs, 4,886 boxes, split 220 / 50 / 94; four refusal probes raised; synthetic scene `red circle` 0.931, `rectangle` 0.695 (`box_iou` 0.944 / 0.964). Grid prior 0.009 / 0.000 / 0.199, generic `cell` 0.075 / 0.027 / 0.514, frozen 0.109 / 0.052 / 0.293 (per phrase AP50 `red blood cell` 0.041, `white blood cell` 0.285, `platelet` 0.000) — all identical to the 2026-09-20 Kaggle T4 run. Six epochs in 879.3 s: validation mAP50 0.161 (frozen) → 0.341 / 0.422 / 0.525 / 0.568 / 0.637 / 0.695, epoch 6 kept, training loss 11,411.0 → 11,312.8. Adapted test mAP50 / mAP75 / recall50 0.630 / 0.461 / 0.957 (delta vs frozen 0.521 / 0.409 / 0.664); AP50 `red blood cell` 0.041 → 0.809, `white blood cell` 0.285 → 0.715, `platelet` 0.000 → 0.366; all four outcome flags true. Adapted scene `red circle` 0.91, `rectangle` 0.517 (`box_iou` 0.883 / 0.924). Adapter 228 tensors, 44,776,600 B; reload parity 8/8.
- **Against the worked answers:** every stated direction holds (validation mAP50 rose in every epoch and epoch 6 was kept; the loss barely moved; mAP50 and mAP75 both rose past both baselines; `platelet` gained least; both scene shapes were still found with lower scores). The adapted 0.630 mAP50 lies inside the quoted 0.570–0.632 spread; mAP75 0.461 against the quoted 0.450, `white blood cell` 0.715 against 0.744 and `platelet` 0.366 against 0.343 are run-to-run differences of non-deterministic GPU training, not a finding.
- **Not exercised:** BYOD (positive and refused input), the Section 10 learning-rate experiment, and a learner walkthrough. Status stays **Candidate**.
