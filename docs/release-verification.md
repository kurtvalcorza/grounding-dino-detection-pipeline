# Release verification

`tutorials/grounding_dino_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 9-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
  prompt + `majority_relabel_baseline`, `pipe.evaluate` on the frozen model with the floor assertion, `pipe.adapt`
  with its explicit hyperparameters, `pipe.evaluate` on the validation and test splits after adaptation with the
  mAP50 assertion, `detect` + `evaluation_report` on the scene after adaptation, the example panels against a
  freshly loaded frozen base, `pipe.save_artifact`, `GroundingDINOPipeline.from_artifact` and the box-parity
  assertion, and the result fields `weight_file` / `weight_format` / `weight_sha256` and the `corpus` block), the
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
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
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
     and the cell's assertion that the frozen model is above the grid prior;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 11,187,460 trainable of 172,249,090 parameters, and
     a six-epoch history with the validation mAP50 rising (build record: 0.162 → 0.352 / 0.423 / 0.516 / 0.573 /
     0.614 / 0.636, `best_epoch` 6; the training loss stays near 11,000 by construction of the upstream criterion);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on the three measures,
     the per-phrase breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test mAP50
     exceeds the frozen one — 0.570 versus 0.109 in the build record, mAP75 0.052 → 0.430, recall 0.29 → 0.96,
     `red blood cell` 0.04 → 0.81, `white blood cell` 0.28 → 0.61, `platelet` 0.00 → 0.29; the adapted model also
     clears both baselines, reported, not asserted);
   - Section 9: the scene re-detected by the adapted model with the `sample-sanity` report,
     `outputs/…_scene_adapted.png` and four example panels under `outputs/…_examples/` written; `pipe.save_artifact`
     writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (228 tensors, 44,776,600 bytes) and
     `GroundingDINOPipeline.from_artifact` reloading it with 8/8 identical scored box lists on eight test images (the
     cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
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
| `grounding_dino_detection_colab.ipynb` (`E2E`) | `832d5d4` / `dc1218be` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 384 files, 698 MB staged from the Hub into a clean cache; comparison {map50: {grid_prior: 0.009, generic_prompt: 0.075, frozen: 0.109, adapted: 0.632}, map75: {grid_prior: 0, generic_prompt: 0.027, frozen: 0.052, adapted: 0.45}, recall50: {grid_prior: 0.199, generic_prompt: 0.514, frozen: 0.293, adapted: 0.951}, delta_vs_frozen: {map50: 0.523, map75: 0.397, recall50: 0.659}, delta_vs_generic_prompt: {map50: 0.557, map75: 0.422, recall50: 0.437}, by_phrase: {red blood cell: {n: 1104, frozen_ap50: 0.041, adapted_ap50: 0.809, frozen_ap75: 0.014, adapted_ap75: 0.597}, white blood cell: {n: 93, frozen_ap50: 0.285, adapted_ap50: 0.744, frozen_ap75: 0.143, adapted_ap75: 0.653}, platelet: {n: 98, frozen_ap50: 0, adapted_ap50: 0.343, frozen_ap75: 0, adapted_ap75: 0.099}}}; reload parity {identical_images: 8, of: 8, boxes_per_image: [136, 136, 136, 146, 165, 99…]}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-grounding-dino-detection/v2/evidence/` in the workspace |
| `grounding_dino_detection_colab.ipynb` (`TASK-INFERENCE`, superseded) | `9bf9e00` / `b1e416c68b76` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v1) | PASS — 8/8 code cells, 271.8 s, 20 files, 690 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/grounding_dino_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/grounding_dino_detection_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `832d5d4` / `dc1218be` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 1503.7 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 384 files, 698 MB staged from the Hub into a clean cache; comparison {map50: {grid_prior: 0.009, generic_prompt: 0.075, frozen: 0.109, adapted: 0.632}, map75: {grid_prior: 0, generic_prompt: 0.027, frozen: 0.052, adapted: 0.45}, recall50: {grid_prior: 0.199, generic_prompt: 0.514, frozen: 0.293, adapted: 0.951}, delta_vs_frozen: {map50: 0.523, map75: 0.397, recall50: 0.659}, delta_vs_generic_prompt: {map50: 0.557, map75: 0.422, recall50: 0.437}, by_phrase: {red blood cell: {n: 1104, frozen_ap50: 0.041, adapted_ap50: 0.809, frozen_ap75: 0.014, adapted_ap75: 0.597}, white blood cell: {n: 93, frozen_ap50: 0.285, adapted_ap50: 0.744, frozen_ap75: 0.143, adapted_ap75: 0.653}, platelet: {n: 98, frozen_ap50: 0, adapted_ap50: 0.343, frozen_ap75: 0, adapted_ap75: 0.099}}}; reload parity {identical_images: 8, of: 8, boxes_per_image: [136, 136, 136, 146, 165, 99…]}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-grounding-dino-detection/v2/evidence/` in the workspace |
| 2026-09-20 | committed template at the candidate revision (blob differs from the committed one in the recorded generating revision only) | Local WSL harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and photographs pre-staged, the CPU unit suite running alongside | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (9 files), 364 photographs re-hashed from the cache, 4,886 boxes split 220 / 50 / 94, four refusal probes raised, the synthetic scene detected (`box_iou` 0.964 / 0.944), grid prior 0.009 and generic `cell` relabelled 0.076, frozen 0.109 / 0.052 / 0.294 in 53.2 s, six epochs 897.8 s (validation mAP50 0.162 → 0.348 / 0.429 / 0.520 / 0.591 / 0.648 / 0.677, epoch 6 kept), adapted 0.608 / 0.450 / 0.952 (`red blood cell` 0.042 → 0.812, `white blood cell` 0.284 → 0.715, `platelet` 0.000 → 0.297), the scene re-detected with both boxes at `box_iou` 0.916 / 0.9, four panels written, adapter 44,776,600 B / 228 tensors, reload parity 8/8, 8 outputs written | 1166.8 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-14 | `9bf9e00` / `b1e416c68b76` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-grounding-dino-detection` v1) | Default sample path | 271.8 s | PASS — 8/8 ok code cells, 20 files, 690 MB staged; not evidence for the `E2E` blob |

## Current status

**Release-grade.** The `E2E` notebook blob `dc1218be` (committed at `832d5d4`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 1503.7 s, 384 files, 698 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

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
