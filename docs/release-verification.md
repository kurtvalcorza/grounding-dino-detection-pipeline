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
| 2026-10-04 | `873e998` / `924cef66` (`DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`, uv isolated environment) | Google Colab (browser, maintainer-run), Tesla T4; kernel Python 3.13.15, isolated CPython 3.12.12 with torch 2.14.0+cu130 / transformers 4.57.6 | Default path (no `# @param` changed), `Run all` | not recorded | **FAILED** — 12/24 code cells ran, counts 1–12 in order; the OWLv2 stage (cell `2f66f39b`) exited 1: `Owlv2ImageProcessor requires the scipy library`, which the hash lock does not contain. Grounding DINO printed metrics identical to the 2026-09-30 T4 run (0.1088 / 0.0523 / 0.0603 / 0.2927 / 5,173) |
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
