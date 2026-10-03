# Grounding DINO tiny E2E Open-Vocabulary Detection Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/grounding-dino-detection-pipeline`  
**Notebook:** `tutorials/grounding_dino_detection_colab.ipynb`  
**Reviewed commit:** `a81adbca3882fa4a931f71f8b67f9733f154367e` (`main`, confirmed with `gh api repos/kurtvalcorza/grounding-dino-detection-pipeline/commits/main`)  
**Notebook Git blob:** `dc1218be8738a8fa631832d44a41d5adaaabb71d`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-20 (commit `832d5d4`). The notebook last changed in `1d7d42e` (PR #3); later commits on `main` touch the separate open-vocabulary workshop notebook, its docs and the validator, not this notebook or its carried modules.  
**Finding prefix:** `GDD`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.  
**Scope:** the generated `E2E` tutorial only. The repository's second notebook, `DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb`, is out of scope (it has its own review and fix history).

## Executive assessment

The engineering is careful, and the default experiment is sound. The notebook carries its three modules verbatim (`tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at this commit). It digest-verifies the 9-file snapshot before construction, fetches 364 digest-pinned BCCD photographs, splits them by image into 220 / 50 / 94 with a decoded-pixel disjointness check, and shows four dataset refusals. It frames the frozen model against two non-adapted baselines (a grid prior and a generic `cell` prompt relabelled), fine-tunes only the decoder and heads with the upstream criterion and validation-mAP50 epoch selection, reports AP50 and AP75 per phrase, and reloads a safetensors adapter with an 8/8 box-parity assertion. The interpretation section says what the gain does and does not establish. A CPU run of the data stage in this review reproduced the recorded split and digests exactly.

| Measure | This review (CPU, direct) | Kaggle T4 record (blob `dc1218be`) |
|---|---|---|
| Code cells completed | carried modules (5, 7, 9) + Section 4 data logic; model stages only on 8/2-image subsets | 11/11 on pass 2; pass 1 stopped at the install guard |
| Photographs / boxes | 364 / 4,886 from the cache, every digest matched | 364 / 4,886 |
| Split / digests | 220 / 50 / 94; train `1ea90ee7…`, val `ee2ec795…`, test `68f2db4d…` | identical val and test digests |
| Refusal probes | 4/4 rejected with the recorded messages | 4/4 |
| Test mAP50 frozen → adapted | not run at full scale | 0.109 → 0.632 |

Four things stop it from being ready. `Run all` needs a manual restart after the install cell, and the `Release-grade` status rests on that two-pass run (GDD-M1). Re-running Section 7 for a documented experiment, or Section 4 for BYOD, continues training the already-adapted model and labels it "frozen" (GDD-M2). The BYOD path needs at least 50 distinct images, not the stated eight, and the refusal does not say so (GDD-M3). The declared `GUIDED` layer is mostly absent, and about 1,600 lines of carried code are unlabelled and uncollapsed (GDD-M4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Revision | `a81adbc` (main); notebook blob `dc1218be`; `metadata.dimer.generated_from.revision` `6a33e750` |
| Spec | NOTEBOOK_SPEC 2.2 baseline; notebook declares 2.0 |
| Profile / mode | `E2E` / `GUIDED`, standalone carrier (metadata and cell 0) |
| Audience | Not stated explicitly. Prerequisites: basic Python, NumPy and PIL, xyxy boxes, IoU and AP |
| Runtime | CUDA GPU (Colab or Kaggle, "Python 3.12"). CPU is called "not practical" |
| Promised outcomes | Cell 0: install, stage/verify snapshot, fetch/validate/split BCCD, scene detection with input manifest, frozen per-phrase AP vs two baselines, bounded decoder fine-tune, held-out re-score, panels, safetensors adapter with reload parity. BYOD as an optional branch |
| Existing evidence | `docs/release-verification.md`: Kaggle T4 run of `dc1218be` on 2026-09-20 (two passes, "1 restart after install cell"), archived at `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-grounding-dino-detection/v2/evidence/`; a local RTX 5070 Ti pre-flight (not promotion evidence). No Colab run of this blob. No BYOD or experiment run recorded |

### Evidence actually obtained

- **Static (direct):** generator `--check` exit 0; `validate_release_assets.py` PASS (P0).
- **Data stage (direct, CPU):** carried cells exec'd from the notebook JSON; Section 4 default logic run against a copy of the local photograph cache (P1).
- **BYOD (direct, CPU):** synthetic zips of 8–50 images (64×48 PNG, one box each, two phrases) pushed through `load_byod_dataset` → `split_dataset` → per-split `validate_dataset` exactly as cell 13 does (P2).
- **Rerun semantics (direct, CPU, real pinned model):** `adapt` called twice on one pipeline over 8 train / 2 validation BCCD images, 1 epoch each (P3).
- **Documented:** Kaggle T4 `run_summary.json` and executed notebooks for blob `dc1218be`.
- **Environment:** `eo-notebook-test` conda env, Python 3.12.14, torch 2.13.0+cpu (one minor below the 2.14.0 pin), transformers 4.57.6, numpy 2.5.3. No installs, no GPU, no downloads (snapshot and photographs were already staged locally).

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Defects on the rerun paths (GDD-M2), the BYOD split sizing (GDD-M3), and BYOD input handling (GDD-m1).
- **Promise fulfilment:** the default promises are met on the T4 record. "Run all … no configuration edit" is not met without a restart (GDD-M1). "At least eight images" for BYOD is not met (GDD-M3). The optional experiments do not measure what they say (GDD-M2).
- **Learner experience:** clear, precise prose with good "Look for / Expect / Watch" notes. The guided scaffolding a `GUIDED` notebook promises is missing (GDD-M4).
- **Spec conformance:** unresolved MUSTs: RUN1, RUN10, ENV6, ENV8, REL2, REL11 (GDD-M1, GDD-m3); DAT12, DAT13, DAT14, DAT19 (GDD-M2, GDD-M3). SHOULD gaps: GDL1–3, 6, 7, 9, 11, 13, 14; EXE2, EXE5; §20 expanded-size limit.

## 3. Promise and objective tracing

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes with no intervention | cell 3 in-kernel `pip install` + stale-import guard | Kaggle pass 1: `RuntimeError … numpy: loaded=2.0.2, installed=2.5.3 … Restart the runtime` | Learner must restart and re-run | **Not met** (GDD-M1) |
| Digest-verified snapshot, no pickle | cell 11 | `verified_files: 9` | Clear | Met (documented) |
| 364 pinned photographs, 220/50/94 split, no leakage | cell 13 | 364 / 4,886; digests reproduced | "Look for" note matches output | Met (direct + documented) |
| Four refusal probes | cell 13 | 4 rejections | Clear | Met (direct) |
| Inference contract on a synthetic scene | cells 15, 23 | sanity checks True; `box_iou` 0.964 / 0.944 | Labelled plumbing, not benchmark | Met (documented) |
| Frozen vs two baselines per phrase | cell 17 | 0.009 / 0.075 / 0.109 | "Expect near the baselines" | Met (documented) |
| Bounded decoder fine-tune, validation selection | cell 19 | val mAP50 0.162 → 0.692, best epoch 6 | "Watch the climb" | Met (documented) |
| Held-out per-phrase comparison | cell 21 | 0.632 / 0.45; per-phrase rows | Reading order given; quoted numbers are another run's (GDD-m3) | Met, with GDD-m3 |
| Adapter export + fresh reload parity | cell 23 | 228 tensors, 8/8 identical | Clear | Met (documented) |
| BYOD: ≥ 8 images through the full sequence | cell 13 branch | 8–49 images refused at split validation | "2 records; 8..5000 are required" | **Not met** (GDD-M3) |
| Optional experiments change one thing | cell 24 text | Re-run continues from adapted weights | Misread as a fresh climb | **Not met** (GDD-M2) |

| Objective | Learner activity | Evidence exercised |
|---|---|---|
| Install the pinned runtime | run cell 3 | needs a restart (GDD-M1) |
| Read what the carried package guarantees | scroll 1,595 carried lines | no guidance on what to read (GDD-M4) |
| Fetch/validate/split without leakage | read probes and digests | yes |
| Read the output contract (sigmoid scores, caller thresholds) | read scene output | yes, prose only; no checkpoint (GDD-M4) |
| Measure frozen per-phrase AP against baselines | read cell 17 | yes |
| Run a bounded fine-tune with validation selection | read epoch history | yes on the default path; experiments invalid (GDD-M2) |
| Evaluate on an image-disjoint test split | read cell 21 | yes |
| Look at adapted vs frozen boxes | open panels | yes (files only, not displayed inline) |
| Export and reload with parity | read cell 23 | yes |

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| First-time learner | Source inspection | Accurate, dense prose. No how-to-use, roadmap, glossary, prediction prompts, checkpoints, troubleshooting or conclusion template; carried cells not marked as infrastructure (GDD-M4). The restart is mentioned in Section 1 but not in cell 0, which promises no intervention (GDD-M1) |
| Clean default | Documented execution (Kaggle T4, blob `dc1218be`): 11/11 on pass 2 after a restart at cell 3. Direct CPU execution of carried modules and the data stage reproduced the split and digests. **No Colab run** | Passes only with a restart |
| Active learning | Direct execution, real pinned model on CPU, 8/2-image subsets | A second `adapt` (as in "lower `LEARNING_RATE` to 1e-5") starts from the adapted weights: its epoch 0, labelled "frozen model", scored val mAP50 0.331, equal to the adapted model, while the first call's epoch 0 scored 0.119. Full-scale experiment numbers not verified |
| Reuse and recovery | Direct execution (BYOD loader, split, validation) + source | 8, 12, 20, 37, 38, 49 images all refused; 50 accepted. Refusal names a split size, not the real minimum. Upload is `google.colab`-only; cancel raises `StopIteration`; same-basename files are silently overwritten. Upload dialog, BYOD past validation, and artifact reload of a BYOD run not verified |

## 5. Findings

### Major

#### GDD-M1 — `Run all` needs a manual restart after the in-kernel install

- **Cell/section:** cell 3 (Section 1), generated by `tools/build_notebook.py` (pinned-install cell with the stale-import guard). Status claims in `STATUS.md`, `README.md`, `tutorials/README.md` and `docs/release-verification.md`.
- **Observed issue:** cell 3 `pip install`s exact pins (`torch==2.14.0`, `numpy==2.5.3`, …) into the running kernel and raises `RuntimeError(... Restart the runtime, then rerun from the top.)` when an already-imported distribution changed. On the recorded Kaggle run it fired (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`), and the run passed only on a second pass. `docs/release-verification.md` step 4 says a restart "is expected", yet cell 0 promises **Run all** needs "no configuration edit (NOTEBOOK_SPEC 2.0 §5)", and the four status documents mark the blob `Release-grade` on that two-pass run.
- **Consequence:** a learner who selects Run all on a hosted runtime whose preinstalled NumPy or torch differ from the pins gets an exception at the first code cell. Every recorded pass depended on a restart the learner must discover and perform.
- **Evidence:** documented execution (`run_summary.json`: pass 1 `ok: false` with the guard's message, pass 2 `ok: true`, `restarted_after_install_cell: true`); source inspection. Colab behaviour is inferred, not verified: the 2026-09-30 Colab run of the *workshop* notebook in this repository reports Colab NumPy 2.1.3, which differs from the `numpy==2.5.3` pin here.
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern. A carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`. Implement it in `tools/build_notebook.py` / `tools/notebook_template.py`, not by hand. Until a one-pass run is recorded, return the registry to `Candidate` in all four status documents.
- **Acceptance check:** a fresh Colab GPU runtime runs the regenerated notebook top to bottom in one pass with no restart and no error output in any cell; the run is recorded in `docs/release-verification.md` with commit, blob, runtime and outcome; no status document claims `Release-grade` for a blob whose only record needed a restart.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11 (MUST).

#### GDD-M2 — Documented reruns continue from the adapted model and label it "frozen"

- **Cell/section:** cell 24 "Optional experiments" (lower `LEARNING_RATE`, raise `EPOCHS`, change `BATCH_SIZE`, BYOD); cell 0 / cell 13 BYOD instruction "set `USE_BYOD = True` in Section 4 and re-run from that cell"; cells 15, 17, 19 run against the one `pipe` object built in cell 11. Library: `GroundingDINOPipeline.adapt` (`src/grounding_dino_detection_pipeline/pipeline.py`, `frozen_state` captured from the current weights; epoch 0 note `"frozen model"`).
- **Observed issue:** `adapt` trains `pipe` in place and keeps the best state. Nothing reloads the base before a rerun. Re-running Section 7 for an experiment therefore starts from the adapted decoder, and its epoch-0 row, printed as `note: 'frozen model'`, is the adapted model. Following the BYOD instruction re-runs Sections 5–6 on the BCCD-adapted `pipe`: the "frozen" scene image, the generic-prompt baseline and `frozen_test` are all measured with the adapted decoder, and the BYOD adaptation continues from BCCD weights. `delta_vs_frozen` and the "frozen" rows in `evaluation_report.json` and `result.json` are then wrong.
- **Consequence:** the experiment the notebook suggests ("lower `LEARNING_RATE` to 1e-5 and read the slower, phrase-uneven climb") shows no climb from 0.16. It starts near the adapted score, so the learner draws the wrong conclusion about learning rate. A BYOD user's frozen-vs-adapted comparison understates the gain and mixes in BCCD training.
- **Evidence:** direct execution, real pinned model, CPU, 8 train / 2 validation images (P3): first call epoch 0 val mAP50 0.119 (base); after it, 0.331; second call epoch 0, still labelled `frozen model`, 0.331; decoder digest `0c72b4bd…` → `dc03e242…`. Source inspection for the BYOD path. Full-scale magnitudes not verified.
- **Recommended correction:** in the generator template, rebuild `pipe = GroundingDINOPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of Section 6 (or Section 4) and again at the top of Section 7 when `pipe.adapter is not None`. Alternatively, give `adapt` a `reset=True` default that restores the snapshot weights first and labels epoch 0 from the restored state. State the rerun scope for each experiment ("re-run from Section 6") and for BYOD.
- **Acceptance check:** after a full default run, following each documented experiment's rerun instruction exactly gives an epoch-0 validation mAP50 equal (within GPU nondeterminism) to the default run's epoch 0. After a BYOD rerun, `frozen_test` equals a fresh `from_pretrained` pipeline's score on the same split. A regression test asserts that a second `adapt` (or Section 6 rerun) begins from the snapshot weights.
- **Spec:** DAT13, DAT14 (MUST); GDL10, UX5, SRC2.

#### GDD-M3 — BYOD needs 50 distinct images, not eight, and the refusal does not say so

- **Cell/section:** cell 0 ("at least eight images"), cell 1 ("a dataset needs 8..5,000 records"), cell 13 BYOD branch: `split_dataset` (val 0.15, test 0.2) followed by `validate_dataset(part, vocabulary)` per split with the default `min_records=MIN_RECORDS` (8). Library: `samples.py` `split_dataset`, `validate_dataset`.
- **Observed issue:** every split must hold eight records, so the validation split (`round(0.15·n)`) needs n ≥ 50. Any BYOD set of 8–49 distinct images is refused after upload with a message such as `2 records; 8..5000 are required`. It names neither the split nor the real minimum, so a learner with 20 images is told they supplied two.
- **Consequence:** the documented BYOD contract is wrong by a factor of six, and the refusal is not actionable. A learner cannot tell what to change.
- **Evidence:** direct execution (P2): n = 8, 12, 20, 37, 38, 49 rejected (`2`, `2`, `4`, `7`, `6`, `7 records; 8..5000 are required`); n = 50 accepted (32 / 8 / 10).
- **Recommended correction:** either validate the BYOD splits with an explicit smaller per-split minimum (`min_records=1` for validation/test, as `adapt` already does for validation), or state the real minimum (50 distinct images) in cells 0, 1 and 13 and check it before splitting with a message that names it. Name the split in every per-split refusal. Fix in `samples.py` / the template, then regenerate.
- **Acceptance check:** the stated BYOD minimum, computed from the code, equals the documented one; a BYOD set of exactly that many distinct images passes cell 13; one image fewer is refused with a message naming the minimum and the image count supplied.
- **Spec:** DAT12, DAT19 (MUST); UX10.

#### GDD-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; template `tools/notebook_template.py`, carrier cells from `tools/build_notebook.py`.
- **Observed issue:** present: precise per-section prose, "Look for / Expect / Watch / Read it in this order" notes (GDL8), optional experiments (UX5), a limits section. Absent: an intended-learner statement (GDL1), "How to use this notebook" (GDL2), a roadmap or fast path (GDL3), a glossary for the many new terms (grounding score, AP50/AP75, Hungarian matching, focal alignment, queries, mAP; GDL6), prediction prompts before the comparisons (GDL7), interpretation checkpoints with collapsible sample answers (GDL9), a Predict → Change → Run → Observe → Explain activity (GDL10), an infrastructure label and collapse for the three carried cells (1,595 lines, 199,043 characters; GDL11), a troubleshooting section for GPU, download, memory, digest and BYOD failures (GDL13), and a conclusion template (GDL14). Cells carry no `collapsed` / `jupyter.source_hidden` metadata.
- **Consequence:** a self-paced learner new to open-vocabulary detection reads a well-written reference, not a guided lesson. The learner runs and reads, but is never asked to predict, check an interpretation or diagnose. The carried code reads as required material.
- **Evidence:** source inspection of all 25 cells and the metadata.
- **Recommended correction:** add the guided layer in `tools/notebook_template.py` (opening audience / how-to-use / roadmap, a glossary, a prediction before Sections 6 and 8, two or three checkpoints with `<details>` sample answers, one structured Predict → Change → Run → Observe → Explain activity built on GDD-M2's corrected rerun scope, troubleshooting, a conclusion template). Mark the carrier cells **Infrastructure — run without studying** and set `metadata.jupyter.source_hidden` / `collapsed` in `tools/build_notebook.py`. Or declare `REFERENCE` if a compact notebook is intended.
- **Acceptance check:** each of GDL1–3, 6, 7, 9–11, 13, 14 maps to a named cell in the regenerated notebook; the three carrier cells are labelled as infrastructure and hidden by default in Colab.
- **Spec:** GDL1–3, GDL6, GDL7, GDL9–11, GDL13, GDL14, UX8 (SHOULD).

### Minor

#### GDD-m1 — BYOD input handling: Colab-only upload, cancel crash, silent basename collisions, no size limit

- **Cell/section:** cell 13 BYOD branch; `samples.py` `load_byod_dataset`.
- **Observed issue:** (a) the branch imports `google.colab.files` and opens an upload dialog; there is no location field, so Kaggle, Jupyter or an executor cannot drive it (EXE2). (b) Cancelling the dialog makes `next(iter(uploaded.items()))` raise a bare `StopIteration`. (c) The loader flattens zip paths to basenames, so `a/img000.png` and `b/img000.png` collapse into one entry and the later file silently replaces the earlier one under the same `boxes.csv` rows. (d) Members are read fully into memory with no expanded-size cap (§20 SHOULD).
- **Consequence:** BYOD is unusable outside Colab, cancellation looks like a crash, and a nested dataset can be trained on the wrong image with no warning.
- **Evidence:** direct execution (P2: a 50-image zip plus `b/img000.png` loaded as 50 records with no message); source inspection for (a), (b), (d).
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}`, read from it when set, and fall back to the upload only when it is empty; turn an empty upload into a `ValueError` naming the next step; refuse duplicate basenames (or keep relative paths); cap the total expanded size and member count before reading.
- **Acceptance check:** BYOD runs from a path on a non-Colab runtime; an empty upload prints an actionable message; a zip with duplicate basenames is refused naming both members; an archive over the cap is refused before decompression.
- **Spec:** EXE2, UX10, §20 (SHOULD); DAT19.

#### GDD-m2 — The comparisons assume the BCCD outcome and crash a legitimate BYOD result

- **Cell/section:** cell 17 (`generic_boxes = … ['cell']`, `assert frozen_test['map50'] > baseline_grid['map50']`), cell 21 (`assert adapted_test['map50'] > frozen_test['map50']`).
- **Observed issue:** the generic-prompt baseline always prompts the word `cell`, which is meaningful only for blood cells, yet the interpretation tells BYOD users to read "the generic-prompt relabelling on *your* labels". A BYOD vocabulary the frozen model already grounds, or a small set the adaptation does not improve, trips a bare `AssertionError` before `adapt` (cell 17) or before the artifact and `result.json` are written (cell 21).
- **Consequence:** a valid negative result, which the interpretation section asks learners to look for, ends in a crash with no export.
- **Evidence:** source inspection.
- **Recommended correction:** keep the asserts on the default BCCD path only (`if not USE_BYOD`), and report the comparison otherwise. Derive the generic prompt for BYOD from a form field (`GENERIC_PROMPT`) or say it is BCCD-specific.
- **Acceptance check:** a BYOD run whose adapted mAP50 does not exceed the frozen one completes and writes all eight outputs with the comparison showing the non-improvement.
- **Spec:** DAT13, RUN9, UX10.

#### GDD-m3 — Quoted results come from a different run, and run-to-run spread is not stated

- **Cell/section:** cells 0, 16, 18, 20, 24.
- **Observed issue:** the prose quotes "the build record" (0.570 mAP50, 0.430 mAP75, `white blood cell` 0.61, validation climbing "to about 0.64") as what the learner will see. The recorded T4 run of this blob gave 0.632 / 0.45, `white blood cell` 0.744, validation 0.692. `docs/release-verification.md` explains that a few hundredths either way is "the expected spread of non-deterministic Hungarian training", but the notebook never says so; it only notes the absence of a dispersion estimate over images.
- **Consequence:** a learner whose numbers differ from the quoted ones cannot tell whether that is normal or a fault.
- **Evidence:** source inspection; documented execution.
- **Recommended correction:** say once, in Sections 7 and 8, that GPU training is not bit-reproducible and that the quoted build-record figures vary by a few hundredths between runs (0.570 / 0.608 / 0.632 mAP50 across the three recorded runs).
- **Acceptance check:** the notebook names the remaining source of run-to-run variation and gives the observed range wherever it quotes a training result.
- **Spec:** ENV8 (MUST), GDL8.

#### GDD-m4 — Stated runtime is Python 3.12; Colab now runs 3.13, and no Colab run of this blob exists

- **Cell/section:** cell 1 ("Google Colab or Kaggle GPU, Python 3.12"), `metadata.language_info.version` `3.12`, `docs/release-verification.md` executor table (Colab is "the supported user path").
- **Observed issue:** the repository's own 2026-09-26 and 2026-09-30 Colab records (workshop notebook) show Python 3.13.15. The only hosted record for this blob is Kaggle.
- **Consequence:** the runtime statement is stale for the primary supported path, and that path is unverified for this notebook.
- **Evidence:** documented execution (`docs/release-verification.md` lines on the Colab workshop runs); source inspection.
- **Recommended correction:** state the supported Python versions as tested, and record a Colab GPU run of the next blob.
- **Acceptance check:** cell 1 names the runtime of a recorded run of the current blob, and that record includes a Colab GPU run.
- **Spec:** ENV3, REL10.

### Suggestions

- **GDD-S1 — Regenerate against NOTEBOOK_SPEC 2.2.** Metadata, cell 0, `NOTEBOOK_SOURCE` and `tutorials/README.md` declare 2.0.
- **GDD-S2 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`** in the notebook. Cell 3 reads it, but no markdown mentions it (EXE5).
- **GDD-S3 — Show, don't only print.** Plot the validation-mAP50 history and a per-phrase frozen-vs-adapted bar, and display one example panel inline. The panels are written only to `outputs/`, and the history is printed as dicts (UX11).
- **GDD-S4 — Print the adapt seed.** `adapt` uses `seed=0` silently; expose it beside `SPLIT_SEED` (FT6, ENV7).

## 6. Readiness

**Needs revision.**

- **Blockers:** none.
- **Open Majors:**
  - GDD-M1: no one-pass `Run all`; the only hosted record needed a restart.
  - GDD-M2: documented reruns compare against and continue from the adapted model.
  - GDD-M3: the BYOD minimum is 50 distinct images, not 8, and the refusal is misleading.
  - GDD-M4: the guided layer is absent in a `GUIDED` notebook.
- **Unresolved applicable MUSTs:** RUN1, RUN10, ENV6, ENV8, REL2, REL11, DAT12, DAT13, DAT14, DAT19.
- **Remaining gates after fixes:**
  1. A one-pass Colab GPU run of the new blob, recorded in `docs/release-verification.md`.
  2. A BYOD run (REL12): a representative set at the stated minimum accepted, one incompatible input rejected clearly, through export and reload.
  3. One documented experiment run following its stated rerun scope, with epoch 0 at the frozen score.

The `Release-grade` status in `STATUS.md`, `README.md`, `tutorials/README.md` and `docs/release-verification.md` should return to `Candidate` until gate 1 is met.

## 7. Verified vs inferred

- **Verified by direct execution (CPU, this review):**
  - the generator `--check` and the static validator pass;
  - the carried modules execute, and the data stage reproduces the recorded split, digests and refusals;
  - the BYOD minimum, refusal text and basename collision;
  - `adapt` rerun semantics on the real pinned model (small subsets).
- **Verified from documented evidence:** the reviewed blob's Kaggle T4 run, including its install-guard failure and second-pass success, and its metrics.
- **Inferred from source:**
  - the Colab restart (expected by the repository's own procedure; not run);
  - the BYOD "frozen" mislabelling;
  - the bare-assert crash on a negative BYOD result;
  - the `StopIteration` on a cancelled upload.
- **Only Kurt or a hosted run can confirm:** Colab behaviour of the install cell, real-model numbers of each experiment, and the BYOD upload dialog.
- **Finding most likely to be wrong:** GDD-M1 on Colab specifically. The restart is recorded on Kaggle and expected by the repository's procedure, but no Colab run of this blob exists; a Colab image whose NumPy and torch were not imported before cell 3 could pass in one go. The finding stands on RUN10/ENV6 either way, because the pattern needs a restart wherever preloaded versions differ.

## Probe bundle

`grounding_dino_detection_colab_Review_Probes.zip` contains:

- `run_probes.py`: P0–P3, run as `python run_probes.py <repo> <weights_root> <out> [--skip-model]`;
- `results.json`;
- `source_manifest.json`: SHA-256 of the 12 inspected source files at `a81adbc`.
