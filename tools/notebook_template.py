"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E open-vocabulary detection fine-tuning workflow: the pinned
IDEA-Research/grounding-dino-tiny snapshot is digest-verified and loaded, the 364 MIT-licensed BCCD blood-smear
photographs are fetched with per-file digests (their 4,886 boxes over three cell types travel inline), the records are
validated and split by image, a synthetic scene is detected through the inference contract, the frozen model's
per-phrase AP over the held-out images is measured beside a grid prior and a generic-prompt baseline, a bounded
fine-tuning of the cross-modality decoder and the heads runs with the upstream Hungarian criterion, the held-out split
is scored again per phrase, the scene and four held-out images are re-detected with the adapted model, and the adapter
is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "grounding_dino_detection_pipeline",
    "repo_name": "grounding-dino-detection-pipeline",
    "stem": "grounding_dino_detection",
    "notebook_name": "grounding_dino_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "GroundingDINOPipeline",
    "weights_key": "grounding-dino-tiny",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "Grounding DINO tiny — DIMER E2E open-vocabulary detection fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/grounding-dino-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/tutorials/grounding_dino_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IDEA--Research%2Fgrounding--dino--tiny-ffcc4d?style=flat",
            "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-IDEA--Research%2FGroundingDINO-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/IDEA-Research/GroundingDINO",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2303.05499-b31b1b.svg", "https://arxiv.org/abs/2303.05499"),
    ],
    "capability": "zero-shot (open-vocabulary, text-prompted) object detection and bounded supervised fine-tuning of the cross-modality decoder on labelled boxes with a phrase vocabulary, using the pinned `IDEA-Research/grounding-dino-tiny` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `IDEA-Research/grounding-dino-tiny` snapshot (a 689 MB `model.safetensors`; no pickle is opened anywhere), "
        "fetches the 364 pinned BCCD blood-smear photographs from GitHub's raw-content host at an immutable commit (about "
        "7.6 MB, each refused on any byte-size or SHA-256 mismatch; their 4,886 boxes travel inline), validates them against "
        "the three-phrase vocabulary and splits them by image into 220 / 50 / 94, detects a synthetic scene through the "
        "inference contract with an input manifest and a rejection probe, measures the frozen model's per-phrase AP50 and "
        "AP75 over the 94 held-out images beside a grid prior and a generic-prompt baseline, runs a bounded fine-tuning of "
        "the cross-modality decoder and the box / contrastive heads with the upstream Hungarian criterion and validation-mAP50 "
        "epoch selection, scores the held-out images again per phrase, re-detects the scene and four held-out images with "
        "the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify box parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On an RTX 5070 Ti the six epochs "
        "took about ten minutes and the whole path about thirteen; a CUDA runtime is used automatically when present, and "
        "**a CPU runtime is not practical for the default path** (Swin-T + BERT at 800-px inputs for 1,320 training steps)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of images plus a `boxes.csv` (`file`, `x0`, `y0`, `x1`, `y1`, `phrase`; one row per box, at most 16 distinct phrases, "
        "at least eight images). The distinct phrases become the prompt vocabulary, and the records pass through the same "
        "validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells "
        "as the BCCD sample. Uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "`IDEA-Research/grounding-dino-tiny` is the Grounding DINO model of Liu et al. (2023) in its smallest published "
        "size: a Swin-T image backbone and a BERT text encoder fused by a feature enhancer, a language-guided query "
        "selection that picks 900 proposals, and a cross-modality decoder whose queries are scored against every token of "
        "the prompt (172,249,090 parameters, of which the decoder and its heads are 11,187,460; published under the "
        "**Apache-2.0** licence). A box's score is the **sigmoid** of its alignment with a phrase's tokens — a grounding "
        "score, not a calibrated probability.\n\n"
        "What this notebook adds to inference is **adaptation with a labelled vocabulary**. An open-vocabulary detector "
        "grounds the words it was trained around; the blood-smear photographs of the BCCD dataset — red blood cells, white "
        "blood cells and platelets under a microscope — are a domain its training captions barely name, and on them the "
        "frozen model reaches a mean AP50 of only **0.109** (the build record's figure on the 94 held-out images: 0.04 for "
        "`red blood cell`, 0.28 for `white blood cell`, 0.00 for `platelet`) against 0.009 for a grid of mean-sized boxes "
        "and 0.076 for the model's own boxes for the generic word `cell` relabelled to the majority phrase. So the honest "
        "question is narrow: does a bounded fine-tuning of the decoder and the heads on 220 labelled images move the held-out "
        "**AP50** and **AP75** on an image-disjoint test split, per phrase, past those two **non-adapted baselines** and the "
        "frozen model? Nothing here is a claim about your images or your vocabulary: it is one seeded split of one small "
        "labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 9-file manifest with the tokenizer files) — no "
        "pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the processor or "
        "the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled detection set with its phrase vocabulary, validate it and split "
        "it by image without leakage; detect a synthetic scene through the public API and read the output contract "
        "correctly (sigmoid grounding scores, caller-owned thresholds, IoU against self-drawn boxes is not a benchmark); "
        "measure the frozen model's per-phrase AP beside two non-adapted baselines and read the breakdown; run a bounded "
        "fine-tuning with the upstream criterion, explicit hyperparameters and validation-based epoch selection; evaluate on "
        "an image-disjoint test split; look at the adapted boxes next to the frozen ones and the labels; and export a "
        "safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "instance or semantic segmentation (see the sibling SAM pipelines), tracking, OCR, captioning, closed-set detection "
        "with a fixed class head (the vocabulary stays text), fine-tuning of the image backbone, the text encoder, the "
        "feature enhancer or the query selection, COCO-style AP averaged over IoU 0.5..0.95 (only AP50 and AP75 are computed), "
        "evaluation on COCO, LVIS, ODinW or any benchmark proper (only one seeded 364-image sample is scored here), and any "
        "claim that three blood-cell phrases stand in for your vocabulary. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime with a CUDA GPU (Google Colab or Kaggle GPU, Python 3.12). The default path uses CUDA automatically when present; one detection costs about 0.2 s per 640×480 image on an RTX 5070 Ti and about 4 s on the build workstation's CPU, and one training step of four images costs about 1.5 s on that GPU. The build record measured 619 s for the six epochs with per-epoch validation scoring and about thirteen minutes for the whole default path on an RTX 5070 Ti with the snapshot and photographs already cached; a Tesla T4 takes roughly twice that, and a CPU runtime would take hours. The pinned `torch==2.14.0` install and the 689 MB checkpoint are the large downloads of the run; the photographs are about 7.6 MB.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union and average precision measure; why a self-drawn scene is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, boxes, labels}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4096 px, `boxes` one or more `[x0, y0, x1, y1]` pixel boxes inside it (at most 300), `labels` one phrase per box drawn from the prompt vocabulary (1..16 distinct phrases of at most 48 characters each, given beside the records). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images plus a `boxes.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image is decoded and every box and phrase checked against the image and the vocabulary, but nothing checks that a box outlines what its phrase says — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches 364 JPEG files from `https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/<commit>/BCCD/JPEGImages/` at the immutable commit `d272fb14…` (about 7.6 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; the boxes (the repository's Pascal VOC annotations at the same commit, two zero-area boxes dropped) travel inline. BCCD is published under the MIT licence; nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. BCCD photographs, the phrase vocabulary and the split\n\n"
                "`fetch_corpus` returns the 364 pinned photographs from the cache under `weights/bccd/` or GitHub's raw-content "
                "host at the pinned commit — every cached file is re-hashed and every fetched file refused on any byte-size or "
                "SHA-256 mismatch — and `read_corpus` turns each into a record with its boxes and the phrase each box is "
                "labelled with; `PROMPTS` is the vocabulary (`red blood cell`, `white blood cell`, `platelet` — the dataset's "
                "class names spelled out as the phrases the detector is prompted with). `build_sample_dataset` draws a seeded "
                "image-level split (220 / 50 / 94). `validate_dataset` then checks every record against the contract and the "
                "vocabulary, `check_split_disjoint` asserts no image (by decoded-pixel digest) is shared, and the training "
                "split's summary table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 364 photographs and 4,886 boxes (about 85 % red blood cells), three digests, and four refusal probes "
                "— a duplicate id, a box outside its image, a label outside the vocabulary, and a dataset too small to use — "
                "each rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records, vocabulary = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, vocabulary, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/bccd')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    vocabulary = list(PROMPTS)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_COMMIT[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'photographs': len(corpus_files), 'bytes': sum(len(v) for v in corpus_files.values()), 'boxes': sum(len(r['boxes']) for r in corpus), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part, vocabulary) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "vocabulary = dataset_manifests['train']['prompts']\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'vocabulary': vocabulary, 'prompt_text': format_prompts(vocabulary)}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'boxes': manifest['n_boxes'], 'label_counts': manifest['label_counts'], 'boxes_per_image': manifest['boxes_per_image'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'n_boxes': len(example['boxes']), 'first_box': [round(v, 1) for v in example['boxes'][0]], 'first_label': example['labels'][0], 'source': example.get('source')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'box outside its image': [{{**train_records[0], 'boxes': [[-5.0, 0.0, 20.0, 20.0], *train_records[0]['boxes'][1:]]}}, *train_records[1:8]],\n"
                "    'label outside the vocabulary': [{{**train_records[0], 'labels': ['stem cell', *train_records[0]['labels'][1:]]}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe, vocabulary)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Detect a synthetic scene through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a deterministic 320×240 scene "
                "drawn in code — grey background, a dark filled rectangle and a red filled disc — with two prompts naming "
                "them and the drawn boxes kept as references; a different image family from the photographs, and a scene the "
                "adapted model will detect again in Section 9. `validate_inputs` applies exactly the checks `detect` applies "
                "(image sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, 1..`MAX_PROMPTS` phrases of at most `MAX_PROMPT_CHARS`, both "
                "thresholds in `[0, 1]`) and returns an input manifest; an over-long phrase is validated too and its rejection "
                "recorded as a finding. `detect` returns `detections` ordered by descending **sigmoid grounding score** — not a "
                "calibrated probability — each with the phrase it grounded to under the processor's text threshold and its "
                "xyxy box; both thresholds are caller-owned request parameters, not calibrations. The per-image "
                "`evaluation_report` against the self-drawn boxes is `sample-sanity` (`box_iou` per reference) — plumbing "
                "evidence, not a measurement; whether the model is *good at blood cells* is what Section 6 measures on 94 "
                "photographs. The inference-only card recorded `red circle` at 0.932 and `rectangle` at 0.698 on this scene."
            ),
            "code": (
                "scene = Image.new('RGB', (320, 240), (128, 128, 128))\n"
                "draw = ImageDraw.Draw(scene)\n"
                "scene_boxes = {{'rectangle': [40.0, 60.0, 140.0, 180.0], 'red circle': [200.0, 80.0, 280.0, 160.0]}}\n"
                "draw.rectangle(scene_boxes['rectangle'], fill=(30, 30, 30))\n"
                "draw.ellipse(scene_boxes['red circle'], fill=(220, 30, 30))\n"
                "scene_prompts = list(scene_boxes)\n"
                "scene_name = 'synthetic_scene_320x240.png'\n"
                "scene_sha256 = hashlib.sha256(np.asarray(scene).tobytes()).hexdigest()\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'BOX_THRESHOLD': BOX_THRESHOLD, 'TEXT_THRESHOLD': TEXT_THRESHOLD, 'SCORE_FLOOR': SCORE_FLOOR, 'MAX_BOXES': MAX_BOXES, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(scene, scene_prompts, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, names=[scene_name])\n"
                "try:\n"
                "    validate_inputs(scene, ['x' * (MAX_PROMPT_CHARS + 1)])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': scene_name, 'sha256': scene_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def detect_scene(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.detect(scene, scene_prompts, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    checks = {{\n"
                "        'sorted_by_score': [d['score'] for d in result['detections']] == sorted((d['score'] for d in result['detections']), reverse=True),\n"
                "        'boxes_inside_image': all(0 <= d['box'][0] <= d['box'][2] <= scene.width and 0 <= d['box'][1] <= d['box'][3] <= scene.height for d in result['detections']),\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'detect output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, scene_boxes, sample_kind='synthetic (authored in this notebook)')\n"
                "    annotated = scene.copy()\n"
                "    marker = ImageDraw.Draw(annotated)\n"
                "    for det in result['detections']:\n"
                "        marker.rectangle(det['box'], outline=(255, 200, 0), width=2)\n"
                "    annotated.save(f'outputs/{stem}_scene_{{label}}.png')\n"
                "    print({{label: {{'seconds': round(elapsed, 3), 'checks': checks, 'detections': [(d['label'], round(d['score'], 3)) for d in result['detections']], 'box_iou': {{m['reference']: round(m['value'], 3) for m in report['metrics']}}, 'verdict': report['verdict']}}}})\n"
                "    return result, report\n\n\n"
                "frozen_scene_result, frozen_scene = detect_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test images\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `detection_metrics` (carried in "
                "`metrics.py`): **AP50** and **AP75** — the Pascal-VOC average precision (all-point interpolation, greedy "
                "one-to-one matching in score order) at IoU 0.5 and 0.75 — per phrase and averaged as **mAP50** / **mAP75**, "
                "with the recall at IoU 0.5 over every box kept. The **grid prior** tiles each image with boxes of the "
                "training split's mean size, all labelled with the majority phrase — no model at all, the floor. The "
                "**generic-prompt** baseline asks the frozen model for the single word `cell` and relabels every box it returns "
                "with the majority phrase — what localisation alone buys without grounding the vocabulary. The **frozen "
                "model** is scored by `pipe.evaluate`, which runs `predict_boxes` on every record: every one of the 900 queries "
                "scored per phrase (the maximum over that phrase's tokens of the query's sigmoid token logits), kept with its "
                "best phrase when that score reaches `SCORE_FLOOR` — the single-phrase-per-box rule the corpus measures use, "
                "distinct from `detect`'s text-threshold labelling. Expect the frozen model **near the baselines**: the build "
                "record measured 0.109 mAP50 against 0.009 for the grid and 0.076 for the generic prompt; read the per-phrase "
                "rows to see that `platelet` grounds nothing at all."
            ),
            "code": (
                "METRICS = ('map50', 'map75', 'recall50')\n"
                "baseline_grid = grid_prior_baseline(train_records, test_records, vocabulary)\n"
                "print({{'grid_prior_baseline': {{k: round(baseline_grid[k], 3) for k in METRICS}}, 'n': baseline_grid['n'], 'note': baseline_grid['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "generic_boxes = [pipe.predict_boxes(r['image'], ['cell']) for r in test_records]\n"
                "baseline_generic = majority_relabel_baseline(generic_boxes, train_records, test_records, vocabulary)\n"
                "print({{'generic_prompt_baseline': {{k: round(baseline_generic[k], 3) for k in METRICS}}, 'note': baseline_generic['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, vocabulary)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'boxes': frozen_test['n_boxes'], 'predictions': frozen_test['n_predictions'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'ap50': round(v['ap50'], 3), 'ap75': round(v['ap75'], 3)}} for c, v in frozen_test['per_category'].items()}}\n"
                "print({{'by_phrase_frozen': frozen_fields}})\n"
                "assert frozen_test['map50'] > baseline_grid['map50']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the decoder and the heads\n\n"
                "`pipe.adapt` trains only the cross-modality decoder (its six layers and reference-point head) and the box and "
                "contrastive heads — 11,187,460 of 172,249,090 parameters — while the Swin-T image backbone, the BERT text "
                "encoder, the feature enhancer and the query selection stay frozen. Every image is paired with the formatted "
                "vocabulary (`red blood cell. white blood cell. platelet.`) and each labelled box with the index of its "
                "phrase, and the **upstream Grounding DINO criterion** scores the 900 queries against them: Hungarian matching, "
                "sigmoid-focal alignment of each matched query to its phrase's tokens (weight 1), L1 (5) and generalised-IoU "
                "(2) box terms — the loss is the checkpoint's own, and its scale (about 11,000) is set by the number of "
                "queries and text tokens it sums over. AdamW without weight decay at a fixed learning rate, gradient clipping at "
                "0.1 (the upstream recipe's norm), seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen "
                "model's validation metrics; every epoch is scored on the 50 validation images, and the epoch with the highest "
                "validation mAP50 is kept.\n\n"
                "Watch the validation mAP50 climb from about 0.16 to about 0.64 over six epochs while the loss barely moves "
                "(it is dominated by the 900-query alignment term): the decoder is learning to bind three new phrases to the "
                "shapes it already proposes, which 220 images are enough to teach. The build record's counter-example — the "
                "same scope at a fifth of the rate — reached 0.32 in four epochs with `white blood cell` drifting down; the "
                "default is the smallest configuration that lifted every phrase."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 1)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, vocabulary, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test images were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6, the four systems are put side by side "
                "on the three measures, and the per-phrase breakdown is repeated. Read it in this order: **mAP50** first (the "
                "measure the epoch was selected on — the build record measured 0.109 → 0.570, past both baselines), then "
                "**mAP75** (0.052 → 0.430: the boxes tightened, not just the labels), then the per-phrase rows, where "
                "`red blood cell` (0.04 → 0.81) and `white blood cell` (0.28 → 0.61) gain most and `platelet` (0.00 → 0.29) "
                "— the smallest and rarest object — gains least. The cell asserts the adapted mAP50 is above the frozen one "
                "and reports whether it is above both baselines. Ninety-four images from one seeded split give **no "
                "dispersion estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a "
                "benchmark, and a gain on three microscope phrases says nothing about other vocabularies or other imaging "
                "until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, vocabulary)\n"
                "adapted_val = pipe.evaluate(val_records, vocabulary)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'ap50': round(v['ap50'], 3), 'ap75': round(v['ap75'], 3)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'grid_prior': round(baseline_grid[metric], 3), 'generic_prompt': round(baseline_generic[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['delta_vs_generic_prompt'] = {{metric: round(adapted_test[metric] - baseline_generic[metric], 3) for metric in METRICS}}\n"
                "comparison['by_phrase'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_ap50': frozen_fields[c]['ap50'], 'adapted_ap50': adapted_fields[c]['ap50'], 'frozen_ap75': frozen_fields[c]['ap75'], 'adapted_ap75': adapted_fields[c]['ap75']}} for c in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'vocabulary': vocabulary,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'grid_prior': baseline_grid, 'generic_prompt': baseline_generic}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['map50'] > frozen_test['map50']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['map50'] > max(baseline_grid['map50'], baseline_generic['map50'])}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the boxes, export the adapter and reload it\n\n"
                "The synthetic scene from Section 5 is detected again by the adapted model — an image family and a vocabulary "
                "the adaptation never saw, so this is a small look at what the adaptation did *outside* its corpus: the build "
                "record kept both self-drawn boxes but with lower grounding scores, a real cost of specialising the decoder to "
                "three phrases to record, not a failure — and four held-out photographs are written as side-by-side panels "
                "(`outputs/{stem}_examples/`: the frozen model's boxes above `BOX_THRESHOLD`, the adapted model's, and the "
                "labelled boxes, coloured by phrase) so the numbers can be checked by eye: the adapted panels should box "
                "the cells the frozen ones missed.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the decoder and the heads, about 45 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the prompt "
                "vocabulary, the training configuration and the epoch history (OUT8). `GroundingDINOPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, refuses any tensor outside the decoder and the heads, and overlays the tensors onto a freshly "
                "loaded base — a new object from files, not the in-memory model (VER2). The cell asserts identical scored "
                "boxes on eight test images (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene_result, adapted_scene = detect_scene(pipe, 'adapted')\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "frozen_base = GroundingDINOPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "palette = {{phrase: colour for phrase, colour in zip(vocabulary, [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 0), (200, 60, 255), (0, 220, 220)] * 3, strict=False)}}\n\n\n"
                "def box_panel(record, boxes):\n"
                "    panel = record['image'].copy()\n"
                "    marker = ImageDraw.Draw(panel)\n"
                "    for _score, phrase, box in boxes:\n"
                "        marker.rectangle(box, outline=palette.get(phrase, (255, 255, 255)), width=2)\n"
                "    return panel\n\n\n"
                "for record in test_records[:4]:\n"
                "    frozen_boxes = [b for b in frozen_base.predict_boxes(record['image'], vocabulary) if b[0] >= BOX_THRESHOLD]\n"
                "    adapted_boxes = [b for b in pipe.predict_boxes(record['image'], vocabulary) if b[0] >= BOX_THRESHOLD]\n"
                "    labelled = [(1.0, phrase, box) for box, phrase in zip(record['boxes'], record['labels'], strict=True)]\n"
                "    panels = [box_panel(record, frozen_boxes), box_panel(record, adapted_boxes), box_panel(record, labelled)]\n"
                "    width, height = record['image'].size\n"
                "    sheet = Image.new('RGB', (width * 3 + 20, height), (255, 255, 255))\n"
                "    for i, panel in enumerate(panels):\n"
                "        sheet.paste(panel, (i * (width + 10), 0))\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panel_order': ['frozen boxes >= BOX_THRESHOLD', 'adapted boxes >= BOX_THRESHOLD', 'labelled boxes'], 'colours': palette}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'prompts': artifact_manifest['adapter']['prompts']}})\n\n"
                "reloaded = GroundingDINOPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.predict_boxes(r['image'], vocabulary) for r in test_records[:8]]\n"
                "after = [reloaded.predict_boxes(r['image'], vocabulary) for r in test_records[:8]]\n"
                "parity = {{'identical_images': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before), 'boxes_per_image': [len(a) for a in before]}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_images'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'commit': CORPUS_COMMIT, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_images': CORPUS_IMAGES, 'pinned_boxes': CORPUS_BOXES, 'vocabulary': vocabulary}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256, 'prompts': scene_prompts}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene, 'output_files': ['outputs/{stem}_scene_frozen.png', 'outputs/{stem}_scene_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "An open-vocabulary detector grounds the words it was trained around, and blood-cell phrases under a microscope are "
        "not among them: the frozen model scores 0.109 mAP50 on BCCD, barely above the model's own boxes for the word `cell` "
        "relabelled to the majority phrase (0.076). A bounded fine-tuning of the decoder and the heads on 220 labelled images "
        "binds the three phrases to the shapes it already proposes (0.570 mAP50 and 0.430 mAP75 in the build record; "
        "`red blood cell` 0.04 → 0.81, `white blood cell` 0.28 → 0.61, `platelet` 0.00 → 0.29), with a 45 MB adapter that "
        "reloads box-for-box. That is the claim: the adaptation contract works end to end on a real labelled set with its "
        "own vocabulary, and the numbers it produces are read on two IoU bars, per phrase, against two non-adapted baselines "
        "and the frozen model rather than in isolation.\n\n"
        "The test split is 94 images from one seeded draw of one 364-image set, the validation split that picks the epoch is "
        "50, and both measures are Pascal-VOC average precision at fixed IoU bars over a single-phrase-per-box scoring rule — "
        "not COCO's averaged AP, not `detect`'s text-threshold labelling. So a gain here says the contract works on three "
        "microscope phrases, not that the adapted model handles other cell types, other stains or magnifications, your "
        "vocabulary, or that its grounding scores are calibrated (they are not, before or after). The rarest and smallest "
        "phrase gained least — a labelled set with 212 platelets teaches less than one with 2,504 red blood cells — and "
        "fine-tuning on a narrow vocabulary also moves the model elsewhere: the synthetic scene re-detected in Section 9 kept "
        "its two boxes at lower scores, one image of evidence that the adapted decoder now expects blood cells, not a "
        "measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the grid prior and the generic-prompt relabelling on "
        "*your* labels are the numbers to read before any adapted one, per phrase. **Vocabulary:** the phrases the model "
        "learns from define what it grounds; spell them the way your deployment will prompt them, and keep every phrase "
        "in the evaluation so a gain on one that costs another is visible. **Leakage:** keep every image in one split (the "
        "contract de-duplicates by decoded pixels) and split by slide or session when your images come from few sources.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled detection set with its "
        "vocabulary, validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning with the upstream criterion, evaluate against two non-adapted baselines and the frozen model on an "
        "image-disjoint split, and emit the shown machine-readable artifacts — without the repository being reachable. It does "
        "**not** establish benchmark superiority, detection quality on any other vocabulary or imaging, calibration of the "
        "grounding scores, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** lower `LEARNING_RATE` to `1e-5` and read the slower, "
        "phrase-uneven climb the build record measured; raise `EPOCHS` and watch the validation mAP50 pick the epoch; change "
        "`BATCH_SIZE` to `8` on a runtime with the memory for it; re-run Section 6 with `pipe.evaluate(test_records, ['erythrocyte', 'leukocyte', 'thrombocyte'])` "
        "against the adapted model to read what the adapter did to synonyms it never saw; or bring your own labelled boxes "
        "through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/IDEA-Research/GroundingDINO\n"
        "- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (Liu et al., ECCV 2024): https://arxiv.org/abs/2303.05499\n"
        "- BCCD blood-cell detection dataset (MIT): https://github.com/Shenggan/BCCD_Dataset\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
