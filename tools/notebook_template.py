"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "grounding_dino_detection_pipeline",
    "repo_name": "grounding-dino-detection-pipeline",
    "stem": "grounding_dino_detection",
    "notebook_name": "grounding_dino_detection_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "GroundingDINOPipeline",
    "weights_key": "grounding-dino-tiny",
    "runtime_imports": ["torch", "transformers"],
    "title": "Grounding DINO tiny — DIMER zero-shot object detection tutorial (standalone)",
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
    "capability": "zero-shot (open-vocabulary, text-prompted) object detection using the pinned `IDEA-Research/grounding-dino-tiny` weights",
    "intro": (
        "At inference the image is resized (shortest edge 800, longest edge 1333) and the caller's phrases are joined "
        "into one lower-cased, period-separated text (`\"rectangle. red circle.\"`); a Swin-T image backbone and a BERT "
        "text encoder are fused, and the decoder proposes boxes whose per-token grounding scores pass a **sigmoid**. "
        "The pipeline returns each surviving box in xyxy pixel coordinates of the input image, the phrase it grounded "
        "to, and its score. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or "
        "preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights, processor and "
        "tokenizer, and the carried pipeline module adds snapshot verification, prompt and image validation with named "
        "ceilings, a fixed output contract and the `box_iou`, `validate_inputs` and `evaluation_report` helpers. The "
        "default sample is a synthetic scene drawn in code; the IoU values reported for it are sanity evidence against "
        "the boxes you drew, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic scene with known object boxes, validate the image and the "
        "prompts into an input manifest through the pipeline's own validation stage, run text-prompted detection through "
        "the public API with explicit caller-owned thresholds, read sigmoid scores and thresholds correctly, produce an "
        "evaluation report that is `sample-sanity` with `box_iou` only when reference boxes exist and `not-measurable` "
        "otherwise, exercise an optional BYOD path, and export machine-readable detections plus an annotated image and provenance."
    ),
    "exclusions": (
        "instance or semantic segmentation (see the sibling SAM 2 pipeline), tracking, OCR, captioning, closed-set "
        "detection with a fixed class list, mAP or precision/recall evaluation (which needs a labelled box set), or any "
        "training. Prompts are free text, so a phrase the model cannot ground still produces boxes for something — the "
        "score, not the label, is your only signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate for one small image: the repository's model card records 6.2 s to load and 4.4 s per `detect` on the 320×240 synthetic scene in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the ~689 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures.",
        "- **Data:** the default sample is a deterministic 320×240 scene drawn in code (grey background, one dark rectangle, one red disc) with two prompts naming them, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus your own comma-separated prompt phrases (1–16 phrases, at most 48 characters each). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference boxes: a deterministic 320×240 RGB scene is "
                "drawn in code — grey background, a dark filled rectangle at `[40, 60, 140, 180]` and a red filled disc whose "
                "bounding box is `[200, 80, 280, 160]` — and the prompts name them (`rectangle`, `red circle`). This is the same "
                "scene and prompt pair the repository's smoke run used. The drawn boxes are the reference for the `box_iou` "
                "sanity check later; they are not a labelled dataset, so nothing here is a precision/recall measurement. The "
                "image digest is printed for the record. BYOD is optional and disabled by default; when enabled, upload one image "
                "and set `BYOD_PROMPTS` to the phrases you want grounded — no reference boxes exist for it, so the evaluation "
                "report will be `not-measurable`.\n\n"
                "The detection thresholds are **caller-owned request parameters**, not pipeline constants: `box_threshold` keeps "
                "a box whose best grounding score reaches it, `text_threshold` keeps the phrase tokens that reach it when the box "
                "is labelled. Their package defaults (`BOX_THRESHOLD = 0.4`, `TEXT_THRESHOLD = 0.3`) follow the upstream card's "
                "usage example, not a calibration; they are exposed here as form parameters and passed explicitly on every call. "
                "Nothing is validated in this cell — the next section hands the image, the prompts and both thresholds to the "
                "pipeline's own validation stage, which is the only checker. Look for a dictionary naming the sample kind, the "
                "image size and digest, the prompts, the thresholds, and the drawn reference boxes."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PROMPTS = 'cat, remote control'  # @param {{type:\"string\"}}\n"
                "box_threshold = 0.4  # @param {{type:\"number\"}}\n"
                "text_threshold = 0.3  # @param {{type:\"number\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    prompts = [phrase.strip() for phrase in BYOD_PROMPTS.split(',') if phrase.strip()]\n"
                "    drawn_boxes = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic scene: no randomness, so no seed is needed and the digest is stable.\n"
                "    image = Image.new('RGB', (320, 240), (128, 128, 128))\n"
                "    draw = ImageDraw.Draw(image)\n"
                "    drawn_boxes = {{'rectangle': [40.0, 60.0, 140.0, 180.0], 'red circle': [200.0, 80.0, 280.0, 160.0]}}\n"
                "    draw.rectangle(drawn_boxes['rectangle'], fill=(30, 30, 30))\n"
                "    draw.ellipse(drawn_boxes['red circle'], fill=(220, 30, 30))\n"
                "    prompts = list(drawn_boxes)\n"
                "    image_name = 'synthetic_scene_320x240.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'prompts': prompts, 'box_threshold': box_threshold, 'text_threshold': text_threshold, 'drawn_boxes': drawn_boxes}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `detect` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, 1..`MAX_PROMPTS` phrases of at most "
                "`MAX_PROMPT_CHARS` characters each, and both thresholds in `[0, 1]` — canonicalises the phrases through the "
                "package's `format_prompts` (lower-cased, period-terminated, space-joined) and returns an **input manifest** "
                "naming the schema and ceilings, the input's observed mode and size, the prompt text actually sent to the "
                "tokenizer, the thresholds, and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To "
                "show what rejection looks like, the cell also validates a deliberately over-long phrase and records the "
                "pipeline's own error message as a finding. `MAX_TEXT_TOKENS` (256) is enforced later, inside the pipeline, "
                "because it counts tokenizer output rather than characters. Inside the pipeline the image is converted to RGB and "
                "resized by the processor; boxes are mapped back to input pixels, and nothing else is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS}}}})\n"
                "input_manifest = validate_inputs(image, prompts, box_threshold=box_threshold, text_threshold=text_threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, ['x' * (MAX_PROMPT_CHARS + 1)])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Detect and read the scores correctly\n\n"
                "`detect` returns a dict with `detections` — a list of `{{box, label, score}}` **ordered by descending score**, "
                "`box` in xyxy pixel coordinates of the input, `label` the grounded phrase text — plus `prompt_text`, the "
                "thresholds used, `width`, `height` and the model identity. Each `score` is a **sigmoid grounding score, not a "
                "calibrated probability**: it was never fitted to the frequency with which a box is correct, so 0.9 does not mean "
                "\"90 % likely\". The thresholds you passed are the only decision rule; the pipeline ships them as defaults, not as "
                "a calibration, and the caller owns them per deployment — raise `box_threshold` when false boxes cost more than "
                "missed ones, lower it for recall, and note that a phrase that grounds nothing well may still surface a box just "
                "above the threshold. Inference is deterministic on a fixed device and dtype (no sampling, `torch.inference_mode`); "
                "CUDA kernel selection can move scores in the third or fourth decimal place and reorder near-ties. As recorded in "
                "the model card, the repository's CPU smoke on this same scene at the default thresholds returned `red circle` at "
                "score 0.932 and `rectangle` at 0.698 with boxes within about 2 px of the drawn ones; that is one observation, not "
                "a calibration point. A materially different result on your runtime is a signal to check the install, not a measurement."
            ),
            "code": (
                "result = pipe.detect(image, prompts, box_threshold=box_threshold, text_threshold=text_threshold)\n"
                "print({{'n_detections': len(result['detections']), 'prompt_text': result['prompt_text'], 'box_threshold': result['box_threshold'], 'text_threshold': result['text_threshold'], 'device': pipe.device}})\n"
                "for rank, det in enumerate(result['detections'], start=1):\n"
                "    print(f\"{{rank:>2}}. score {{det['score']:.4f}}  label {{det['label']!r}}  box {{[round(v, 1) for v in det['box']]}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No detection metric "
                "is reported by default: mean average precision needs a labelled box set with a matching vocabulary, and this "
                "repository ships none. The repository's only metric helper is `box_iou(a, b)` (intersection-over-union of two "
                "xyxy boxes), the building block a caller would use to compute mAP on their own labelled data; when reference "
                "boxes are supplied the report carries one `box_iou` entry per reference — its value, which detection matched it "
                "best and whether that detection's label agrees — with the verdict `sample-sanity`. On the synthetic path those "
                "references are shapes **you drew yourself**, so a high IoU proves only that the input contract, prompt "
                "formatting, forward pass and coordinate mapping round-trip. On BYOD no reference exists, the verdict is "
                "`not-measurable`, and the report states what would make the task measurable: labelled boxes on your own images "
                "with a phrase vocabulary matching the prompts. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, drawn_boxes, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference boxes exist for this input, so box_iou is not computed; inspect the annotated PNG instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (score-ordered detections with boxes and labels, prompt text, "
                "thresholds), the evaluation report, the input manifest, the sample identity, digest and drawn boxes, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the immutable "
                "model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The "
                "detections are also written as CSV with explicit `image`, `rank`, `label`, `score`, `x0`, `y0`, `x1`, `y1` "
                "columns so score ordering survives downstream use, and an annotated PNG draws every returned box for visual "
                "inspection (a supplement to, not a replacement for, the machine-readable files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in result['detections']:\n"
                "    draw.rectangle(det['box'], outline=(0, 255, 0), width=2)\n"
                "    draw.text((det['box'][0] + 2, det['box'][1] + 2), f\"{{det['label']}} {{det['score']:.2f}}\", fill=(0, 255, 0))\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'prompts': prompts, 'drawn_boxes': drawn_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_detections.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, det in enumerate(result['detections'], start=1):\n"
                "        writer.writerow([image_name, rank, det['label'], f\"{{det['score']:.6f}}\", *[f\"{{v:.2f}}\" for v in det['box']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The boxes are grounded to free-text phrases: the label tells you which phrase the box scored best against, not that the "
        "object is really there, and the sigmoid score is uncalibrated. The thresholds are request parameters you own; the "
        "defaults are the upstream usage example, not a tuned operating point. On the synthetic scene the `box_iou` values in the "
        "evaluation report compare detections to shapes you drew yourself and the verdict is `sample-sanity`, which proves only "
        "that the input contract, prompt formatting, forward pass and coordinate mapping work; they say nothing about "
        "photographs, small or occluded objects, crowded scenes, or vocabulary the model has never grounded, and a BYOD result is "
        "a single-image observation with the verdict `not-measurable`. Long or many prompts are refused at the stated ceilings, "
        "and phrases are lower-cased and period-joined before encoding, which can merge or split labels in ways you should "
        "inspect in `prompt_text`. The pipeline provides no segmentation, tracking, OCR, captioning, mAP evaluation, or training "
        "capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** lower `box_threshold` to 0.2 and count how many extra boxes appear on the same scene (the smoke "
        "run found none, but a photograph behaves differently); add a phrase that names nothing in the image (`\"bicycle\"`) and "
        "watch where its box lands and how its score compares; enable `USE_BYOD` with a photograph, hand-label a few boxes and "
        "pass them to `evaluation_report` to see the verdict switch to `sample-sanity` — the first step towards a real "
        "precision/recall number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/grounding-dino-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/IDEA-Research/GroundingDINO\n"
        "- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (Liu et al., 2023): https://arxiv.org/abs/2303.05499"
    ),
}
