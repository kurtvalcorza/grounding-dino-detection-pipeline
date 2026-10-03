"""Model stages of DIMER_Open_Vocabulary_Object_Detection_Workshop.ipynb, run in its isolated environment.

The notebook carries this file byte-for-byte, writes it next to the hash-locked requirements and runs
each model stage as ``python ovd_workshop.py --stage <name> --job <job.json> --out <result.json>`` with
the interpreter of a uv-built Python 3.12 virtual environment. Nothing is installed into the notebook
kernel: it keeps the data loading, the common evaluator, the diagnostics, the panels and the exports,
and reads the JSON each stage writes.

The adapter functions are the notebook's former cell code (cells 48be2c06, 53f2f300, 2f66f39b and
3f9f36da), moved unchanged except that the score-floor defaults of ``gdino_predict`` and
``owlv2_predict`` are read when called (the floor arrives in the job file after import), and the
former inline load, loop and experiment code is wrapped in ``load_gdino``/``load_owlv2``/``run_model``.
``normalize_prompt_list`` and ``validate_detections`` are copies of the kernel's own cells, so a
malformed prediction is refused on both sides of the process boundary.
"""
# ruff: noqa: B905,E501,E701,E702
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

STAGE_FORMAT = "dimer.ovd-workshop.stage.v1"

# Heavy libraries load only in the stages that use them (see load_libraries).
torch = None
AutoModelForZeroShotObjectDetection = AutoProcessor = None
Owlv2ForObjectDetection = Owlv2Processor = None
hf_hub_download = None
DEVICE = "cpu"

# ---- Settings (overwritten by configure() from the notebook's job file) -----------------------------
EVAL_SCORE_FLOOR = 0.05
GDINO_BOX_THRESHOLD = 0.40
GDINO_TEXT_THRESHOLD = 0.30
OWLV2_THRESHOLD = 0.10
RUN_PROMPT_EXPERIMENT = True
RUN_THRESHOLD_SWEEP = False
PROMPT_VARIANTS = {}
PROMPT_VARIANT_TO_CANONICAL = {}
REORDERED_PROMPTS = ["platelet", "red blood cell", "white blood cell"]

MAX_PROMPTS = 16
MAX_PROMPT_CHARS = 48
MANIFEST_NAME = "dimer-base-manifest.json"
GDINO_DIR = Path("weights/grounding-dino-tiny")
OWLV2_DIR = Path("weights/owlv2-base-patch16-ensemble")
GDINO_REVISION = OWLV2_REVISION = None
GDINO_SPECIAL_TOKEN_IDS = (101, 102, 1012, 1029, 0)
GDINO_MAX_TEXT_TOKENS = 256
OWLV2_MAX_TOKENS = 16

gdino_model = gdino_processor = owlv2_model = owlv2_processor = None

PINS = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "torchaudio": "2.11.0",
    "transformers": "4.57.6",
    "safetensors": "0.8.0",
    "numpy": "2.1.3",
    "pillow": "11.3.0",
    "huggingface-hub": "0.36.2",
}


def configure(job):
    """Apply the notebook controls carried in the job file."""
    global EVAL_SCORE_FLOOR, GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD, OWLV2_THRESHOLD
    global RUN_PROMPT_EXPERIMENT, RUN_THRESHOLD_SWEEP, PROMPT_VARIANTS, PROMPT_VARIANT_TO_CANONICAL
    global REORDERED_PROMPTS, GDINO_DIR, OWLV2_DIR, GDINO_REVISION, OWLV2_REVISION
    settings = job.get("settings", {})
    EVAL_SCORE_FLOOR = float(settings.get("eval_score_floor", EVAL_SCORE_FLOOR))
    GDINO_BOX_THRESHOLD = float(settings.get("gdino_box_threshold", GDINO_BOX_THRESHOLD))
    GDINO_TEXT_THRESHOLD = float(settings.get("gdino_text_threshold", GDINO_TEXT_THRESHOLD))
    OWLV2_THRESHOLD = float(settings.get("owlv2_threshold", OWLV2_THRESHOLD))
    RUN_PROMPT_EXPERIMENT = bool(settings.get("run_prompt_experiment", RUN_PROMPT_EXPERIMENT))
    RUN_THRESHOLD_SWEEP = bool(settings.get("run_threshold_sweep", RUN_THRESHOLD_SWEEP))
    PROMPT_VARIANTS = settings.get("prompt_variants", PROMPT_VARIANTS)
    PROMPT_VARIANT_TO_CANONICAL = settings.get("prompt_variant_to_canonical", PROMPT_VARIANT_TO_CANONICAL)
    REORDERED_PROMPTS = settings.get("reordered_prompts", REORDERED_PROMPTS)
    if "gdino" in job:
        GDINO_DIR = Path(job["gdino"]["dir"])
        GDINO_REVISION = job["gdino"]["manifest"]["revision"]
    if "owlv2" in job:
        OWLV2_DIR = Path(job["owlv2"]["dir"])
        OWLV2_REVISION = job["owlv2"]["manifest"]["revision"]


def load_libraries():
    """Import the model libraries; only stages that run a model call this."""
    global torch, AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor
    global hf_hub_download, DEVICE
    import torch as _torch
    from huggingface_hub import hf_hub_download as _download
    from transformers import AutoModelForZeroShotObjectDetection as _gdino_model
    from transformers import AutoProcessor as _processor
    from transformers import Owlv2ForObjectDetection as _owl_model
    from transformers import Owlv2Processor as _owl_processor
    torch = _torch
    hf_hub_download = _download
    AutoModelForZeroShotObjectDetection, AutoProcessor = _gdino_model, _processor
    Owlv2ForObjectDetection, Owlv2Processor = _owl_model, _owl_processor
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---- Copies of kernel helpers -------------------------------------------------------------------------

def normalize_prompt_list(prompts):
    if isinstance(prompts, str) or not isinstance(prompts, (list, tuple)):
        raise TypeError("prompts must be a list/tuple of strings")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError("prompt count outside 1..16")
    cleaned = []
    for p in prompts:
        if not isinstance(p, str):
            raise TypeError("every prompt must be str")
        x = " ".join(p.strip().rstrip(".").strip().lower().split())
        if not x or len(x) > MAX_PROMPT_CHARS:
            raise ValueError(f"invalid prompt {p!r}")
        cleaned.append(x)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("prompts must be distinct")
    return cleaned


def validate_detections(preds, image_size=None, vocab=None, source="detector"):
    """Reject malformed detector output before it is thresholded, scored, drawn or exported.

    A detection is (score, phrase, [x0, y0, x1, y1]) in original-image pixels. Both detectors emit
    sigmoid scores, so a score must be finite and in [0, 1]. A box needs four finite, ordered
    coordinates and, when the image size is given, must lie inside the image. The phrase must be
    in `vocab` when one is given. An empty list is a valid result (nothing found).
    """
    if not isinstance(preds, (list, tuple)):
        raise TypeError(f"{source}: predictions for one image must be a list, got {type(preds).__name__}")
    for n, det in enumerate(preds):
        if not isinstance(det, (list, tuple)) or len(det) != 3:
            raise ValueError(f"{source}: detection {n} is not (score, phrase, box): {det!r}")
        score, label, box = det
        if (isinstance(score, bool) or not isinstance(score, (int, float, np.floating))
                or not np.isfinite(score) or not 0.0 <= float(score) <= 1.0):
            raise ValueError(f"{source}: detection {n} score {score!r} is not a finite value in [0, 1]")
        if not isinstance(label, str) or (vocab is not None and label not in vocab):
            raise ValueError(f"{source}: detection {n} phrase {label!r} is outside the vocabulary {list(vocab or [])}")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"{source}: detection {n} box {box!r} does not have four coordinates")
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            raise ValueError(f"{source}: detection {n} box {box!r} is not numeric") from None
        if not all(np.isfinite([x0, y0, x1, y1])) or x0 > x1 or y0 > y1:
            raise ValueError(f"{source}: detection {n} box {box!r} is non-finite or unordered")
        if image_size is not None:
            width, height = image_size
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                raise ValueError(f"{source}: detection {n} box {box!r} lies outside the {width}x{height} image")
    return preds


def image_digest(image):
    rgb = image.convert("RGB")
    payload = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_images(items):
    """Decode the images the kernel verified and refuse any whose pixels differ from its digest."""
    records = []
    for item in items:
        with Image.open(item["path"]) as source:
            source.load()
            image = source.convert("RGB")
        if image_digest(image) != item["pixel_sha256"]:
            raise RuntimeError(f"{item['id']}: decoded pixels differ from the notebook kernel's verified image")
        records.append({"id": item["id"], "image": image})
    return records


def public_version_matches(actual, expected):
    from packaging.version import Version
    return actual is not None and Version(str(actual)).public == Version(str(expected)).public


# ---- Immutable model provenance (former cell 48be2c06) ------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_and_verify(root, manifest):
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=manifest["modelId"],
                filename=entry["path"],
                revision=manifest["revision"],
                local_dir=str(root),
            )
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{manifest['modelId']} {entry['path']} size mismatch")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{manifest['modelId']} {entry['path']} SHA-256 mismatch")
    return {
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest["totalBytes"],
    }


# ---- Grounding DINO adapter (former cell 53f2f300) ------------------------------------------------------

def gdino_prompt_text(prompts):
    p = normalize_prompt_list(prompts)
    return " ".join(f"{x}." for x in p), p


def phrase_token_groups(input_ids):
    groups, current = [], []
    for position, token in enumerate(input_ids):
        if int(token) in GDINO_SPECIAL_TOKEN_IDS:
            if current:
                groups.append(current)
                current = []
        else:
            current.append(position)
    if current:
        groups.append(current)
    return groups


def gdino_predict(image, prompts, score_floor=None):
    score_floor = EVAL_SCORE_FLOOR if score_floor is None else score_floor
    text, vocab = gdino_prompt_text(prompts)
    inputs = gdino_processor(images=image.convert("RGB"), text=text, return_tensors="pt")
    if int(inputs["input_ids"].shape[1]) > GDINO_MAX_TEXT_TOKENS:
        raise ValueError("Grounding DINO combined prompt exceeds 256 tokens")
    groups = phrase_token_groups(inputs["input_ids"][0].tolist())
    if len(groups) != len(vocab):
        raise RuntimeError(f"Grounding DINO token groups {len(groups)} != phrases {len(vocab)}")
    model_inputs = inputs.to(DEVICE)
    with torch.inference_mode():
        outputs = gdino_model(**model_inputs)
    # Validate the raw output before thresholding, so a numerical failure cannot look like
    # "no objects found". Only active phrase-token positions are checked: positions past the text
    # (and special tokens) are legitimately masked with -inf by the model.
    logits, pred_boxes = outputs.logits, outputs.pred_boxes
    if (logits.ndim != 3 or pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4
            or tuple(logits.shape[:2]) != tuple(pred_boxes.shape[:2])):
        raise RuntimeError(
            f"Grounding DINO output shapes {tuple(logits.shape)} / {tuple(pred_boxes.shape)} are malformed"
        )
    active = sorted({position for group in groups for position in group})
    if active[-1] >= logits.shape[-1]:
        raise RuntimeError("Grounding DINO logits do not cover every phrase token")
    if not torch.isfinite(logits[0][:, active]).all():
        raise FloatingPointError(
            "Grounding DINO produced non-finite logits at active phrase tokens; "
            "this is a numerical failure, not an empty detection result"
        )
    if not torch.isfinite(pred_boxes).all() or ((pred_boxes < 0) | (pred_boxes > 1)).any():
        raise FloatingPointError("Grounding DINO produced non-finite or out-of-range normalized boxes")
    probs = outputs.logits[0].sigmoid().cpu()
    phrase_scores = torch.stack(
        [probs[:, group].max(dim=1).values for group in groups], dim=1
    )
    score, index = phrase_scores.max(dim=1)
    boxes = outputs.pred_boxes[0].detach().cpu()
    width, height = image.size
    xyxy = torch.stack([
        (boxes[:,0] - boxes[:,2]/2) * width,
        (boxes[:,1] - boxes[:,3]/2) * height,
        (boxes[:,0] + boxes[:,2]/2) * width,
        (boxes[:,1] + boxes[:,3]/2) * height,
    ], dim=1)
    xyxy[:,0::2].clamp_(0, float(width))
    xyxy[:,1::2].clamp_(0, float(height))
    keep = score >= float(score_floor)
    out = [
        (float(s), vocab[int(k)], [float(v) for v in b])
        for s,k,b in zip(score[keep], index[keep], xyxy[keep], strict=True)
    ]
    out.sort(key=lambda x: -x[0])
    return validate_detections(out, image.size, vocab, "Grounding DINO")


def gdino_native_visual(image, prompts):
    text, vocab = gdino_prompt_text(prompts)
    inputs = gdino_processor(images=image.convert("RGB"), text=text, return_tensors="pt")
    model_inputs = inputs.to(DEVICE)
    with torch.inference_mode():
        outputs = gdino_model(**model_inputs)
    result = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=GDINO_BOX_THRESHOLD,
        text_threshold=GDINO_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]
    labels = result.get("text_labels") or result.get("labels")
    # Display-only path: native labels may span merged phrases and boxes are not clipped, so only
    # finite scores and ordered finite boxes are required here.
    return validate_detections([
        (float(score), str(label).strip().rstrip(".").lower(), [float(v) for v in box.tolist()])
        for box, score, label in zip(result["boxes"], result["scores"], labels, strict=True)
    ], None, None, "Grounding DINO native view")


def load_gdino():
    global gdino_processor, gdino_model
    gdino_processor = AutoProcessor.from_pretrained(
        str(GDINO_DIR),
        revision=GDINO_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(GDINO_DIR),
        revision=GDINO_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    ).to(DEVICE).eval()
    for p in gdino_model.parameters():
        p.requires_grad_(False)
    p = None  # Release the final parameter alias before the model cleanup stage.


# ---- OWLv2 adapter (former cell 2f66f39b) -----------------------------------------------------------

def owlv2_validate_tokens(prompts):
    queries = normalize_prompt_list(prompts)
    encoded = owlv2_processor.tokenizer(
        queries,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    lengths = [len(x) for x in encoded["input_ids"]]
    if any(n > OWLV2_MAX_TOKENS for n in lengths):
        raise ValueError(f"OWLv2 phrase exceeds {OWLV2_MAX_TOKENS} tokens: {list(zip(queries,lengths))}")
    return queries


def owlv2_predict(image, prompts, threshold=None):
    threshold = EVAL_SCORE_FLOOR if threshold is None else threshold
    queries = owlv2_validate_tokens(prompts)
    inputs = owlv2_processor(text=[queries], images=image.convert("RGB"), return_tensors="pt")
    model_inputs = inputs.to(DEVICE)
    with torch.inference_mode():
        outputs = owlv2_model(**model_inputs)
    logits, pred_boxes = outputs.logits, outputs.pred_boxes
    if (logits.ndim != 3 or pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4
            or tuple(logits.shape[:2]) != tuple(pred_boxes.shape[:2]) or logits.shape[-1] < len(queries)):
        raise RuntimeError(f"OWLv2 output shapes {tuple(logits.shape)} / {tuple(pred_boxes.shape)} are malformed")
    if not torch.isfinite(logits[0, :, :len(queries)]).all() or not torch.isfinite(pred_boxes).all():
        raise FloatingPointError(
            "OWLv2 produced non-finite logits or boxes; this is a numerical failure, not an empty result"
        )
    result = owlv2_processor.post_process_grounded_object_detection(
        outputs,
        threshold=float(threshold),
        target_sizes=[image.size[::-1]],
        text_labels=[queries],
    )[0]
    # OWLv2 pads each image to a square before prediction; the upstream post-processing rescales by
    # the longer side, so a box can extend past the shorter edge. Clip to the original image: the
    # same coordinate policy as the Grounding DINO adapter.
    width, height = image.size
    out = []
    for box, label, score in zip(result["boxes"], result["text_labels"], result["scores"], strict=True):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        clipped = [min(max(x0, 0.0), width), min(max(y0, 0.0), height),
                   min(max(x1, 0.0), width), min(max(y1, 0.0), height)]
        out.append((float(score), str(label).strip().rstrip(".").lower(), clipped))
    out.sort(key=lambda x: -x[0])
    if len(out) > 3600:
        raise RuntimeError("OWLv2 returned more than 3,600 patch candidates")
    return validate_detections(out, image.size, queries, "OWLv2")


def load_owlv2():
    global owlv2_processor, owlv2_model
    owlv2_processor = Owlv2Processor.from_pretrained(
        str(OWLV2_DIR),
        revision=OWLV2_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    owlv2_model = Owlv2ForObjectDetection.from_pretrained(
        str(OWLV2_DIR),
        revision=OWLV2_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    ).to(DEVICE).eval()
    for p in owlv2_model.parameters():
        p.requires_grad_(False)
    p = None  # Release the final parameter alias before the model cleanup stage.


# ---- Shared frozen-inference run (former loops of cells 53f2f300 and 2f66f39b) -------------------------

def run_model(name, predict, test_records, prompt_records, canonical_prompts, visual):
    """Warm-up, timed evaluation pass, native-threshold views, prompt experiments and optional sweep."""
    # Warm-up (excluded from timings).
    _ = predict(test_records[0]["image"], canonical_prompts)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    predictions = []
    times = []

    for i, record in enumerate(test_records):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        preds = predict(record["image"], canonical_prompts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        predictions.append(preds)
        if (i+1) % 20 == 0:
            print(f"{name}: {i+1}/{len(test_records)}", flush=True)

    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    # Native-threshold visual predictions for deterministic examples.
    visual_predictions = {r["id"]: visual(r["image"], canonical_prompts) for r in prompt_records}

    # Prompt-variant and prompt-order observations while the model is resident.
    prompt_results = []
    if RUN_PROMPT_EXPERIMENT:
        for variant_name, variant_prompts in PROMPT_VARIANTS.items():
            alias = {k.lower(): v for k,v in PROMPT_VARIANT_TO_CANONICAL[variant_name].items()}
            for record in prompt_records:
                preds = predict(record["image"], variant_prompts)
                normalized = [(s, alias[label.lower()], b) for s,label,b in preds if label.lower() in alias]
                prompt_results.append((variant_name, record["id"], normalized))
        for record in prompt_records[:1]:
            canonical_order = predict(record["image"], canonical_prompts)
            reordered = predict(record["image"], REORDERED_PROMPTS)
            prompt_results.append(("order_canonical", record["id"], canonical_order))
            prompt_results.append(("order_reordered", record["id"], reordered))

    threshold_sweep = []
    if RUN_THRESHOLD_SWEEP:
        for floor in (0.03, 0.05, 0.10, 0.20):
            preds = [predict(r["image"], canonical_prompts, floor) for r in prompt_records]
            threshold_sweep.append((floor, preds))

    return {
        "predictions": predictions,
        "times": times,
        "peak_gpu_memory_bytes": peak_gpu,
        "visual_predictions": visual_predictions,
        "prompt_results": prompt_results,
        "threshold_sweep": threshold_sweep,
    }


# ---- BYOD inference (former predict_byod of cell 3f9f36da) -------------------------------------------

def release_byod_models():
    for name in ('gdino_model','gdino_processor','owlv2_model','owlv2_processor','gp','gm','op','om','p'):
        globals().pop(name,None)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def predict_byod(records,prompts):
    global gdino_processor,gdino_model,owlv2_processor,owlv2_model
    release_byod_models()
    try:
        # Both token contracts are checked before loading either model.
        gdino_processor=AutoProcessor.from_pretrained(str(GDINO_DIR),revision=GDINO_REVISION,local_files_only=True,trust_remote_code=False)
        owlv2_processor=Owlv2Processor.from_pretrained(str(OWLV2_DIR),revision=OWLV2_REVISION,local_files_only=True,trust_remote_code=False)
        text,vocab=gdino_prompt_text(prompts)
        ids=gdino_processor.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
        if len(ids)>GDINO_MAX_TEXT_TOKENS or len(phrase_token_groups(ids))!=len(vocab):
            raise ValueError('Grounding DINO vocabulary exceeds token limit or phrase boundaries')
        owlv2_validate_tokens(prompts)
        gdino_model=AutoModelForZeroShotObjectDetection.from_pretrained(str(GDINO_DIR),revision=GDINO_REVISION,local_files_only=True,trust_remote_code=False).to(DEVICE).eval()
        gd=[gdino_predict(r['image'],prompts) for r in records]
        del gdino_model,gdino_processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        owlv2_model=Owlv2ForObjectDetection.from_pretrained(str(OWLV2_DIR),revision=OWLV2_REVISION,local_files_only=True,trust_remote_code=False).to(DEVICE).eval()
        owl=[owlv2_predict(r['image'],prompts) for r in records]
        return gd,owl
    except Exception as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        release_byod_models()


# ---- Stages ------------------------------------------------------------------------------------------

def stage_runtime(job):
    """Report the isolated interpreter, library versions and accelerator; refuse drift from the pins."""
    load_libraries()
    import importlib.metadata as importlib_metadata
    installed = {name: importlib_metadata.version(name) for name in PINS}
    bad = {k: (installed[k], v) for k, v in PINS.items() if not public_version_matches(installed[k], v)}
    if bad:
        raise RuntimeError(f"Isolated environment does not match the pinned versions: {bad}")
    import huggingface_hub
    import torchvision
    import transformers
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": np.__version__,
        "pillow": importlib_metadata.version("pillow"),
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
    }
    if torch.cuda.is_available():
        runtime["gpu_name"] = torch.cuda.get_device_name(0)
        runtime["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return {"runtime": runtime, "executable": sys.executable}


def stage_weights(job):
    """Stage both pinned snapshots and verify every file's size and SHA-256."""
    load_libraries()
    return {
        "gdino": stage_and_verify(Path(job["gdino"]["dir"]), job["gdino"]["manifest"]),
        "owlv2": stage_and_verify(Path(job["owlv2"]["dir"]), job["owlv2"]["manifest"]),
    }


def stage_gdino(job):
    load_libraries()
    test_records = load_images(job["test_images"])
    by_id = {r["id"]: r for r in test_records}
    prompt_records = [by_id[i] for i in job["prompt_image_ids"]]
    canonical_prompts = job["canonical_prompts"]
    t0 = time.perf_counter()
    load_gdino()
    load_seconds = time.perf_counter() - t0
    parameter_count = sum(p.numel() for p in gdino_model.parameters())
    # Validate token ceiling before the full run.
    probe_inputs = gdino_processor(
        images=test_records[0]["image"],
        text=gdino_prompt_text(canonical_prompts)[0],
        return_tensors="pt",
    )
    if int(probe_inputs["input_ids"].shape[1]) > GDINO_MAX_TEXT_TOKENS:
        raise RuntimeError("Canonical Grounding DINO prompt exceeds token ceiling")
    result = run_model("Grounding DINO", gdino_predict, test_records, prompt_records, canonical_prompts,
                       gdino_native_visual)
    return {"load_seconds": load_seconds, "parameter_count": parameter_count, "device": DEVICE, **result}


def stage_owlv2(job):
    load_libraries()
    test_records = load_images(job["test_images"])
    by_id = {r["id"]: r for r in test_records}
    prompt_records = [by_id[i] for i in job["prompt_image_ids"]]
    canonical_prompts = job["canonical_prompts"]
    t0 = time.perf_counter()
    load_owlv2()
    load_seconds = time.perf_counter() - t0
    parameter_count = sum(p.numel() for p in owlv2_model.parameters())
    owlv2_validate_tokens(canonical_prompts)
    result = run_model("OWLv2", owlv2_predict, test_records, prompt_records, canonical_prompts,
                       lambda image, prompts: owlv2_predict(image, prompts, OWLV2_THRESHOLD))
    return {"load_seconds": load_seconds, "parameter_count": parameter_count, "device": DEVICE, **result}


def stage_byod(job):
    load_libraries()
    records = load_images(job["images"])
    gd, owl = predict_byod(records, job["prompts"])
    return {"grounding_dino": gd, "owlv2": owl}


STAGES = {
    "runtime": stage_runtime,
    "weights": stage_weights,
    "gdino": stage_gdino,
    "owlv2": stage_owlv2,
    "byod": stage_byod,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    configure(job)
    result = STAGES[args.stage](job)
    payload = {"format": STAGE_FORMAT, "stage": args.stage, **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON: a NaN/Infinity that slipped past validation fails here instead of being written.
    args.out.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
