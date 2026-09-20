from __future__ import annotations

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width
import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "grounding-dino-tiny"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Thresholds: the values the upstream model card's usage example passes to
# post_process_grounded_object_detection (box_threshold=0.4, text_threshold=0.3). Both gate a sigmoid
# score that is not calibrated; the deployment owns tuning them on its own labelled data.
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
# Input ceilings. The processor resizes to shortest edge 800 / longest edge 1333 (preprocessor_config.json),
# so image cost is bounded; the text side is bounded by config.json "max_text_len": 256 tokens.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
MAX_PROMPT_CHARS = 48
MAX_TEXT_TOKENS = 256
PARAMETER_COUNT = 172_249_090
DECODER_PARAMETERS = 11_187_460  # the cross-modality decoder, its reference-point head and the box / contrastive heads
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
WEIGHT_FILE = "model.safetensors"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
SCORE_FLOOR = 0.05  # queries below this per-phrase probability are dropped before the corpus measures
GRAD_CLIP = 0.1  # the upstream training recipe's max gradient norm
_TRAINABLE_PREFIXES = ("model.decoder.", "bbox_embed.", "class_embed.")
_SPECIAL_TOKEN_IDS = (101, 102, 1012, 1029, 0)  # [CLS], [SEP], ".", "?", [PAD] — the phrase delimiters


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block for any caller-side mAP."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def format_prompts(prompts: Sequence[str]) -> str:
    """Validate a list of phrases and join them the way the upstream card instructs: lowercase, each
    phrase terminated by a period, separated by a space ("a cat. a remote control.")."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a list of phrases, not a single string")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"prompt count {len(prompts)} outside 1..MAX_PROMPTS {MAX_PROMPTS}")
    cleaned: list[str] = []
    for phrase in prompts:
        if not isinstance(phrase, str):
            raise TypeError(f"prompt must be str, got {type(phrase).__name__}")
        text = phrase.strip().rstrip(".").strip().lower()
        if not text:
            raise ValueError("prompt phrases must not be empty")
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"prompt {text[:12]!r}... is {len(text)} chars > MAX_PROMPT_CHARS {MAX_PROMPT_CHARS}"
            )
        cleaned.append(text)
    return " ".join(f"{text}." for text in cleaned)


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL.Image.Image (any mode, converted to RGB) plus 1..MAX_PROMPTS free-text phrases",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "prompts": [1, MAX_PROMPTS],
    "prompt_chars": [1, MAX_PROMPT_CHARS],
    "prompt_tokens": [1, MAX_TEXT_TOKENS],
    "box_threshold": [0.0, 1.0],
    "text_threshold": [0.0, 1.0],
    "preprocessing": (
        "image converted to RGB; phrases stripped, lower-cased, period-terminated and space-joined "
        "(format_prompts); the processor resizes to shortest edge 800 / longest edge 1333 and returned "
        "boxes are mapped back to input pixels"
    ),
}


def _check_inputs(
    image: Any, prompts: Any, box_threshold: Any, text_threshold: Any
) -> tuple[Image.Image, str, float, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``detect`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    text = format_prompts(prompts)
    box_t = _check_threshold("box_threshold", box_threshold)
    text_t = _check_threshold("text_threshold", text_threshold)
    return rgb, text, box_t, text_t


def validate_inputs(
    image: Image.Image,
    prompts: Sequence[str],
    *,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``detect`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, text, box_t, text_t = _check_inputs(image, prompts, box_threshold, text_threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "image-0",
                "mode": image.mode,
                "size": list(image.size),
                "n_prompts": len(prompts),
            }
        ],
        "prompt_text": text,
        "box_threshold": box_t,
        "text_threshold": text_t,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[float]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth_boxes`` (phrase -> xyxy reference box) the report carries one ``box_iou``
    entry per reference as sample-sanity geometry evidence; without them the verdict is
    ``not-measurable`` and the report says what labelled data would make the task measurable.
    """
    detections = list(result["detections"])
    base = {
        "task": "zero-shot (open-vocabulary, text-prompted) object detection",
        "decision_rule": (
            "a box survives when its best grounding score reaches box_threshold and its phrase "
            "tokens reach text_threshold; the score is an uncalibrated sigmoid, not a probability"
        ),
        "box_threshold": result.get("box_threshold", BOX_THRESHOLD),
        "text_threshold": result.get("text_threshold", TEXT_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth boxes were supplied for the evaluated image",
            "needs": (
                "labelled boxes on your own images with a phrase vocabulary matching the prompts, "
                "scored per object with box_iou and aggregated into precision/recall or mean average "
                "precision at a stated IoU threshold; no such labelled set ships with this repository"
            ),
        }
    metrics = []
    for phrase, box in ground_truth_boxes.items():
        ious = [box_iou(det["box"], box) for det in detections]
        best = max(range(len(ious)), key=ious.__getitem__) if ious else None
        metrics.append(
            {
                "id": "box_iou",
                "reference": phrase,
                "value": ious[best] if best is not None else 0.0,
                "matched_label": detections[best]["label"] if best is not None else None,
                "label_matches_reference": (detections[best]["label"] == phrase)
                if best is not None
                else False,
                "estimation": "one reference box per phrase on a single scene, no dispersion estimate",
            }
        )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial sample; geometry sanity evidence, "
            "not a detection benchmark"
        ),
        "needs": (
            "a labelled box set from the deployment domain with a matching phrase vocabulary for any "
            "mean-average-precision or precision/recall claim"
        ),
    }


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHT_FILE), None)


def _trainable_names(model: Any) -> list[str]:
    """The cross-modality decoder (its layers and reference-point head) plus the box and contrastive heads; the
    image backbone, the text backbone, the feature enhancer and the query selection stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


def _phrase_token_groups(input_ids: Sequence[int]) -> list[list[int]]:
    """Token positions of each phrase in a formatted prompt, in prompt order: the runs between delimiter tokens
    (the same rule the upstream loss uses to map labels to tokens)."""
    groups: list[list[int]] = []
    current: list[int] = []
    for position, token in enumerate(input_ids):
        if int(token) in _SPECIAL_TOKEN_IDS:
            if current:
                groups.append(current)
                current = []
        else:
            current.append(position)
    if current:
        groups.append(current)
    return groups


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError("artifact tensors must all belong to the decoder or the box / contrastive heads")
    prompts = (manifest.get("adapter") or {}).get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError("artifact adapter.prompts must list the phrases the adapter was trained on")


@dataclass
class GroundingDINOPipeline:
    """Text-prompted (open-vocabulary) object detection over the pinned Grounding DINO tiny checkpoint."""

    _runner: Callable[[Image.Image, str, float, float], list[dict[str, Any]]]
    device: str
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GroundingDINOPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()
        for param in model.parameters():
            param.requires_grad_(False)

        def runner(image: Image.Image, text: str, box_threshold: float, text_threshold: float) -> list[dict]:
            inputs = processor(images=image, text=text, return_tensors="pt")
            n_tokens = int(inputs["input_ids"].shape[1])
            if n_tokens > MAX_TEXT_TOKENS:
                raise ValueError(f"prompt text is {n_tokens} tokens > MAX_TEXT_TOKENS {MAX_TEXT_TOKENS}")
            inputs = inputs.to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            labels = result.get("text_labels") or result.get("labels")
            return [
                {"box": [float(v) for v in box.tolist()], "label": str(label), "score": float(score)}
                for box, label, score in zip(result["boxes"], labels, result["scores"], strict=True)
            ]

        return cls(runner, resolved_device, model, processor, _weight_digest(root))

    def detect(
        self,
        image: Image.Image,
        prompts: Sequence[str],
        *,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> dict[str, Any]:
        """Detect the phrases in `prompts`; boxes are xyxy pixel coordinates in the input image."""
        rgb, text, box_t, text_t = _check_inputs(image, prompts, box_threshold, text_threshold)
        detections = self._runner(rgb, text, box_t, text_t)
        for det in detections:
            if set(det) != {"box", "label", "score"} or len(det["box"]) != 4:
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "prompt_text": text,
            "box_threshold": box_t,
            "text_threshold": text_t,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained for evaluate/adapt")
        return self._model, self._processor


    def predict_boxes(self, image: Image.Image, prompts: Sequence[str], *, score_floor: float = SCORE_FLOOR) -> list[tuple[float, str, list[float]]]:
        """Every query scored per phrase — the maximum over that phrase's tokens of the query's sigmoid token logits —
        and kept as `(score, phrase, [x0, y0, x1, y1])` with its best phrase when the score reaches `score_floor`. This
        is the single-phrase-per-box scoring the corpus measures use; `detect` keeps the processor's text-threshold
        labelling instead."""
        model, processor = self._require_model()  # refuse before importing torch
        import torch

        rgb, text, _b, _t = _check_inputs(image, prompts, BOX_THRESHOLD, TEXT_THRESHOLD)
        if not isinstance(score_floor, int | float) or not 0.0 <= float(score_floor) <= 1.0:
            raise ValueError("score_floor must be in [0, 1]")
        vocabulary = [p.strip().rstrip(".").strip().lower() for p in prompts]
        inputs = processor(images=rgb, text=text, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) > MAX_TEXT_TOKENS:
            raise ValueError(f"prompt text is {int(inputs['input_ids'].shape[1])} tokens > MAX_TEXT_TOKENS {MAX_TEXT_TOKENS}")
        groups = _phrase_token_groups(inputs["input_ids"][0].tolist())
        if len(groups) != len(vocabulary):
            raise RuntimeError(f"tokeniser produced {len(groups)} phrase groups for {len(vocabulary)} prompts")
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        probs = outputs.logits[0].sigmoid().cpu()
        phrase_scores = torch.stack([probs[:, group].max(dim=1).values for group in groups], dim=1)
        score, index = phrase_scores.max(dim=1)
        boxes = outputs.pred_boxes[0].cpu()
        width, height = rgb.size
        xyxy = torch.stack(
            [(boxes[:, 0] - boxes[:, 2] / 2) * width, (boxes[:, 1] - boxes[:, 3] / 2) * height, (boxes[:, 0] + boxes[:, 2] / 2) * width, (boxes[:, 1] + boxes[:, 3] / 2) * height],
            dim=1,
        ).clamp(min=0.0)
        xyxy[:, 2].clamp_(max=float(width))
        xyxy[:, 3].clamp_(max=float(height))
        keep = score >= float(score_floor)
        out = [(float(s), vocabulary[int(k)], [float(v) for v in b]) for s, k, b in zip(score[keep], index[keep], xyxy[keep], strict=True)]
        return sorted(out, key=lambda item: -item[0])


    def evaluate(self, records: Sequence[Mapping[str, Any]], prompts: Sequence[str], *, score_floor: float = SCORE_FLOOR, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Predict every validated record's boxes for the prompt vocabulary and score them with
        `metrics.detection_metrics` (AP50 / AP75 per phrase and their means)."""
        from .metrics import detection_metrics
        from .samples import validate_dataset

        manifest = validate_dataset(records, prompts, min_records=1, max_records=MAX_EVAL_RECORDS)
        checked, vocabulary = manifest["records"], manifest["prompts"]
        started = time.perf_counter()
        predictions = []
        for i, record in enumerate(checked):
            predictions.append(self.predict_boxes(record["image"], vocabulary, score_floor=score_floor))
            if progress is not None:
                progress(i + 1, len(checked))
        metrics = detection_metrics(predictions, checked, vocabulary)
        metrics.update(
            {
                "prompts": vocabulary,
                "score_floor": float(score_floor),
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics


    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        prompts: Sequence[str],
        *,
        epochs: int = 6,
        lr: float = 5e-5,
        batch_size: int = 4,
        seed: int = 0,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the cross-modality decoder and the box / contrastive heads on labelled boxes: every
        image is paired with the formatted prompt vocabulary, each labelled box with the index of its phrase, and the
        upstream Grounding DINO criterion — Hungarian matching, sigmoid-focal alignment of queries to phrase tokens, L1
        and generalised-IoU box terms — scores the 900 queries against them; the image backbone, the text backbone, the
        feature enhancer and the query selection stay frozen. AdamW (no weight decay), gradient clipping at `GRAD_CLIP`,
        seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen model's validation metrics; the epoch
        with the highest validation mAP50 is kept (the final one without a validation split). On any exception the
        frozen weights are restored."""
        model, processor = self._require_model()  # refuse before importing torch
        import torch

        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        train_manifest = validate_dataset(train, prompts)
        train_checked, vocabulary = train_manifest["records"], train_manifest["prompts"]
        val_checked = validate_dataset(val, vocabulary, min_records=1)["records"] if val is not None else None
        text = format_prompts(vocabulary)
        names = _trainable_names(model)
        name_set = set(names)
        device = torch.device(self.device)
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_adapter = self.adapter
        cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # repeatable on one device
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            result = self.evaluate(val_checked, vocabulary)
            return {"map50": result["map50"], "map75": result["map75"], "recall50": result["recall50"], "n": result["n"]}

        def _targets(record: Mapping[str, Any]) -> dict[str, Any]:
            width, height = record["image"].size
            boxes = torch.tensor(record["boxes"], dtype=torch.float32)
            cxcywh = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / width, (boxes[:, 1] + boxes[:, 3]) / 2 / height, (boxes[:, 2] - boxes[:, 0]) / width, (boxes[:, 3] - boxes[:, 1]) / height], dim=1)
            return {"class_labels": torch.tensor([vocabulary.index(label) for label in record["labels"]], dtype=torch.long, device=device), "boxes": cxcywh.to(device)}

        try:
            for param in model.parameters():
                param.requires_grad_(False)
            params = []
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("map50", -1.0)
            best_state = frozen_state
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                for module in model.modules():  # normalisation statistics stay frozen: small batches, parameters-only adapter
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                order = list(range(len(train_checked)))
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[k] for k in order[start : start + batch_size]]
                    inputs = processor(images=[r["image"] for r in batch], text=[text] * len(batch), return_tensors="pt", padding=True).to(device)
                    outputs = model(**inputs, labels=[_targets(r) for r in batch])
                    loss = outputs.loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["map50"] > best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("map50", -1.0)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            model.load_state_dict(best_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "prompts": list(vocabulary),
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "highest validation mAP50" if val_checked is not None else "final epoch (no validation split)",
            "loss": "upstream Grounding DINO criterion: Hungarian matching, sigmoid-focal query-to-phrase alignment (weight 1), L1 (5) and generalised-IoU (2) box terms",
            "lr": float(lr),
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)


    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests, the prompt vocabulary
        and the training configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out


    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the decoder and the heads."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)


    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GroundingDINOPipeline:
        """Load the verified base snapshot, then overlay the adapter (verified before deserialising)."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
