"""Corpus-level detection measures and two non-adapted baselines, in numpy.

``detection_metrics`` scores one prediction list per record — ``(score, phrase, [x0, y0, x1, y1])`` triples — against
``record['boxes']`` / ``record['labels']``: **AP50** and **AP75**, the Pascal-VOC average precision (all-point
interpolation, greedy one-to-one matching in score order) at IoU 0.5 and 0.75, per phrase and averaged (**mAP50**,
**mAP75**), plus the recall at IoU 0.5 over every prediction kept. The baselines answer without adaptation: a fixed grid
of boxes of the training split's mean size carrying the majority phrase, or the frozen model's own detections for a
generic prompt relabelled to the majority phrase.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .pipeline import box_iou

METRIC_DEFINITIONS = {
    "map50": "mean over the prompt vocabulary of the Pascal-VOC average precision (all-point interpolation) at IoU >= 0.5; in 0..1, higher is better",
    "map75": "the same average precision at IoU >= 0.75 — the stricter localisation bar",
    "recall50": "share of labelled boxes matched by a prediction of the same phrase at IoU >= 0.5, over every prediction kept; in 0..1",
}
IOU_THRESHOLDS = (0.5, 0.75)


def _average_precision(predictions: Sequence[Sequence[tuple[float, str, Sequence[float]]]], records: Sequence[Mapping[str, Any]], phrase: str, iou_threshold: float) -> tuple[float, int, int]:
    """AP of one phrase over the corpus: predictions sorted by score, each matched greedily to the best unmatched
    labelled box of that phrase in its image at IoU >= `iou_threshold`."""
    detections = []
    n_gt = 0
    for index, (preds, record) in enumerate(zip(predictions, records, strict=True)):
        n_gt += sum(1 for label in record["labels"] if label == phrase)
        detections.extend((float(score), index, box) for score, label, box in preds if label == phrase)
    detections.sort(key=lambda item: -item[0])
    matched: dict[int, set[int]] = {}
    hits = []
    for _score, index, box in detections:
        taken = matched.setdefault(index, set())
        record = records[index]
        best_iou, best_j = 0.0, None
        for j, (gt_box, gt_label) in enumerate(zip(record["boxes"], record["labels"], strict=True)):
            if gt_label != phrase or j in taken:
                continue
            value = box_iou(box, gt_box)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_j is not None and best_iou >= iou_threshold:
            taken.add(best_j)
            hits.append(1)
        else:
            hits.append(0)
    if n_gt == 0 or not hits:
        return 0.0, n_gt, int(sum(hits))
    tp = np.cumsum(np.asarray(hits, dtype=np.float64))
    recall = tp / n_gt
    precision = tp / np.arange(1, len(hits) + 1)
    m_recall = np.concatenate([[0.0], recall, [1.0]])
    m_precision = np.concatenate([[0.0], precision, [0.0]])
    for k in range(len(m_precision) - 2, -1, -1):
        m_precision[k] = max(m_precision[k], m_precision[k + 1])
    steps = np.where(m_recall[1:] != m_recall[:-1])[0]
    return float(np.sum((m_recall[steps + 1] - m_recall[steps]) * m_precision[steps + 1])), n_gt, int(tp[-1])


def detection_metrics(predictions: Sequence[Sequence[tuple[float, str, Sequence[float]]]], records: Sequence[Mapping[str, Any]], prompts: Sequence[str]) -> dict[str, Any]:
    """Score one prediction list per record against its labelled boxes: AP50 / AP75 per phrase, their means and the
    recall at IoU 0.5. Raises when the lengths differ or nothing is scored."""
    if len(predictions) != len(records) or not predictions:
        raise ValueError("predictions and records must be non-empty and the same length")
    for preds in predictions:
        for item in preds:
            if len(item) != 3 or len(item[2]) != 4:
                raise ValueError("each prediction must be (score, phrase, [x0, y0, x1, y1])")
    per_phrase: dict[str, dict[str, float | int]] = {}
    for phrase in prompts:
        ap50, n_gt, matched = _average_precision(predictions, records, phrase, 0.5)
        ap75, _n, _m = _average_precision(predictions, records, phrase, 0.75)
        per_phrase[phrase] = {"n": n_gt, "ap50": ap50, "ap75": ap75, "recall50": (matched / n_gt) if n_gt else 0.0, "n_predictions": sum(1 for preds in predictions for _s, label, _b in preds if label == phrase)}
    total_gt = sum(v["n"] for v in per_phrase.values())
    return {
        "n": len(records),
        "n_boxes": total_gt,
        "map50": float(np.mean([v["ap50"] for v in per_phrase.values()])),
        "map75": float(np.mean([v["ap75"] for v in per_phrase.values()])),
        "recall50": float(sum(v["recall50"] * v["n"] for v in per_phrase.values()) / total_gt) if total_gt else 0.0,
        "n_predictions": sum(len(preds) for preds in predictions),
        "per_category": per_phrase,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def majority_phrase(train: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for record in train:
        for label in record["labels"]:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("train must hold at least one labelled box")
    return max(counts, key=lambda k: (counts[k], k))


def grid_prior_baseline(train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], prompts: Sequence[str]) -> dict[str, Any]:
    """No model at all: a grid of boxes of the training split's mean box size tiled over each image with a constant
    score, every one carrying the majority phrase — the floor a detector must clear."""
    widths, heights = [], []
    for record in train:
        for x0, y0, x1, y1 in record["boxes"]:
            widths.append(x1 - x0)
            heights.append(y1 - y0)
    if not widths:
        raise ValueError("train must hold at least one labelled box")
    mean_w, mean_h = float(np.mean(widths)), float(np.mean(heights))
    phrase = majority_phrase(train)
    predictions = []
    for record in records:
        width, height = record["image"].size
        boxes = []
        y = 0.0
        while y + mean_h <= height + 1e-6:
            x = 0.0
            while x + mean_w <= width + 1e-6:
                boxes.append((1.0, phrase, [x, y, min(x + mean_w, width), min(y + mean_h, height)]))
                x += mean_w
            y += mean_h
        predictions.append(boxes)
    return {**detection_metrics(predictions, records, prompts), "baseline": f"a grid of {mean_w:.0f}x{mean_h:.0f} px boxes labelled {phrase!r}"}


def majority_relabel_baseline(predictions: Sequence[Sequence[tuple[float, str, Sequence[float]]]], train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], prompts: Sequence[str]) -> dict[str, Any]:
    """Boxes from any predictor (the tutorial uses the frozen model prompted with one generic word) with every phrase
    replaced by the training split's majority phrase — what localisation alone buys without grounding the vocabulary."""
    phrase = majority_phrase(train)
    relabelled = [[(score, phrase, box) for score, _label, box in preds] for preds in predictions]
    return {**detection_metrics(relabelled, records, prompts), "baseline": f"every predicted box relabelled {phrase!r}"}
