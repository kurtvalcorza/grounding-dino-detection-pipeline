"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #14 head 728fea7.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "grounding_dino_detection_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'few hundredths away from the quoted one is that spread',
        'The difference from the default (about 0.6)',
        'a CPU runtime would take hours (Swin-T',
        'a CPU runtime would take hours. The locked install',
        'training results by a few hundredths',
        'move only in the third decimal between GPUs',
        'vary by a few hundredths between runs',
    ):
        assert gone not in md, gone
    for kept in (
        'can be that spread, not a fault.',
        '0.570 to 0.632 in the six recorded GPU runs',
        'would take hours (an estimate: Swin-T',
        'would take hours (an estimate). The locked install',
        'the recorded training runs differed from each other',
        'lists six test mAP50s from 0.570 to 0.632); yours may differ.',
    ):
        assert kept in md, kept
