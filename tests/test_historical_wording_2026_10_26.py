"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Kaggle T4 of blob dc1218be (2026-09-20); Colab T4 of blobs b539602b (2026-10-04),
81eb95a4 and a04b58a2 (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "grounding_dino_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_generalisations_are_gone():
    md = _markdown()
    for stale in ("epochs take about 15 minutes", "the whole session about 21 to 25 minutes",
                  "Expect the same shape over six epochs", "Your run can land a few hundredths away",
                  "the adapted panels should box", "the smallest gain in every run", "0.951 to 0.957"):
        assert stale not in md, stale


def test_confirming_run_of_a04b58a2_is_recorded():
    md = _markdown()
    assert "`81eb95a4` of revision `2fb41e2` and `a04b58a2` of revision `03f4a1a`" in md
    assert "**0.570, 0.608, 0.632, 0.630, 0.610 and 0.614**" in md
    assert "recall 0.293 → 0.951 to 0.964" in md and "0.285 → 0.695 to 0.744" in md
    assert "862 s to 952.8 s" in md and "between 21.3 and 25.1 minutes" in md
