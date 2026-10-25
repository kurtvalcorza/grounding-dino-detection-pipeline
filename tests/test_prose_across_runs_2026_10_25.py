"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Runs: Colab T4 of blobs b539602b (a2bdaaf, 2026-10-04) and 81eb95a4 (2fb41e2, 2026-10-25);
Kaggle T4 of blob dc1218be (2026-09-20).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "grounding_dino_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_untraceable_or_single_run_claims_are_gone():
    md = _markdown()
    for stale in ("0.319 / 0.426 / 0.516 / 0.577 / 0.637 / 0.692", "with `box_iou` 0.964 and 0.944",
                  "(0.931 → 0.910) and `rectangle` (0.695 → 0.517), with `box_iou` 0.883 and 0.924 ",
                  "In the recorded T4 run", "931 s on the T4 record", "almost as well as the 0.931 one"):
        assert stale not in md, stale


def test_box_iou_pairs_match_the_printed_output_and_ranges_name_the_runs():
    md = _markdown()
    assert "`red circle` scored 0.931 with `box_iou` 0.944" in md
    assert "`rectangle` 0.695 with `box_iou` 0.964" in md
    assert "0.341 / 0.422 / 0.525 / 0.568 / 0.637 / 0.695" in md
    assert "0.324 / 0.425 / 0.519 / 0.574 / 0.623 / 0.672" in md
    assert "`platelet` 0.000 → 0.289 to 0.366" in md
    assert "(0.931 → 0.910 and 0.843)" in md and "(0.695 → 0.517 and 0.524)" in md
