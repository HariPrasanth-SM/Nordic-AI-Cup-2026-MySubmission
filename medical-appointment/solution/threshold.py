"""Threshold logic, kept as pure functions with zero ML dependencies so it's
trivially unit-testable and reusable from scripts/calibrate.py.

Two things live here:

1. ``dynamic_yes_threshold`` -- the metric-aware formula derived from the
   official score (Score = 0.4*Accuracy + 0.6*mean_tIoU, exactly balanced
   yes/no): answer yes iff p(yes) > 1 / (2 + 3*t_hat), where t_hat is the
   tIoU you'd expect to achieve if the true answer is yes. This assumes
   p(yes) is an actually-calibrated probability. The similarity verifier in
   experiments 1-2 does *not* produce one (cosine similarity isn't a
   probability), so it doesn't use this function -- it uses a single
   grid-searched constant instead (``ACCURACY_WEIGHT``/``TIOU_WEIGHT`` below
   plus ``official_score``). This function is here, tested, and ready for
   experiment 3's LLM verifier, whose yes/no logprobs *are* a real
   probability.

2. ``official_score`` -- a re-implementation of the exact arithmetic in the
   official ``utils.py`` (accuracy + mean tIoU over annotated-yes questions,
   0.4/0.6 weights), used by calibrate.py's grid search so "what score would
   this threshold get" never depends on a live server round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6


def dynamic_yes_threshold(expected_tiou: float) -> float:
    """p(yes) threshold above which answering yes has higher expected score
    than answering no, given you'd achieve ``expected_tiou`` if right.

    Derivation (N questions, P = N/2 positives, exactly balanced yes/no):
        EV(yes) = p * (0.4/N + 0.6/P * t) = p * (0.4 + 1.2*t) / N
        EV(no)  = (1-p) * 0.4/N
        EV(yes) > EV(no)  <=>  p > 1 / (2 + 3*t)
    """

    t = max(0.0, min(1.0, expected_tiou))
    return 1.0 / (2.0 + 3.0 * t)


def temporal_iou(gold: Tuple[float, float], predicted: Optional[Tuple[float, float]]) -> float:
    """Mirrors utils.temporal_iou exactly -- duplicated rather than imported
    so this module has no dependency on the official harness and can be
    unit-tested in isolation. Keep in sync if utils.py ever changes; a test
    in tests/test_threshold.py cross-checks the two against each other.
    """

    if predicted is None:
        return 0.0
    gt_start, gt_end = gold
    pred_start, pred_end = predicted
    intersection = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    if union <= 0:
        return 0.0
    return intersection / union


@dataclass
class ScoredQuestion:
    """One row's worth of what's needed to compute the official score."""

    label: int                                  # 1 = true yes, 0 = true no
    gold_span: Optional[Tuple[float, float]]     # None unless label == 1
    predicted_score: float                       # similarity score (or p_yes)
    predicted_span: Optional[Tuple[float, float]]


def official_score(
    rows: Sequence[ScoredQuestion],
    threshold: float,
) -> Tuple[float, float, float]:
    """(score, accuracy, mean_tiou) if we answer yes wherever
    predicted_score > threshold, exactly reproducing the weighting in
    utils.py / local_evaluator.py.
    """

    if not rows:
        return 0.0, 0.0, 0.0

    correct = 0
    tiou_scores: List[float] = []

    for row in rows:
        predicted_yes = row.predicted_score > threshold
        true_yes = row.label == 1
        if predicted_yes == true_yes:
            correct += 1

        if true_yes and row.gold_span is not None:
            predicted_span = row.predicted_span if predicted_yes else None
            tiou_scores.append(temporal_iou(row.gold_span, predicted_span))

    accuracy = correct / len(rows)
    mean_tiou = sum(tiou_scores) / len(tiou_scores) if tiou_scores else 0.0
    score = ACCURACY_WEIGHT * accuracy + TIOU_WEIGHT * mean_tiou
    return score, accuracy, mean_tiou


def grid_search_threshold(
    rows: Sequence[ScoredQuestion],
    candidates: Sequence[float],
) -> Tuple[float, float, float, float]:
    """(best_threshold, best_score, accuracy_at_best, mean_tiou_at_best)."""

    if not candidates:
        raise ValueError("candidates must be non-empty")

    best = (candidates[0], -1.0, 0.0, 0.0)
    for threshold in candidates:
        score, accuracy, mean_tiou = official_score(rows, threshold)
        if score > best[1]:
            best = (threshold, score, accuracy, mean_tiou)
    return best
