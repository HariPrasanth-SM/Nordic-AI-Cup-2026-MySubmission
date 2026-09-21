import math

import pytest

from solution.threshold import (
    ScoredQuestion,
    dynamic_yes_threshold,
    grid_search_threshold,
    official_score,
    temporal_iou,
)


# --------------------------------------------------------------------------- #
# dynamic_yes_threshold: p > 1 / (2 + 3t)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "expected_tiou,expected_threshold",
    [
        (0.0, 0.5),
        (0.5, 1 / 3.5),
        (0.6, 1 / 3.8),
        (1.0, 0.2),
    ],
)
def test_dynamic_threshold_matches_derivation(expected_tiou, expected_threshold):
    assert dynamic_yes_threshold(expected_tiou) == pytest.approx(expected_threshold, abs=1e-9)


def test_dynamic_threshold_is_monotonically_decreasing_in_tiou():
    thresholds = [dynamic_yes_threshold(t) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_dynamic_threshold_clamps_out_of_range_input():
    assert dynamic_yes_threshold(-1.0) == dynamic_yes_threshold(0.0)
    assert dynamic_yes_threshold(2.0) == dynamic_yes_threshold(1.0)


# --------------------------------------------------------------------------- #
# temporal_iou: must match the official utils.temporal_iou exactly, including
# the four worked examples the README itself gives.
# --------------------------------------------------------------------------- #

README_EXAMPLES = [
    ((21.62, 26.24), (21.62, 26.24), 1.0),
    ((21.62, 26.24), (22.00, 27.00), 0.788),
    ((21.62, 26.24), (24.00, 34.00), 0.181),
    ((21.62, 26.24), (0.00, 120.0), 0.038),
    ((21.62, 26.24), (40.00, 45.00), 0.0),
]


@pytest.mark.parametrize("gold,predicted,expected", README_EXAMPLES)
def test_temporal_iou_matches_readme_examples(gold, predicted, expected):
    assert temporal_iou(gold, predicted) == pytest.approx(expected, abs=1e-2)


def test_temporal_iou_none_prediction_is_zero():
    assert temporal_iou((0.0, 1.0), None) == 0.0


def test_temporal_iou_matches_official_utils_implementation():
    """Cross-check against the actual utils.py this competition scores
    against, so this module can never silently drift from it.
    """
    from utils import temporal_iou as official_temporal_iou

    cases = [
        ((21.62, 26.24), (21.62, 26.24)),
        ((21.62, 26.24), (22.00, 27.00)),
        ((10.0, 15.0), None),
        ((0.0, 5.0), (5.0, 10.0)),   # touching, zero overlap
        ((0.0, 5.0), (100.0, 105.0)),
    ]
    for gold, predicted in cases:
        assert temporal_iou(gold, predicted) == pytest.approx(
            official_temporal_iou(gold, predicted), abs=1e-9
        )


# --------------------------------------------------------------------------- #
# official_score: reproduces the README's own worked table.
# --------------------------------------------------------------------------- #

def _rows(n_yes: int, n_no: int, tiou_if_answered_yes: float, answer_all_yes: bool) -> list:
    rows = []
    # predicted_score is 1.0 if we mean to answer yes, 0.0 otherwise, scored
    # against threshold=0.5 -- a simple way to force a specific answer
    # pattern through official_score's threshold comparison in these tests.
    score = 1.0 if answer_all_yes else 0.0
    for _ in range(n_yes):
        rows.append(
            ScoredQuestion(
                label=1,
                gold_span=(0.0, 10.0),
                predicted_score=score,
                predicted_span=(0.0, 10.0 * tiou_if_answered_yes) if answer_all_yes else None,
            )
        )
    for _ in range(n_no):
        rows.append(ScoredQuestion(label=0, gold_span=None, predicted_score=score, predicted_span=None))
    return rows


def test_shipped_baseline_all_yes_no_spans():
    rows = [
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=1.0, predicted_span=None)
        for _ in range(5)
    ] + [
        ScoredQuestion(label=0, gold_span=None, predicted_score=1.0, predicted_span=None) for _ in range(5)
    ]
    score, accuracy, mean_tiou = official_score(rows, threshold=0.5)
    assert accuracy == pytest.approx(0.5)
    assert mean_tiou == pytest.approx(0.0)
    assert score == pytest.approx(0.2)


def test_perfect_answers_no_spans():
    rows = [
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=1.0, predicted_span=None)
        for _ in range(5)
    ] + [
        ScoredQuestion(label=0, gold_span=None, predicted_score=0.0, predicted_span=None) for _ in range(5)
    ]
    score, accuracy, mean_tiou = official_score(rows, threshold=0.5)
    assert accuracy == pytest.approx(1.0)
    assert mean_tiou == pytest.approx(0.0)
    assert score == pytest.approx(0.4)


def test_all_yes_perfect_spans():
    rows = [
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=1.0, predicted_span=(0.0, 10.0))
        for _ in range(5)
    ] + [
        ScoredQuestion(label=0, gold_span=None, predicted_score=1.0, predicted_span=None) for _ in range(5)
    ]
    score, accuracy, mean_tiou = official_score(rows, threshold=0.5)
    assert accuracy == pytest.approx(0.5)
    assert mean_tiou == pytest.approx(1.0)
    assert score == pytest.approx(0.8)


def test_missed_positive_scores_zero_on_both_halves_for_that_row():
    """A true-yes row answered no must not simply drop out of the tIoU
    average -- utils.mean_temporal_iou keeps it in the denominator.
    """
    rows = [
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=1.0, predicted_span=(0.0, 10.0)),
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=0.0, predicted_span=None),  # missed
    ]
    score, accuracy, mean_tiou = official_score(rows, threshold=0.5)
    assert accuracy == pytest.approx(0.5)
    # one perfect (1.0) and one missed (0.0) -> mean 0.5, not 1.0
    assert mean_tiou == pytest.approx(0.5)


def test_no_answered_alongside_span_neither_helps_nor_hurts():
    baseline = ScoredQuestion(label=0, gold_span=None, predicted_score=0.0, predicted_span=None)
    with_span = ScoredQuestion(label=0, gold_span=None, predicted_score=0.0, predicted_span=(1.0, 2.0))
    score_a, acc_a, tiou_a = official_score([baseline], threshold=0.5)
    score_b, acc_b, tiou_b = official_score([with_span], threshold=0.5)
    assert (score_a, acc_a, tiou_a) == (score_b, acc_b, tiou_b)


# --------------------------------------------------------------------------- #
# grid_search_threshold
# --------------------------------------------------------------------------- #

def test_grid_search_finds_perfectly_separable_threshold():
    rows = [
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=0.9, predicted_span=(0.0, 10.0)),
        ScoredQuestion(label=1, gold_span=(0.0, 10.0), predicted_score=0.8, predicted_span=(0.0, 10.0)),
        ScoredQuestion(label=0, gold_span=None, predicted_score=0.2, predicted_span=None),
        ScoredQuestion(label=0, gold_span=None, predicted_score=0.1, predicted_span=None),
    ]
    threshold, score, accuracy, mean_tiou = grid_search_threshold(rows, [0.15, 0.3, 0.5, 0.7, 0.85])
    assert accuracy == pytest.approx(1.0)
    assert mean_tiou == pytest.approx(1.0)
    assert score == pytest.approx(1.0)


def test_grid_search_requires_candidates():
    with pytest.raises(ValueError):
        grid_search_threshold([], [])
