"""The response we send when something has gone wrong or run out of time.

One failed request costs ten marks -- every question about that conversation
-- so this always has to produce something valid, never raise, and be cheap
enough to compute with no budget left. Answering all-yes with no spans
matches the shipped baseline (0.500 accuracy, 0 tIoU): a coin toss's worth of
marks is far better than a timeout or an unparseable body, which score
nothing on both halves.
"""

from __future__ import annotations

from dtos import ASRQuestionResponseDto


def safe_guess(question_count: int) -> ASRQuestionResponseDto:
    return ASRQuestionResponseDto(
        answers=[True] * question_count,
        evidence_start=[None] * question_count,
        evidence_end=[None] * question_count,
    )
