"""A thin extension of the official local_evaluator.py for scripts/analyze.py.

local_evaluator.py's own replay() only keeps aggregate Statistics -- exactly
what it needs to print its report, and correctly nothing more. analyze.py
needs the per-question detail (which conversation, which question, gold vs.
predicted span) to build a "worst offenders" report worth reading. Rather
than reimplement request handling, timeout accounting and scoring a second
time -- and risk it silently drifting from what the real evaluation service
computes -- this module calls straight into local_evaluator's own _ask(),
Statistics, and gold_evidence(), and only adds the bookkeeping on top.

If local_evaluator.py is ever updated upstream, this module keeps working
as long as those names still exist; it deliberately does not copy their
logic.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

import local_evaluator as official  # noqa: E402
from utils import Span, encode_audio, gold_evidence, group_questions_by_conversation, load_sample_audio  # noqa: E402


@dataclass
class DetailRow:
    audio_filename: str
    question_id: str
    question: str
    question_type: str
    label: int
    prediction: int              # official.YES (1), 0, or official.UNANSWERED (-1)
    gold_span: Optional[Span]
    predicted_span: Optional[Span]
    tiou: Optional[float]        # None unless label == YES and gold_span is not None
    request_latency_ms: Optional[float]
    request_failed: bool
    request_timed_out: bool


def replay_with_detail(
    url: str = official.DEFAULT_URL,
    verbose: bool = False,
) -> Tuple["official.Statistics", List[DetailRow]]:
    """Same replay as local_evaluator.replay(), plus a DetailRow per question."""

    statistics = official.Statistics()
    details: List[DetailRow] = []
    session = requests.Session()
    consecutive_timeouts = 0

    conversations = group_questions_by_conversation()
    attempt_budget = len(conversations) * official.ATTEMPT_BUDGET_SECONDS_PER_CONVERSATION
    started_attempt = time.time()

    for audio_filename, rows in conversations:
        if time.time() - started_attempt > attempt_budget:
            statistics.aborted = True
            break

        questions = [row["question"] for row in rows]
        payload = {
            "audio_base64": encode_audio(load_sample_audio(audio_filename)),
            "audio_filename": audio_filename,
            "questions": questions,
        }

        answers, spans, latency_ms, error, timed_out = official._ask(
            session, url, payload, len(questions)
        )
        consecutive_timeouts = consecutive_timeouts + 1 if timed_out else 0

        statistics.record_request(
            question_count=len(questions),
            latency_ms=latency_ms,
            failed=error is not None,
            timed_out=timed_out,
        )

        for row, prediction, predicted_span in zip(rows, answers, spans):
            label = int(row["label"])
            gold = gold_evidence(row)
            iou = statistics.record(row["question_type"], label, prediction, gold, predicted_span)

            details.append(
                DetailRow(
                    audio_filename=audio_filename,
                    question_id=row["question_id"],
                    question=row["question"],
                    question_type=row["question_type"],
                    label=label,
                    prediction=prediction,
                    gold_span=gold,
                    predicted_span=predicted_span,
                    tiou=iou if (label == official.YES and gold is not None) else None,
                    request_latency_ms=latency_ms,
                    request_failed=error is not None,
                    request_timed_out=timed_out,
                )
            )

        if verbose:
            print(f"  {audio_filename}: {len(questions)} questions replayed")

        if consecutive_timeouts >= official.MAX_CONSECUTIVE_TIMEOUTS:
            statistics.aborted = True
            break

    return statistics, details
