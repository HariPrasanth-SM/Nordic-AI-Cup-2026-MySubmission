"""Unit test for solution/worker_client.py's fallback behavior.

Deliberately does not spawn a real worker process or test the actual
multiprocessing round-trip here: that needs a real Python interpreter doing
a real 'spawn', which is slow and, more importantly, would only be testing
plumbing, not the real models -- the thing worth verifying on real hardware
(see scripts/debug_gpu_repro.py and scripts/run_local_eval.py), not in the
unit suite. This just covers that predict() degrades safely when no worker
is running, which is directly testable and matters on its own: if the
worker process ever dies mid-attempt, this is the code path that keeps the
server answering (with a guess) instead of hanging or crashing.
"""

from utils import validate_response

import solution.worker_client as worker_client
from dtos import ASRQuestionRequestDto


def _request(question_count: int = 10) -> ASRQuestionRequestDto:
    return ASRQuestionRequestDto(
        audio_base64="not real audio",
        audio_filename="x.mp3",
        questions=["q"] * question_count,
    )


def test_predict_falls_back_when_worker_process_is_none(monkeypatch):
    monkeypatch.setattr(worker_client, "_worker_process", None)
    response = worker_client.predict(_request(10))
    validate_response(response, expected_count=10)
    assert all(a is True for a in response.answers)
    assert all(s is None for s in response.evidence_start)


class _DeadProcess:
    def is_alive(self) -> bool:
        return False


def test_predict_falls_back_when_worker_process_has_died(monkeypatch):
    monkeypatch.setattr(worker_client, "_worker_process", _DeadProcess())
    response = worker_client.predict(_request(7))
    validate_response(response, expected_count=7)
    assert all(a is True for a in response.answers)
