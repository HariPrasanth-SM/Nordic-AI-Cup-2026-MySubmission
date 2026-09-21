"""What example.py imports. Spawns and talks to the dedicated GPU worker
process (solution/gpu_worker.py) -- see that module's docstring for why
this exists.

This module itself never touches the GPU and never imports
solution.pipeline directly; it only ever runs inside api.py's process,
which is deliberately kept GPU-free so that whatever it is about the
FastAPI/uvicorn process that broke CTranslate2 has nothing to touch.

Testing hook: set MEDICAL_APPT_SKIP_MODEL_LOAD=1 to skip spawning the
worker entirely (predict() then always returns a safe guess). Matches the
same env var solution/pipeline.py itself checks, for a different reason:
that one governs whether the *worker* loads real models once it's running;
this one governs whether a worker is started at all. tests/conftest.py sets
this by default, so unit tests never spawn a process.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing
import os

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto

from solution.fallback import safe_guess
from solution.gpu_worker import worker_main

logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 900.0   # generous: first-time model download + load + warm-up, worst case
_REQUEST_TIMEOUT_S = 65.0  # a few seconds past the evaluator's own 60s budget

_SKIP_MODEL_LOAD = os.environ.get("MEDICAL_APPT_SKIP_MODEL_LOAD") == "1"

_ctx = multiprocessing.get_context("spawn")
_request_queue = _ctx.Queue()
_response_queue = _ctx.Queue()
_ready_queue = _ctx.Queue()
_request_ids = itertools.count()

_worker_process = None

# Only the true original process should ever start the worker. When
# 'spawn' launches a child process, it re-executes this same import chain
# (api.py -> example.py -> solution.worker_client) a second time *inside
# that child*, to rebuild enough context to safely unpickle the target
# function -- this is Python's own bootstrap mechanism, not a bug in how
# we're calling it. Without this guard, that re-execution would reach this
# exact line again and try to start a second worker from within the first
# worker's own startup, and so on -- which is exactly what
# multiprocessing's built-in safety check refused to allow. Checking
# against "MainProcess" is the standard, documented way to tell "am I the
# original process" apart from "am I a spawned child currently
# bootstrapping its own re-import of the main module".
_is_main_process = multiprocessing.current_process().name == "MainProcess"

if not _SKIP_MODEL_LOAD and _is_main_process:
    logger.info("Starting GPU worker process ...")
    _worker_process = _ctx.Process(
        target=worker_main,
        args=(_request_queue, _response_queue, _ready_queue),
        daemon=True,
        name="medical-appointment-gpu-worker",
    )
    _worker_process.start()

    status, detail = _ready_queue.get(timeout=_READY_TIMEOUT_S)
    if status != "ready":
        raise RuntimeError(f"GPU worker process failed to start: {detail}")
    logger.info("GPU worker process is up (pid=%s).", _worker_process.pid)
elif not _SKIP_MODEL_LOAD:
    # We're inside a spawned child's bootstrap re-import, not the real
    # process -- nothing to do here, this module is only being re-imported
    # so worker_main is available to unpickle, not to be executed again.
    logger.debug(
        "Not the main process (name=%s); not starting a GPU worker here.",
        multiprocessing.current_process().name,
    )
else:
    logger.warning("MEDICAL_APPT_SKIP_MODEL_LOAD=1: GPU worker NOT started (test mode only).")


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    question_count = len(request.questions)

    if _worker_process is None or not _worker_process.is_alive():
        logger.error(
            "GPU worker process is not running for %s; returning safe guess.",
            request.audio_filename,
        )
        return safe_guess(question_count)

    request_id = next(_request_ids)
    try:
        _request_queue.put((request_id, request.model_dump()))

        while True:
            got_id, payload, error = _response_queue.get(timeout=_REQUEST_TIMEOUT_S)
            if got_id != request_id:
                # A late response to a request we already gave up on and
                # safe-guessed (our own timeout fired before the worker's
                # answer arrived). Discard it and keep waiting for ours --
                # the worker processes one request at a time, so this can
                # only ever be a stale answer, never a wrong-order one.
                continue
            if error is not None:
                logger.error(
                    "GPU worker reported an error for %s: %s", request.audio_filename, error
                )
                return safe_guess(question_count)
            return ASRQuestionResponseDto.model_validate(payload)

    except Exception:
        logger.exception(
            "predict() via GPU worker failed for %s; returning safe guess.", request.audio_filename
        )
        return safe_guess(question_count)
