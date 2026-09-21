"""Runs solution.pipeline in a dedicated, separate OS process.

Why a whole separate process rather than a thread: every attempt to keep
GPU inference in-process alongside FastAPI/uvicorn failed with
CUBLAS_STATUS_NOT_SUPPORTED, even after (in order): pinning every GPU call
to one dedicated thread; ruling out a CTranslate2/PyTorch device-placement
conflict; confirming the library/driver stack is current; and removing a
uvicorn app-string-resolution anomaly. A plain, single-process script
exercising the exact same models (scripts/debug_gpu_repro.py) never failed
once across all 39 real conversations, with real compute timings. The one
variable never tested in isolation was FastAPI/uvicorn's process itself --
so rather than keep guessing at which specific thing about that process is
responsible, this sidesteps the question by not sharing a process with it
at all.

The worker is a plain Python process, started via multiprocessing with the
'spawn' method (not 'fork' -- forking after any CUDA use in the parent is
unsafe, and the whole point here is a clean, freshly-imported process with
no FastAPI/asyncio machinery anywhere in its ancestry, matching
debug_gpu_repro.py as closely as possible). It imports solution.pipeline
normally -- which triggers that module's existing import-time model
loading and warm-up exactly as before -- then loops, answering one request
at a time from api.py's process via two multiprocessing.Queue objects.
solution/pipeline.py itself did not need to change: it already does the
right thing when it owns its own process, which is exactly the role it's
given here.
"""

from __future__ import annotations

import logging
import multiprocessing

from dtos import ASRQuestionRequestDto

logger = logging.getLogger(__name__)


def worker_main(
    request_queue: "multiprocessing.Queue",
    response_queue: "multiprocessing.Queue",
    ready_queue: "multiprocessing.Queue",
) -> None:
    """Entry point for the worker process.

    Must stay at module level, not a closure or lambda: multiprocessing's
    'spawn' method pickles the target by import path and re-imports it in
    the child, which only works for something importable by name.
    """

    logging.basicConfig(level=logging.INFO)

    try:
        import solution.pipeline as pipeline   # import-time model load + warm-up happens here
    except Exception as exc:
        logger.exception("GPU worker failed to load models")
        ready_queue.put(("error", repr(exc)))
        return

    ready_queue.put(("ready", None))
    logger.info("GPU worker process ready, waiting for requests.")

    while True:
        message = request_queue.get()
        if message is None:   # sentinel: shut down
            logger.info("GPU worker received shutdown sentinel, exiting.")
            break

        request_id, payload = message
        try:
            request = ASRQuestionRequestDto.model_validate(payload)
            response = pipeline.predict(request)
            response_queue.put((request_id, response.model_dump(), None))
        except Exception as exc:
            # pipeline.predict() already has its own internal fallback and
            # essentially never raises -- this is a second, outer safety
            # net for anything even more fundamental (e.g. a malformed
            # payload that fails model_validate before predict() is even
            # reached).
            logger.exception("GPU worker failed to answer request %s", request_id)
            response_queue.put((request_id, None, repr(exc)))
