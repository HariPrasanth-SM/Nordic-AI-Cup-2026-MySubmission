"""The actual model/prediction logic. Runs inside the dedicated GPU worker
process (solution/gpu_worker.py), never inside api.py's own process -- see
that module's docstring for why. example.py no longer imports this module
directly; it imports solution/worker_client.py instead, which spawns the
worker and talks to it over a queue.

Import-time behaviour matters here: the README is explicit that there is no
warm-up grace period on a real evaluation run, so every model this module
needs is constructed and exercised once *at import time* -- before
``predict`` is ever called for a real request. In this architecture that
import happens inside the worker process, at the moment gpu_worker.py's
worker_main() does ``import solution.pipeline``, before it starts pulling
requests off its queue.

Every GPU call -- model construction, warm-up, and every ``predict()`` call
-- still runs on one dedicated worker *thread* within this process
(``_GPU_EXECUTOR``), a defensive leftover from when this module ran inside
FastAPI/uvicorn directly: it no longer shares a process with anything else
that could call it from an unexpected thread, but pinning costs nothing and
there is no reason to remove a correct safeguard.

Testing hook: set MEDICAL_APPT_SKIP_MODEL_LOAD=1 to import this module
without constructing any model (used by tests that only exercise the
orchestration logic with monkeypatched components, and by
solution/worker_client.py's own equally-named check to decide whether to
start a worker process at all -- see that module for the distinction).
Never set this for a real server -- gpu_worker.py never sets it, so a real
deployment always loads for real.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from solution import experiment_trace
import concurrent.futures
import logging
import os
import time
from typing import List, Optional

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio, validate_response

from solution.asr import AsrModel, warmup as asr_warmup
from solution.candidates import build_candidate_windows
from solution.config import Config, latest_calibrated_name, load_config
from solution.fallback import safe_guess
from solution.retrieval import Retriever
from solution.seed import seed_everything
from solution.span import clamp_span
from solution.threshold import dynamic_yes_threshold
from solution.types import VerifierResult
from solution.verify import SimilarityVerifier, Verifier

logger = logging.getLogger(__name__)


def _build_verifier(config: Config, retriever: Retriever) -> Verifier:
    # Always built: used directly for verifier.kind == "similarity", and as
    # the LLM verifier's fallback for verifier.kind == "llm" -- cheap to
    # construct (no model of its own beyond the already-loaded retriever),
    # so there's no cost to always having a working fallback on hand.
    similarity = SimilarityVerifier(
        retriever=retriever,
        retrieval_config=config.retrieval,
        span_config=config.span,
    )

    if os.environ.get("MEDICAL_APPT_EXPERIMENT") in {"exp12", "exp13", "exp14", "exp15", "exp16", "exp17", "exp18"}:
        from solution.grounding12 import AnchoredVerifier
        verifier = AnchoredVerifier(fallback=similarity)
        verifier.client.warmup()
        mode = os.environ.get("MEDICAL_APPT_EXPERIMENT")
        settings14 = None
        if mode in {"exp14", "exp15", "exp16", "exp17", "exp18"}:
            if mode in {"exp15", "exp16", "exp17", "exp18"}:
                from solution.calibration15 import load_settings, CalibratedVerifier
            else:
                from solution.calibration14 import load_settings, CalibratedVerifier
            settings14 = load_settings()
        if mode == "exp13" or (settings14 and settings14["base_experiment"] == "exp13"):
            if mode == "exp16":
                from solution.refinement16 import ContrastiveVerifier
            else:
                from solution.refinement13 import ContrastiveVerifier
            verifier = ContrastiveVerifier(base=verifier)
        if settings14:
            verifier = CalibratedVerifier(verifier, settings14)
        if mode == "exp18":
            from solution.rescue18 import RescueVerifier
            verifier = RescueVerifier(verifier)
        if mode == "exp17":
            from solution.reader17 import ReaderVerifier
            verifier = ReaderVerifier(verifier)
        return verifier

    if config.verifier.kind == "similarity":
        return similarity

    if config.verifier.kind == "llm":
        from solution.llm import LlmModel
        from solution.verify import LLMVerifier

        logger.info(
            "Loading LLM verifier: %s",
            config.llm.local_path or f"{config.llm.repo_id}/{config.llm.filename}",
        )
        llm_model = LlmModel(config.llm)
        return LLMVerifier(
            llm_model=llm_model,
            llm_config=config.llm,
            fallback=similarity,
            retriever=retriever,
            span_config=config.span,
        )

    raise NotImplementedError(f"verifier.kind={config.verifier.kind!r} is not implemented.")


def _decide_threshold(config: Config, result: VerifierResult) -> float:
    if config.threshold.mode == "dynamic":
        return dynamic_yes_threshold(result.expected_tiou)
    return config.threshold.fixed_value


# --------------------------------------------------------------------------- #
# Module-level load: runs once, at import time, on the dedicated GPU thread.
# --------------------------------------------------------------------------- #

CONFIG: Config = load_config(calibrated_name=latest_calibrated_name())
seed_everything(CONFIG.seed)

_SKIP_MODEL_LOAD = os.environ.get("MEDICAL_APPT_SKIP_MODEL_LOAD") == "1"

# max_workers=1 is the point: exactly one OS thread ever touches the GPU,
# for the lifetime of the process, whether that's model construction at
# import time or any later predict() call.
_GPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")

ASR_MODEL: Optional[AsrModel] = None
RETRIEVER: Optional[Retriever] = None
VERIFIER: Optional[Verifier] = None
ALIGNER = None
EXPERIMENT = os.environ.get("MEDICAL_APPT_EXPERIMENT", "baseline")


def _load_models() -> None:
    global ASR_MODEL, RETRIEVER, VERIFIER, ALIGNER
    logger.info("Loading models for profile=%s ...", CONFIG.profile)
    ASR_MODEL = AsrModel(CONFIG.asr)
    RETRIEVER = Retriever(CONFIG.retrieval)
    VERIFIER = _build_verifier(CONFIG, RETRIEVER)
    if EXPERIMENT in {"exp10", "exp11"}:
        from solution.evidence_experiment import EvidenceExperiment
        if CONFIG.verifier.kind != "llm":
            raise ValueError("Experiments require the LLM verifier")
        VERIFIER = EvidenceExperiment(VERIFIER, CONFIG.threshold.fixed_value if CONFIG.threshold.mode == "fixed" else 0.0)
    if EXPERIMENT == "exp11":
        from solution.forced_timing import ForcedTiming
        ALIGNER = ForcedTiming()


    logger.info("Warming up ASR and retrieval models ...")
    asr_warmup(ASR_MODEL)
    RETRIEVER.embed_queries(["warm up"])
    logger.info("Warm-up complete; ready to accept requests.")


if not _SKIP_MODEL_LOAD:
    # .result() blocks import until warm-up finishes, same as before this
    # change -- the only difference is that this now runs on _GPU_EXECUTOR's
    # one worker thread rather than whatever thread imported the module.
    _GPU_EXECUTOR.submit(_load_models).result()
else:
    logger.warning("MEDICAL_APPT_SKIP_MODEL_LOAD=1: models NOT loaded (test mode only).")


# --------------------------------------------------------------------------- #
# Per-request logic.
# --------------------------------------------------------------------------- #

def _predict_impl(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """The real work. Always executed on _GPU_EXECUTOR's one dedicated
    thread -- see ``predict`` below. Never raises: any failure falls back to
    ``safe_guess``, which is always a valid, schema-passing response. A
    guess is worth half a mark on average; an exception is worth nothing
    and, if it happens five times in a row, ends the whole attempt.
    """

    experiment_trace.reset()
    start = time.monotonic()
    question_count = len(request.questions)

    try:
        audio_bytes = decode_audio(request.audio_base64)
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if EXPERIMENT in {"exp10", "exp11", "exp12", "exp13", "exp14", "exp15", "exp16", "exp17", "exp18"}:
            VERIFIER.audio_hash = audio_hash

        duration = audio_duration_seconds(audio_bytes)

        logger.info(
            "%s: %.1fs audio, %d questions",
            request.audio_filename,
            duration if duration is not None else float("nan"),
            question_count,
        )

        segments = ASR_MODEL.transcribe_bytes(audio_bytes)
        windows = build_candidate_windows(segments, CONFIG.retrieval.coarse_merge_sizes)
        # Computed once and reused for all ten questions -- window text doesn't
        # depend on the question, only the query side of each retrieval does.
        coarse_vectors = (
            RETRIEVER.embed_passages([w.text for w in windows])
            if windows
            else RETRIEVER.embed_passages([])
        )

        deadline = start + CONFIG.timing.hard_timeout_s
        remaining = deadline - time.monotonic()

        if remaining < CONFIG.timing.min_question_budget_s:
            logger.warning("%.1fs left before verification would even start; guessing.", remaining)
            results = [
                VerifierResult(p_yes=1.0, span=None, expected_tiou=0.0) for _ in request.questions
            ]
        else:
            try:
                results = VERIFIER.verify_batch(
                    request.questions, segments, windows, coarse_vectors, deadline
                )
            except Exception:
                # Every verifier is expected to catch its own failures
                # internally (SimilarityVerifier has nothing further to
                # fall back to; LLMVerifier falls back to SimilarityVerifier
                # -- see solution/verify.py) -- this is a last-resort net
                # for anything even more fundamental, e.g. a bug in the
                # verifier construction itself.
                logger.exception("Verifier raised unexpectedly; guessing for the whole conversation.")
                results = [
                    VerifierResult(p_yes=1.0, span=None, expected_tiou=0.0) for _ in request.questions
                ]

        experiment_trace.event("pre_alignment", results=[asdict(r) for r in results])
        if ALIGNER is not None:
            if deadline - time.monotonic() >= 12:
                results = ALIGNER.refine(audio_bytes, segments, results)
            else:
                experiment_trace.event("forced_alignment_skipped", reason="less than 12 seconds remain")

        answers: List[bool] = []
        evidence_start: List[Optional[float]] = []
        evidence_end: List[Optional[float]] = []

        for result in results:
            threshold = _decide_threshold(CONFIG, result)
            is_yes = result.p_yes > threshold
            span = clamp_span(result.span, duration) if (is_yes and result.span is not None) else None

            answers.append(is_yes)
            evidence_start.append(span.start if span is not None else None)
            evidence_end.append(span.end if span is not None else None)

        response = ASRQuestionResponseDto(
            answers=answers,
            evidence_start=evidence_start,
            evidence_end=evidence_end,
        )
        validate_response(response, expected_count=question_count)
        experiment_trace.finish(request, audio_hash, segments, results, response,
                                CONFIG, time.monotonic() - start)
        return response

    except Exception:
        logger.exception("predict() failed for %s; returning safe guess.", request.audio_filename)
        return safe_guess(question_count)


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Public entry point -- what example.py exposes to api.py.

    Dispatches to _GPU_EXECUTOR's dedicated thread and waits for the result,
    regardless of which thread called us from. The timeout here is a coarse
    whole-request safety net on top of _predict_impl's own internal
    per-question budget: if the GPU call itself hangs rather than raising
    (a real possibility with a driver-level issue, not just a Python
    exception), this still returns a valid response instead of hanging the
    HTTP response indefinitely. It cannot forcibly kill a stuck thread --
    Python doesn't support that -- but every future submitted to the
    executor is independently timeout-bounded, so a single stuck call
    degrades later requests to "always falls back promptly" rather than
    "hangs forever too."
    """

    question_count = len(request.questions)
    try:
        future = _GPU_EXECUTOR.submit(_predict_impl, request)
        return future.result(timeout=CONFIG.timing.hard_timeout_s + 2.0)
    except Exception:
        logger.exception(
            "predict() executor call failed for %s; returning safe guess.", request.audio_filename
        )
        return safe_guess(question_count)
