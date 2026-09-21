"""Turns (questions, transcript, candidate windows) into one VerifierResult
per question.

SimilarityVerifier is experiments 1-2: retrieve the best coarse window per
question, tighten it with span.fine_tighten, use the resulting similarity
score directly as ``p_yes`` -- calibrated against a single grid-searched
constant threshold (configs/calibrated/*.yaml), not threshold.py's dynamic
formula, because a raw cosine similarity is not a calibrated probability.

LLMVerifier is experiment 3: one batched LLM call answers every question in
the conversation at once, given the full transcript (not pre-filtered
windows -- see solution/prompt.py for why), with instructions to check
exact numeric/negation/unit detail rather than topical similarity, and to
quote its evidence verbatim. Span tightening is two-pass: quote alignment
(solution/span.py) finds *which region* of the transcript the quote occurs
in -- that needs the LLM's own reasoning -- then fine_tighten searches
*inside only that region* for the sub-phrase that best matches the
question, the same embedding-based mechanism the similarity verifier uses.
Quote alignment alone produced spans shaped like whatever the LLM happened
to quote rather than how tightly a human annotator drew the gold span
(confirmed in practice: strong accuracy, mean tIoU well under what
accuracy alone would suggest was possible) -- the second pass exists
specifically to close that gap. Its ``p_yes`` is the model's self-reported
confidence, treated exactly the same way as the similarity score -- an
uncalibrated number that scripts/calibrate.py fits a threshold against, not
threshold.py's dynamic formula (see solution/types.py's VerifierResult for
why that's a deliberate choice, not an oversight).

Both implement ``verify_batch`` -- the whole point of batching is one LLM
call per conversation instead of ten, so the interface is batch-shaped from
the start rather than being a per-question call the LLM verifier awkwardly
loops. pipeline.py depends only on this interface.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Protocol, Sequence

import numpy as np

from solution.config import LlmConfig, RetrievalConfig, SpanConfig
from solution.retrieval import Retriever, top_k
from solution.span import align_quote_to_words, fine_tighten
from solution.types import Segment, Span, VerifierResult, Window, Word

logger = logging.getLogger(__name__)

# Used as a diagnostic default for VerifierResult.expected_tiou -- not part
# of the yes/no decision in fixed threshold mode (see module docstring),
# but shown in analyze.py's reports and available if threshold.mode is ever
# switched to "dynamic" against a genuinely calibrated probability.
DEFAULT_EXPECTED_TIOU = 0.6


class Verifier(Protocol):
    def verify_batch(
        self,
        questions: Sequence[str],
        segments: Sequence[Segment],
        coarse_windows: Sequence[Window],
        coarse_vectors: np.ndarray,
        deadline: float,
    ) -> List[VerifierResult]:
        """One VerifierResult per question, same order as ``questions``.

        ``segments``: the full conversation transcript, in order -- what
        LLMVerifier actually reasons over. ``coarse_windows``/
        ``coarse_vectors``: candidates.py's merged-segment windows and
        their passage embeddings (computed once per conversation by the
        caller and reused across questions) -- what SimilarityVerifier
        uses, whether as the primary verifier or as LLMVerifier's fallback.
        ``deadline``: an absolute time.monotonic() value; implementations
        should degrade gracefully (return trivial no-evidence results, or
        hand off to a faster fallback) rather than run past it.
        """
        ...


class SimilarityVerifier:
    def __init__(
        self,
        retriever: Retriever,
        retrieval_config: RetrievalConfig,
        span_config: SpanConfig,
        expected_tiou: float = DEFAULT_EXPECTED_TIOU,
    ):
        self.retriever = retriever
        self.retrieval_config = retrieval_config
        self.span_config = span_config
        self.expected_tiou = expected_tiou

    def verify_batch(
        self,
        questions: Sequence[str],
        segments: Sequence[Segment],
        coarse_windows: Sequence[Window],
        coarse_vectors: np.ndarray,
        deadline: float,
    ) -> List[VerifierResult]:
        del segments   # unused -- this verifier reasons over windows, not the raw transcript

        if not coarse_windows or not questions:
            return [
                VerifierResult(p_yes=1.0, span=None, expected_tiou=self.expected_tiou)
                for _ in questions
            ]

        # Batched once for all questions -- cheap, and avoids N separate
        # embedding calls for what's otherwise an essentially free step.
        query_vecs = self.retriever.embed_queries(list(questions))

        results: List[VerifierResult] = []
        for i, question in enumerate(questions):
            if time.monotonic() > deadline:
                # Matches safe_guess()'s convention (and the organizer's own
                # baseline) of guessing "yes" when giving up, not "no" --
                # symmetric in expected accuracy on a balanced dataset, but
                # every other fallback path in this codebase already picked
                # "yes", and there's no reason to introduce an inconsistent
                # second convention here.
                results.append(VerifierResult(p_yes=1.0, span=None, expected_tiou=self.expected_tiou))
                continue

            ranked = top_k(query_vecs[i], coarse_vectors, k=self.retrieval_config.coarse_top_k)
            ranked_windows: List[Window] = [coarse_windows[j] for j, _ in ranked]
            top_score = ranked[0][1]

            tight_window = fine_tighten(
                question=question,
                coarse_windows_ranked=ranked_windows,
                retriever=self.retriever,
                span_config=self.span_config,
            )
            span = Span(start=tight_window.start, end=tight_window.end) if tight_window else None
            results.append(VerifierResult(p_yes=top_score, span=span, expected_tiou=self.expected_tiou))

        return results


class LLMVerifier:
    def __init__(
        self,
        llm_model,                       # solution.llm.LlmModel
        llm_config: LlmConfig,
        fallback: Verifier,
        retriever: Retriever,
        span_config: SpanConfig,
        fewshot_examples=None,
        expected_tiou: float = DEFAULT_EXPECTED_TIOU,
    ):
        from solution.prompt import load_fewshot_examples

        self.llm_model = llm_model
        self.llm_config = llm_config
        self.fallback = fallback
        self.retriever = retriever
        self.span_config = span_config
        self.fewshot_examples = (
            fewshot_examples
            if fewshot_examples is not None
            else load_fewshot_examples(llm_config.fewshot_path)
        )
        self.expected_tiou = expected_tiou

    def verify_batch(
        self,
        questions: Sequence[str],
        segments: Sequence[Segment],
        coarse_windows: Sequence[Window],
        coarse_vectors: np.ndarray,
        deadline: float,
    ) -> List[VerifierResult]:
        if not questions:
            return []

        remaining = deadline - time.monotonic()
        if remaining < 0:
            logger.warning("No time left before the LLM call would even start; using fallback verifier.")
            return self.fallback.verify_batch(questions, segments, coarse_windows, coarse_vectors, deadline)

        try:
            return self._verify_with_llm(questions, segments)
        except Exception:
            logger.exception("LLM verification failed; falling back to the similarity verifier.")
            return self.fallback.verify_batch(questions, segments, coarse_windows, coarse_vectors, deadline)

    def _verify_with_llm(
        self,
        questions: Sequence[str],
        segments: Sequence[Segment],
    ) -> List[VerifierResult]:
        from solution.prompt import build_messages, parse_batch_answers

        messages = build_messages(segments, questions, self.fewshot_examples)
        raw = self.llm_model.generate_json(messages)
        answers = parse_batch_answers(raw, expected_count=len(questions))

        all_words = _flatten_words(segments)

        results: List[VerifierResult] = []
        for question, item in zip(questions, answers):
            # item.confidence is "how sure the model is in the answer it
            # gave" (yes OR no), not P(yes) directly -- flip it for "no"
            # answers, or a confident no would score as a high p_yes and
            # invert the threshold decision.
            p_yes = item.confidence / 100.0 if item.answer == "yes" else 1.0 - item.confidence / 100.0

            span: Optional[Span] = None
            quote: Optional[str] = None
            if item.answer == "yes" and item.quote.strip():
                quote = item.quote
                span = self._locate_span(question, item.quote, all_words)

            results.append(
                VerifierResult(p_yes=p_yes, span=span, expected_tiou=self.expected_tiou, quote=quote)
            )

        return results

    def _locate_span(self, question: str, quote: str, all_words: List[Word]) -> Optional[Span]:
        """Quote-align, then fine-trim within the matched region.

        Two stages, not one: align_quote_to_words finds *which region* of
        the transcript the LLM's quote actually occurs in -- that's the
        part that needs the LLM's own reasoning, since only it knows what
        text supports the answer. But the resulting span is exactly as
        long as whatever the LLM happened to quote, which isn't
        necessarily how tightly a human annotator drew the gold span.
        fine_tighten then searches *inside* that region (only that region
        -- passed as the sole candidate window, so it can't drift outside
        what the LLM actually pointed at) for the sub-phrase that best
        matches the question, the same embedding-based mechanism that
        already tightens spans for the similarity verifier. If alignment
        fails outright, there's no region to trim, so this returns None
        and the caller keeps the yes/no answer without a span -- a missing
        span costs nothing on the accuracy half of the score and is
        strictly better than a wrong one.
        """

        coarse_span = align_quote_to_words(quote, all_words)
        if coarse_span is None:
            logger.info(
                "LLM quote didn't align to the transcript closely enough "
                "(kept the yes/no answer, dropped the span): %r", quote,
            )
            return None

        matched_words = [
            w for w in all_words if w.start >= coarse_span.start - 0.01 and w.end <= coarse_span.end + 0.01
        ]
        if not matched_words:
            return coarse_span

        coarse_window = Window(
            start=coarse_span.start,
            end=coarse_span.end,
            text=" ".join(w.text for w in matched_words),
            words=matched_words,
        )
        tightened = fine_tighten(
            question=question,
            coarse_windows_ranked=[coarse_window],
            retriever=self.retriever,
            span_config=self.span_config,
        )
        return Span(start=tightened.start, end=tightened.end)


def _flatten_words(segments: Sequence[Segment]) -> List[Word]:
    return [w for seg in segments for w in seg.words]
