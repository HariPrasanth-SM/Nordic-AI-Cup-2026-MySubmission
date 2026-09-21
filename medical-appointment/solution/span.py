"""Tighten a coarse window down to a short, well-centred evidence span.

This is where most of the score lives (see the project's scoring notes: a
whole-segment span scores tIoU ~0.29 against the median annotated span, a
well-centred ~3s window scores ~0.75-0.9). The mechanism is a second,
finer-grained retrieval pass: generate short word-level sub-windows *inside*
only the top few coarse windows (not the whole conversation -- that would be
both slower and noisier), embed them with the same model used for coarse
retrieval, and keep whichever sub-window scores best against the question.

No LLM involved yet -- that's experiment 3's "quote the exact phrase, align
it back onto the words" upgrade, which will plug into this module without
changing its public interface (fine_tighten still returns a Window).
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

from solution.config import SpanConfig
from solution.retrieval import Retriever, top_k
from solution.types import Span, Window, Word


def generate_fine_windows(
    coarse_window: Window,
    span_config: SpanConfig,
) -> List[Window]:
    """Sliding sub-windows of ``coarse_window``, sized close to the
    configured duration variants, stepped every ``stride_words`` words.

    Bounded by construction: candidate count is O(words / stride *
    duration_variants), not a combinatorial search over every possible
    (start, end) pair.
    """

    words = coarse_window.words
    if not words:
        return [coarse_window]   # no word timestamps available; fall back whole

    windows: List[Window] = []
    n = len(words)

    for start_idx in range(0, n, max(1, span_config.stride_words)):
        for target_len in span_config.duration_variants_s:
            end_idx = _extend_to_duration(words, start_idx, target_len)
            if end_idx is None or end_idx <= start_idx:
                continue

            chunk = words[start_idx : end_idx + 1]
            duration = chunk[-1].end - chunk[0].start
            if duration < span_config.min_duration_s or duration > span_config.max_duration_s:
                continue

            windows.append(
                Window(
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=" ".join(w.text for w in chunk),
                    words=list(chunk),
                    segment_indices=coarse_window.segment_indices,
                )
            )

    return windows or [coarse_window]


def _extend_to_duration(words: Sequence[Word], start_idx: int, target_len: float) -> Optional[int]:
    start_time = words[start_idx].start
    for idx in range(start_idx, len(words)):
        if words[idx].end - start_time >= target_len:
            return idx
    return len(words) - 1 if len(words) > start_idx else None


def fine_tighten(
    question: str,
    coarse_windows_ranked: Sequence[Window],
    retriever: Retriever,
    span_config: SpanConfig,
) -> Window:
    """Search inside the top few coarse windows for the best-scoring, tight
    sub-window. Ties (within a small margin) are broken toward the window
    closest to ``target_duration_s``, since tIoU rewards matching both
    length and centre, and the median annotated span is close to that
    target.
    """

    search_pool = coarse_windows_ranked[: max(1, span_config.fine_search_top_k_coarse)]

    fine_windows: List[Window] = []
    for cw in search_pool:
        fine_windows.extend(generate_fine_windows(cw, span_config))

    if not fine_windows:
        return search_pool[0]

    query_vec = retriever.embed_queries([question])[0]
    passage_vecs = retriever.embed_passages([w.text for w in fine_windows])
    ranked = top_k(query_vec, passage_vecs, k=len(fine_windows))

    best_idx, best_score = ranked[0]
    tie_margin = 0.01
    tied = [(i, s) for i, s in ranked if best_score - s <= tie_margin]
    tied.sort(key=lambda pair: abs(fine_windows[pair[0]].duration - span_config.target_duration_s))

    chosen_idx = tied[0][0]
    return fine_windows[chosen_idx]


def clamp_span(span: Span, duration: Optional[float]) -> Span:
    """Clamp into [0, duration], enforce end > start, round to 2 decimals.

    ``duration`` may be None if utils.audio_duration_seconds couldn't read
    the MP3 header -- in that case we only enforce start >= 0 and end > start.
    """

    start = max(0.0, span.start)
    end = max(start + 0.01, span.end)
    if duration is not None:
        end = min(end, duration)
        start = min(start, max(0.0, end - 0.01))
    return Span(start=round(start, 2), end=round(end, 2))


def _normalize_for_matching(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- so matching
    isn't thrown off by the LLM copying a quote with slightly different
    capitalization or punctuation than faster-whisper produced, which
    happens even when the LLM is instructed to copy verbatim.
    """

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def align_quote_to_words(
    quote: str,
    all_words: Sequence[Word],
    min_match_score: float = 60.0,
) -> Optional[Span]:
    """Find where ``quote`` best matches a contiguous run of ``all_words``
    (the full conversation's words, in order) and return the corresponding
    timestamp span.

    This is experiment 3's span-tightening mechanism, replacing
    fine_tighten's embedding-based sliding window search for the LLM
    verifier: rather than searching for the best-scoring *window shape*,
    it searches for where the LLM's own verbatim quote actually occurs,
    which naturally produces a span shaped like the quote itself rather
    than a pre-defined duration bucket.

    Bounded, cheap search: for a conversation of a few hundred words and a
    quote of a handful to ~20 words, this is at most a few thousand fuzzy
    comparisons -- milliseconds, not a meaningful fraction of the request
    budget. Returns None (caller falls back) if the quote is empty, there
    are no words to search, or nothing matches well enough to trust --
    a bad span is worse than admitting the alignment failed.
    """

    from rapidfuzz import fuzz

    quote_norm = _normalize_for_matching(quote)
    if not quote_norm or not all_words:
        return None

    quote_word_count = len(quote_norm.split())
    n = len(all_words)

    best_score = -1.0
    best_span: Optional[Span] = None

    for length in range(max(1, quote_word_count - 2), quote_word_count + 3):
        if length > n:
            continue
        for start in range(0, n - length + 1):
            candidate = all_words[start : start + length]
            candidate_norm = _normalize_for_matching(" ".join(w.text for w in candidate))
            score = fuzz.ratio(quote_norm, candidate_norm)
            if score > best_score:
                best_score = score
                best_span = Span(start=candidate[0].start, end=candidate[-1].end)

    if best_score < min_match_score or best_span is None:
        return None
    return best_span
