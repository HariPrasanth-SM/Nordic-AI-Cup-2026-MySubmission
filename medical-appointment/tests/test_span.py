import numpy as np
import pytest

from solution.config import SpanConfig
from solution.span import align_quote_to_words, clamp_span, fine_tighten, generate_fine_windows
from solution.types import Span, Window, Word


def _word_window(text: str, start: float, word_duration: float = 0.5) -> Window:
    tokens = text.split()
    words = [
        Word(text=tok, start=start + i * word_duration, end=start + (i + 1) * word_duration)
        for i, tok in enumerate(tokens)
    ]
    return Window(start=words[0].start, end=words[-1].end, text=text, words=words)


# --------------------------------------------------------------------------- #
# generate_fine_windows
# --------------------------------------------------------------------------- #

def test_generate_fine_windows_falls_back_to_whole_window_without_words():
    empty = Window(start=0.0, end=5.0, text="no word timestamps", words=[])
    config = SpanConfig()
    result = generate_fine_windows(empty, config)
    assert result == [empty]


def test_generate_fine_windows_respects_duration_bounds():
    # 30 words, 0.5s each = 15s total
    window = _word_window(" ".join(f"w{i}" for i in range(30)), start=0.0, word_duration=0.5)
    config = SpanConfig(
        min_duration_s=1.0,
        max_duration_s=4.0,
        duration_variants_s=[2.0, 3.0],
        stride_words=2,
    )
    windows = generate_fine_windows(window, config)
    assert windows   # non-empty
    for w in windows:
        assert config.min_duration_s <= w.duration <= config.max_duration_s + 1e-9


def test_generate_fine_windows_never_exceeds_source_window():
    window = _word_window(" ".join(f"w{i}" for i in range(6)), start=10.0, word_duration=0.5)
    config = SpanConfig(min_duration_s=0.1, max_duration_s=10.0, duration_variants_s=[10.0], stride_words=1)
    windows = generate_fine_windows(window, config)
    for w in windows:
        assert w.start >= window.start - 1e-9
        assert w.end <= window.end + 1e-9


# --------------------------------------------------------------------------- #
# fine_tighten, with a fake retriever so this stays a fast unit test
# --------------------------------------------------------------------------- #

class _FakeRetriever:
    """Scores any window containing the word 'target' highest; otherwise by
    inverse length, so the tie-break-toward-target-duration logic has
    something real to break a tie between.
    """

    def embed_queries(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)

    def embed_passages(self, texts):
        vectors = []
        for t in texts:
            base = 1.0 if "target" in t.split() else 0.1
            vectors.append([base])
        return np.array(vectors, dtype=np.float32)


def test_fine_tighten_prefers_window_containing_the_answer():
    words_text = "the quick brown fox target jumps over the lazy dog"
    coarse = _word_window(words_text, start=0.0, word_duration=1.0)
    config = SpanConfig(
        min_duration_s=1.0,
        max_duration_s=5.0,
        target_duration_s=2.0,
        duration_variants_s=[2.0, 3.0],
        stride_words=1,
        fine_search_top_k_coarse=1,
    )
    chosen = fine_tighten("does it mention the target?", [coarse], _FakeRetriever(), config)
    assert "target" in chosen.text.split()
    assert chosen.duration <= config.max_duration_s + 1e-9


def test_fine_tighten_empty_pool_returns_first_coarse_window():
    coarse = Window(start=0.0, end=1.0, text="", words=[])
    config = SpanConfig()
    chosen = fine_tighten("anything", [coarse], _FakeRetriever(), config)
    assert chosen is coarse


# --------------------------------------------------------------------------- #
# clamp_span
# --------------------------------------------------------------------------- #

def test_clamp_span_within_bounds_is_unchanged_up_to_rounding():
    span = clamp_span(Span(1.234, 5.678), duration=10.0)
    assert span.start == pytest.approx(1.23)
    assert span.end == pytest.approx(5.68)


def test_clamp_span_negative_start_is_clamped_to_zero():
    span = clamp_span(Span(-2.0, 3.0), duration=10.0)
    assert span.start == 0.0


def test_clamp_span_end_past_duration_is_clamped():
    span = clamp_span(Span(8.0, 20.0), duration=10.0)
    assert span.end == 10.0


def test_clamp_span_end_never_before_start():
    span = clamp_span(Span(5.0, 5.0), duration=10.0)
    assert span.end > span.start


def test_clamp_span_handles_unknown_duration():
    span = clamp_span(Span(1.0, 2.0), duration=None)
    assert span.start == 1.0
    assert span.end == 2.0


# --------------------------------------------------------------------------- #
# align_quote_to_words
# --------------------------------------------------------------------------- #

def _flat_words(text: str, word_duration: float = 0.4, start: float = 0.0) -> list:
    words = []
    t = start
    for tok in text.split():
        words.append(Word(text=tok, start=t, end=t + word_duration))
        t += word_duration
    return words


def test_align_quote_finds_exact_contiguous_match():
    words = _flat_words("the patient has been diagnosed with asthma and started treatment")
    span = align_quote_to_words("diagnosed with asthma", words)
    assert span is not None
    assert span.start == pytest.approx(1.6)
    assert span.end == pytest.approx(2.8)


def test_align_quote_is_robust_to_case_and_punctuation():
    words = _flat_words("the patient has been diagnosed with asthma today")
    a = align_quote_to_words("diagnosed with asthma", words)
    b = align_quote_to_words("Diagnosed With Asthma,", words)
    assert a == b


def test_align_quote_returns_none_when_nothing_matches():
    words = _flat_words("the patient has been diagnosed with asthma today")
    assert align_quote_to_words("completely unrelated statement about something else", words) is None


def test_align_quote_returns_none_for_empty_quote_or_words():
    words = _flat_words("some words here")
    assert align_quote_to_words("", words) is None
    assert align_quote_to_words("some words", []) is None
