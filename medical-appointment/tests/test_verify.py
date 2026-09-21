import hashlib
import time

import numpy as np
import pytest

from solution.config import LlmConfig, RetrievalConfig, SpanConfig
from solution.types import Segment, VerifierResult, Window, Word
from solution.verify import LLMVerifier, SimilarityVerifier, _flatten_words


FAR_FUTURE_DEADLINE = time.monotonic() + 3600
PAST_DEADLINE = time.monotonic() - 1


def _segment(text: str, start: float, word_duration: float = 0.4) -> Segment:
    words = []
    t = start
    for tok in text.split():
        words.append(Word(text=tok, start=t, end=t + word_duration))
        t += word_duration
    return Segment(start=start, end=t, text=text, words=words)


class _FakeRetriever:
    """Deterministic, vocabulary-agnostic stand-in: encodes text as an
    L2-normalized, hashed bag-of-words vector (a real numpy array, so this
    composes correctly with the real solution.retrieval.top_k and
    solution.span.fine_tighten, not a mock of those too). More shared
    words with the query -> higher cosine similarity, which is all that's
    needed for ranking to be meaningful in these tests -- not a real
    embedding model.
    """

    _DIM = 64

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._DIM, dtype=np.float32)
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._DIM
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_queries(self, texts):
        return np.array([self._encode_one(t) for t in texts], dtype=np.float32)

    def embed_passages(self, texts):
        return np.array([self._encode_one(t) for t in texts], dtype=np.float32)


class _FakeLlmModel:
    """Returns a pre-scripted response (or raises) instead of running real
    inference -- what's under test here is LLMVerifier's own logic
    (confidence conversion, fallback triggers, quote handling), not
    llama.cpp itself.
    """

    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        self.last_messages = None

    def generate_json(self, messages):
        self.last_messages = messages
        if self.exception is not None:
            raise self.exception
        return self.response


class _FakeFallbackVerifier:
    def __init__(self):
        self.called_with = None

    def verify_batch(self, questions, segments, coarse_windows, coarse_vectors, deadline):
        self.called_with = questions
        return [
            VerifierResult(p_yes=0.0, span=None, expected_tiou=0.6) for _ in questions
        ]


def _make_verifier(llm_response=None, llm_exception=None, span_config=None, retriever=None):
    fake_llm = _FakeLlmModel(response=llm_response, exception=llm_exception)
    fallback = _FakeFallbackVerifier()
    verifier = LLMVerifier(
        llm_model=fake_llm,
        llm_config=LlmConfig(),
        fallback=fallback,
        retriever=retriever if retriever is not None else _FakeRetriever(),
        span_config=span_config if span_config is not None else SpanConfig(),
        fewshot_examples=[],   # no fewshot file needed for these tests
    )
    return verifier, fake_llm, fallback


# --------------------------------------------------------------------------- #
# Confidence -> p_yes conversion -- exactly the bug this needs to guard against
# --------------------------------------------------------------------------- #

def test_confident_yes_gives_high_p_yes():
    response = {"answers": [{"index": 0, "answer": "yes", "confidence": 90, "quote": ""}]}
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["q1"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].p_yes == pytest.approx(0.90)


def test_confident_no_gives_low_p_yes_not_high():
    """The regression test for the actual bug found while building this:
    'confidence' is confidence in whichever answer was given, not P(yes)
    directly -- a confident 'no' must score a LOW p_yes, not a high one,
    or the threshold decision inverts silently.
    """
    response = {"answers": [{"index": 0, "answer": "no", "confidence": 95, "quote": ""}]}
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["q1"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].p_yes == pytest.approx(0.05)


def test_unsure_no_gives_p_yes_near_half():
    response = {"answers": [{"index": 0, "answer": "no", "confidence": 55, "quote": ""}]}
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["q1"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].p_yes == pytest.approx(0.45)


# --------------------------------------------------------------------------- #
# Quote -> span alignment integration
# --------------------------------------------------------------------------- #

def test_yes_with_good_quote_gets_a_span():
    segments = [_segment("the patient was diagnosed with asthma today", start=10.0)]
    response = {
        "answers": [
            {"index": 0, "answer": "yes", "confidence": 88, "quote": "diagnosed with asthma"}
        ]
    }
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["has asthma?"], segments, [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].span is not None
    assert results[0].quote == "diagnosed with asthma"


def test_yes_with_unmatchable_quote_keeps_answer_but_drops_span():
    segments = [_segment("the patient was diagnosed with asthma today", start=10.0)]
    response = {
        "answers": [
            {"index": 0, "answer": "yes", "confidence": 88, "quote": "something never actually said"}
        ]
    }
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["q1"], segments, [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].p_yes == pytest.approx(0.88)   # the yes/no decision is untouched
    assert results[0].span is None                    # only the span is dropped


def test_no_answer_never_gets_a_span_even_with_a_quote_field():
    segments = [_segment("the patient was diagnosed with asthma today", start=10.0)]
    response = {"answers": [{"index": 0, "answer": "no", "confidence": 80, "quote": "asthma"}]}
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["q1"], segments, [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].span is None


def test_fine_trim_narrows_a_loose_quote_to_the_query_relevant_part():
    """The actual mechanism upgrade this file exists to guard: quote
    alignment alone returns a span shaped like whatever the LLM happened
    to quote. When that quote is looser than the minimal supporting
    phrase, fine_tighten should narrow it down toward the sub-phrase that
    best matches the question -- not just return the raw quote's span.
    """
    from solution.span import align_quote_to_words

    segments = [
        _segment(
            "well let me look here the patient was diagnosed with asthma "
            "and prescribed an inhaler today",
            start=0.0,
            word_duration=0.4,
        )
    ]
    question = "was the patient diagnosed with asthma"
    quote = "the patient was diagnosed with asthma and prescribed an inhaler today"
    response = {"answers": [{"index": 0, "answer": "yes", "confidence": 88, "quote": quote}]}

    span_config = SpanConfig(
        min_duration_s=0.4, max_duration_s=8.0, target_duration_s=1.6,
        duration_variants_s=[1.2, 1.6, 2.0], stride_words=1, fine_search_top_k_coarse=1,
    )
    verifier, _, _ = _make_verifier(llm_response=response, span_config=span_config)

    results = verifier.verify_batch([question], segments, [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)

    raw_span = align_quote_to_words(quote, _flatten_words(segments))
    assert results[0].span is not None
    assert results[0].span.duration < raw_span.duration   # actually narrowed, not just re-returned
    assert results[0].span.start == pytest.approx(raw_span.start)   # anchored at the same region


# --------------------------------------------------------------------------- #
# Fallback cascade
# --------------------------------------------------------------------------- #

def test_falls_back_when_llm_raises():
    verifier, _, fallback = _make_verifier(llm_exception=RuntimeError("simulated GPU failure"))
    results = verifier.verify_batch(["q1", "q2"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert fallback.called_with == ["q1", "q2"]
    assert len(results) == 2


def test_falls_back_on_malformed_json():
    verifier, _, fallback = _make_verifier(llm_response={"not_the_expected_shape": True})
    results = verifier.verify_batch(["q1"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert fallback.called_with == ["q1"]
    assert len(results) == 1


def test_falls_back_on_wrong_answer_count():
    # asked one question, LLM answered as if there were two
    response = {
        "answers": [
            {"index": 0, "answer": "yes", "confidence": 80, "quote": "x"},
            {"index": 1, "answer": "no", "confidence": 80, "quote": ""},
        ]
    }
    verifier, _, fallback = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["only one question"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert fallback.called_with == ["only one question"]
    assert len(results) == 1


def test_falls_back_immediately_when_deadline_already_passed():
    verifier, fake_llm, fallback = _make_verifier(llm_response={"answers": []})
    verifier.verify_batch(["q1"], [], [], np.zeros((0, 1)), PAST_DEADLINE)
    assert fallback.called_with == ["q1"]
    assert fake_llm.last_messages is None   # never even attempted the call


def test_answers_returned_in_question_order_even_if_llm_reorders_indices():
    response = {
        "answers": [
            {"index": 1, "answer": "no", "confidence": 70, "quote": ""},
            {"index": 0, "answer": "yes", "confidence": 90, "quote": ""},
        ]
    }
    verifier, _, _ = _make_verifier(llm_response=response)
    results = verifier.verify_batch(["q0", "q1"], [], [], np.zeros((0, 1)), FAR_FUTURE_DEADLINE)
    assert results[0].p_yes == pytest.approx(0.90)   # matches index 0, not list position 0
    assert results[1].p_yes == pytest.approx(0.30)


# --------------------------------------------------------------------------- #
# _flatten_words
# --------------------------------------------------------------------------- #

def test_flatten_words_preserves_order_across_segments():
    segments = [_segment("hello there", start=0.0), _segment("how are you", start=1.0)]
    words = _flatten_words(segments)
    assert [w.text for w in words] == ["hello", "there", "how", "are", "you"]


# --------------------------------------------------------------------------- #
# SimilarityVerifier.verify_batch
# --------------------------------------------------------------------------- #

def test_similarity_verifier_empty_windows_guesses_yes_no_span():
    verifier = SimilarityVerifier(_FakeRetriever(), RetrievalConfig(), SpanConfig())
    results = verifier.verify_batch(["q1", "q2"], [], [], [], FAR_FUTURE_DEADLINE)
    assert len(results) == 2
    assert all(r.p_yes == 1.0 and r.span is None for r in results)


def test_similarity_verifier_past_deadline_guesses_yes_no_span():
    # Non-empty coarse_windows/vectors so the deadline branch (not the
    # empty-windows branch) is what's actually exercised.
    verifier = SimilarityVerifier(_FakeRetriever(), RetrievalConfig(), SpanConfig())
    fake_window = Window(start=0.0, end=3.0, text="asthma follow up visit", words=[])
    coarse_vectors = _FakeRetriever().embed_passages([fake_window.text])
    results = verifier.verify_batch(["q1"], [], [fake_window], coarse_vectors, PAST_DEADLINE)
    assert results[0].p_yes == 1.0
    assert results[0].span is None
