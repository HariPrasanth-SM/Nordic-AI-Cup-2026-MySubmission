from utils import validate_response

from solution.fallback import safe_guess


def test_safe_guess_has_correct_length_for_any_count():
    for n in (1, 5, 10, 17):
        response = safe_guess(n)
        assert len(response.answers) == n
        assert len(response.evidence_start) == n
        assert len(response.evidence_end) == n


def test_safe_guess_answers_all_true_with_null_spans():
    response = safe_guess(10)
    assert all(a is True for a in response.answers)
    assert all(s is None for s in response.evidence_start)
    assert all(e is None for e in response.evidence_end)


def test_safe_guess_passes_official_validation():
    response = safe_guess(10)
    validate_response(response, expected_count=10)   # must not raise


def test_safe_guess_zero_questions_is_still_valid():
    response = safe_guess(0)
    validate_response(response, expected_count=0)
