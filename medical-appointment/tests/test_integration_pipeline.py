"""Integration test: runs the real pipeline (real ASR, real embeddings)
against one real training conversation.

Deliberately not mocked. Mocking Whisper or the embedding model would test
the mock, not the system -- the whole point of this file is to catch the
class of bug the unit tests structurally cannot: a wrong faster-whisper
argument name, a model that fails to download, a real conversation that
breaks an assumption a synthetic Word/Segment fixture didn't.

Skips itself (not a failure) if the ML stack isn't installed or no real
training audio is present -- e.g. in a CI box or a lightweight checkout that
only has the placeholder file under data/audio/. Run this for real on
whichever machine you intend to calibrate or submit from before trusting a
calibration run; conftest.py's MEDICAL_APPT_SKIP_MODEL_LOAD=1 default is
overridden here specifically so this file always attempts a real load.
"""

from __future__ import annotations

import base64
import importlib
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _skip_reason():
    try:
        import faster_whisper  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "faster-whisper / sentence-transformers not installed (pip install -r requirements.txt)"

    audio_dir = Path(__file__).resolve().parent.parent / "data" / "audio"
    real_audio = list(audio_dir.glob("*.mp3"))
    if not real_audio:
        return "no real training audio under data/audio/ (this checkout only has the placeholder file)"

    return None


@pytest.fixture(scope="module")
def real_pipeline():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    os.environ["MEDICAL_APPT_SKIP_MODEL_LOAD"] = "0"
    os.environ.setdefault("MEDICAL_APPT_PROFILE", "dev_8gb")

    import solution.pipeline as pipeline_module

    importlib.reload(pipeline_module)   # force a real model load, ignoring conftest's test-mode default
    return pipeline_module


def test_predict_on_one_real_conversation(real_pipeline):
    from dtos import ASRQuestionRequestDto
    from utils import encode_audio, group_questions_by_conversation, load_sample_audio, validate_response

    audio_filename, rows = group_questions_by_conversation()[0]
    request = ASRQuestionRequestDto(
        audio_base64=encode_audio(load_sample_audio(audio_filename)),
        audio_filename=audio_filename,
        questions=[row["question"] for row in rows],
    )

    started = time.monotonic()
    response = real_pipeline.predict(request)
    elapsed = time.monotonic() - started

    validate_response(response, expected_count=len(rows))
    assert len(response.answers) == len(rows)
    assert elapsed < real_pipeline.CONFIG.timing.hard_timeout_s, (
        f"predict() took {elapsed:.1f}s, over the configured "
        f"{real_pipeline.CONFIG.timing.hard_timeout_s}s budget"
    )


def test_predict_never_raises_on_unparseable_audio(real_pipeline):
    from dtos import ASRQuestionRequestDto
    from utils import validate_response

    request = ASRQuestionRequestDto(
        audio_base64=base64.b64encode(b"not actually an mp3 file").decode("utf-8"),
        audio_filename="garbage.mp3",
        questions=["Is this a real question?"] * 10,
    )
    response = real_pipeline.predict(request)   # must not raise -- the whole point of fallback.py
    validate_response(response, expected_count=10)
    assert all(a is True for a in response.answers)   # the safe_guess shape specifically
