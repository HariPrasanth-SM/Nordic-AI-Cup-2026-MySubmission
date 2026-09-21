"""Local ASR: faster-whisper with word-level timestamps.

Word timestamps are not an optional extra -- they're what span.py trims
against to hit the ~3s evidence spans the score rewards. Never join segments
into a single string; keep the structure.

Like retrieval.py, the heavy import (faster_whisper) is deferred into
functions so this module is cheap to import for anything that doesn't
actually need to transcribe.
"""

from __future__ import annotations

import logging
import tempfile
from typing import List, TYPE_CHECKING

from solution.config import AsrConfig
from solution.types import Segment, Word

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class AsrModel:
    def __init__(self, config: AsrConfig):
        from faster_whisper import WhisperModel

        self.config = config
        self._model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe_bytes(self, audio_bytes: bytes) -> List[Segment]:
        """Transcribe raw MP3 bytes into segments with word timestamps.

        Writes to a temp file rather than passing bytes/ndarray directly --
        matches the pattern demonstrated in the official README, which is
        the one code path the organizers have actually verified against
        their exact faster-whisper version.
        """

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(audio_bytes)
            f.flush()
            return self._transcribe_path(f.name)

    def transcribe_array(self, audio: "np.ndarray", sample_rate: int = 16000) -> List[Segment]:
        """Transcribe an in-memory waveform. Used for the import-time warm-up
        so we don't need a real audio file on disk just to exercise the
        model once before the server accepts traffic.
        """

        return self._run(audio)

    def _transcribe_path(self, path: str) -> List[Segment]:
        return self._run(path)

    def _run(self, audio) -> List[Segment]:
        segments_iter, _info = self._model.transcribe(
            audio,
            language=self.config.language,
            word_timestamps=True,
            vad_filter=self.config.vad_filter,
            beam_size=self.config.beam_size,
            condition_on_previous_text=self.config.condition_on_previous_text,
            temperature=self.config.temperature,
            initial_prompt=self.config.initial_prompt or None,
        )

        segments: List[Segment] = []
        for seg in segments_iter:
            words = [
                Word(text=w.word.strip(), start=w.start, end=w.end, probability=w.probability)
                for w in (seg.words or [])
                if w.word and w.word.strip()
            ]
            segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words))

        return segments


def warmup(model: AsrModel) -> None:
    """Exercise the model once before the server accepts traffic.

    The README is explicit that there is no separate warm-up period on the
    real evaluation run -- the first inference is the slowest one, so it
    must happen at import time, not on the first real request.
    """

    import numpy as np

    silence = np.zeros(16000 * 2, dtype=np.float32)   # 2s of silence at 16kHz
    try:
        model.transcribe_array(silence)
    except Exception:
        logger.exception("ASR warm-up failed; continuing, first real request will be slower")
