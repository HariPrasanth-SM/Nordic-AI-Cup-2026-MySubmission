"""Plain data structures shared across the solution package.

Deliberately dependency-free (stdlib only) so every module -- including the
ones that must stay lightweight for unit testing, like threshold.py and
span.py -- can import these without pulling in torch, faster-whisper or
sentence-transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Word:
    """One recognised word with its timing, as ASR produces it."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True)
class Segment:
    """One ASR segment: a short run of speech with its own words."""

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass(frozen=True)
class Window:
    """A candidate span of the conversation being scored against a question.

    ``segment_indices`` records which ASR segments a coarse window was built
    from, purely for debugging/analysis -- it plays no role in scoring.
    """

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    segment_indices: tuple = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Span:
    """A predicted evidence span: just the two timestamps the API wants."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class VerifierResult:
    """What a verifier hands back for one question.

    ``p_yes`` is always treated as an uncalibrated score, not a real
    probability -- true for the similarity verifier's raw cosine score, and
    deliberately true for the LLM verifier's self-reported confidence too.
    LLM-reported confidence is well known to be poorly calibrated on its
    own; rather than trust it directly (or take on the real complexity of
    extracting token-level logprobs from a batched, JSON-structured
    generation -- more fragile machinery than this project can afford to
    debug again this week), both verifiers are calibrated the same
    (proven) way: scripts/calibrate.py grid-searches a threshold against
    the official score on the 390 training rows, whatever produced the
    score. threshold.py's dynamic formula (p > 1/(2+3t)) stays unused for
    now for the same reason -- it needs a genuinely calibrated probability,
    which self-reported confidence isn't.

    ``quote`` is the verbatim supporting text the verifier found (LLM
    verifier only; None for the similarity verifier) -- kept for
    debugging/analyze.py, not used in scoring.
    """

    p_yes: float
    span: Optional[Span]
    expected_tiou: float
    quote: Optional[str] = None
