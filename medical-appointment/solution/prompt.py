"""Builds the messages sent to the LLM for one batched verification call,
and validates its response.

Design choice worth stating plainly: the LLM gets the FULL timestamped
transcript for the conversation, not a pre-filtered set of retrieved
windows. A 14B-class model's context comfortably fits a whole conversation
(a few hundred to ~2000 words), so there's no reason to risk the retrieval
step hiding the right evidence from the LLM before it even gets to reason
about it -- that limitation only existed because the similarity verifier
had nothing smarter available to it. Retrieval/windows are still built and
used by the fallback path (solution/verify.py's SimilarityVerifier).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator

from solution.types import Segment

SYSTEM_PROMPT = """\
You are verifying yes/no questions about a real doctor-patient conversation, \
using the conversation transcript as your only evidence. Answer strictly from \
what is explicitly stated -- never infer, assume, or fill in from typical \
medical practice.

The critical skill being tested: telling apart a question genuinely supported \
by the transcript from one that closely resembles something said but is wrong \
on a specific detail. Many questions will differ from the transcript in a \
number, dose, unit, drug name, timing, frequency, body site, or negation \
(for example "the dose was increased" vs "the dose was not increased" -- \
opposite meanings, nearly identical wording). Check these details explicitly \
before answering yes; a topically-related excerpt is not enough on its own.

For each question, decide:
- "answer": "yes" only if the transcript explicitly and correctly supports it \
in every specific detail; "no" otherwise, including when the transcript \
doesn't mention it, or almost matches but differs on a detail.
- "confidence": an integer 0-100 for how sure you are in this specific answer.
- "quote": if answer is "yes", the SHORTEST exact excerpt from the transcript \
that supports it, copied verbatim word for word -- never paraphrased, never \
summarized. This should be a short phrase or clause, not a full sentence and \
never surrounding context -- typically well under 15 words, only the minimal \
words that directly support the answer. If answer is "no", use an empty string.

Respond with a single JSON object and nothing else:
{"answers": [{"index": 0, "answer": "yes", "confidence": 87, "quote": "..."}, ...]}
with exactly one entry per question, in the same order given, indexed from 0.\
"""


def format_transcript(segments: Sequence[Segment]) -> str:
    """One timestamped line per ASR segment. This -- not the words the LLM
    happens to quote -- is what a quote gets fuzzy-matched back against for
    timing (see solution/span.py's align_quote_to_words), so the model
    never needs to produce or reason about timestamps itself.
    """

    lines = [f"[{seg.start:6.2f}s - {seg.end:6.2f}s] {seg.text}" for seg in segments]
    return "\n".join(lines)


def format_questions(questions: Sequence[str]) -> str:
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions))


class FewshotExample(BaseModel):
    """One worked example, auto-selected from real training data by
    scripts/build_fewshot_examples.py -- never hand-written, so the prompt
    never contains fabricated clinical content.
    """

    transcript_excerpt: str
    question: str
    answer: str          # "yes" | "no"
    confidence: int
    quote: str = ""


def load_fewshot_examples(path) -> List[FewshotExample]:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return [FewshotExample.model_validate(item) for item in data.get("examples", [])]


def build_messages(
    transcript_segments: Sequence[Segment],
    questions: Sequence[str],
    fewshot_examples: Sequence[FewshotExample],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example in fewshot_examples:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Transcript excerpt:\n{example.transcript_excerpt}\n\n"
                    f"Question:\n0. {example.question}"
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": (
                    '{"answers": [{"index": 0, "answer": "%s", "confidence": %d, "quote": %s}]}'
                    % (example.answer, example.confidence, _json_string(example.quote))
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                f"Transcript:\n{format_transcript(transcript_segments)}\n\n"
                f"Questions:\n{format_questions(questions)}"
            ),
        }
    )
    return messages


def _json_string(text: str) -> str:
    import json

    return json.dumps(text)


class LlmAnswerItem(BaseModel):
    index: int
    answer: str
    confidence: int = Field(ge=0, le=100)
    quote: str = ""

    @field_validator("answer")
    @classmethod
    def _normalize_answer(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"answer must be 'yes' or 'no', got {v!r}")
        return v


class LlmBatchAnswers(BaseModel):
    answers: List[LlmAnswerItem]


def parse_batch_answers(raw: dict, expected_count: int) -> List[LlmAnswerItem]:
    """Validate and order the LLM's parsed JSON against how many questions
    were actually asked.

    Ordered by ``index`` rather than list position -- the model is asked to
    echo the index, and trusting that over raw order is cheap insurance
    against an occasional reordered or skipped entry. Raises on a genuine
    mismatch (wrong count, duplicate/out-of-range indices) so the caller
    (LLMVerifier) can fall back rather than silently score a misaligned
    answer against the wrong question.
    """

    parsed = LlmBatchAnswers.model_validate(raw)
    by_index = {item.index: item for item in parsed.answers}

    if len(by_index) != expected_count or set(by_index) != set(range(expected_count)):
        raise ValueError(
            f"Expected answers for indices 0..{expected_count - 1}, "
            f"got indices {sorted(by_index)}"
        )

    return [by_index[i] for i in range(expected_count)]
