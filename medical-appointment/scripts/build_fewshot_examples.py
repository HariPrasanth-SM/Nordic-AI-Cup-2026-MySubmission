"""Auto-select and format few-shot examples for the LLM verifier prompt,
from real cached transcripts and the training CSV -- never hand-written.

Why auto-selected rather than written by hand: the actual conversation
content isn't available outside this competition's data, so any
hand-written example would have to be invented, which is exactly the kind
of fabricated clinical content that shouldn't end up baked into a prompt.
Selecting real, verified rows keeps every example grounded in something
that actually happened in the training data.

Selects (by default): 1 positive with a short, clean gold span; up to 2
hard_negative rows, preferring one whose question contains a digit (to
illustrate numeric mismatch specifically) and one that doesn't (negation/
lexical mismatch); 1 short off_topic row. Requires scripts/preprocess.py to
have already cached transcripts for the model_size in use.

Writes configs/fewshot_examples.yaml, which solution/prompt.py reads at
LLM-verifier construction time. Review the output before committing --
this picks reasonable, verified examples, not necessarily the ideal ones;
hand-editing the generated YAML (or re-running with different --seed-like
selection tweaks) is expected, not a sign anything is broken.

Usage:
    python scripts/build_fewshot_examples.py --profile workstation_32gb
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from solution.candidates import build_candidate_windows  # noqa: E402
from solution.config import load_config  # noqa: E402
from solution.retrieval import Retriever, top_k  # noqa: E402
from solution.seed import seed_everything  # noqa: E402
from solution.types import Segment, Word  # noqa: E402
from utils import group_questions_by_conversation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_fewshot_examples")

HAS_DIGIT = re.compile(r"\d")


def _load_cached_segments(transcripts_dir: Path, model_size: str, audio_filename: str) -> List[Segment]:
    stem = Path(audio_filename).stem
    path = transcripts_dir / model_size / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached transcript at {path}. Run "
            f"'python scripts/preprocess.py --model-size {model_size}' first."
        )
    payload = json.loads(path.read_text())
    segments = []
    for s in payload["segments"]:
        words = [Word(text=w["text"], start=w["start"], end=w["end"], probability=w["probability"]) for w in s["words"]]
        segments.append(Segment(start=s["start"], end=s["end"], text=s["text"], words=words))
    return segments


def _gold_quote_text(segments: List[Segment], start: float, end: float) -> str:
    words = [w for seg in segments for w in seg.words if w.start >= start - 0.05 and w.end <= end + 0.05]
    return " ".join(w.text for w in words)


def _best_window_excerpt(segments: List[Segment], question: str, retriever: Retriever, merge_sizes) -> str:
    windows = build_candidate_windows(segments, merge_sizes)
    if not windows:
        return " ".join(seg.text for seg in segments)
    query_vec = retriever.embed_queries([question])[0]
    passage_vecs = retriever.embed_passages([w.text for w in windows])
    ranked = top_k(query_vec, passage_vecs, k=1)
    return windows[ranked[0][0]].text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model-size", default=None, help="which cached transcripts to build excerpts from")
    parser.add_argument("--max-positive-span-s", type=float, default=4.0)
    args = parser.parse_args()

    overrides = {"asr": {"model_size": args.model_size}} if args.model_size else None
    config = load_config(profile=args.profile, overrides=overrides)
    seed_everything(config.seed)

    transcripts_dir = config.paths.resolve("transcripts_dir")
    model_size = config.asr.model_size

    conversations = group_questions_by_conversation()
    segments_by_file: Dict[str, List[Segment]] = {}

    def segments_for(audio_filename: str) -> List[Segment]:
        if audio_filename not in segments_by_file:
            segments_by_file[audio_filename] = _load_cached_segments(transcripts_dir, model_size, audio_filename)
        return segments_by_file[audio_filename]

    positive_candidate = None
    hard_negative_with_digit = None
    hard_negative_without_digit = None
    off_topic_candidate = None

    for audio_filename, rows in conversations:
        for row in rows:
            qtype = row["question_type"]
            question = row["question"]

            if qtype == "positive" and positive_candidate is None:
                start, end = float(row["evidence_start"]), float(row["evidence_end"])
                if end - start <= args.max_positive_span_s:
                    positive_candidate = (audio_filename, row, start, end)

            elif qtype == "hard_negative":
                if HAS_DIGIT.search(question) and hard_negative_with_digit is None:
                    hard_negative_with_digit = (audio_filename, row)
                elif not HAS_DIGIT.search(question) and hard_negative_without_digit is None:
                    hard_negative_without_digit = (audio_filename, row)

            elif qtype == "off_topic" and off_topic_candidate is None:
                off_topic_candidate = (audio_filename, row)

        if all([positive_candidate, hard_negative_with_digit, hard_negative_without_digit, off_topic_candidate]):
            break

    if positive_candidate is None:
        raise RuntimeError(
            f"No positive row found with a gold span <= {args.max_positive_span_s}s. "
            "Loosen --max-positive-span-s."
        )

    logger.info("Loading embedding model %s ...", config.retrieval.embedding_model)
    retriever = Retriever(config.retrieval)

    examples = []

    audio_filename, row, start, end = positive_candidate
    segments = segments_for(audio_filename)
    excerpt = _best_window_excerpt(segments, row["question"], retriever, config.retrieval.coarse_merge_sizes)
    quote = _gold_quote_text(segments, start, end)
    examples.append(
        {
            "transcript_excerpt": excerpt,
            "question": row["question"],
            "answer": "yes",
            "confidence": 92,
            "quote": quote,
        }
    )
    logger.info("positive: %s / %s -> quote=%r", audio_filename, row["question"][:60], quote[:60])

    for candidate, label in (
        (hard_negative_with_digit, "hard_negative (has a number)"),
        (hard_negative_without_digit, "hard_negative (no number)"),
    ):
        if candidate is None:
            logger.warning("No %s example found; skipping that slot.", label)
            continue
        audio_filename, row = candidate
        segments = segments_for(audio_filename)
        excerpt = _best_window_excerpt(segments, row["question"], retriever, config.retrieval.coarse_merge_sizes)
        examples.append(
            {
                "transcript_excerpt": excerpt,
                "question": row["question"],
                "answer": "no",
                "confidence": 90,
                "quote": "",
            }
        )
        logger.info("%s: %s / %s", label, audio_filename, row["question"][:60])

    if off_topic_candidate is not None:
        audio_filename, row = off_topic_candidate
        segments = segments_for(audio_filename)
        excerpt = _best_window_excerpt(segments, row["question"], retriever, config.retrieval.coarse_merge_sizes)
        examples.append(
            {
                "transcript_excerpt": excerpt,
                "question": row["question"],
                "answer": "no",
                "confidence": 95,
                "quote": "",
            }
        )
        logger.info("off_topic: %s / %s", audio_filename, row["question"][:60])

    out_path = Path(config.llm.fewshot_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump({"examples": examples}, sort_keys=False, allow_unicode=True))

    logger.info("Wrote %d examples to %s -- review before committing.", len(examples), out_path)


if __name__ == "__main__":
    main()
