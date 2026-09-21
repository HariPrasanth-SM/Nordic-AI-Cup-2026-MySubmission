"""Calibrate the yes/no threshold (and, for the similarity verifier, the
evidence-span target length) against the 390 training questions.

Requires scripts/preprocess.py to have already cached transcripts for the
model_size this profile/config points at -- this script never re-runs ASR.

Two modes, chosen by the configured verifier.kind (see configs/*.yaml):

  similarity: for each candidate span target length, score every question
      once, then pick the best fixed threshold via 5-fold cross-validation
      grouped by conversation. The cross-validated score selects the span
      target; the threshold for that target is refit on the full 390 rows.
      Fast -- no model calls beyond the already-loaded embedding model.

  llm: quote-based span tightening doesn't use span.target_duration_s (see
      solution/span.py's align_quote_to_words), so there's nothing to
      sweep there -- one pass scores all 390 questions through the real
      LLM verifier (batched per conversation, exactly like production),
      then the same threshold cross-validation applies. Much slower than
      the similarity path (39 real LLM calls) -- expect minutes, not
      seconds; progress is logged per conversation.

Writes:
    configs/calibrated/<timestamp>_<git-sha>.yaml  -- picked up automatically
        by solution/config.py's load_config() (most recent by filename sort)
    reports/calibrate_<timestamp>_<git-sha>.md      -- what was tried and why

Usage:
    python scripts/calibrate.py --profile workstation_32gb
    python scripts/calibrate.py --profile workstation_32gb --model-size large-v3-turbo
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from solution.candidates import build_candidate_windows  # noqa: E402
from solution.config import Config, load_config  # noqa: E402
from solution.retrieval import Retriever  # noqa: E402
from solution.seed import seed_everything  # noqa: E402
from solution.threshold import ScoredQuestion, grid_search_threshold, official_score  # noqa: E402
from solution.types import Segment, Window, Word  # noqa: E402
from solution.verify import SimilarityVerifier, Verifier  # noqa: E402
from utils import group_questions_by_conversation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calibrate")

THRESHOLD_CANDIDATES = [round(0.05 + 0.01 * i, 2) for i in range(91)]   # 0.05 .. 0.95
SPAN_TARGET_CANDIDATES = [2.0, 2.5, 3.0, 3.5, 4.0]
N_FOLDS = 5
# Calibration scores against cached data, never a live time budget -- a
# generous fixed deadline so a verifier's own deadline-awareness never
# kicks in and quietly truncates a calibration run.
_CALIBRATION_DEADLINE = lambda: time.monotonic() + 600.0


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


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def _grouped_folds(conv_ids: Sequence[str], n_folds: int) -> List[int]:
    unique_convs = sorted(set(conv_ids))
    fold_of = {cid: i % n_folds for i, cid in enumerate(unique_convs)}
    return [fold_of[cid] for cid in conv_ids]


def _cross_validated_score(rows: List[ScoredQuestion], conv_ids: List[str], n_folds: int = N_FOLDS) -> float:
    fold_assignment = _grouped_folds(conv_ids, n_folds)
    fold_scores = []
    for fold in range(n_folds):
        train_rows = [r for r, f in zip(rows, fold_assignment) if f != fold]
        held_rows = [r for r, f in zip(rows, fold_assignment) if f == fold]
        if not train_rows or not held_rows:
            continue
        threshold, _, _, _ = grid_search_threshold(train_rows, THRESHOLD_CANDIDATES)
        score, _, _ = official_score(held_rows, threshold)
        fold_scores.append(score)
    return sum(fold_scores) / len(fold_scores) if fold_scores else 0.0


def _score_rows_for_verifier(
    verifier: Verifier,
    conversations,
    cached_segments: List[List[Segment]],
    cached_windows: List[List[Window]],
    cached_coarse_vectors,
    log_progress: bool = False,
) -> Tuple[List[ScoredQuestion], List[str]]:
    """Score all 390 questions through ``verifier``, one verify_batch call
    per conversation -- exactly how production calls it, so calibration
    never scores something subtly different from what actually gets served.
    """

    rows: List[ScoredQuestion] = []
    conv_ids: List[str] = []

    total = len(conversations)
    for i, ((audio_filename, question_rows), segments, windows, coarse_vectors) in enumerate(
        zip(conversations, cached_segments, cached_windows, cached_coarse_vectors)
    ):
        if log_progress:
            logger.info("  [%d/%d] %s", i + 1, total, audio_filename)

        questions = [row["question"] for row in question_rows]
        results = verifier.verify_batch(
            questions, segments, windows, coarse_vectors, _CALIBRATION_DEADLINE()
        )

        for row, result in zip(question_rows, results):
            label = int(row["label"])
            gold_span = None
            if row.get("evidence_start") and row.get("evidence_end"):
                gold_span = (float(row["evidence_start"]), float(row["evidence_end"]))
            predicted_span = (result.span.start, result.span.end) if result.span is not None else None

            rows.append(
                ScoredQuestion(
                    label=label,
                    gold_span=gold_span,
                    predicted_score=result.p_yes,
                    predicted_span=predicted_span,
                )
            )
            conv_ids.append(row["transcript_id"])

    return rows, conv_ids


def _calibrate_similarity(
    config: Config,
    conversations,
    cached_segments: List[List[Segment]],
    cached_windows: List[List[Window]],
    retriever: Retriever,
) -> Tuple[dict, Dict[float, dict]]:
    cached_coarse_vectors = [
        retriever.embed_passages([w.text for w in windows]) for windows in cached_windows
    ]

    results: Dict[float, dict] = {}
    for target in SPAN_TARGET_CANDIDATES:
        logger.info("Scoring span target_duration_s=%.1f ...", target)
        span_config = config.span.model_copy(update={"target_duration_s": target})
        verifier = SimilarityVerifier(retriever=retriever, retrieval_config=config.retrieval, span_config=span_config)

        rows, conv_ids = _score_rows_for_verifier(
            verifier, conversations, cached_segments, cached_windows, cached_coarse_vectors
        )
        cv_score = _cross_validated_score(rows, conv_ids)
        full_threshold, full_score, full_acc, full_tiou = grid_search_threshold(rows, THRESHOLD_CANDIDATES)
        results[target] = {
            "cv_score": cv_score,
            "full_threshold": full_threshold,
            "full_score": full_score,
            "full_accuracy": full_acc,
            "full_mean_tiou": full_tiou,
        }
        logger.info(
            "  target=%.1fs  cv_score=%.4f  full_score=%.4f (acc=%.3f tiou=%.3f) @ threshold=%.2f",
            target, cv_score, full_score, full_acc, full_tiou, full_threshold,
        )

    best_target = max(results, key=lambda t: results[t]["cv_score"])
    best = dict(results[best_target])
    best["span_target_duration_s"] = best_target
    return best, results


def _calibrate_llm(
    config: Config,
    conversations,
    cached_segments: List[List[Segment]],
    cached_windows: List[List[Window]],
    retriever: Retriever,
) -> Tuple[dict, Dict[float, dict]]:
    from solution.llm import LlmModel
    from solution.verify import LLMVerifier

    cached_coarse_vectors = [
        retriever.embed_passages([w.text for w in windows]) for windows in cached_windows
    ]

    logger.info(
        "Loading LLM: %s ... (this and the 39-conversation pass below can take a while)",
        config.llm.local_path or f"{config.llm.repo_id}/{config.llm.filename}",
    )
    llm_model = LlmModel(config.llm)
    fallback = SimilarityVerifier(retriever=retriever, retrieval_config=config.retrieval, span_config=config.span)
    verifier = LLMVerifier(
        llm_model=llm_model,
        llm_config=config.llm,
        fallback=fallback,
        retriever=retriever,
        span_config=config.span,
    )

    t0 = time.monotonic()
    rows, conv_ids = _score_rows_for_verifier(
        verifier, conversations, cached_segments, cached_windows, cached_coarse_vectors, log_progress=True
    )
    elapsed = time.monotonic() - t0
    logger.info("Scored all conversations through the LLM verifier in %.1fs.", elapsed)

    cv_score = _cross_validated_score(rows, conv_ids)
    full_threshold, full_score, full_acc, full_tiou = grid_search_threshold(rows, THRESHOLD_CANDIDATES)
    best = {
        "cv_score": cv_score,
        "full_threshold": full_threshold,
        "full_score": full_score,
        "full_accuracy": full_acc,
        "full_mean_tiou": full_tiou,
        "elapsed_seconds": elapsed,
    }
    logger.info(
        "cv_score=%.4f  full_score=%.4f (acc=%.3f tiou=%.3f) @ threshold=%.2f",
        cv_score, full_score, full_acc, full_tiou, full_threshold,
    )
    return best, {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model-size", default=None, help="which cached transcripts to calibrate against")
    parser.add_argument(
        "--verifier",
        default=None,
        choices=["similarity", "llm"],
        help="override verifier.kind for this run without editing configs/*.yaml",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.model_size:
        overrides["asr"] = {"model_size": args.model_size}
    if args.verifier:
        overrides["verifier"] = {"kind": args.verifier}
    config = load_config(profile=args.profile, overrides=overrides or None)
    seed_everything(config.seed)

    transcripts_dir = config.paths.resolve("transcripts_dir")
    model_size = config.asr.model_size

    conversations = group_questions_by_conversation()
    logger.info("%d conversations, %d questions", len(conversations), sum(len(r) for _, r in conversations))

    logger.info("Loading cached transcripts for model_size=%s ...", model_size)
    cached_segments = [_load_cached_segments(transcripts_dir, model_size, fn) for fn, _ in conversations]
    cached_windows = [build_candidate_windows(segs, config.retrieval.coarse_merge_sizes) for segs in cached_segments]

    logger.info("Loading embedding model %s ...", config.retrieval.embedding_model)
    retriever = Retriever(config.retrieval)

    if config.verifier.kind == "llm":
        best, span_results = _calibrate_llm(config, conversations, cached_segments, cached_windows, retriever)
    else:
        best, span_results = _calibrate_similarity(config, conversations, cached_segments, cached_windows, retriever)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short_sha()
    name = f"{timestamp}_{sha}"

    calibrated_dir = config.paths.resolve("calibrated_dir")
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    calibrated_payload = {
        "asr": {"model_size": model_size},
        "verifier": {"kind": config.verifier.kind},
        "threshold": {"mode": "fixed", "fixed_value": best["full_threshold"]},
    }
    if "span_target_duration_s" in best:
        calibrated_payload["span"] = {"target_duration_s": best["span_target_duration_s"]}
    (calibrated_dir / f"{name}.yaml").write_text(yaml.safe_dump(calibrated_payload, sort_keys=False))

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    lines = [
        f"# Calibration report -- {name}",
        "",
        f"- verifier: `{config.verifier.kind}`",
        f"- model_size: `{model_size}`",
        f"- profile: `{config.profile}`",
        f"- git commit: `{sha}`",
    ]
    if "span_target_duration_s" in best:
        lines.append(f"- selected span target_duration_s: **{best['span_target_duration_s']}**")
    lines += [
        f"- selected threshold: **{best['full_threshold']:.2f}**",
        f"- cross-validated score ({N_FOLDS}-fold, grouped by conversation): **{best['cv_score']:.4f}**",
        f"- full-data score at selection: **{best['full_score']:.4f}** "
        f"(accuracy={best['full_accuracy']:.3f}, mean_tIoU={best['full_mean_tiou']:.3f})",
        "",
    ]
    if span_results:
        lines += [
            "## All span variants tried",
            "",
            "| target_duration_s | cv_score | full_score | full_accuracy | full_mean_tiou | full_threshold |",
            "|---|---|---|---|---|---|",
        ]
        for target, r in sorted(span_results.items()):
            lines.append(
                f"| {target} | {r['cv_score']:.4f} | {r['full_score']:.4f} | "
                f"{r['full_accuracy']:.3f} | {r['full_mean_tiou']:.3f} | {r['full_threshold']:.2f} |"
            )
        lines.append("")
    lines += [
        f"Config written to `configs/calibrated/{name}.yaml`. `solution/pipeline.py` picks up the "
        "most recently written calibrated config automatically (by filename sort order, which is "
        "also chronological order here); pass `calibrated_name=\"" + name + "\"` to `load_config()` "
        "explicitly if you need to pin this one over a later run.",
    ]
    (reports_dir / f"calibrate_{name}.md").write_text("\n".join(lines))

    logger.info("Wrote configs/calibrated/%s.yaml", name)
    logger.info("Wrote reports/calibrate_%s.md", name)


if __name__ == "__main__":
    main()
