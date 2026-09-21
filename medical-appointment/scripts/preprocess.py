"""Transcribe every training conversation once and cache the result.

Run this before calibrate.py or analyze.py -- both read from the cache
instead of re-running ASR, so iterating on thresholds/span parameters takes
seconds, not minutes. Re-run with --force whenever the ASR model,
compute_type, or profile changes in a way that should invalidate the cache.

Also the natural place to A/B large-v3 against large-v3-turbo (experiment 1):
run this twice with different --model-size values, each caching to its own
subdirectory, then point calibrate.py/analyze.py at whichever cache you want
to score.

Usage:
    python scripts/preprocess.py --profile workstation_32gb
    python scripts/preprocess.py --profile workstation_32gb --model-size large-v3-turbo
    python scripts/preprocess.py --profile workstation_32gb --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solution.asr import AsrModel  # noqa: E402
from solution.config import load_config  # noqa: E402
from solution.seed import seed_everything  # noqa: E402
from solution.types import Segment  # noqa: E402
from utils import group_questions_by_conversation, load_sample_audio  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess")


def _cache_path(transcripts_dir: Path, model_size: str, audio_filename: str) -> Path:
    stem = Path(audio_filename).stem
    return transcripts_dir / model_size / f"{stem}.json"


def _segments_to_json(segments: list[Segment]) -> list[dict]:
    return [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": [
                {"text": w.text, "start": w.start, "end": w.end, "probability": w.probability}
                for w in s.words
            ],
        }
        for s in segments
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None, help="dev_8gb | workstation_32gb (default: $MEDICAL_APPT_PROFILE or dev_8gb)")
    parser.add_argument("--model-size", default=None, help="override asr.model_size, e.g. large-v3-turbo")
    parser.add_argument("--force", action="store_true", help="re-transcribe even if a cache file already exists")
    args = parser.parse_args()

    overrides = {"asr": {"model_size": args.model_size}} if args.model_size else None
    config = load_config(profile=args.profile, overrides=overrides)
    seed_everything(config.seed)

    transcripts_dir = config.paths.resolve("transcripts_dir")
    model_size = config.asr.model_size
    out_dir = transcripts_dir / model_size
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading ASR model=%s profile=%s compute_type=%s device=%s",
        model_size, config.profile, config.asr.compute_type, config.asr.device,
    )
    model = AsrModel(config.asr)

    conversations = group_questions_by_conversation()
    logger.info("%d conversations found in data/question_train.csv", len(conversations))

    timings: list[float] = []
    for audio_filename, _rows in conversations:
        out_path = _cache_path(transcripts_dir, model_size, audio_filename)
        if out_path.exists() and not args.force:
            logger.info("skip (already cached): %s", audio_filename)
            continue

        audio_bytes = load_sample_audio(audio_filename)
        t0 = time.monotonic()
        segments = model.transcribe_bytes(audio_bytes)
        elapsed = time.monotonic() - t0
        timings.append(elapsed)

        payload = {
            "audio_filename": audio_filename,
            "model_size": model_size,
            "compute_type": config.asr.compute_type,
            "device": config.asr.device,
            "elapsed_seconds": elapsed,
            "segments": _segments_to_json(segments),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        logger.info("%-32s %2d segments  %5.1fs  -> %s", audio_filename, len(segments), elapsed, out_path)

    if timings:
        timings_sorted = sorted(timings)
        p95 = timings_sorted[int(0.95 * (len(timings_sorted) - 1))]
        logger.info(
            "ASR timing over %d newly-transcribed conversations: mean=%.1fs  p95=%.1fs  max=%.1fs",
            len(timings), sum(timings) / len(timings), p95, max(timings),
        )
    else:
        logger.info("Nothing new to transcribe (use --force to re-run everything).")


if __name__ == "__main__":
    main()
