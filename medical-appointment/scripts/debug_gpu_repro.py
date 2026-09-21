"""One-off diagnostic: does the CUBLAS_STATUS_NOT_SUPPORTED crash reproduce
without FastAPI/uvicorn in the picture at all?

Loads the exact same pipeline as the real server -- both models, via
solution.pipeline, which reads MEDICAL_APPT_PROFILE the same way api.py
does -- in a plain synchronous script, then replays all 39 training
conversations sequentially: the same thing scripts/preprocess.py does for
ASR alone, but exercising the full pipeline (ASR + retrieval) this time,
and never swallowing an exception, so a crash here prints the real
traceback instead of silently falling back to a guess.

This is the one cell of the "which variable actually matters" grid that
hasn't been tested cleanly yet. The retrieval.device=cpu test and the
thread-pinning fix both still ran *inside* FastAPI -- neither actually
isolated "both models loaded together" from "running inside FastAPI/
uvicorn" as separate variables. This script has no FastAPI, no uvicorn, no
asyncio event loop, and no thread pool of its own at all.

  - If this crashes too: the problem is loading both models together in one
    process, independent of any web framework. Worth trying next: swap the
    construction order in solution/pipeline.py's _load_models() (Retriever
    before AsrModel, not after) -- import/construction order has caused
    exactly this class of cross-library CUDA issue before.
  - If this runs clean: FastAPI/uvicorn's own process is implicated -- most
    likely uvloop, which uvicorn uses automatically when installed instead
    of the standard asyncio loop. Worth trying next: force
    uvicorn.run(..., loop="asyncio") in api.py, or -- more robust, more
    work -- move inference to a genuinely separate OS process from the
    FastAPI server rather than just a separate thread within it.

Usage:
    MEDICAL_APPT_PROFILE=workstation_32gb python scripts/debug_gpu_repro.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import group_questions_by_conversation, load_sample_audio  # noqa: E402


def main() -> int:
    # Importing here, not at module top, is what actually constructs and
    # warms up both models -- exactly what happens when api.py imports
    # example.predict -- on this script's own single thread, nothing else
    # involved.
    from solution.candidates import build_candidate_windows
    from solution.pipeline import ASR_MODEL, CONFIG, RETRIEVER

    conversations = group_questions_by_conversation()
    print(f"{len(conversations)} conversations, sequential, no server involved ...\n")

    failures = 0
    for audio_filename, _rows in conversations:
        try:
            audio_bytes = load_sample_audio(audio_filename)
            t0 = time.monotonic()
            segments = ASR_MODEL.transcribe_bytes(audio_bytes)
            windows = build_candidate_windows(segments, CONFIG.retrieval.coarse_merge_sizes)
            if windows:
                RETRIEVER.embed_passages([w.text for w in windows])
            elapsed = time.monotonic() - t0
            print(f"  {audio_filename:<32} {elapsed:6.1f}s  ok  ({len(segments)} segments)")
        except Exception:
            failures += 1
            print(f"  {audio_filename:<32}  FAILED:")
            traceback.print_exc()
            print()

    print(f"\n{failures}/{len(conversations)} conversations raised an exception.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
