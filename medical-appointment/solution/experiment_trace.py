"""Local-run-only trace. No gold, audio, or question IDs enter this module."""
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

_EVENTS = []


def reset():
    _EVENTS.clear()


def event(stage, **data):
    if os.environ.get('MEDICAL_LOCAL_TRACE'):
        _EVENTS.append({'stage': stage, **data})


def finish(request, audio_hash, segments, results, response, config, elapsed):
    destination = os.environ.get('MEDICAL_LOCAL_TRACE')
    if not destination:
        return
    try:
        record = dict(audio_filename=request.audio_filename, audio_sha256=audio_hash,
                      questions=request.questions, segments=[asdict(s) for s in segments],
                      results=[asdict(r) for r in results], response=response.model_dump(),
                      config=config.model_dump(), elapsed_s=elapsed, events=list(_EVENTS))
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
    except Exception:
        logging.getLogger(__name__).exception('Could not write local diagnostic trace')
