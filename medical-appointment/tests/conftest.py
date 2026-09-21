"""Shared pytest setup.

Two things every test needs, done once here instead of in every file:

1. medical-appointment/ on sys.path, so `import dtos`, `import utils` and
   `import solution.*` all resolve regardless of where pytest is invoked
   from (mirrors what running `python api.py` gets for free from the
   interpreter, which pytest does not).

2. MEDICAL_APPT_SKIP_MODEL_LOAD=1 by default, so importing solution.pipeline
   in a unit test doesn't try to download/load a Whisper model and an
   embedding model. test_integration_pipeline.py explicitly unsets this and
   skips itself if the real models aren't available -- see that file.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDICAL_APPT_SKIP_MODEL_LOAD", "1")
