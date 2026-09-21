"""Make runs reproducible on a single machine.

Call ``seed_everything`` once, as early as possible (solution/pipeline.py
does this at import time, before any model loads).

Honest limit: this makes a given machine deterministic run-to-run. It does
*not* guarantee bit-identical floating-point results across different GPU
architectures (a 4060 and a 5090 can differ in the last few decimal places
of an embedding or a logit even with identical seeds and identical code --
that's a property of how each GPU's kernels sum floats, not a bug). That
difference is far too small to change any yes/no decision or move a span by
a meaningful amount, so it doesn't affect which calibrated config we ship --
it just means "identical to the last bit across machines" isn't a promise
this function can make, and it shouldn't be treated as one.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism over speed: disable cuDNN's input-size-based kernel
        # autotuning (which picks different, non-deterministic algorithms
        # run to run) and force it to only use deterministic kernels.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            # warn_only=True: some ops (e.g. certain CUDA scatter/reduce
            # kernels used deep in transformer backends) have no
            # deterministic implementation. We'd rather know about it in the
            # logs than have the server crash on an operation we don't
            # directly call.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
