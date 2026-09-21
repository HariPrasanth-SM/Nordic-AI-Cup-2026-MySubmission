"""Local LLM inference via llama.cpp.

llama.cpp/GGUF over vLLM or raw transformers, deliberately: after a week
mostly spent on GPU reliability, not model quality, simplicity and
maturity on consumer GPUs mattered more than vLLM's extra throughput.
GGUF quantization also sidesteps this project's one hardware-confirmed
landmine directly -- INT8 is confirmed broken on this GPU in one inference
engine (CTranslate2); GGUF's K-quants dequantize to fp16 for the actual
matmul rather than using INT8 tensor-core paths, which is a different
enough code path that the same bug class is much less likely to apply, but
worth testing deliberately on real hardware regardless of that reasoning
-- see scripts/preprocess.py's caution about assuming rather than
measuring.

Deferred import (llama_cpp) so this module is cheap to import without the
package installed, matching solution/asr.py and solution/retrieval.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from solution.config import LlmConfig

logger = logging.getLogger(__name__)


class LlmModel:
    def __init__(self, config: LlmConfig):
        from llama_cpp import Llama

        self.config = config

        # verbose is config-driven (default off) -- see LlmConfig.verbose.
        # It's what prints "offloaded N/N layers to GPU" at load time and
        # per-call performance stats; useful once while confirming GPU
        # offload actually happened, a flood on every call after that's
        # already confirmed.
        if config.local_path:
            self._model = Llama(
                model_path=config.local_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                verbose=config.verbose,
            )
        else:
            self._model = Llama.from_pretrained(
                repo_id=config.repo_id,
                filename=config.filename,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                verbose=config.verbose,
            )

    def generate_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """One chat-completion call, parsed as JSON.

        Tries llama-cpp-python's JSON-object response format first (which
        constrains decoding via a grammar so the output is guaranteed
        syntactically valid JSON, when the installed version supports it),
        and falls back to plain generation plus manual extraction of the
        first {...} block if that's unavailable or rejected -- older
        llama-cpp-python versions, or models whose chat template doesn't
        play well with grammar constraints, shouldn't hard-fail the whole
        verifier over a version mismatch.

        Raises on genuine failure (caller -- LLMVerifier -- is responsible
        for catching this and falling back to the similarity verifier; this
        function does not know about that fallback and shouldn't hide
        failures from it).
        """

        try:
            completion = self._model.create_chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        except TypeError:
            # Older llama-cpp-python without response_format support.
            logger.warning("response_format unsupported by this llama-cpp-python; retrying without it.")
            completion = self._model.create_chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        raw_text = completion["choices"][0]["message"]["content"]
        return _parse_json_object(raw_text)


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Fallback: the model wrapped the JSON in prose or a markdown code
    # fence despite instructions not to -- take the first balanced-looking
    # {...} block and try that instead of giving up immediately.
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {raw_text[:200]!r}")
    return json.loads(match.group(0))
