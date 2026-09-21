"""Typed configuration for the solution.

Merge order (later wins): configs/base.yaml -> configs/<profile>.yaml ->
configs/calibrated/<name>.yaml (optional). This is the only file that knows
about the YAML layout; everything else imports the ``Config`` object and
never touches paths under configs/ directly.

Profile selection:
    - Scripts (preprocess.py, calibrate.py, run_local_eval.py, analyze.py)
      take ``--profile`` on the command line.
    - The server entry point (solution/pipeline.py, loaded by example.py at
      import time) has no command line, so it reads the
      ``MEDICAL_APPT_PROFILE`` environment variable instead, defaulting to
      ``workstation_32gb`` if unset -- see DEFAULT_PROFILE below for why.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

SOLUTION_ROOT = Path(__file__).resolve().parent.parent   # medical-appointment/
CONFIGS_DIR = SOLUTION_ROOT / "configs"
DEFAULT_PROFILE_ENV_VAR = "MEDICAL_APPT_PROFILE"
# Deliberately workstation_32gb, not dev_8gb: dev_8gb's compute_type
# (int8_float16) is confirmed broken on the RTX 5090 -- CTranslate2
# explicitly disables INT8 on Blackwell GPUs because of it -- and the 5090
# is the machine this actually gets submitted from. Silently defaulting to
# a compute type known to crash on the real deployment hardware is exactly
# the kind of landmine that should never be the *unset* behavior. Always
# pass --profile (scripts) or set MEDICAL_APPT_PROFILE explicitly (server)
# regardless; this default is a safety net, not a substitute for that.
DEFAULT_PROFILE = "workstation_32gb"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    transcripts_dir: str = "transcripts"
    models_dir: str = "models"
    calibrated_dir: str = "configs/calibrated"

    def resolve(self, name: str) -> Path:
        return SOLUTION_ROOT / getattr(self, name)


class AsrConfig(BaseModel):
    model_size: str = "large-v3"
    language: str = "en"
    device: str = "cpu"
    compute_type: str = "int8"
    vad_filter: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    initial_prompt: str = ""


class RetrievalConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    query_prefix: str = ""
    passage_prefix: str = ""
    coarse_merge_sizes: List[int] = Field(default_factory=lambda: [1, 2, 3])
    coarse_top_k: int = 6


class SpanConfig(BaseModel):
    fine_search_top_k_coarse: int = 3
    min_duration_s: float = 1.2
    max_duration_s: float = 6.0
    target_duration_s: float = 3.0
    duration_variants_s: List[float] = Field(
        default_factory=lambda: [2.0, 2.5, 3.0, 3.5, 4.5]
    )
    stride_words: int = 2


class ThresholdConfig(BaseModel):
    mode: str = "fixed"          # "fixed" | "dynamic"
    fixed_value: float = 0.5


class TimingConfig(BaseModel):
    soft_deadline_s: float = 45.0
    hard_timeout_s: float = 58.0
    # Below this much remaining budget, skip verification entirely for the
    # rest of the questions in this conversation and guess -- a coin toss
    # beats a timeout that costs all ten questions and counts toward the
    # five-consecutive-timeouts abort.
    min_question_budget_s: float = 1.5
    # How long the GPU worker waits for a request before issuing a trivial
    # dummy GPU call instead, to keep the CUDA/cuBLAS context from sitting
    # fully idle. See solution/gpu_worker.py for why this exists.
    gpu_heartbeat_interval_s: float = 20.0
    # Below this much remaining budget, skip the LLM call entirely for this
    # conversation and go straight to the (fast) similarity fallback -- an
    # LLM call that gets cut off mid-generation wastes the time spent on it
    # for nothing, where the fallback alone would have produced a scored
    # answer. Generous relative to observed batched-call latency, not a
    # tight fit -- see solution/verify.py's LLMVerifier.
    llm_min_batch_budget_s: float = 15.0


class VerifierConfig(BaseModel):
    kind: str = "similarity"     # "similarity" (exp 1-2) | "llm" (exp 3+)


class LlmConfig(BaseModel):
    # A local GGUF checkpoint. Either a filesystem path (models_dir-relative
    # or absolute) or a "repo_id/filename" pair for llama-cpp-python's
    # Llama.from_pretrained, which downloads and caches from Hugging Face
    # Hub on first use -- see solution/llm.py.
    repo_id: str = "bartowski/Qwen2.5-14B-Instruct-GGUF"
    filename: str = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    local_path: str = ""          # if set, used instead of repo_id/filename
    n_ctx: int = 8192              # comfortably covers a full transcript + 10 questions + few-shot
    n_gpu_layers: int = -1         # -1 = offload every layer to GPU
    # Greedy decoding, matching the project's reproducibility stance for
    # every other model (see solution/seed.py, solution/asr.py's
    # temperature=0.0).
    temperature: float = 0.0
    # The real bounding mechanism on how long a call can run: llama.cpp
    # enforces this natively, no timeout/threading machinery needed. A
    # deliberate choice after this week -- not adding a second timeout
    # layer around the LLM call itself (signal- or thread-based) given the
    # existing outer timeout in solution/worker_client.py already provides
    # a proven backstop, and thread-based interruption of a blocking GPU
    # call is exactly the category of fragility this project just spent a
    # long debugging thread getting out of.
    max_tokens: int = 2048
    fewshot_path: str = "configs/fewshot_examples.yaml"
    # llama.cpp's own C++ logging (per-token perf stats, the full GGUF
    # metadata dump at load time) -- genuinely useful once, to confirm GPU
    # offload actually happened (look for "offloaded N/N layers to GPU"),
    # a flood on every single call after that's confirmed. Off by default;
    # flip to true only when actively debugging a GPU/load problem.
    verbose: bool = False


class Config(BaseModel):
    seed: int = 1234
    profile: str = DEFAULT_PROFILE
    paths: PathsConfig = Field(default_factory=PathsConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    span: SpanConfig = Field(default_factory=SpanConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)

    # extra="forbid" catches typos in YAML immediately (e.g. a stray key that
    # doesn't match any field) instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Dicts are merged key-by-key; any other type (including lists) is
    replaced wholesale by the override's value. That keeps merge semantics
    predictable -- a profile that sets ``asr.compute_type`` doesn't need to
    repeat every other ``asr.*`` key, but a profile that wants to change
    ``span.duration_variants_s`` replaces the whole list rather than
    attempting to splice it.
    """

    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    profile: Optional[str] = None,
    calibrated_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load and validate the merged configuration.

    Args:
        profile: e.g. "dev_8gb" or "workstation_32gb". Defaults to the
            MEDICAL_APPT_PROFILE environment variable, then "dev_8gb".
        calibrated_name: filename (without .yaml) under
            configs/calibrated/, e.g. "2026-09-18_exp2". If omitted, no
            calibrated overrides are applied and threshold/span defaults
            from base.yaml are used as-is -- fine for smoke-testing the
            pipeline, not for a real submission.
        overrides: an optional dict merged in last, for ad-hoc CLI/test
            overrides without writing a YAML file.
    """

    profile = profile or os.environ.get(DEFAULT_PROFILE_ENV_VAR, DEFAULT_PROFILE)

    merged = _load_yaml(CONFIGS_DIR / "base.yaml")
    merged = _deep_merge(merged, _load_yaml(CONFIGS_DIR / f"{profile}.yaml"))

    if calibrated_name:
        calibrated_path = CONFIGS_DIR / "calibrated" / f"{calibrated_name}.yaml"
        if not calibrated_path.exists():
            raise FileNotFoundError(
                f"Calibrated config {calibrated_path} not found. Run "
                "scripts/calibrate.py first, or omit calibrated_name to use "
                "base defaults (not recommended for a real submission)."
            )
        merged = _deep_merge(merged, _load_yaml(calibrated_path))

    # A lower-priority override than the explicit `overrides` param below,
    # for the one setting worth a dedicated env var: testing the LLM
    # verifier against a running server (e.g. via scripts/run_local_eval.py
    # --verifier llm) without first needing a calibrated config that sets
    # it, and without editing configs/*.yaml by hand.
    env_verifier_kind = os.environ.get("MEDICAL_APPT_VERIFIER_KIND")
    if env_verifier_kind:
        merged = _deep_merge(merged, {"verifier": {"kind": env_verifier_kind}})

    experiment = os.environ.get("MEDICAL_APPT_EXPERIMENT", "baseline")
    if experiment not in {"baseline", "exp10", "exp11", "exp12", "exp13", "exp14", "exp15", "exp16", "exp17", "exp18"}:
        raise ValueError(f"Unknown experiment: {experiment}")
    if experiment != "baseline":
        experiment_path = CONFIGS_DIR / "experiments" / f"{experiment}.yaml"
        if not experiment_path.exists():
            raise FileNotFoundError(experiment_path)
        merged = _deep_merge(merged, _load_yaml(experiment_path))

    if overrides:
        merged = _deep_merge(merged, overrides)

    merged["profile"] = profile
    return Config.model_validate(merged)


def latest_calibrated_name() -> Optional[str]:
    """The most recently written calibrated config, by filename sort order.

    scripts/calibrate.py names files ``<UTC timestamp>_<git short sha>.yaml``
    (see that script), so a plain sort is also a chronological sort. Returns
    None if nothing has been calibrated yet.
    """

    calibrated_dir = CONFIGS_DIR / "calibrated"
    candidates = sorted(p.stem for p in calibrated_dir.glob("*.yaml"))
    return candidates[-1] if candidates else None
