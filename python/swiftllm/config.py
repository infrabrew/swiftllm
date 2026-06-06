# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      config.py
# PATH:      /python/swiftllm/config.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - (stdlib only — no internal imports)
# USED BY:
#   - python/swiftllm/__init__.py      public re-exports
#   - python/swiftllm/engine.py        EngineConfig, SamplingParams, new inference configs
#   - python/swiftllm/training.py      GrpoConfig, CgarConfig, PrmConfig, LongRewardConfig
#   - python/swiftllm/sampling.py      SelfConsistencyConfig
#   - python/swiftllm/cli.py           argument → config mapping
# SEE ALSO:
#   - crates/swiftllm-core/src/sampling/self_consistency.rs   Rust counterpart for SelfConsistencyConfig
#   - crates/swiftllm-core/src/inference/refinement.rs        Rust counterpart for RefinementConfig
#   - crates/swiftllm-core/src/inference/verification.rs      Rust counterpart for VerificationConfig
#   - crates/swiftllm-core/src/serving/disaggregated.rs       Rust counterpart for DisaggregatedServingConfig
#   - crates/swiftllm-training/src/grpo.rs                    Rust counterpart for GrpoConfig
#   - crates/swiftllm-training/src/curriculum.rs              Rust counterpart for CgarConfig
#   - crates/swiftllm-training/src/process_reward.rs          Rust counterpart for PrmConfig
#   - crates/swiftllm-training/src/long_reward.rs             Rust counterpart for LongRewardConfig
#   - crates/swiftllm-models/src/layers/rlm.rs                Rust counterpart for RlmConfig
#   - crates/swiftllm-models/src/layers/dense_verification.rs Rust counterpart for DenseVerificationConfig
# ------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SwiftLLM Configuration Classes

This module provides configuration classes for the SwiftLLM inference engine,
training pipeline, and all research-derived features added in Phases 1-3:

  Phase 1 — Hybrid Model Architecture:
    (configured at model-load time via JambaConfig in Rust)

  Phase 2 — Training:
    GrpoConfig         Group Relative Policy Optimization (GRPO)
    CgarConfig         Curriculum-Guided Adaptive Recursion (CGAR)
    PrmConfig          Process Reward Model configuration
    LongRewardConfig   LongR dense token-level rewards

  Phase 3 — Inference:
    SelfConsistencyConfig      Self-consistency majority voting (Wang et al., 2022)
    RefinementConfig           Multi-round self-refinement (Madaan et al., 2023)
    VerificationConfig         Best-of-N dense verification & reranking
    DisaggregatedServingConfig Disaggregated prefill/decode serving (Splitwise)

  Phase 3 — Model-level Reasoning (new):
    RlmConfig                  Recursive Language Model (REPL state, variable binding)
    DenseVerificationConfig    Dense Verification Layer (full-capacity eval pass)

All configuration options can be overridden via environment variables with the
``SWIFTLLM_`` prefix.  For example, ``SWIFTLLM_GPU_MEMORY_UTILIZATION=0.95``
overrides ``EngineConfig.gpu_memory_utilization``.  Boolean values accept
``1/true/yes`` (case-insensitive).

See the README "Environment Variables" section for the full reference.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _env(name: str, default=None):
    """Read a SWIFTLLM_ environment variable (case-insensitive value)."""
    return os.environ.get(f"SWIFTLLM_{name}", default)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean SWIFTLLM_ environment variable."""
    val = _env(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Read an integer SWIFTLLM_ environment variable."""
    val = _env(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Read a float SWIFTLLM_ environment variable."""
    val = _env(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


class DataType(Enum):
    """Data type for model weights and computations."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    AUTO = "auto"


class QuantizationMethod(Enum):
    """Quantization method for model compression."""
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
    GGUF = "gguf"
    TURBOQUANT = "turboquant"
    BNB_4BIT = "4bit"
    BNB_8BIT = "8bit"


class SchedulerPolicy(Enum):
    """Scheduling policy for request batching."""
    FCFS = "fcfs"           # First Come First Served
    SJF = "sjf"             # Shortest Job First
    PRIORITY = "priority"   # Priority-based


class PreemptionMode(Enum):
    """Mode for handling preemption."""
    SWAP = "swap"           # Swap to CPU memory
    RECOMPUTE = "recompute" # Recompute from beginning


# ---------------------------------------------------------------------------
# Phase 3 — Inference Enhancements
# ---------------------------------------------------------------------------

class AnswerExtractor(Enum):
    """Strategy for extracting the final answer from generated text.

    Used by SelfConsistencyConfig. Mirrors ``AnswerExtractor`` in
    ``crates/swiftllm-core/src/sampling/self_consistency.rs``.
    """
    HEURISTIC = "heuristic"       # Last number / boxed expression heuristic
    AFTER_SENTINEL = "sentinel"   # Text after a configurable sentinel string
    LAST_LINE = "last_line"       # Last non-empty line of the output
    XML_TAG = "xml_tag"           # Content of a configurable XML tag


class StoppingCriterion(Enum):
    """When to stop the refinement loop.

    Mirrors ``StoppingCriterion`` in
    ``crates/swiftllm-core/src/inference/refinement.rs``.
    """
    MAX_ROUNDS = "max_rounds"           # Stop after a fixed number of rounds
    MIN_IMPROVEMENT = "min_improvement" # Stop when improvement falls below threshold
    EITHER = "either"                   # Stop when either condition is met


class ImprovementMetric(Enum):
    """How to measure improvement between refinement rounds.

    Mirrors ``ImprovementMetric`` in
    ``crates/swiftllm-core/src/inference/refinement.rs``.
    """
    EDIT_DISTANCE = "edit_distance"   # Normalised Levenshtein distance
    ANY_CHANGE = "any_change"         # 1.0 if text changed, 0.0 otherwise
    EXTERNAL_SCORE = "external_score" # Provided by caller via callback


class ScoringStrategy(Enum):
    """How to score candidates for Best-of-N selection.

    Mirrors ``ScoringStrategy`` in
    ``crates/swiftllm-core/src/inference/verification.rs``.
    """
    RULE_BASED = "rule_based"             # Heuristic rule scoring only
    NEURAL = "neural"                     # Neural PRM scorer only
    ENSEMBLE = "ensemble"                 # Weighted mix of rule, neural, logprob
    SEQUENCE_LOG_PROB = "sequence_logprob" # Raw sequence log-probability


class DisaggregatedPolicy(Enum):
    """Scheduling policy for disaggregated prefill/decode workers.

    Mirrors ``SchedulingPolicy`` in
    ``crates/swiftllm-core/src/serving/disaggregated.rs``.
    """
    ROUND_ROBIN = "round_robin"       # Cycle through workers in order
    LEAST_LOADED = "least_loaded"     # Route to worker with fewest active requests
    LOCALITY_AWARE = "locality_aware" # Prefer worker that already holds the KV cache


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency majority voting (Wang et al., 2022).

    Generates ``num_samples`` independent reasoning chains and returns the
    plurality-majority answer.  Corresponds to
    ``SelfConsistencyConfig`` in ``sampling/self_consistency.rs``.

    Attributes:
        num_samples: Number of independent samples to generate (≥ 2).
        extractor: Strategy to extract the final answer from each sample.
        answer_sentinel: Sentinel string used by ``AFTER_SENTINEL`` extractor
            (e.g. ``"The answer is"``).
        answer_tag: XML tag name used by ``XML_TAG`` extractor
            (e.g. ``"answer"`` matches ``<answer>…</answer>``).
        temperature: Sampling temperature (should be > 0 for diversity).
    """
    num_samples: int = 8
    extractor: AnswerExtractor = AnswerExtractor.HEURISTIC
    answer_sentinel: str = "The answer is"
    answer_tag: str = "answer"
    temperature: float = 0.8

    def __post_init__(self):
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be >= 2, got {self.num_samples}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0 for self-consistency, got {self.temperature}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "extractor": self.extractor.value,
            "answer_sentinel": self.answer_sentinel,
            "answer_tag": self.answer_tag,
            "temperature": self.temperature,
        }


@dataclass
class RefinementConfig:
    """Configuration for multi-round self-refinement (Madaan et al., 2023).

    Corresponds to ``RefinementConfig`` in ``inference/refinement.rs``.

    Attributes:
        max_rounds: Maximum number of critique→revision cycles (1–20).
        min_improvement: Improvement threshold below which refinement stops
            (used by ``MIN_IMPROVEMENT`` and ``EITHER`` criteria, 0.0–1.0).
        stopping_criterion: When to stop refinement.
        improvement_metric: How to measure improvement between rounds.
        critique_template: Optional prompt template prepended to each critique
            call; ``{output}`` is replaced with the current candidate text.
    """
    max_rounds: int = 3
    min_improvement: float = 0.05
    stopping_criterion: StoppingCriterion = StoppingCriterion.EITHER
    improvement_metric: ImprovementMetric = ImprovementMetric.EDIT_DISTANCE
    critique_template: Optional[str] = None

    def __post_init__(self):
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if not 0.0 <= self.min_improvement <= 1.0:
            raise ValueError(f"min_improvement must be in [0, 1], got {self.min_improvement}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "min_improvement": self.min_improvement,
            "stopping_criterion": self.stopping_criterion.value,
            "improvement_metric": self.improvement_metric.value,
            "critique_template": self.critique_template,
        }


@dataclass
class VerificationConfig:
    """Configuration for dense verification and Best-of-N reranking.

    Corresponds to ``VerificationConfig`` in ``inference/verification.rs``.

    Attributes:
        num_candidates: Number of candidates to generate and then rank (≥ 2).
        scoring_strategy: How to score and rank candidates.
        rule_weight: Weight for the rule-based score in ``ENSEMBLE`` mode.
        neural_weight: Weight for the neural PRM score in ``ENSEMBLE`` mode.
        logprob_weight: Weight for the sequence log-prob in ``ENSEMBLE`` mode.
        neural_model: Path to the neural PRM model (``NEURAL``/``ENSEMBLE`` only).
    """
    num_candidates: int = 8
    scoring_strategy: ScoringStrategy = ScoringStrategy.RULE_BASED
    rule_weight: float = 0.5
    neural_weight: float = 0.3
    logprob_weight: float = 0.2
    neural_model: Optional[str] = None

    def __post_init__(self):
        if self.num_candidates < 2:
            raise ValueError(f"num_candidates must be >= 2, got {self.num_candidates}")
        if self.scoring_strategy in (ScoringStrategy.NEURAL, ScoringStrategy.ENSEMBLE):
            if self.neural_model is None:
                import warnings
                warnings.warn(
                    "scoring_strategy requires a neural model; set neural_model= "
                    "or switch to RULE_BASED / SEQUENCE_LOG_PROB.",
                    UserWarning,
                    stacklevel=2,
                )
        if abs(self.rule_weight + self.neural_weight + self.logprob_weight - 1.0) > 1e-5:
            import warnings
            warnings.warn(
                f"Ensemble weights sum to {self.rule_weight + self.neural_weight + self.logprob_weight:.4f}, "
                "not 1.0; results will be rescaled internally.",
                UserWarning,
                stacklevel=2,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_candidates": self.num_candidates,
            "scoring_strategy": self.scoring_strategy.value,
            "rule_weight": self.rule_weight,
            "neural_weight": self.neural_weight,
            "logprob_weight": self.logprob_weight,
            "neural_model": self.neural_model,
        }


@dataclass
class DisaggregatedServingConfig:
    """Configuration for disaggregated prefill/decode serving (Splitwise/DistServe).

    Routes prefill (compute-bound) and decode (bandwidth-bound) onto dedicated
    worker pools.  Corresponds to ``DisaggregatedConfig`` in
    ``serving/disaggregated.rs``.

    Attributes:
        num_prefill_workers: Number of dedicated prefill workers (≥ 1).
        num_decode_workers: Number of dedicated decode workers (≥ 1).
        scheduling_policy: How to assign requests to workers.
        kv_transfer_timeout_ms: Timeout for KV-cache transfers between workers (ms).
        enable_auto_ratio: Automatically compute the optimal prefill/decode ratio
            using ``optimal_worker_ratio()`` at startup.
    """
    num_prefill_workers: int = 2
    num_decode_workers: int = 6
    scheduling_policy: DisaggregatedPolicy = DisaggregatedPolicy.LEAST_LOADED
    kv_transfer_timeout_ms: int = 100
    enable_auto_ratio: bool = False

    def __post_init__(self):
        if self.num_prefill_workers < 1:
            raise ValueError(f"num_prefill_workers must be >= 1, got {self.num_prefill_workers}")
        if self.num_decode_workers < 1:
            raise ValueError(f"num_decode_workers must be >= 1, got {self.num_decode_workers}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_prefill_workers": self.num_prefill_workers,
            "num_decode_workers": self.num_decode_workers,
            "scheduling_policy": self.scheduling_policy.value,
            "kv_transfer_timeout_ms": self.kv_transfer_timeout_ms,
            "enable_auto_ratio": self.enable_auto_ratio,
        }


# ---------------------------------------------------------------------------
# TurboQuant — KV Cache Compression (ICLR 2026)
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression.

    TurboQuant (Zandieh et al., ICLR 2026) is an online vector quantization
    algorithm that compresses KV cache vectors via random rotation followed
    by per-coordinate scalar quantization against a precomputed
    Beta-distribution codebook. It achieves near-optimal distortion with no
    training required.

    Mirrors ``TurboQuantConfig`` in ``crates/swiftllm-core/src/config.rs``.

    Attributes:
        key_bits: Bit-width per channel for key cache (1-16, default 4).
        value_bits: Bit-width per channel for value cache (1-16, default 4).
        residual_length: Rotation matrix residual length. None = head_dim.
        use_inner_product_variant: Use TurboQuantProd (unbiased dot products).
        group_size: Per-group quantization size (0 = per-head, recommended).
        quantize_prefill: Whether to quantize KV during prefill phase.
        seed: Random seed for the rotation matrix.
    """
    key_bits: int = 4
    value_bits: int = 4
    residual_length: Optional[int] = None
    use_inner_product_variant: bool = False
    group_size: int = 0
    quantize_prefill: bool = True
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if not (1 <= self.key_bits <= 16):
            raise ValueError(f"key_bits must be in [1, 16], got {self.key_bits}")
        if not (1 <= self.value_bits <= 16):
            raise ValueError(f"value_bits must be in [1, 16], got {self.value_bits}")
        if self.use_inner_product_variant and (self.key_bits < 2 or self.value_bits < 2):
            raise ValueError(
                "TurboQuantProd requires at least 2 bits (1 for MSE + 1 for JL sign)"
            )

    @classmethod
    def quality_neutral(cls) -> "TurboQuantConfig":
        """Preset: 4-bit keys + 3-bit values (3.5-bit avg, ~4x compression).

        Matches full-precision quality on most benchmarks.
        """
        return cls(key_bits=4, value_bits=3)

    @classmethod
    def aggressive(cls) -> "TurboQuantConfig":
        """Preset: 3-bit keys + 2-bit values (2.5-bit avg, ~5x compression).

        Marginal quality degradation (approx 0.1 ppl on LLaMA-2-7B).
        """
        return cls(key_bits=3, value_bits=2)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to FP16 storage."""
        return (self.key_bits + self.value_bits) / 2.0 / 16.0

    @property
    def memory_reduction(self) -> float:
        """Memory reduction factor (e.g. 4.0 means 4x less memory)."""
        return 1.0 / self.compression_ratio if self.compression_ratio > 0 else float('inf')

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "residual_length": self.residual_length,
            "use_inner_product_variant": self.use_inner_product_variant,
            "group_size": self.group_size,
            "quantize_prefill": self.quantize_prefill,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Phase 3 — Model-level Reasoning (RLM + Dense Verification)
# ---------------------------------------------------------------------------

class RlmMode(Enum):
    """Operating mode for the Recursive Language Model layer.

    Mirrors ``RlmMode`` in ``crates/swiftllm-models/src/layers/rlm.rs``.
    """
    DISABLED   = "disabled"    # Pass-through; no recursion or REPL side-effects
    SHALLOW    = "shallow"     # max_depth=1; single sub-call, no REPL state
    REASONING  = "reasoning"   # max_depth=3; full REPL + variable binding
    AGENTIC    = "agentic"     # max_depth=5; for multi-step agentic tasks


class VerificationStrategy(Enum):
    """Scoring strategy for the Dense Verification Layer.

    Mirrors ``VerificationStrategy`` in
    ``crates/swiftllm-models/src/layers/dense_verification.rs``.
    """
    DISABLED       = "disabled"        # Skip verification entirely
    SCORE_ONLY     = "score_only"      # Score but always accept
    GATE           = "gate"            # Reject drafts below min_confidence
    GATE_AND_REGEN = "gate_and_regen"  # Reject and regenerate (up to max_attempts)


@dataclass
class RlmConfig:
    """Configuration for the Recursive Language Model (RLM) layer.

    The RLM extends autoregressive generation with bounded recursive self-
    calling and a symbolic REPL sandbox.  Complex sub-problems are solved
    recursively at shallower depth and the sub-solutions are integrated back
    into the main hidden state via a learned gating mechanism.

    Corresponds to ``RlmConfig`` in
    ``crates/swiftllm-models/src/layers/rlm.rs``.

    References:
        "Architecting the Next-Generation Agentic Paradigm: A Hybrid
         Synthesis of Mamba-3, Mixture of Experts, Recursive Language
         Models, and Dense Verification" (2024)

    Attributes:
        mode: Operating mode (disabled / shallow / reasoning / agentic).
        max_depth: Maximum recursion depth (0 = direct solve, no sub-calls).
            Paper recommendation: 2–4 for math/coding, 1 for language tasks.
        enable_repl: Enable the symbolic REPL sandbox with variable binding.
            When False the RLM acts as a plain pass-through MLP.
        var_binding_slots: Number of soft key-value memory slots in the
            REPL variable binding table.  Default: 32.
        depth_hidden_size: Hidden size of the complexity-classifier MLP.
            Defaults to d_model // 4 (set by the Rust layer at construction).
        early_exit_threshold: Confidence threshold for skipping deeper recursion.
            If the scheduler assigns depth 0 with confidence ≥ threshold the
            sub-call is skipped even if the model predicted deeper recursion.
        d_subproblem: Projection size for sub-problem embeddings passed to
            recursive sub-calls.  Defaults to d_model // 2.
    """
    mode: RlmMode = RlmMode.REASONING
    max_depth: int = 3
    enable_repl: bool = True
    var_binding_slots: int = 32
    depth_hidden_size: Optional[int] = None   # None → d_model // 4 (set in Rust)
    early_exit_threshold: float = 0.92
    d_subproblem: Optional[int] = None         # None → d_model // 2 (set in Rust)

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {self.max_depth}")
        if self.var_binding_slots < 1:
            raise ValueError(f"var_binding_slots must be >= 1, got {self.var_binding_slots}")
        if not 0.0 < self.early_exit_threshold <= 1.0:
            raise ValueError(
                f"early_exit_threshold must be in (0, 1], got {self.early_exit_threshold}"
            )
        if self.mode == RlmMode.DISABLED:
            self.max_depth = 0
            self.enable_repl = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "max_depth": self.max_depth,
            "enable_repl": self.enable_repl,
            "var_binding_slots": self.var_binding_slots,
            "depth_hidden_size": self.depth_hidden_size,
            "early_exit_threshold": self.early_exit_threshold,
            "d_subproblem": self.d_subproblem,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RlmConfig":
        d = dict(d)
        if "mode" in d:
            d["mode"] = RlmMode(d["mode"])
        return cls(**d)


@dataclass
class DenseVerificationConfig:
    """Configuration for the Dense Verification Layer.

    After generation completes the Dense Verification Layer performs one
    additional read-over pass on the full output, scoring each token and
    reasoning step against the REPL execution trace.  Outputs below
    ``min_confidence`` trigger re-generation up to ``max_regen_attempts`` times.

    Corresponds to ``DenseVerificationConfig`` in
    ``crates/swiftllm-models/src/layers/dense_verification.rs``.

    References:
        "Hybrid_Mamba3-RLM-Reasoning-Architecture.pdf", §3.5
        "Let's Verify Step by Step" — Lightman et al. 2023

    Attributes:
        strategy: When / how to act on verification scores.
        num_verification_heads: Cross-attention heads for draft ↔ REPL-trace
            attention.  Default: 8.
        min_confidence: Global score threshold below which the draft is
            rejected.  Range (0, 1].  Default: 0.80 for reasoning tasks.
        max_regen_attempts: Maximum re-generation attempts when
            strategy = GATE_AND_REGEN.  Default: 3.
        score_repl_steps: Also compute per-REPL-step confidence scores
            (uses explicit ``Verify`` step confidences from the trace).
    """
    strategy: VerificationStrategy = VerificationStrategy.GATE_AND_REGEN
    num_verification_heads: int = 8
    min_confidence: float = 0.80
    max_regen_attempts: int = 3
    score_repl_steps: bool = True

    def __post_init__(self):
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in (0, 1], got {self.min_confidence}"
            )
        if self.max_regen_attempts < 1:
            raise ValueError(
                f"max_regen_attempts must be >= 1, got {self.max_regen_attempts}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "num_verification_heads": self.num_verification_heads,
            "min_confidence": self.min_confidence,
            "max_regen_attempts": self.max_regen_attempts,
            "score_repl_steps": self.score_repl_steps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DenseVerificationConfig":
        d = dict(d)
        if "strategy" in d:
            d["strategy"] = VerificationStrategy(d["strategy"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Phase 2 — Training Enhancements
# ---------------------------------------------------------------------------

class PrmAggregation(Enum):
    """How to aggregate per-step PRM scores into a single reward.

    Mirrors ``PrmAggregation`` in
    ``crates/swiftllm-training/src/process_reward.rs``.
    """
    MIN = "min"
    MEAN = "mean"
    PRODUCT = "product"
    LAST_STEP = "last_step"
    WEIGHTED_MEAN = "weighted_mean"


class DenseAggregation(Enum):
    """How to aggregate token-level dense rewards into a scalar.

    Mirrors ``DenseAggregation`` in
    ``crates/swiftllm-training/src/long_reward.rs``.
    """
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    LAST = "last"


@dataclass
class GrpoConfig:
    """Configuration for Group Relative Policy Optimization (GRPO).

    GRPO fine-tunes a model using RL without a critic model by computing
    group-relative advantages.  Corresponds to ``GrpoConfig`` in
    ``crates/swiftllm-training/src/grpo.rs``.

    Attributes:
        group_size: Number of samples per prompt in each group (G, ≥ 2).
        clip_eps: PPO-style probability ratio clipping threshold (ε).
        kl_coeff: KL-divergence penalty coefficient (β).
        correctness_weight: Weight for the correctness reward.
        format_weight: Weight for the format / structure reward.
        length_penalty_weight: Weight for the length-deviation penalty.
        reference_model: Path to the frozen reference model for KL divergence.
            Defaults to the same model as the policy when None.
    """
    group_size: int = 8
    clip_eps: float = 0.2
    kl_coeff: float = 0.04
    correctness_weight: float = 1.0
    format_weight: float = 0.2
    length_penalty_weight: float = 0.1
    reference_model: Optional[str] = None

    def __post_init__(self):
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {self.group_size}")
        if self.clip_eps <= 0:
            raise ValueError(f"clip_eps must be > 0, got {self.clip_eps}")
        if self.kl_coeff < 0:
            raise ValueError(f"kl_coeff must be >= 0, got {self.kl_coeff}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_size": self.group_size,
            "clip_eps": self.clip_eps,
            "kl_coeff": self.kl_coeff,
            "correctness_weight": self.correctness_weight,
            "format_weight": self.format_weight,
            "length_penalty_weight": self.length_penalty_weight,
            "reference_model": self.reference_model,
        }


@dataclass
class CgarConfig:
    """Configuration for Curriculum-Guided Adaptive Recursion (CGAR).

    Implements a phased depth curriculum: shallow → medium → full depth.
    Corresponds to ``CgarConfig`` in
    ``crates/swiftllm-training/src/curriculum.rs``.

    Attributes:
        shallow_end: Training fraction at which shallow phase ends (0–1).
        medium_end: Training fraction at which medium phase ends (0–1).
        min_layers: Minimum number of active layers during shallow phase.
        max_layers: Maximum layers (= total model layers at full depth).
        enable_phased_specialisation: Also apply phased attention/SSM
            specialisation (for Jamba-style hybrid models).
        attention_lead_end: Fraction at which attention-lead phase ends
            (only used when ``enable_phased_specialisation`` is True).
    """
    shallow_end: float = 0.30
    medium_end: float = 0.60
    min_layers: Optional[int] = None   # None → num_layers // 3
    max_layers: Optional[int] = None   # None → num_layers
    enable_phased_specialisation: bool = False
    attention_lead_end: float = 0.40

    def __post_init__(self):
        if not 0.0 < self.shallow_end < self.medium_end < 1.0:
            raise ValueError(
                f"Require 0 < shallow_end ({self.shallow_end}) < medium_end ({self.medium_end}) < 1"
            )
        if self.min_layers is not None and self.min_layers < 1:
            raise ValueError(f"min_layers must be >= 1, got {self.min_layers}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shallow_end": self.shallow_end,
            "medium_end": self.medium_end,
            "min_layers": self.min_layers,
            "max_layers": self.max_layers,
            "enable_phased_specialisation": self.enable_phased_specialisation,
            "attention_lead_end": self.attention_lead_end,
        }


@dataclass
class PrmConfig:
    """Configuration for the Process Reward Model (PRM).

    Provides step-level feedback on reasoning chains.  Corresponds to
    ``PrmConfig`` in ``crates/swiftllm-training/src/process_reward.rs``.

    Attributes:
        aggregation: How to combine per-step scores into a single reward.
        outcome_weight: Weight for the outcome (final-answer) reward (0–1).
        prm_weight: Weight for the PRM (step-level) reward (0–1).
        step_separator: String that delineates steps in the reasoning chain.
        neural_model: Path to a neural PRM model; if None, uses the rule-based
            heuristic (``RulePrm``) which requires no additional model.
    """
    aggregation: PrmAggregation = PrmAggregation.LAST_STEP
    outcome_weight: float = 0.5
    prm_weight: float = 0.5
    step_separator: str = "\n\n"
    neural_model: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.outcome_weight <= 1.0:
            raise ValueError(f"outcome_weight must be in [0, 1], got {self.outcome_weight}")
        if not 0.0 <= self.prm_weight <= 1.0:
            raise ValueError(f"prm_weight must be in [0, 1], got {self.prm_weight}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation": self.aggregation.value,
            "outcome_weight": self.outcome_weight,
            "prm_weight": self.prm_weight,
            "step_separator": self.step_separator,
            "neural_model": self.neural_model,
        }


@dataclass
class LongRewardConfig:
    """Configuration for LongR dense token-level rewards.

    Computes per-token relative information gain vs. a reference model:
    ``r_t = NLL_ref − NLL_model``.  Yields a 9% gain on LongBench v2.
    Corresponds to ``LongRewardConfig`` in
    ``crates/swiftllm-training/src/long_reward.rs``.

    Attributes:
        weight: Scalar weight applied to the dense reward before adding to the
            total reward (0.0 = disabled).
        aggregation: How to reduce token-level rewards to a scalar.
        normalise: Whether to z-score-normalise rewards within each batch.
        reference_model: Path to the frozen reference model for NLL computation.
            Defaults to the same model as the policy when None.
    """
    weight: float = 0.1
    aggregation: DenseAggregation = DenseAggregation.MEAN
    normalise: bool = True
    reference_model: Optional[str] = None

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight": self.weight,
            "aggregation": self.aggregation.value,
            "normalise": self.normalise,
            "reference_model": self.reference_model,
        }


@dataclass
class SamplingParams:
    """Parameters for text generation sampling.

    Attributes:
        temperature: Sampling temperature. Higher values produce more random outputs.
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling. Only consider top k tokens.
        min_p: Minimum probability threshold for sampling.
        max_tokens: Maximum number of tokens to generate.
        min_tokens: Minimum number of tokens to generate.
        stop: List of stop strings. Generation stops when any is encountered.
        stop_token_ids: List of token IDs that trigger stop.
        presence_penalty: Penalty for token presence in generated text.
        frequency_penalty: Penalty for token frequency in generated text.
        repetition_penalty: Multiplicative penalty for repetition.
        seed: Random seed for reproducibility.
        skip_special_tokens: Whether to skip special tokens in output.
        include_stop_str_in_output: Whether to include stop string in output.
        logprobs: Number of log probabilities to return per token.
        prompt_logprobs: Number of prompt log probabilities to return.
        best_of: Number of sequences to generate and return the best.
        n: Number of output sequences to return.
        use_beam_search: Whether to use beam search instead of sampling.
        length_penalty: Penalty for sequence length in beam search.
        early_stopping: Whether to stop beam search early.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 256
    min_tokens: int = 0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    skip_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    best_of: int = 1
    n: int = 1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False

    def __post_init__(self):
        """Validate sampling parameters."""
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {self.top_k}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.best_of < self.n:
            raise ValueError(f"best_of must be >= n, got best_of={self.best_of}, n={self.n}")
        if self.use_beam_search and self.temperature != 0:
            raise ValueError("temperature must be 0 when using beam search")
        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be <= max_tokens ({self.max_tokens})"
            )
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingParams":
        """Create SamplingParams from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "stop": self.stop,
            "stop_token_ids": self.stop_token_ids,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "skip_special_tokens": self.skip_special_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "best_of": self.best_of,
            "n": self.n,
            "use_beam_search": self.use_beam_search,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }


@dataclass
class EngineConfig:
    """Configuration for the SwiftLLM inference engine.

    Attributes:
        model: Path to the model or HuggingFace model ID.
        tokenizer: Path to tokenizer. Defaults to model path.
        dtype: Data type for model weights.
        quantization: Quantization method if any.
        max_model_len: Maximum sequence length for the model.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of pipeline parallel stages.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
        block_size: Block size for PagedAttention.
        swap_space: Swap space in GiB for CPU offloading.
        max_num_seqs: Maximum number of concurrent sequences.
        max_num_batched_tokens: Maximum tokens per batch.
        enable_prefix_caching: Enable automatic prefix caching.
        enable_chunked_prefill: Enable chunked prefill for long prompts.
        max_paddings: Maximum padding tokens allowed.
        scheduler_policy: Scheduling policy for requests.
        preemption_mode: Mode for handling preemption.
        trust_remote_code: Trust remote code from HuggingFace.
        download_dir: Directory for downloading models.
        seed: Random seed for reproducibility.
        device: Device to use ('cuda', 'cpu', 'auto').
        rlm: Recursive Language Model config (Phase 3). None = disabled.
        dense_verification: Dense Verification Layer config (Phase 3). None = disabled.
    """
    model: str = ""
    tokenizer: Optional[str] = None
    dtype: DataType = field(default_factory=lambda: DataType(_env("DTYPE", "auto")))
    quantization: QuantizationMethod = field(default_factory=lambda: QuantizationMethod(_env("QUANTIZATION", "none")))
    max_model_len: Optional[int] = field(default_factory=lambda: _env_int("MAX_MODEL_LEN"))
    tensor_parallel_size: int = field(default_factory=lambda: _env_int("TENSOR_PARALLEL_SIZE", 1))
    pipeline_parallel_size: int = field(default_factory=lambda: _env_int("PIPELINE_PARALLEL_SIZE", 1))
    gpu_memory_utilization: float = field(default_factory=lambda: _env_float("GPU_MEMORY_UTILIZATION", 0.90))
    block_size: int = field(default_factory=lambda: _env_int("BLOCK_SIZE", 16))
    swap_space: float = field(default_factory=lambda: _env_float("SWAP_SPACE", 4.0))
    max_num_seqs: int = field(default_factory=lambda: _env_int("MAX_NUM_SEQS", 256))
    max_num_batched_tokens: Optional[int] = field(default_factory=lambda: _env_int("MAX_NUM_BATCHED_TOKENS"))
    enable_prefix_caching: bool = field(default_factory=lambda: _env_bool("ENABLE_PREFIX_CACHING"))
    enable_chunked_prefill: bool = field(default_factory=lambda: _env_bool("ENABLE_CHUNKED_PREFILL"))
    max_paddings: int = field(default_factory=lambda: _env_int("MAX_PADDINGS", 256))
    scheduler_policy: SchedulerPolicy = field(default_factory=lambda: SchedulerPolicy(_env("SCHEDULER_POLICY", "fcfs")))
    preemption_mode: PreemptionMode = field(default_factory=lambda: PreemptionMode(_env("PREEMPTION_MODE", "swap")))
    trust_remote_code: bool = field(default_factory=lambda: _env_bool("TRUST_REMOTE_CODE"))
    download_dir: Optional[str] = field(default_factory=lambda: _env("MODEL_DIR"))
    seed: int = field(default_factory=lambda: _env_int("SEED", 0))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    # Speculative decoding
    speculative_model: Optional[str] = field(default_factory=lambda: _env("SPECULATIVE_MODEL"))
    num_speculative_tokens: int = field(default_factory=lambda: _env_int("NUM_SPECULATIVE_TOKENS", 5))
    speculative_max_model_len: Optional[int] = field(default_factory=lambda: _env_int("SPECULATIVE_MAX_MODEL_LEN"))

    # LoRA
    enable_lora: bool = field(default_factory=lambda: _env_bool("ENABLE_LORA"))
    max_loras: int = field(default_factory=lambda: _env_int("MAX_LORAS", 1))
    max_lora_rank: int = field(default_factory=lambda: _env_int("MAX_LORA_RANK", 16))
    lora_dtype: Optional[DataType] = None

    # Performance tuning
    enforce_eager: bool = field(default_factory=lambda: _env_bool("ENFORCE_EAGER"))
    kv_cache_dtype: str = field(default_factory=lambda: _env("KV_CACHE_DTYPE", "auto"))
    num_gpu_layers: Optional[int] = field(default_factory=lambda: _env_int("NUM_GPU_LAYERS"))
    cpu_offload_gb: float = field(default_factory=lambda: _env_float("CPU_OFFLOAD_GB", 0.0))
    max_parallel_loading: int = field(default_factory=lambda: _env_int("MAX_PARALLEL_LOADING", 1))
    gpu_overhead_mb: int = field(default_factory=lambda: _env_int("GPU_OVERHEAD_MB", 0))
    flash_attention: bool = field(default_factory=lambda: _env_bool("FLASH_ATTENTION", True))
    keep_alive_secs: int = field(default_factory=lambda: _env_int("KEEP_ALIVE", 300))
    num_parallel_slots: int = field(default_factory=lambda: _env_int("NUM_PARALLEL", 1))
    max_loaded_models: int = field(default_factory=lambda: _env_int("MAX_LOADED_MODELS", 1))

    # Phase 3 — Inference enhancements (all optional; None = feature disabled)
    self_consistency: Optional[SelfConsistencyConfig] = None
    """Self-consistency majority voting config. Set to enable generate_with_self_consistency()."""
    refinement: Optional[RefinementConfig] = None
    """Multi-round self-refinement config. Set to enable generate_with_refinement()."""
    verification: Optional[VerificationConfig] = None
    """Best-of-N dense verification config. Set to enable generate_best_of_n()."""
    disaggregated_serving: Optional[DisaggregatedServingConfig] = None
    """Disaggregated prefill/decode serving config. Set to enable disaggregated mode."""

    # Phase 3 — Model-level reasoning (RLM + Dense Verification)
    rlm: Optional[RlmConfig] = None
    """Recursive Language Model config. Set to enable generate_with_rlm()."""
    dense_verification: Optional[DenseVerificationConfig] = None
    """Dense Verification Layer config. Set to enable generate_with_dense_verification()."""

    # TurboQuant KV cache compression
    turbo_quant: Optional[TurboQuantConfig] = None
    """TurboQuant KV cache compression (ICLR 2026). Set to enable compressed KV cache."""

    def __post_init__(self):
        """Validate configuration."""
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if self.block_size not in [8, 16, 32]:
            raise ValueError(f"block_size must be 8, 16, or 32, got {self.block_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.tokenizer is None:
            self.tokenizer = self.model

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        """Create EngineConfig from a dictionary (does not mutate input)."""
        d = dict(d)  # Avoid mutating caller's dict
        # Convert string enums
        if "dtype" in d and isinstance(d["dtype"], str):
            d["dtype"] = DataType(d["dtype"])
        if "quantization" in d and isinstance(d["quantization"], str):
            d["quantization"] = QuantizationMethod(d["quantization"])
        if "scheduler_policy" in d and isinstance(d["scheduler_policy"], str):
            d["scheduler_policy"] = SchedulerPolicy(d["scheduler_policy"])
        if "preemption_mode" in d and isinstance(d["preemption_mode"], str):
            d["preemption_mode"] = PreemptionMode(d["preemption_mode"])
        # Deserialise nested Phase-3 configs
        if "self_consistency" in d and isinstance(d["self_consistency"], dict):
            sc = dict(d["self_consistency"])
            if "extractor" in sc:
                sc["extractor"] = AnswerExtractor(sc["extractor"])
            d["self_consistency"] = SelfConsistencyConfig(**sc)
        if "refinement" in d and isinstance(d["refinement"], dict):
            rc = dict(d["refinement"])
            if "stopping_criterion" in rc:
                rc["stopping_criterion"] = StoppingCriterion(rc["stopping_criterion"])
            if "improvement_metric" in rc:
                rc["improvement_metric"] = ImprovementMetric(rc["improvement_metric"])
            d["refinement"] = RefinementConfig(**rc)
        if "verification" in d and isinstance(d["verification"], dict):
            vc = dict(d["verification"])
            if "scoring_strategy" in vc:
                vc["scoring_strategy"] = ScoringStrategy(vc["scoring_strategy"])
            d["verification"] = VerificationConfig(**vc)
        if "disaggregated_serving" in d and isinstance(d["disaggregated_serving"], dict):
            ds = dict(d["disaggregated_serving"])
            if "scheduling_policy" in ds:
                ds["scheduling_policy"] = DisaggregatedPolicy(ds["scheduling_policy"])
            d["disaggregated_serving"] = DisaggregatedServingConfig(**ds)
        if "rlm" in d and isinstance(d["rlm"], dict):
            d["rlm"] = RlmConfig.from_dict(d["rlm"])
        if "dense_verification" in d and isinstance(d["dense_verification"], dict):
            d["dense_verification"] = DenseVerificationConfig.from_dict(d["dense_verification"])
        if "turbo_quant" in d and isinstance(d["turbo_quant"], dict):
            d["turbo_quant"] = TurboQuantConfig(**d["turbo_quant"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Enum):
                value = value.value
            elif hasattr(value, "to_dict"):
                value = value.to_dict()
            result[field_name] = value
        return result


@dataclass
class ServerConfig:
    """Configuration for the SwiftLLM HTTP server.

    Attributes:
        host: Host to bind to.
        port: Port to bind to.
        api_key: API key for authentication.
        root_path: Root path for the API.
        ssl_keyfile: Path to SSL key file.
        ssl_certfile: Path to SSL certificate file.
        cors_allow_origins: Allowed CORS origins.
        max_log_len: Maximum log length for requests.
        response_role: Default role for responses.
        served_model_name: Name to use for the served model.
    """
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    api_key: Optional[str] = field(default_factory=lambda: _env("API_KEY"))
    root_path: str = field(default_factory=lambda: _env("ROOT_PATH", ""))
    ssl_keyfile: Optional[str] = field(default_factory=lambda: _env("SSL_KEYFILE"))
    ssl_certfile: Optional[str] = field(default_factory=lambda: _env("SSL_CERTFILE"))
    cors_allow_origins: List[str] = field(default_factory=lambda: (_env("CORS_ALLOW_ORIGINS", "*")).split(","))
    max_log_len: Optional[int] = field(default_factory=lambda: _env_int("MAX_LOG_LEN"))
    response_role: str = field(default_factory=lambda: _env("RESPONSE_ROLE", "assistant"))
    served_model_name: Optional[str] = field(default_factory=lambda: _env("SERVED_MODEL_NAME"))

    # Limits
    max_model_len_limit: Optional[int] = field(default_factory=lambda: _env_int("MAX_MODEL_LEN_LIMIT"))
    max_num_seqs_limit: Optional[int] = field(default_factory=lambda: _env_int("MAX_NUM_SEQS_LIMIT"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServerConfig":
        """Create ServerConfig from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LoRARequest:
    """Request to use a specific LoRA adapter.

    Attributes:
        lora_name: Unique name for this LoRA adapter.
        lora_path: Path to the LoRA adapter weights.
        lora_local_path: Local path if different from lora_path.
    """
    lora_name: str
    lora_path: str
    lora_local_path: Optional[str] = None

    @property
    def lora_int_id(self) -> int:
        """Return a unique integer ID for this LoRA."""
        return hash(self.lora_name) & 0xFFFFFFFF

# ------------------------------------------------------------------------------
# END OF FILE: config.py
# REPO PATH:   /swiftllm/python/swiftllm/config.py
# INTEGRATES:  engine.py · training.py · sampling.py · cli.py · __init__.py
#              Rust: self_consistency.rs · refinement.rs · verification.rs
#              Rust: disaggregated.rs · grpo.rs · curriculum.rs
#              Rust: process_reward.rs · long_reward.rs
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
