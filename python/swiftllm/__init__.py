# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      __init__.py
# PATH:      /python/swiftllm/__init__.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/engine.py    LLM, AsyncLLM, LLMEngine, output types
#   - python/swiftllm/config.py    all config dataclasses
#   - python/swiftllm/sampling.py  SamplingStrategy, SelfConsistencySampler
#   - python/swiftllm/training.py  Trainer, GrpoTrainer, convenience functions
# USED BY:
#   - user code  (top-level public API)
#   - python/swiftllm/cli.py       lazy imports via __getattr__
# SEE ALSO:
#   - crates/swiftllm-core/src/lib.rs             Rust core library public API
#   - crates/swiftllm-training/src/lib.rs         Rust training library public API
#   - crates/swiftllm-models/src/layers/rlm.rs            RlmLayer
#   - crates/swiftllm-models/src/layers/dense_verification.rs  DenseVerificationLayer
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

"""SwiftLLM - High-performance LLM Inference Engine

SwiftLLM is a fast and memory-efficient inference engine for large language models,
featuring PagedAttention, continuous batching, and multi-GPU support.

Phase 2 (training) and Phase 3 (inference) research additions are also exposed here:

  Inference (Phase 3):
    generate_with_self_consistency()     — majority voting over N chains
    generate_with_refinement()           — iterative self-refinement
    generate_best_of_n()                 — Best-of-N dense verification
    generate_with_rlm()                  — Recursive Language Model with REPL sandbox
    generate_with_dense_verification()   — cross-attention token/step confidence scoring

  Training (Phase 2):
    GrpoTrainer / grpo_train()        — GRPO RL fine-tuning
    GrpoConfig, CgarConfig, PrmConfig, LongRewardConfig

Example usage:
    >>> from swiftllm import LLM, SamplingParams
    >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
    >>> outputs = llm.generate(["Hello, how are you?"], SamplingParams(temperature=0.7))
    >>> print(outputs[0].outputs[0].text)

    >>> from swiftllm.config import SelfConsistencyConfig
    >>> results = llm.generate_with_self_consistency("What is 12 × 15?",
    ...     config=SelfConsistencyConfig(num_samples=8))
    >>> print(results[0].answer)
"""

from .engine import (
    LLM,
    AsyncLLM,
    LLMEngine,
    RequestOutput,
    CompletionOutput,
    RefinementOutput,
    VerifiedOutput,
    RlmOutput,
    DenseVerificationOutput,
)
from .config import (
    SamplingParams,
    EngineConfig,
    ServerConfig,
    LoRARequest,
    # Phase 3 — Inference configs
    SelfConsistencyConfig,
    RefinementConfig,
    VerificationConfig,
    DisaggregatedServingConfig,
    AnswerExtractor,
    StoppingCriterion,
    ImprovementMetric,
    ScoringStrategy,
    DisaggregatedPolicy,
    # Phase 3 — Model-level reasoning (RLM + Dense Verification)
    RlmConfig,
    RlmMode,
    DenseVerificationConfig,
    VerificationStrategy,
    # Phase 2 — Training configs
    GrpoConfig,
    CgarConfig,
    PrmConfig,
    LongRewardConfig,
    PrmAggregation,
    DenseAggregation,
)
from .sampling import SamplingStrategy, create_sampler, SelfConsistencySampler, SelfConsistencyResult
from .model_resolver import resolve_model

__version__ = "2.0.0a1"
__all__ = [
    # Main classes
    "LLM",
    "AsyncLLM",
    "LLMEngine",
    # Output types
    "RequestOutput",
    "CompletionOutput",
    "RefinementOutput",
    "VerifiedOutput",
    "RlmOutput",
    "DenseVerificationOutput",
    # Configuration — core
    "SamplingParams",
    "EngineConfig",
    "ServerConfig",
    "LoRARequest",
    # Configuration — Phase 3 Inference
    "SelfConsistencyConfig",
    "RefinementConfig",
    "VerificationConfig",
    "DisaggregatedServingConfig",
    "AnswerExtractor",
    "StoppingCriterion",
    "ImprovementMetric",
    "ScoringStrategy",
    "DisaggregatedPolicy",
    # Configuration — Phase 3 Model-level reasoning
    "RlmConfig",
    "RlmMode",
    "DenseVerificationConfig",
    "VerificationStrategy",
    # Configuration — Phase 2 Training
    "GrpoConfig",
    "CgarConfig",
    "PrmConfig",
    "LongRewardConfig",
    "PrmAggregation",
    "DenseAggregation",
    # Model resolution
    "resolve_model",
    # Sampling
    "SamplingStrategy",
    "create_sampler",
    "SelfConsistencySampler",
    "SelfConsistencyResult",
    # Training (lazy-loaded below)
    "Trainer",
    "TrainingConfig",
    "LoRAConfig",
    "fine_tune",
    "GrpoTrainer",
    "grpo_train",
    # Version
    "__version__",
]

# Training imports (lazy to avoid import overhead for inference-only usage)
def __getattr__(name: str):
    _training_names = {
        "Trainer", "TrainingConfig", "LoRAConfig", "fine_tune",
        "GrpoTrainer", "grpo_train",
    }
    if name in _training_names:
        from .training import (
            Trainer,
            TrainingConfig,
            LoRAConfig,
            fine_tune,
            GrpoTrainer,
            grpo_train,
        )
        return {
            "Trainer": Trainer,
            "TrainingConfig": TrainingConfig,
            "LoRAConfig": LoRAConfig,
            "fine_tune": fine_tune,
            "GrpoTrainer": GrpoTrainer,
            "grpo_train": grpo_train,
        }[name]
    raise AttributeError(f"module 'swiftllm' has no attribute {name!r}")


def version() -> str:
    """Get the SwiftLLM version."""
    return __version__

# ------------------------------------------------------------------------------
# END OF FILE: __init__.py
# REPO PATH:   /swiftllm/python/swiftllm/__init__.py
# INTEGRATES:  engine.py · config.py · sampling.py · training.py · model_resolver.py
#              Rust: swiftllm-core · swiftllm-training · swiftllm-models
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
