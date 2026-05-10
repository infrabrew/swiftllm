# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      model_config.py
# PATH:      /python/swiftllm/model_config.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - (stdlib only — dataclasses, enum, math, typing)
# USED BY:
#   - python/swiftllm/hybrid_model.py   HybridModelBuilder, build_* presets
#   - python/swiftllm/__init__.py       public re-exports
#   - python/swiftllm/engine.py         EngineConfig.model_config field
#   - python/swiftllm/config.py         re-exports RlmConfig, DenseVerificationConfig
# SEE ALSO:
#   - crates/swiftllm-models/src/layers/mamba.rs              MambaConfig struct + presets
#   - crates/swiftllm-models/src/layers/moe.rs                MoeConfig / LatentMoeConfig structs
#   - crates/swiftllm-models/src/layers/rlm.rs                RlmConfig struct + presets
#   - crates/swiftllm-models/src/layers/dense_verification.rs DenseVerificationConfig struct
#   - crates/swiftllm-models/src/architectures/jamba.rs       JambaConfig + HybridLayerType
#   - crates/swiftllm-models/src/lib.rs                       ModelConfig base struct
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

"""SwiftLLM Hybrid Model Configuration

Python-side mirror of the Rust configuration structs defined in
``crates/swiftllm-models/src/``.  Every field name, default value, and
derived-quantity rule matches the Rust implementation exactly so that
configs serialised here can be passed to the Rust core without translation.

Architecture graph (all-Mamba-3 reasoning model)::

    Tokens
      │
      ▼
    Embedding
      │
      ├─ [Mamba-3 SSM Block] × N_mamba
      │      selective scan · complex states · MIMO multi-head
      │      exponential-trapezoidal discretisation
      │
      ├─ [LatentMoE Block] × N_moe  (every moe_period layers)
      │      latent compress (÷8) → token router → sparse experts
      │      → latent expand (×8)  — 87.5 % less inter-GPU traffic
      │
      ├─ [RLM Block]  (final reasoning layers)
      │      REPL state · variable binding · recursion depth scheduler
      │      early exit on confidence ≥ threshold
      │
      └─ [Dense Verification]  (post-decode)
             cross-attention over REPL trace
             token / step / global confidence scoring
             re-generation on low confidence

Usage::

    from swiftllm.model_config import (
        MambaConfig, LatentMoeConfig, RlmConfig,
        DenseVerificationConfig, HybridModelConfig,
        HybridLayerType, RoutingStrategy,
    )

    # Quick preset — production reasoning model
    cfg = HybridModelConfig.mamba3_reasoning(d_model=2048, num_layers=32)

    # Fine-grained control
    mamba = MambaConfig.mamba3(d_model=2048)
    moe   = LatentMoeConfig.deepseek_style(d_model=2048, num_experts=64)
    rlm   = RlmConfig.reasoning(d_model=2048)
    dv    = DenseVerificationConfig.standard(d_model=2048)

    cfg = HybridModelConfig(
        d_model=2048,
        num_layers=32,
        vocab_size=32000,
        mamba_config=mamba,
        moe_config=moe,
        rlm_config=rlm,
        dense_verification_config=dv,
        layer_schedule=HybridModelConfig.make_schedule(
            num_layers=32,
            moe_period=4,
        ),
    )
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Enums (mirror crates/swiftllm-models/src/layers/moe.rs)
# ---------------------------------------------------------------------------

class RoutingStrategy(str, Enum):
    """Expert-routing algorithm used inside MoE / LatentMoE blocks.

    Mirrors ``RoutingStrategy`` in ``crates/swiftllm-models/src/layers/moe.rs``.

    Attributes
    ----------
    TOP_K:
        Standard Top-K gating (Shazeer et al., 2017).  Each token is sent
        to its *k* highest-scoring experts.
    EXPERT_CHOICE:
        Expert-choice routing (Zhou et al., 2022).  Each expert selects the
        top-*capacity* tokens, guaranteeing perfect load balance.
    RELU_GATING:
        ReLU-gated sparse routing — experts activate only for tokens where
        the router logit exceeds zero.  Naturally sparse; no fixed-k needed.
    """

    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    RELU_GATING = "relu_gating"


class HybridLayerType(str, Enum):
    """Per-layer type in a hybrid model schedule.

    Mirrors ``HybridLayerType`` in
    ``crates/swiftllm-models/src/architectures/jamba.rs``.

    Attributes
    ----------
    ATTENTION:
        Standard multi-head (or grouped-query) attention block.
    MAMBA:
        Mamba SSM block (Mamba-2 or Mamba-3 depending on ``MambaConfig``).
    MAMBA_MOE:
        Mamba SSM block whose FFN is replaced by a sparse MoE / LatentMoE.
    ATTENTION_MOE:
        Attention block whose FFN is replaced by a sparse MoE / LatentMoE.
    """

    ATTENTION = "attention"
    MAMBA = "mamba"
    MAMBA_MOE = "mamba_moe"
    ATTENTION_MOE = "attention_moe"


# ---------------------------------------------------------------------------
# MambaConfig
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    """Configuration for a single Mamba SSM block.

    Mirrors ``MambaConfig`` in
    ``crates/swiftllm-models/src/layers/mamba.rs``.

    Parameters
    ----------
    d_model : int
        Model (residual stream) dimension.
    expand : int
        Inner-state expansion factor.  Inner dimension = ``d_model * expand``.
    d_state : int
        SSM state size per head.  Mamba-3 uses 128; Mamba-2 uses 16.
    d_conv : int
        Depthwise convolution kernel width (applied before SSM scan).
    dt_rank : int or None
        Rank of the Δt projection.  Defaults to ``ceil(d_model / 16)``
        when *None*.
    dt_min / dt_max : float
        Clamp range for the discretised time step Δt.
    bias : bool
        Whether to add bias to the main input/output projections.
    conv_bias : bool
        Whether to add bias to the depthwise convolution.
    use_trapezoidal_disc : bool
        Use exponential-trapezoidal discretisation (Mamba-3).  When
        *False* the standard ZOH (zero-order hold) scheme is used.
    use_complex_states : bool
        Use complex-valued SSM states (Mamba-3 MIMO).
    use_mimo : bool
        Enable multi-input / multi-output multi-head formulation (Mamba-3).
    num_heads : int or None
        Number of SSM heads.  Defaults to ``d_model // 64`` when *None*.
    ns_steps : int
        Number of Newton-style refinement steps for the exponential-
        trapezoidal integrator (Mamba-3 only; ignored otherwise).
    """

    d_model: int
    expand: int = 2
    d_state: int = 128          # 128 for Mamba-3; 16 for Mamba-2
    d_conv: int = 4
    dt_rank: Optional[int] = None   # None → ceil(d_model / 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    bias: bool = False
    conv_bias: bool = True
    use_trapezoidal_disc: bool = True    # Mamba-3
    use_complex_states: bool = True      # Mamba-3
    use_mimo: bool = True                # Mamba-3
    num_heads: Optional[int] = None      # None → d_model // 64
    ns_steps: int = 5

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.num_heads is None:
            self.num_heads = max(1, self.d_model // 64)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def mamba3(cls, d_model: int) -> "MambaConfig":
        """Full Mamba-3 preset: complex states, MIMO, trapezoidal disc.

        Matches the ``mamba3()`` convenience function in
        ``crates/swiftllm-models/src/layers/mamba.rs``.
        """
        return cls(
            d_model=d_model,
            expand=2,
            d_state=128,
            d_conv=4,
            dt_min=0.001,
            dt_max=0.1,
            bias=False,
            conv_bias=True,
            use_trapezoidal_disc=True,
            use_complex_states=True,
            use_mimo=True,
            ns_steps=5,
        )

    @classmethod
    def mamba2(cls, d_model: int) -> "MambaConfig":
        """Mamba-2 preset: real states, no MIMO, ZOH discretisation.

        Matches the ``mamba2()`` convenience function in
        ``crates/swiftllm-models/src/layers/mamba.rs``.
        """
        return cls(
            d_model=d_model,
            expand=2,
            d_state=16,
            d_conv=4,
            dt_min=0.001,
            dt_max=0.1,
            bias=False,
            conv_bias=True,
            use_trapezoidal_disc=False,
            use_complex_states=False,
            use_mimo=False,
            ns_steps=0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "expand": self.expand,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "dt_rank": self.dt_rank,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "bias": self.bias,
            "conv_bias": self.conv_bias,
            "use_trapezoidal_disc": self.use_trapezoidal_disc,
            "use_complex_states": self.use_complex_states,
            "use_mimo": self.use_mimo,
            "num_heads": self.num_heads,
            "ns_steps": self.ns_steps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MambaConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# MoeConfig  (standard / dense MoE — used when use_latent_moe=False)
# ---------------------------------------------------------------------------

@dataclass
class MoeConfig:
    """Configuration for a standard sparse MoE FFN.

    Mirrors ``MoeConfig`` in ``crates/swiftllm-models/src/layers/moe.rs``.
    Use :class:`LatentMoeConfig` for the latent-compressed variant.

    Parameters
    ----------
    d_model : int
        Model (residual stream) dimension.
    d_ffn : int
        Hidden dimension of each expert FFN.  Defaults to ``4 * d_model``.
    num_experts : int
        Total number of expert networks (``E``).
    num_experts_per_token : int
        How many experts each token is routed to (``k`` in Top-K).
    num_shared_experts : int
        Number of always-active "shared" experts (DeepSeek-MoE style).
        0 means no shared experts.
    routing : RoutingStrategy
        Which gating algorithm to use.
    bias_update_rate : float
        Learning rate for the dynamic per-expert bias used in DeepSeek-V3
        aux-loss-free load balancing.  Ignored when routing ≠ TOP_K.
    aux_loss_coeff : float
        Coefficient for the auxiliary load-balancing loss (0 = disabled).
    capacity_factor : float
        Expert capacity = ``(tokens_per_batch / num_experts) * capacity_factor``.
        Tokens that overflow are dropped or passed through.
    router_temperature : float
        Softmax temperature applied to router logits before gating.
    normalise_weights : bool
        Normalise gating weights so they sum to 1 across activated experts.
    """

    d_model: int
    d_ffn: Optional[int] = None         # None → 4 * d_model
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_shared_experts: int = 0
    routing: RoutingStrategy = RoutingStrategy.TOP_K
    bias_update_rate: float = 0.001     # DeepSeek-V3 dynamic bias rate
    aux_loss_coeff: float = 0.01
    capacity_factor: float = 1.25
    router_temperature: float = 1.0
    normalise_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_ffn is None:
            self.d_ffn = 4 * self.d_model

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def deepseek_style(
        cls,
        d_model: int,
        num_experts: int = 64,
        num_experts_per_token: int = 6,
        num_shared_experts: int = 2,
    ) -> "MoeConfig":
        """DeepSeek-V3-style aux-loss-free dynamic bias load balancing."""
        return cls(
            d_model=d_model,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            num_shared_experts=num_shared_experts,
            routing=RoutingStrategy.TOP_K,
            bias_update_rate=0.001,
            aux_loss_coeff=0.0,     # aux-loss-free — bias handles balance
            capacity_factor=1.25,
            router_temperature=1.0,
            normalise_weights=True,
        )

    @classmethod
    def mixtral_style(
        cls,
        d_model: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
    ) -> "MoeConfig":
        """Mixtral-style sparse MoE (aux-loss load balancing)."""
        return cls(
            d_model=d_model,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            num_shared_experts=0,
            routing=RoutingStrategy.TOP_K,
            bias_update_rate=0.0,
            aux_loss_coeff=0.01,
            capacity_factor=1.25,
            router_temperature=1.0,
            normalise_weights=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "d_ffn": self.d_ffn,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "num_shared_experts": self.num_shared_experts,
            "routing": self.routing.value,
            "bias_update_rate": self.bias_update_rate,
            "aux_loss_coeff": self.aux_loss_coeff,
            "capacity_factor": self.capacity_factor,
            "router_temperature": self.router_temperature,
            "normalise_weights": self.normalise_weights,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoeConfig":
        d = dict(d)
        if "routing" in d and isinstance(d["routing"], str):
            d["routing"] = RoutingStrategy(d["routing"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# LatentMoeConfig  (latent-compressed variant — default for hybrid models)
# ---------------------------------------------------------------------------

@dataclass
class LatentMoeConfig:
    """Configuration for the LatentMoE FFN block.

    LatentMoE compresses the residual stream by ``latent_compression_ratio``
    before routing, reducing inter-GPU expert-parallelism traffic by the
    same factor (87.5 % for the default ratio of 8).

    Mirrors ``LatentMoeConfig`` in
    ``crates/swiftllm-models/src/layers/moe.rs``.

    Parameters
    ----------
    d_model : int
        Model (residual stream) dimension.
    latent_compression_ratio : int
        Factor by which the residual stream is compressed before routing.
        Default is 8 (matches DeepSeek-V3 latent attention ratio).
    moe : MoeConfig
        The underlying MoE configuration applied inside the latent space.
    """

    d_model: int
    latent_compression_ratio: int = 8
    moe: MoeConfig = field(default_factory=lambda: MoeConfig(d_model=1))  # reset in __post_init__

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.latent_compression_ratio <= 0:
            raise ValueError(
                f"latent_compression_ratio must be positive, got {self.latent_compression_ratio}"
            )
        d_latent = self.d_model // self.latent_compression_ratio
        if d_latent <= 0:
            raise ValueError(
                f"d_model ({self.d_model}) // latent_compression_ratio "
                f"({self.latent_compression_ratio}) = {d_latent} — must be > 0"
            )
        # If the moe config was left at the placeholder default, build a proper one
        if self.moe.d_model != d_latent:
            # Rebuild moe with correct latent d_model
            self.moe = MoeConfig(
                d_model=d_latent,
                d_ffn=self.moe.d_ffn if self.moe.d_model != 1 else None,
                num_experts=self.moe.num_experts,
                num_experts_per_token=self.moe.num_experts_per_token,
                num_shared_experts=self.moe.num_shared_experts,
                routing=self.moe.routing,
                bias_update_rate=self.moe.bias_update_rate,
                aux_loss_coeff=self.moe.aux_loss_coeff,
                capacity_factor=self.moe.capacity_factor,
                router_temperature=self.moe.router_temperature,
                normalise_weights=self.moe.normalise_weights,
            )

    @property
    def d_latent(self) -> int:
        """Dimension of the compressed latent space."""
        return self.d_model // self.latent_compression_ratio

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def deepseek_style(
        cls,
        d_model: int,
        num_experts: int = 64,
        num_experts_per_token: int = 6,
        num_shared_experts: int = 2,
        latent_compression_ratio: int = 8,
    ) -> "LatentMoeConfig":
        """DeepSeek-V3-inspired LatentMoE with aux-loss-free load balancing."""
        d_latent = d_model // latent_compression_ratio
        moe = MoeConfig(
            d_model=d_latent,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            num_shared_experts=num_shared_experts,
            routing=RoutingStrategy.TOP_K,
            bias_update_rate=0.001,
            aux_loss_coeff=0.0,
            capacity_factor=1.25,
            router_temperature=1.0,
            normalise_weights=True,
        )
        obj = cls.__new__(cls)
        obj.d_model = d_model
        obj.latent_compression_ratio = latent_compression_ratio
        obj.moe = moe
        return obj

    @classmethod
    def small(cls, d_model: int) -> "LatentMoeConfig":
        """Lightweight 8-expert LatentMoE for smaller models."""
        d_latent = d_model // 8
        moe = MoeConfig(
            d_model=d_latent,
            num_experts=8,
            num_experts_per_token=2,
            num_shared_experts=0,
            routing=RoutingStrategy.TOP_K,
            bias_update_rate=0.001,
            aux_loss_coeff=0.01,
            capacity_factor=1.25,
            router_temperature=1.0,
            normalise_weights=True,
        )
        obj = cls.__new__(cls)
        obj.d_model = d_model
        obj.latent_compression_ratio = 8
        obj.moe = moe
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "latent_compression_ratio": self.latent_compression_ratio,
            "moe": self.moe.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LatentMoeConfig":
        moe = MoeConfig.from_dict(d["moe"])
        obj = cls.__new__(cls)
        obj.d_model = d["d_model"]
        obj.latent_compression_ratio = d.get("latent_compression_ratio", 8)
        obj.moe = moe
        return obj


# ---------------------------------------------------------------------------
# RlmConfig
# ---------------------------------------------------------------------------

@dataclass
class RlmConfig:
    """Configuration for the Recursive Language Model (RLM) block.

    The RLM layer adds a lightweight REPL state machine and variable-binding
    table to the residual stream, enabling iterative sub-problem decomposition
    without external tool calls.

    Mirrors ``RlmConfig`` in ``crates/swiftllm-models/src/layers/rlm.rs``.

    Parameters
    ----------
    d_model : int
        Model (residual stream) dimension.
    max_depth : int
        Maximum recursion depth.  Use 3 for reasoning models; 1 for
        standard language models (disables actual recursion).
    enable_repl : bool
        Enable the REPL state machine (``Assign`` / ``Compute`` /
        ``Verify`` / ``Recurse`` execution steps).
    var_binding_slots : int
        Number of variable-binding slots in the binding table.
    depth_hidden_size : int or None
        Hidden size of the depth-embedding MLP.  Defaults to
        ``d_model // 4`` when *None*.
    early_exit_threshold : float
        Stop recursion when the step-confidence score ≥ this value.
        Must be in ``(0, 1]``.
    d_subproblem : int or None
        Embedding dimension used to represent a sub-problem query.
        Defaults to ``d_model // 2`` when *None*.
    """

    d_model: int
    max_depth: int = 3
    enable_repl: bool = True
    var_binding_slots: int = 32
    depth_hidden_size: Optional[int] = None     # None → d_model // 4
    early_exit_threshold: float = 0.92
    d_subproblem: Optional[int] = None          # None → d_model // 2

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if not (0 < self.early_exit_threshold <= 1.0):
            raise ValueError(
                f"early_exit_threshold must be in (0, 1], got {self.early_exit_threshold}"
            )
        if self.depth_hidden_size is None:
            self.depth_hidden_size = max(1, self.d_model // 4)
        if self.d_subproblem is None:
            self.d_subproblem = max(1, self.d_model // 2)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def reasoning(cls, d_model: int) -> "RlmConfig":
        """Full reasoning preset: REPL enabled, depth-3 recursion.

        Matches the ``reasoning()`` preset in
        ``crates/swiftllm-models/src/layers/rlm.rs``.
        """
        return cls(
            d_model=d_model,
            max_depth=3,
            enable_repl=True,
            var_binding_slots=32,
            early_exit_threshold=0.92,
        )

    @classmethod
    def language(cls, d_model: int) -> "RlmConfig":
        """Standard language model preset: REPL disabled, depth-1.

        Matches the ``language()`` preset in
        ``crates/swiftllm-models/src/layers/rlm.rs``.
        """
        return cls(
            d_model=d_model,
            max_depth=1,
            enable_repl=False,
            var_binding_slots=0,
            early_exit_threshold=0.92,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "max_depth": self.max_depth,
            "enable_repl": self.enable_repl,
            "var_binding_slots": self.var_binding_slots,
            "depth_hidden_size": self.depth_hidden_size,
            "early_exit_threshold": self.early_exit_threshold,
            "d_subproblem": self.d_subproblem,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RlmConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DenseVerificationConfig
# ---------------------------------------------------------------------------

@dataclass
class DenseVerificationConfig:
    """Configuration for the Dense Verification post-decode pass.

    The DenseVerification layer performs a second forward pass (full-capacity,
    no sampling) over the draft output, cross-attending to the REPL execution
    trace produced by the RLM block.  It outputs per-token, per-step, and
    global confidence scores and triggers selective re-generation for low-
    confidence spans.

    Mirrors ``DenseVerificationConfig`` in
    ``crates/swiftllm-models/src/layers/dense_verification.rs``.

    Parameters
    ----------
    d_model : int
        Model (residual stream) dimension.
    num_verification_heads : int
        Number of cross-attention heads in the verification pass.
    d_v_head : int or None
        Per-head dimension for verification cross-attention.
        Defaults to ``d_model // 8`` when *None*.
    min_confidence : float
        Global confidence threshold below which re-generation is triggered.
        Must be in ``(0, 1]``.
    max_regen_attempts : int
        Maximum number of re-generation attempts per request.
    score_repl_steps : bool
        Whether to produce per-REPL-step confidence scores in addition to
        per-token scores.  Requires an active RLM block.
    """

    d_model: int
    num_verification_heads: int = 8
    d_v_head: Optional[int] = None      # None → d_model // 8
    min_confidence: float = 0.80
    max_regen_attempts: int = 3
    score_repl_steps: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if not (0 < self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in (0, 1], got {self.min_confidence}"
            )
        if self.d_v_head is None:
            self.d_v_head = max(1, self.d_model // 8)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def standard(cls, d_model: int) -> "DenseVerificationConfig":
        """Standard dense verification with REPL-step scoring."""
        return cls(
            d_model=d_model,
            num_verification_heads=8,
            min_confidence=0.80,
            max_regen_attempts=3,
            score_repl_steps=True,
        )

    @classmethod
    def strict(cls, d_model: int) -> "DenseVerificationConfig":
        """High-confidence threshold (0.90) with up to 5 re-gen attempts."""
        return cls(
            d_model=d_model,
            num_verification_heads=8,
            min_confidence=0.90,
            max_regen_attempts=5,
            score_repl_steps=True,
        )

    @classmethod
    def lightweight(cls, d_model: int) -> "DenseVerificationConfig":
        """Fast verification: 4 heads, no per-step scoring."""
        return cls(
            d_model=d_model,
            num_verification_heads=4,
            min_confidence=0.75,
            max_regen_attempts=2,
            score_repl_steps=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_verification_heads": self.num_verification_heads,
            "d_v_head": self.d_v_head,
            "min_confidence": self.min_confidence,
            "max_regen_attempts": self.max_regen_attempts,
            "score_repl_steps": self.score_repl_steps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DenseVerificationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ModelBaseConfig  (mirrors ModelConfig in crates/swiftllm-models/src/lib.rs)
# ---------------------------------------------------------------------------

@dataclass
class ModelBaseConfig:
    """Base configuration shared by all model architectures.

    Mirrors ``ModelConfig`` in ``crates/swiftllm-models/src/lib.rs``.

    Parameters
    ----------
    architecture : str
        Architecture identifier, e.g. ``"mamba3_hybrid"`` or
        ``"mamba3_reasoning"``.
    hidden_size : int
        Model dimension (synonymous with ``d_model`` in SSM literature).
    intermediate_size : int
        FFN intermediate dimension (unused when all layers are MoE/LatentMoE).
    num_attention_heads : int
        Number of attention heads (0 for pure-Mamba models).
    num_key_value_heads : int
        Number of KV heads for grouped-query attention (0 for pure-Mamba).
    num_hidden_layers : int
        Total number of transformer/Mamba layers.
    vocab_size : int
        Vocabulary size.
    max_position_embeddings : int
        Maximum sequence length.
    rms_norm_eps : float
        Epsilon for RMSNorm layers.
    rope_theta : float
        Base for RoPE positional embeddings (used only when attention layers
        are present).
    head_dim : int or None
        Per-attention-head dimension.  Defaults to
        ``hidden_size // num_attention_heads`` when *None* and
        ``num_attention_heads > 0``.
    attention_bias : bool
        Add bias to Q/K/V projections.
    mlp_bias : bool
        Add bias to MLP / FFN projections.
    """

    architecture: str = "mamba3_hybrid"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 0        # 0 = pure-Mamba (no attention)
    num_key_value_heads: int = 0
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    head_dim: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None and self.num_attention_heads > 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "head_dim": self.head_dim,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelBaseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# HybridModelConfig  (top-level config — mirrors JambaConfig in jamba.rs)
# ---------------------------------------------------------------------------

@dataclass
class HybridModelConfig:
    """Top-level configuration for a hybrid (Mamba-3 + LatentMoE + RLM + DV) model.

    This is the Python mirror of ``JambaConfig`` in
    ``crates/swiftllm-models/src/architectures/jamba.rs``, extended with
    the RLM and DenseVerification configs that sit on top of the backbone.

    The :meth:`make_schedule` class method generates the per-layer type list
    that tells the Rust backend which block to instantiate at each depth.
    With ``attn_period=0`` (the default for pure-Mamba models) every layer
    is either ``MAMBA`` or ``MAMBA_MOE`` — no attention blocks are created.

    Parameters
    ----------
    d_model : int
        Model dimension (must match ``ModelBaseConfig.hidden_size``).
    num_layers : int
        Total number of backbone layers.
    vocab_size : int
        Vocabulary size.
    mamba_config : MambaConfig or None
        SSM configuration.  If *None*, a Mamba-3 preset is auto-created.
    moe_config : LatentMoeConfig or None
        LatentMoE configuration.  Set to *None* to disable MoE entirely
        (all layers become plain Mamba).
    use_latent_moe : bool
        When *True* (default) the MoE blocks use latent compression.
        When *False*, standard dense MoE is used instead (``moe_config``
        must then be a :class:`MoeConfig` instance).
    rlm_config : RlmConfig or None
        RLM configuration.  Set to *None* to disable reasoning layers.
    dense_verification_config : DenseVerificationConfig or None
        Dense Verification configuration.  Set to *None* to skip the
        post-decode verification pass.
    layer_schedule : list of HybridLayerType or None
        Per-layer type list with length ``num_layers``.  If *None*, a
        schedule is generated automatically via :meth:`make_schedule`
        with ``moe_period=4``.
    base : ModelBaseConfig or None
        Low-level base config.  Auto-populated from other fields if *None*.
    max_seq_len : int
        Maximum sequence length.
    tie_embeddings : bool
        Tie input and output embedding weights.
    """

    d_model: int
    num_layers: int
    vocab_size: int = 32000
    mamba_config: Optional[MambaConfig] = None
    moe_config: Optional[Any] = None               # LatentMoeConfig | MoeConfig | None
    use_latent_moe: bool = True
    rlm_config: Optional[RlmConfig] = None
    dense_verification_config: Optional[DenseVerificationConfig] = None
    layer_schedule: Optional[List[HybridLayerType]] = None
    base: Optional[ModelBaseConfig] = None
    max_seq_len: int = 131072
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        # --- Mamba config ---
        if self.mamba_config is None:
            self.mamba_config = MambaConfig.mamba3(self.d_model)

        # --- MoE config ---
        if self.moe_config is None and self.use_latent_moe:
            self.moe_config = LatentMoeConfig.deepseek_style(self.d_model)
        elif self.moe_config is None:
            self.moe_config = MoeConfig.deepseek_style(self.d_model)

        # --- Layer schedule ---
        if self.layer_schedule is None:
            self.layer_schedule = self.make_schedule(
                num_layers=self.num_layers,
                moe_period=4,
                attn_period=0,
            )
        if len(self.layer_schedule) != self.num_layers:
            raise ValueError(
                f"layer_schedule length ({len(self.layer_schedule)}) "
                f"must equal num_layers ({self.num_layers})"
            )

        # --- Base config ---
        if self.base is None:
            self.base = ModelBaseConfig(
                architecture="mamba3_reasoning" if self.rlm_config is not None else "mamba3_hybrid",
                hidden_size=self.d_model,
                intermediate_size=4 * self.d_model,
                num_attention_heads=0,      # pure-Mamba — no attention
                num_key_value_heads=0,
                num_hidden_layers=self.num_layers,
                vocab_size=self.vocab_size,
                max_position_embeddings=self.max_seq_len,
            )

    # ------------------------------------------------------------------
    # Layer schedule generator
    # ------------------------------------------------------------------

    @staticmethod
    def make_schedule(
        num_layers: int,
        moe_period: int = 4,
        attn_period: int = 0,
    ) -> List[HybridLayerType]:
        """Generate a per-layer type schedule.

        Parameters
        ----------
        num_layers : int
            Total number of layers in the model.
        moe_period : int
            Every ``moe_period``-th layer (1-indexed) becomes a MoE/LatentMoE
            layer.  ``moe_period=0`` disables MoE entirely (all Mamba).
        attn_period : int
            Every ``attn_period``-th layer becomes an attention layer instead
            of a Mamba layer.  ``attn_period=0`` disables attention entirely
            — the result is a **pure-Mamba** model.

        Returns
        -------
        list of HybridLayerType
            Length ``num_layers``.

        Examples
        --------
        >>> HybridModelConfig.make_schedule(8, moe_period=4, attn_period=0)
        [MAMBA, MAMBA, MAMBA, MAMBA_MOE, MAMBA, MAMBA, MAMBA, MAMBA_MOE]

        >>> HybridModelConfig.make_schedule(8, moe_period=4, attn_period=8)
        [MAMBA, MAMBA, MAMBA, MAMBA_MOE, MAMBA, MAMBA, MAMBA, ATTENTION_MOE]
        """
        schedule = []
        for i in range(1, num_layers + 1):
            is_moe = (moe_period > 0) and (i % moe_period == 0)
            is_attn = (attn_period > 0) and (i % attn_period == 0)

            if is_attn and is_moe:
                schedule.append(HybridLayerType.ATTENTION_MOE)
            elif is_attn:
                schedule.append(HybridLayerType.ATTENTION)
            elif is_moe:
                schedule.append(HybridLayerType.MAMBA_MOE)
            else:
                schedule.append(HybridLayerType.MAMBA)
        return schedule

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_moe_layers(self) -> int:
        """Number of MoE / LatentMoE layers in the schedule."""
        return sum(
            1 for t in (self.layer_schedule or [])
            if t in (HybridLayerType.MAMBA_MOE, HybridLayerType.ATTENTION_MOE)
        )

    @property
    def num_mamba_layers(self) -> int:
        """Number of plain Mamba layers in the schedule."""
        return sum(
            1 for t in (self.layer_schedule or [])
            if t in (HybridLayerType.MAMBA, HybridLayerType.MAMBA_MOE)
        )

    @property
    def num_attention_layers(self) -> int:
        """Number of attention layers in the schedule."""
        return sum(
            1 for t in (self.layer_schedule or [])
            if t in (HybridLayerType.ATTENTION, HybridLayerType.ATTENTION_MOE)
        )

    @property
    def is_pure_mamba(self) -> bool:
        """True if the model has zero attention layers."""
        return self.num_attention_layers == 0

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def mamba3_reasoning(
        cls,
        d_model: int = 2048,
        num_layers: int = 32,
        vocab_size: int = 32000,
        num_experts: int = 64,
        moe_period: int = 4,
    ) -> "HybridModelConfig":
        """Production preset: Mamba-3 + LatentMoE + RLM + Dense Verification.

        This is the flagship all-Mamba-3 reasoning architecture — no
        transformer attention blocks.  Layers alternate between plain
        Mamba-3 SSM blocks and LatentMoE blocks (every ``moe_period``
        layers).  An RLM block sits on top of the backbone and a Dense
        Verification pass is applied post-decode.

        Parameters
        ----------
        d_model : int
            Model dimension.  Typical values: 1024, 2048, 4096, 7168.
        num_layers : int
            Total backbone layers.  Typical: 24 (small), 32 (medium), 48 (large).
        vocab_size : int
            Vocabulary size.
        num_experts : int
            Number of LatentMoE experts.
        moe_period : int
            LatentMoE appears every ``moe_period`` layers.
        """
        return cls(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            mamba_config=MambaConfig.mamba3(d_model),
            moe_config=LatentMoeConfig.deepseek_style(
                d_model, num_experts=num_experts, num_experts_per_token=6
            ),
            use_latent_moe=True,
            rlm_config=RlmConfig.reasoning(d_model),
            dense_verification_config=DenseVerificationConfig.standard(d_model),
            layer_schedule=cls.make_schedule(num_layers, moe_period=moe_period, attn_period=0),
        )

    @classmethod
    def mamba3_hybrid_attention(
        cls,
        d_model: int = 2048,
        num_layers: int = 32,
        vocab_size: int = 32000,
        attn_period: int = 8,
        moe_period: int = 4,
    ) -> "HybridModelConfig":
        """Hybrid preset: mostly Mamba-3, with sparse attention layers.

        Attention layers appear every ``attn_period`` layers (e.g., every
        8th layer for Jamba-style architectures).
        """
        return cls(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            mamba_config=MambaConfig.mamba3(d_model),
            moe_config=LatentMoeConfig.deepseek_style(d_model),
            use_latent_moe=True,
            rlm_config=None,
            dense_verification_config=None,
            layer_schedule=cls.make_schedule(
                num_layers, moe_period=moe_period, attn_period=attn_period
            ),
        )

    @classmethod
    def mamba3_pure(
        cls,
        d_model: int = 2048,
        num_layers: int = 32,
        vocab_size: int = 32000,
    ) -> "HybridModelConfig":
        """Pure Mamba-3 baseline: no MoE, no attention, no reasoning layers."""
        return cls(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            mamba_config=MambaConfig.mamba3(d_model),
            moe_config=None,
            use_latent_moe=False,
            rlm_config=None,
            dense_verification_config=None,
            layer_schedule=[HybridLayerType.MAMBA] * num_layers,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        def _moe_dict(cfg):
            if cfg is None:
                return None
            if isinstance(cfg, LatentMoeConfig):
                return {"type": "latent", **cfg.to_dict()}
            return {"type": "standard", **cfg.to_dict()}

        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "mamba_config": self.mamba_config.to_dict() if self.mamba_config else None,
            "moe_config": _moe_dict(self.moe_config),
            "use_latent_moe": self.use_latent_moe,
            "rlm_config": self.rlm_config.to_dict() if self.rlm_config else None,
            "dense_verification_config": (
                self.dense_verification_config.to_dict()
                if self.dense_verification_config else None
            ),
            "layer_schedule": [t.value for t in (self.layer_schedule or [])],
            "base": self.base.to_dict() if self.base else None,
            "max_seq_len": self.max_seq_len,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HybridModelConfig":
        mamba = MambaConfig.from_dict(d["mamba_config"]) if d.get("mamba_config") else None

        raw_moe = d.get("moe_config")
        moe: Optional[Any] = None
        if raw_moe is not None:
            moe_type = raw_moe.pop("type", "latent")
            if moe_type == "latent":
                moe = LatentMoeConfig.from_dict(raw_moe)
            else:
                moe = MoeConfig.from_dict(raw_moe)

        rlm = RlmConfig.from_dict(d["rlm_config"]) if d.get("rlm_config") else None
        dv = (
            DenseVerificationConfig.from_dict(d["dense_verification_config"])
            if d.get("dense_verification_config") else None
        )
        base = ModelBaseConfig.from_dict(d["base"]) if d.get("base") else None
        schedule = (
            [HybridLayerType(t) for t in d["layer_schedule"]]
            if d.get("layer_schedule") else None
        )
        obj = cls.__new__(cls)
        obj.d_model = d["d_model"]
        obj.num_layers = d["num_layers"]
        obj.vocab_size = d.get("vocab_size", 32000)
        obj.mamba_config = mamba
        obj.moe_config = moe
        obj.use_latent_moe = d.get("use_latent_moe", True)
        obj.rlm_config = rlm
        obj.dense_verification_config = dv
        obj.layer_schedule = schedule
        obj.base = base
        obj.max_seq_len = d.get("max_seq_len", 131072)
        obj.tie_embeddings = d.get("tie_embeddings", False)
        return obj

    def summary(self) -> str:
        """Return a human-readable architecture summary string."""
        attn_count = self.num_attention_layers
        mamba_count = self.num_mamba_layers
        moe_count = self.num_moe_layers

        lines = [
            f"HybridModelConfig — {self.base.architecture if self.base else 'unknown'}",
            f"  d_model       : {self.d_model}",
            f"  num_layers    : {self.num_layers}",
            f"  vocab_size    : {self.vocab_size}",
            f"  max_seq_len   : {self.max_seq_len:,}",
            "",
            f"  Backbone layers ({self.num_layers} total):",
            f"    Mamba-3 SSM     : {mamba_count}",
            f"    Attention       : {attn_count}  {'← pure-Mamba, no attention' if attn_count == 0 else ''}",
            f"    MoE / LatentMoE : {moe_count}",
            "",
        ]
        if self.mamba_config:
            mc = self.mamba_config
            lines += [
                "  MambaConfig:",
                f"    d_state         : {mc.d_state}",
                f"    num_heads       : {mc.num_heads}",
                f"    complex_states  : {mc.use_complex_states}",
                f"    mimo            : {mc.use_mimo}",
                f"    trapezoidal_disc: {mc.use_trapezoidal_disc}",
                "",
            ]
        if self.moe_config:
            label = "LatentMoeConfig" if isinstance(self.moe_config, LatentMoeConfig) else "MoeConfig"
            mc2 = self.moe_config
            lines += [
                f"  {label}:",
            ]
            if isinstance(mc2, LatentMoeConfig):
                lines.append(f"    d_latent        : {mc2.d_latent}  (compression ÷{mc2.latent_compression_ratio})")
                mc2 = mc2.moe
            lines += [
                f"    num_experts     : {mc2.num_experts}",
                f"    top_k           : {mc2.num_experts_per_token}",
                f"    shared_experts  : {mc2.num_shared_experts}",
                f"    routing         : {mc2.routing.value}",
                "",
            ]
        if self.rlm_config:
            rc = self.rlm_config
            lines += [
                "  RlmConfig:",
                f"    max_depth       : {rc.max_depth}",
                f"    enable_repl     : {rc.enable_repl}",
                f"    var_slots       : {rc.var_binding_slots}",
                f"    early_exit_thr  : {rc.early_exit_threshold}",
                "",
            ]
        if self.dense_verification_config:
            dv = self.dense_verification_config
            lines += [
                "  DenseVerificationConfig:",
                f"    verif_heads     : {dv.num_verification_heads}",
                f"    min_confidence  : {dv.min_confidence}",
                f"    max_regen       : {dv.max_regen_attempts}",
                f"    score_repl_steps: {dv.score_repl_steps}",
                "",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "RoutingStrategy",
    "HybridLayerType",
    # Layer configs
    "MambaConfig",
    "MoeConfig",
    "LatentMoeConfig",
    "RlmConfig",
    "DenseVerificationConfig",
    # Architecture config
    "ModelBaseConfig",
    "HybridModelConfig",
]

# ------------------------------------------------------------------------------
# END OF FILE: model_config.py
# REPO PATH:   /swiftllm/python/swiftllm/model_config.py
# MIRRORS:     crates/swiftllm-models/src/layers/{mamba,moe,rlm,dense_verification}.rs
#              crates/swiftllm-models/src/architectures/jamba.rs
#              crates/swiftllm-models/src/lib.rs  (ModelConfig)
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
