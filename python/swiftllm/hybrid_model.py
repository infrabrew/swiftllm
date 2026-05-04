# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      hybrid_model.py
# PATH:      /python/swiftllm/hybrid_model.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/model_config.py   all config dataclasses
#   - python/swiftllm/engine.py         LLM / EngineConfig (optional, for loading)
# USED BY:
#   - python/swiftllm/__init__.py       public re-exports
#   - examples/hybrid_model.py          usage examples
# SEE ALSO:
#   - crates/swiftllm-models/src/architectures/jamba.rs   JambaConfig (Rust backbone)
#   - crates/swiftllm-models/src/layers/mamba.rs           MambaConfig
#   - crates/swiftllm-models/src/layers/moe.rs             MoeConfig / LatentMoeConfig
#   - crates/swiftllm-models/src/layers/rlm.rs             RlmConfig
#   - crates/swiftllm-models/src/layers/dense_verification.rs  DenseVerificationConfig
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

"""SwiftLLM Hybrid Model Builder

High-level builder API for constructing, saving, and loading hybrid
Mamba-3 + LatentMoE + RLM + Dense Verification model configurations.

This module sits between user code and the Rust core:

  User code
    │
    ▼  (this module)
  HybridModelBuilder / build_* helpers
    │  serialises HybridModelConfig → JSON
    ▼
  swiftllm._core  (PyO3 bindings)
    │  deserialises JSON → Rust JambaConfig
    ▼
  crates/swiftllm-models  (Rust computation)

Usage::

    from swiftllm.hybrid_model import (
        build_mamba3_reasoning_model,
        build_mamba3_base_model,
        HybridModelBuilder,
    )

    # ── Quickstart — flagship reasoning model ────────────────────────
    cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)
    print(cfg.summary())

    # ── Fine-grained builder ─────────────────────────────────────────
    cfg = (
        HybridModelBuilder(d_model=2048, num_layers=32)
        .with_mamba3()
        .with_latent_moe(num_experts=64, moe_period=4)
        .with_rlm(max_depth=3)
        .with_dense_verification(min_confidence=0.85)
        .build()
    )

    # ── Save / load ──────────────────────────────────────────────────
    cfg.to_json("my_model_config.json")
    cfg2 = HybridModelConfig.from_json("my_model_config.json")

    # ── Integrate with LLM engine ────────────────────────────────────
    from swiftllm import LLM, EngineConfig
    engine_cfg = EngineConfig(model="./weights/", model_config=cfg)
    llm = LLM(engine_config=engine_cfg)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .model_config import (
    DenseVerificationConfig,
    HybridLayerType,
    HybridModelConfig,
    LatentMoeConfig,
    MambaConfig,
    ModelBaseConfig,
    MoeConfig,
    RlmConfig,
    RoutingStrategy,
)


# ---------------------------------------------------------------------------
# HybridForwardResult
# ---------------------------------------------------------------------------

@dataclass
class HybridForwardResult:
    """Output produced by a hybrid model forward pass.

    Returned by the optional Python-level forward shim when using PyTorch as
    an interim compute backend (before the full Rust GPU kernel bridge is
    available).  The Rust core returns an equivalent struct via PyO3.

    Attributes
    ----------
    logits : list of list of float or None
        Raw vocabulary logits, shape ``[batch, vocab_size]``.  *None* when
        the backend is Rust-native and logits are kept on-device.
    hidden_states : list of list of float or None
        Final hidden states, shape ``[batch, seq_len, d_model]``.  *None*
        unless ``output_hidden_states=True`` was passed to the engine.
    repl_trace : list of dict or None
        Serialised REPL execution trace produced by the RLM block.  Each
        entry has keys ``step_type`` (str), ``depth`` (int), and
        ``confidence`` (float).  *None* when the model has no RLM block.
    verification_result : dict or None
        Dense Verification output with keys:
        - ``global_score`` (float)
        - ``token_scores`` (list of float)
        - ``step_scores`` (list of float, only when ``score_repl_steps=True``)
        - ``is_accepted`` (bool)
        - ``low_confidence_positions`` (list of int)
        - ``low_confidence_steps`` (list of int)
        *None* when the model has no Dense Verification layer.
    metadata : dict
        Arbitrary metadata forwarded by the backend (timing, memory, etc.).
    """

    logits: Optional[List[List[float]]] = None
    hidden_states: Optional[List[List[List[float]]]] = None
    repl_trace: Optional[List[Dict[str, Any]]] = None
    verification_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        """True if Dense Verification accepted the draft output."""
        if self.verification_result is None:
            return True     # no verification → always accepted
        return bool(self.verification_result.get("is_accepted", True))

    @property
    def global_confidence(self) -> float:
        """Global confidence score from Dense Verification (0–1)."""
        if self.verification_result is None:
            return 1.0
        return float(self.verification_result.get("global_score", 1.0))


# ---------------------------------------------------------------------------
# HybridModelBuilder  (fluent builder)
# ---------------------------------------------------------------------------

class HybridModelBuilder:
    """Fluent builder for :class:`~swiftllm.model_config.HybridModelConfig`.

    Build an architecture step by step with method chaining::

        cfg = (
            HybridModelBuilder(d_model=2048, num_layers=32, vocab_size=32000)
            .with_mamba3()                          # Mamba-3 SSM backbone
            .with_latent_moe(num_experts=64)        # LatentMoE every 4 layers
            .with_rlm(max_depth=3)                  # RLM reasoning block
            .with_dense_verification()              # post-decode verification
            .build()
        )

    Every ``with_*`` method returns ``self`` so calls can be chained.  Call
    :meth:`build` at the end to produce a validated
    :class:`~swiftllm.model_config.HybridModelConfig`.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        vocab_size: int = 32000,
        max_seq_len: int = 131072,
        tie_embeddings: bool = False,
    ) -> None:
        self._d_model = d_model
        self._num_layers = num_layers
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._tie_embeddings = tie_embeddings

        # component configs — populated by with_* methods
        self._mamba_config: Optional[MambaConfig] = None
        self._moe_config: Optional[Any] = None
        self._use_latent_moe: bool = True
        self._rlm_config: Optional[RlmConfig] = None
        self._dv_config: Optional[DenseVerificationConfig] = None
        self._layer_schedule: Optional[List[HybridLayerType]] = None
        self._base: Optional[ModelBaseConfig] = None

        # schedule params — applied lazily in build()
        self._moe_period: int = 4
        self._attn_period: int = 0

    # ------------------------------------------------------------------
    # with_* methods
    # ------------------------------------------------------------------

    def with_mamba3(
        self,
        d_state: int = 128,
        expand: int = 2,
        num_heads: Optional[int] = None,
        ns_steps: int = 5,
    ) -> "HybridModelBuilder":
        """Add a Mamba-3 SSM backbone (complex states, MIMO, trapezoidal disc).

        Parameters
        ----------
        d_state : int
            SSM state size per head.
        expand : int
            Inner expansion factor.
        num_heads : int or None
            Number of SSM heads; defaults to ``d_model // 64``.
        ns_steps : int
            Newton refinement steps for the trapezoidal integrator.
        """
        self._mamba_config = MambaConfig(
            d_model=self._d_model,
            expand=expand,
            d_state=d_state,
            use_complex_states=True,
            use_mimo=True,
            use_trapezoidal_disc=True,
            num_heads=num_heads,
            ns_steps=ns_steps,
        )
        return self

    def with_mamba2(
        self,
        d_state: int = 16,
        expand: int = 2,
    ) -> "HybridModelBuilder":
        """Add a Mamba-2 SSM backbone (real states, ZOH discretisation)."""
        self._mamba_config = MambaConfig.mamba2(self._d_model)
        self._mamba_config.d_state = d_state
        self._mamba_config.expand = expand
        return self

    def with_latent_moe(
        self,
        num_experts: int = 64,
        num_experts_per_token: int = 6,
        num_shared_experts: int = 2,
        latent_compression_ratio: int = 8,
        moe_period: int = 4,
        routing: RoutingStrategy = RoutingStrategy.TOP_K,
        aux_loss_coeff: float = 0.0,
    ) -> "HybridModelBuilder":
        """Add a LatentMoE block with DeepSeek-style load balancing.

        Parameters
        ----------
        num_experts : int
            Total expert count.
        num_experts_per_token : int
            Top-K experts activated per token.
        num_shared_experts : int
            Always-active shared experts (DeepSeek-MoE style).
        latent_compression_ratio : int
            Compression factor before routing (reduces inter-GPU traffic).
        moe_period : int
            LatentMoE layers appear every ``moe_period`` backbone layers.
        routing : RoutingStrategy
            Gating algorithm.
        aux_loss_coeff : float
            Load-balancing auxiliary loss coefficient (0 = aux-loss-free).
        """
        self._moe_period = moe_period
        self._use_latent_moe = True
        d_latent = self._d_model // latent_compression_ratio
        inner_moe = MoeConfig(
            d_model=d_latent,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            num_shared_experts=num_shared_experts,
            routing=routing,
            bias_update_rate=0.001 if routing == RoutingStrategy.TOP_K else 0.0,
            aux_loss_coeff=aux_loss_coeff,
        )
        obj = LatentMoeConfig.__new__(LatentMoeConfig)
        obj.d_model = self._d_model
        obj.latent_compression_ratio = latent_compression_ratio
        obj.moe = inner_moe
        self._moe_config = obj
        return self

    def with_standard_moe(
        self,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        num_shared_experts: int = 0,
        moe_period: int = 4,
    ) -> "HybridModelBuilder":
        """Add a standard (non-latent) sparse MoE block."""
        self._moe_period = moe_period
        self._use_latent_moe = False
        self._moe_config = MoeConfig(
            d_model=self._d_model,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            num_shared_experts=num_shared_experts,
        )
        return self

    def with_attention(self, attn_period: int = 8) -> "HybridModelBuilder":
        """Interleave sparse attention layers into the schedule.

        Parameters
        ----------
        attn_period : int
            Attention layers appear every ``attn_period`` backbone layers.
            Set to 0 to remove all attention (pure-Mamba).
        """
        self._attn_period = attn_period
        return self

    def with_rlm(
        self,
        max_depth: int = 3,
        enable_repl: bool = True,
        var_binding_slots: int = 32,
        early_exit_threshold: float = 0.92,
    ) -> "HybridModelBuilder":
        """Add the Recursive Language Model reasoning block.

        Parameters
        ----------
        max_depth : int
            Maximum recursion depth (3 for reasoning, 1 for language).
        enable_repl : bool
            Enable the REPL state machine.
        var_binding_slots : int
            Variable binding table size.
        early_exit_threshold : float
            Stop recursion when step-confidence ≥ this value.
        """
        self._rlm_config = RlmConfig(
            d_model=self._d_model,
            max_depth=max_depth,
            enable_repl=enable_repl,
            var_binding_slots=var_binding_slots,
            early_exit_threshold=early_exit_threshold,
        )
        return self

    def with_dense_verification(
        self,
        num_verification_heads: int = 8,
        min_confidence: float = 0.80,
        max_regen_attempts: int = 3,
        score_repl_steps: bool = True,
    ) -> "HybridModelBuilder":
        """Add the post-decode Dense Verification pass.

        Parameters
        ----------
        num_verification_heads : int
            Cross-attention heads in the verification pass.
        min_confidence : float
            Threshold below which re-generation is triggered.
        max_regen_attempts : int
            Maximum re-generation attempts.
        score_repl_steps : bool
            Produce per-REPL-step confidence scores (requires RLM block).
        """
        self._dv_config = DenseVerificationConfig(
            d_model=self._d_model,
            num_verification_heads=num_verification_heads,
            min_confidence=min_confidence,
            max_regen_attempts=max_regen_attempts,
            score_repl_steps=score_repl_steps,
        )
        return self

    def with_layer_schedule(
        self, schedule: List[HybridLayerType]
    ) -> "HybridModelBuilder":
        """Provide a fully custom per-layer schedule (overrides make_schedule)."""
        self._layer_schedule = schedule
        return self

    def with_base_config(self, base: ModelBaseConfig) -> "HybridModelBuilder":
        """Override the auto-generated ModelBaseConfig."""
        self._base = base
        return self

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self) -> HybridModelConfig:
        """Validate all settings and return a :class:`HybridModelConfig`.

        Raises
        ------
        ValueError
            If the configuration is incomplete or contradictory.
        """
        # Apply defaults for any component not explicitly set
        if self._mamba_config is None:
            self._mamba_config = MambaConfig.mamba3(self._d_model)

        schedule = self._layer_schedule or HybridModelConfig.make_schedule(
            num_layers=self._num_layers,
            moe_period=self._moe_period if self._moe_config is not None else 0,
            attn_period=self._attn_period,
        )

        cfg = HybridModelConfig.__new__(HybridModelConfig)
        cfg.d_model = self._d_model
        cfg.num_layers = self._num_layers
        cfg.vocab_size = self._vocab_size
        cfg.mamba_config = self._mamba_config
        cfg.moe_config = self._moe_config
        cfg.use_latent_moe = self._use_latent_moe
        cfg.rlm_config = self._rlm_config
        cfg.dense_verification_config = self._dv_config
        cfg.layer_schedule = schedule
        cfg.max_seq_len = self._max_seq_len
        cfg.tie_embeddings = self._tie_embeddings

        # Build base config
        if self._base is not None:
            cfg.base = self._base
        else:
            arch = "mamba3_reasoning" if self._rlm_config is not None else "mamba3_hybrid"
            cfg.base = ModelBaseConfig(
                architecture=arch,
                hidden_size=self._d_model,
                intermediate_size=4 * self._d_model,
                num_attention_heads=0,
                num_key_value_heads=0,
                num_hidden_layers=self._num_layers,
                vocab_size=self._vocab_size,
                max_position_embeddings=self._max_seq_len,
            )

        # Validate schedule length
        if len(cfg.layer_schedule) != cfg.num_layers:
            raise ValueError(
                f"layer_schedule length ({len(cfg.layer_schedule)}) "
                f"≠ num_layers ({cfg.num_layers})"
            )

        # Warn if score_repl_steps is True but no RLM block exists
        if (
            cfg.dense_verification_config is not None
            and cfg.dense_verification_config.score_repl_steps
            and cfg.rlm_config is None
        ):
            import warnings
            warnings.warn(
                "DenseVerificationConfig.score_repl_steps=True but no RlmConfig "
                "is present — per-step scoring will be skipped at runtime.",
                stacklevel=2,
            )

        return cfg

    def __repr__(self) -> str:
        return (
            f"HybridModelBuilder("
            f"d_model={self._d_model}, "
            f"num_layers={self._num_layers}, "
            f"vocab_size={self._vocab_size})"
        )


# ---------------------------------------------------------------------------
# Convenience preset functions
# ---------------------------------------------------------------------------

def build_mamba3_reasoning_model(
    d_model: int = 2048,
    num_layers: int = 32,
    vocab_size: int = 32000,
    num_experts: int = 64,
    moe_period: int = 4,
    max_seq_len: int = 131072,
) -> HybridModelConfig:
    """Build the flagship all-Mamba-3 reasoning architecture.

    Combines:
    - **Mamba-3 SSM** backbone (complex states, MIMO, trapezoidal discretisation)
    - **LatentMoE** every ``moe_period`` layers (DeepSeek-V3-style, aux-loss-free)
    - **RLM** reasoning block (REPL, variable binding, depth-3 recursion)
    - **Dense Verification** post-decode pass (cross-attention over REPL trace)

    No transformer attention blocks are used — this is a pure SSM architecture.

    Parameters
    ----------
    d_model : int
        Model dimension.  Common sizes: 1024 (small), 2048 (medium),
        4096 (large), 7168 (XL).
    num_layers : int
        Total backbone layers.  Typical: 24 (S), 32 (M), 48 (L), 64 (XL).
    vocab_size : int
        Vocabulary size.
    num_experts : int
        Total number of LatentMoE experts.
    moe_period : int
        LatentMoE appears every ``moe_period`` layers.
    max_seq_len : int
        Maximum context length in tokens.

    Returns
    -------
    HybridModelConfig
        Validated configuration ready to pass to the engine or serialise.

    Example
    -------
    ::

        cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)
        print(cfg.summary())
    """
    return (
        HybridModelBuilder(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        .with_mamba3()
        .with_latent_moe(
            num_experts=num_experts,
            num_experts_per_token=6,
            num_shared_experts=2,
            latent_compression_ratio=8,
            moe_period=moe_period,
        )
        .with_rlm(max_depth=3, enable_repl=True)
        .with_dense_verification(min_confidence=0.80, score_repl_steps=True)
        .build()
    )


def build_mamba3_base_model(
    d_model: int = 2048,
    num_layers: int = 32,
    vocab_size: int = 32000,
    num_experts: int = 64,
    moe_period: int = 4,
    max_seq_len: int = 131072,
) -> HybridModelConfig:
    """Build a Mamba-3 + LatentMoE base model (no reasoning layers).

    Use this as a foundation for supervised fine-tuning before adding
    the RLM and Dense Verification layers via GRPO training.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_layers : int
        Total backbone layers.
    vocab_size : int
        Vocabulary size.
    num_experts : int
        Total LatentMoE experts.
    moe_period : int
        LatentMoE period.
    max_seq_len : int
        Maximum sequence length.

    Returns
    -------
    HybridModelConfig
    """
    return (
        HybridModelBuilder(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        .with_mamba3()
        .with_latent_moe(
            num_experts=num_experts,
            num_experts_per_token=6,
            num_shared_experts=2,
            moe_period=moe_period,
        )
        .build()
    )


def build_mamba3_pure_model(
    d_model: int = 2048,
    num_layers: int = 32,
    vocab_size: int = 32000,
    max_seq_len: int = 131072,
) -> HybridModelConfig:
    """Build a pure Mamba-3 model (no MoE, no attention, no reasoning).

    Useful as an ablation baseline or for resource-constrained deployments.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_layers : int
        Total backbone layers.
    vocab_size : int
        Vocabulary size.
    max_seq_len : int
        Maximum sequence length.

    Returns
    -------
    HybridModelConfig
    """
    return (
        HybridModelBuilder(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        .with_mamba3()
        .build()
    )


def build_mamba3_hybrid_attention_model(
    d_model: int = 2048,
    num_layers: int = 32,
    vocab_size: int = 32000,
    attn_period: int = 8,
    moe_period: int = 4,
    max_seq_len: int = 131072,
) -> HybridModelConfig:
    """Build a mostly-Mamba-3 model with sparse attention layers.

    Mimics the Jamba architecture: majority Mamba SSM layers with one
    attention layer every ``attn_period`` layers, and LatentMoE every
    ``moe_period`` layers.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_layers : int
        Total backbone layers.
    vocab_size : int
        Vocabulary size.
    attn_period : int
        Attention layers appear every ``attn_period`` backbone layers.
    moe_period : int
        LatentMoE period.
    max_seq_len : int
        Maximum sequence length.

    Returns
    -------
    HybridModelConfig
    """
    return (
        HybridModelBuilder(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        .with_mamba3()
        .with_latent_moe(num_experts=64, moe_period=moe_period)
        .with_attention(attn_period=attn_period)
        .build()
    )


# ---------------------------------------------------------------------------
# JSON helpers patched onto HybridModelConfig
# ---------------------------------------------------------------------------

def _to_json(self, path: Union[str, "os.PathLike[str]"], indent: int = 2) -> None:
    """Serialise this config to a JSON file.

    Parameters
    ----------
    path : str or path-like
        Destination file path.
    indent : int
        JSON indentation width.

    Example
    -------
    ::

        cfg = build_mamba3_reasoning_model(d_model=2048)
        cfg.to_json("mamba3_reasoning_2048.json")
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(self.to_dict(), fh, indent=indent)


@classmethod
def _from_json(cls, path: Union[str, "os.PathLike[str]"]) -> "HybridModelConfig":
    """Load a :class:`HybridModelConfig` from a JSON file.

    Parameters
    ----------
    path : str or path-like
        Source JSON file previously written by :meth:`to_json`.

    Returns
    -------
    HybridModelConfig
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    return cls.from_dict(d)


# Patch JSON helpers onto the dataclass (avoids modifying model_config.py)
HybridModelConfig.to_json = _to_json         # type: ignore[attr-defined]
HybridModelConfig.from_json = _from_json     # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Model size estimator
# ---------------------------------------------------------------------------

def estimate_parameters(cfg: HybridModelConfig) -> Dict[str, int]:
    """Estimate the parameter count for a hybrid model configuration.

    This is an approximation — exact counts depend on the Rust implementation
    details (layer norm, bias terms, embedding tying, etc.).

    Parameters
    ----------
    cfg : HybridModelConfig
        The model configuration to estimate.

    Returns
    -------
    dict
        Keys: ``"embedding"``, ``"mamba_layers"``, ``"moe_layers"``,
        ``"rlm"``, ``"dense_verification"``, ``"lm_head"``, ``"total"``.

    Example
    -------
    ::

        cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)
        params = estimate_parameters(cfg)
        print(f"Total: {params['total'] / 1e9:.1f}B parameters")
    """
    d = cfg.d_model
    V = cfg.vocab_size
    L = cfg.num_layers

    # Embedding table
    embedding = d * V

    # Mamba-3 SSM layer params (approximate)
    # Per layer: in_proj (d → d*expand*2), conv (d_inner, d_conv),
    #            A_log/B/C/dt (d_inner, d_state), out_proj (d_inner → d)
    mc = cfg.mamba_config
    d_inner = d * (mc.expand if mc else 2)
    d_state = mc.d_state if mc else 128
    params_per_mamba = (
        d * d_inner * 2       # in_proj (×2 for gate)
        + d_inner             # conv weight
        + d_inner * d_state   # A_log
        + d_inner * d_state   # B
        + d_inner * d_state   # C
        + d_inner             # dt
        + d_inner * d         # out_proj
        + d * 2               # RMSNorm
    )
    n_plain_mamba = cfg.num_mamba_layers - cfg.num_moe_layers
    n_mamba_moe = cfg.num_moe_layers
    mamba_params = n_plain_mamba * params_per_mamba

    # LatentMoE layer params (approximate)
    moe_params = 0
    if cfg.moe_config is not None and n_mamba_moe > 0:
        if isinstance(cfg.moe_config, LatentMoeConfig):
            d_lat = cfg.moe_config.d_latent
            moe_cfg = cfg.moe_config.moe
            # compress + expand projections
            proj_params = d * d_lat * 2
        else:
            d_lat = d
            moe_cfg = cfg.moe_config
            proj_params = 0
        d_ffn = moe_cfg.d_ffn or (4 * d_lat)
        E = moe_cfg.num_experts
        # router + E × (gate, up, down)
        expert_params = (d_lat * E) + E * (d_lat * d_ffn * 3)
        params_per_moe_layer = (
            params_per_mamba     # SSM part (Mamba-3 SSM still present)
            + proj_params        # compress/expand
            + expert_params      # MoE experts + router
            + d * 2              # extra RMSNorm
        )
        moe_params = n_mamba_moe * params_per_moe_layer

    # RLM block
    rlm_params = 0
    if cfg.rlm_config is not None:
        rc = cfg.rlm_config
        depth_hidden = rc.depth_hidden_size or (d // 4)
        d_sub = rc.d_subproblem or (d // 2)
        rlm_params = (
            d * depth_hidden       # depth MLP (in)
            + depth_hidden * d     # depth MLP (out)
            + rc.var_binding_slots * d   # binding table
            + d * d_sub            # sub-problem embed
            + d * 2                # RMSNorm
        )

    # Dense Verification layer
    dv_params = 0
    if cfg.dense_verification_config is not None:
        dvc = cfg.dense_verification_config
        d_vhead = dvc.d_v_head or (d // 8)
        H = dvc.num_verification_heads
        # Q/K/V projections + output proj
        dv_params = (
            d * d_vhead * H * 3   # Q, K, V
            + d_vhead * H * d     # out proj
            + d                   # scalar scorer
            + d * 2               # RMSNorm
        )

    # LM head
    lm_head = 0 if cfg.tie_embeddings else d * V

    total = embedding + mamba_params + moe_params + rlm_params + dv_params + lm_head

    return {
        "embedding": embedding,
        "mamba_layers": mamba_params,
        "moe_layers": moe_params,
        "rlm": rlm_params,
        "dense_verification": dv_params,
        "lm_head": lm_head,
        "total": total,
    }


def parameter_summary(cfg: HybridModelConfig) -> str:
    """Return a human-readable parameter count summary.

    Example
    -------
    ::

        cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)
        print(parameter_summary(cfg))
    """
    counts = estimate_parameters(cfg)
    total = counts["total"]

    def fmt(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        return f"{n:,}"

    lines = [
        f"Parameter Estimate  ({fmt(total)} total)",
        f"  Embedding         : {fmt(counts['embedding'])}",
        f"  Mamba-3 layers    : {fmt(counts['mamba_layers'])}",
        f"  MoE layers        : {fmt(counts['moe_layers'])}",
        f"  RLM block         : {fmt(counts['rlm'])}",
        f"  Dense Verif.      : {fmt(counts['dense_verification'])}",
        f"  LM head           : {fmt(counts['lm_head'])}",
        f"  ─────────────────────────────",
        f"  Total             : {fmt(total)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Builder
    "HybridModelBuilder",
    # Result type
    "HybridForwardResult",
    # Preset functions
    "build_mamba3_reasoning_model",
    "build_mamba3_base_model",
    "build_mamba3_pure_model",
    "build_mamba3_hybrid_attention_model",
    # Utilities
    "estimate_parameters",
    "parameter_summary",
]

# ------------------------------------------------------------------------------
# END OF FILE: hybrid_model.py
# REPO PATH:   /swiftllm/python/swiftllm/hybrid_model.py
# BRIDGES:     model_config.py ↔ crates/swiftllm-models (via PyO3 JSON bridge)
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
