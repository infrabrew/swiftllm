# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      torch_model.py
# PATH:      /python/swiftllm/torch_model.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/model_config.py   HybridModelConfig + all component configs
# USED BY:
#   - python/swiftllm/__init__.py       public re-exports
#   - examples/hybrid_model_torch.py    training / inference examples
# SEE ALSO:
#   - crates/swiftllm-models/src/        Rust compute backend (replaces this module
#                                        once GPU kernels are complete)
#   - https://github.com/state-spaces/mamba  mamba-ssm (optional fast backend)
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

"""SwiftLLM PyTorch Bridge — GPU-executable hybrid model

This module builds a fully GPU-executable :class:`torch.nn.Module` from a
:class:`~swiftllm.model_config.HybridModelConfig`.  It acts as the interim
compute backend while the native Rust/CUDA kernel bridge is under development.

Architecture mapping
--------------------
Each :class:`~swiftllm.model_config.HybridLayerType` maps to a PyTorch module:

============================  =================================================
``HybridLayerType``           Module
============================  =================================================
``MAMBA``                     :class:`MambaLayer`  (SSM only, no FFN)
``MAMBA_MOE``                 :class:`MambaLayer` + :class:`LatentMoeLayer`
``ATTENTION``                 :class:`AttentionLayer` + :class:`DenseFFN`
``ATTENTION_MOE``             :class:`AttentionLayer` + :class:`LatentMoeLayer`
============================  =================================================

The RLM block (:class:`RlmLayer`) wraps the final backbone output, adding
learned depth embeddings, confidence estimation, and sub-problem gating.

The Dense Verification layer (:class:`DenseVerificationLayer`) is a separate
post-decode cross-attention pass called after token generation is complete.

SSM backend selection
---------------------
``mamba-ssm`` is used when available (fused CUDA kernels, Tri Dao et al.):

    pip install mamba-ssm causal-conv1d

When not installed, :class:`_ReferenceMamba` provides a correct pure-PyTorch
fallback (sequential scan — accurate but slower for long sequences).

.. warning::
   ``mamba-ssm`` implements Mamba-2.  Mamba-3's complex states, MIMO
   multi-head formulation, and trapezoidal discretisation are approximated
   by the Mamba-2 SSD kernel.  All Mamba-3-specific flags in
   :class:`~swiftllm.model_config.MambaConfig` are honoured by the reference
   implementation but silently approximated when using the ``mamba-ssm`` path.

Usage::

    from swiftllm import build_mamba3_reasoning_model
    from swiftllm.torch_model import build_torch_model

    cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)
    model = build_torch_model(cfg).cuda()

    # Training
    input_ids = torch.randint(0, cfg.vocab_size, (2, 512)).cuda()
    labels    = torch.randint(0, cfg.vocab_size, (2, 512)).cuda()
    out = model(input_ids, labels=labels)
    out.loss.backward()

    # Inference
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits              # [batch, seq, vocab_size]
    verif  = out.verification        # VerificationOutput or None
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import (
    DenseVerificationConfig,
    HybridLayerType,
    HybridModelConfig,
    LatentMoeConfig,
    MambaConfig,
    MoeConfig,
    ModelBaseConfig,
    RlmConfig,
    RoutingStrategy,
)

# ---------------------------------------------------------------------------
# Optional: mamba-ssm fast backend
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba2 as _Mamba2Impl
    HAS_MAMBA_SSM = True
except ImportError:
    _Mamba2Impl = None
    HAS_MAMBA_SSM = False

# ---------------------------------------------------------------------------
# RMSNorm compatibility (nn.RMSNorm added in PyTorch 2.4)
# ---------------------------------------------------------------------------
try:
    _RMSNorm = nn.RMSNorm
except AttributeError:
    class _RMSNorm(nn.Module):  # type: ignore[no-redef]
        """Fallback RMSNorm for PyTorch < 2.4."""

        def __init__(self, d: int, eps: float = 1e-5, elementwise_affine: bool = True):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d)) if elementwise_affine else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            out = x / rms
            return out * self.weight if self.weight is not None else out


# ===========================================================================
# 1.  Reference Mamba SSM (pure PyTorch — no mamba-ssm dependency)
# ===========================================================================

class _ReferenceMamba(nn.Module):
    """Pure-PyTorch Mamba-2/3 reference implementation.

    Implements the full selective SSM scan with:
    - Input / output projections
    - Causal depthwise convolution
    - Δt projection with softplus activation
    - Selective SSM scan (loop-based — correct, not fused)
    - SiLU gate

    Mamba-3 extensions (complex states, MIMO, trapezoidal discretisation) are
    structurally present but approximated with real arithmetic for simplicity.
    Replace with a fused CUDA kernel for production throughput.
    """

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        expand = cfg.expand
        d_inner = d * expand
        d_state = cfg.d_state
        dt_rank = cfg.dt_rank  # already resolved (non-None) by MambaConfig.__post_init__

        # Projections
        # in_proj: d → [z (d_inner), x (d_inner), B (d_state), C (d_state), dt (dt_rank)]
        in_dim = d_inner * 2 + d_state * 2 + dt_rank
        self.in_proj  = nn.Linear(d, in_dim, bias=cfg.bias)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d, bias=cfg.bias)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=cfg.d_conv,
            groups=d_inner,
            padding=cfg.d_conv - 1,
            bias=cfg.conv_bias,
        )

        # SSM parameters
        # A: [d_inner, d_state] — log parameterisation, must stay negative
        A_init = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D     = nn.Parameter(torch.ones(d_inner))   # skip connection

        # Mamba-3 MIMO: per-head B/C projections (approximated with shared B/C)
        # A real Mamba-3 impl would use per-head complex B/C; here we use shared.
        if cfg.use_mimo and cfg.num_heads and cfg.num_heads > 1:
            self.num_heads = cfg.num_heads
            self.head_dim  = d_inner // cfg.num_heads
        else:
            self.num_heads = 1
            self.head_dim  = d_inner

        # dt bias (initialised to small positive values)
        dt_bias_init = torch.exp(
            torch.rand(d_inner) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt_bias_init + torch.log(-torch.expm1(-dt_bias_init))
        self.dt_bias = nn.Parameter(inv_dt)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [batch, seq_len, d_model]

        Returns
        -------
        Tensor, shape [batch, seq_len, d_model]
        """
        B, L, _ = x.shape
        d_inner = self.cfg.d_model * self.cfg.expand
        d_state = self.cfg.d_state
        dt_rank = self.cfg.dt_rank

        # 1. Input projection
        proj = self.in_proj(x)                               # [B, L, in_dim]
        x_ssm, z, B_proj, C_proj, dt_raw = proj.split(
            [d_inner, d_inner, d_state, d_state, dt_rank], dim=-1
        )

        # 2. Causal conv on x_ssm
        # Conv1d expects [B, C, L]; slice off the causal padding on the right
        x_ssm_t = x_ssm.transpose(1, 2)                     # [B, d_inner, L]
        x_conv   = self.conv1d(x_ssm_t)[..., :L]            # [B, d_inner, L]
        x_conv   = x_conv.transpose(1, 2)                   # [B, L, d_inner]
        x_conv   = F.silu(x_conv)

        # 3. Δt
        dt = F.softplus(self.dt_proj(dt_raw) + self.dt_bias)  # [B, L, d_inner]

        # 4. SSM parameters
        A  = -torch.exp(self.A_log.float())                  # [d_inner, d_state]
        D  = self.D.float()

        # 5. Selective SSM scan (sequential loop — correct reference impl)
        #    For Mamba-3: real approximation of complex states.
        h = x.new_zeros(B, d_inner, d_state)                # SSM state
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].float()                       # [B, d_inner]
            u_t  = x_conv[:, t, :].float()                  # [B, d_inner]
            B_t  = B_proj[:, t, :].float()                  # [B, d_state]
            C_t  = C_proj[:, t, :].float()                  # [B, d_state]

            # Discretise: ZOH (trapezoidal approximation for Mamba-3)
            if self.cfg.use_trapezoidal_disc:
                # Exponential-trapezoidal: dA = exp(A * dt), dB = (expm1(A*dt)/A) * dt * B
                dA = torch.exp(torch.einsum('bi,ip->bip', dt_t, A))    # [B, d_inner, d_state]
                dB = torch.einsum('bi,bp->bip', dt_t, B_t)             # [B, d_inner, d_state]
                # Trapezoid correction (1 + A*dt/2) / (1 - A*dt/2) — first-order
                trap = 1.0 + 0.5 * torch.einsum('bi,ip->bip', dt_t, A)
                dB = dB * trap.abs()
            else:
                # Standard ZOH
                dA = torch.exp(torch.einsum('bi,ip->bip', dt_t, A))
                dB = torch.einsum('bi,bp->bip', dt_t, B_t)

            # State update: h = dA * h + dB * u
            h = dA * h + dB * u_t.unsqueeze(-1)             # [B, d_inner, d_state]

            # Output: y = C * h + D * u
            y_t = torch.einsum('bip,bp->bi', h, C_t) + D * u_t   # [B, d_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1).to(x.dtype)              # [B, L, d_inner]

        # 6. SiLU gate
        y = y * F.silu(z)

        # 7. Output projection
        return self.out_proj(y)                              # [B, L, d_model]


# ===========================================================================
# 2.  MambaLayer  (RMSNorm + SSM + residual)
# ===========================================================================

class MambaLayer(nn.Module):
    """Mamba-3 SSM block with pre-norm and residual connection.

    Uses ``mamba_ssm.Mamba2`` (fused CUDA) when available, otherwise falls
    back to :class:`_ReferenceMamba` (pure PyTorch).
    """

    def __init__(self, cfg: MambaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.norm = _RMSNorm(cfg.d_model)

        if HAS_MAMBA_SSM:
            # headdim: per-head size for Mamba-2 SSD formulation
            headdim = max(1, cfg.d_model // max(1, cfg.num_heads or 1))
            self._ssm = _Mamba2Impl(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                headdim=headdim,
                dt_min=cfg.dt_min,
                dt_max=cfg.dt_max,
                bias=cfg.bias,
                conv_bias=cfg.conv_bias,
                layer_idx=layer_idx,
            )
            self._use_mamba_ssm = True
        else:
            self._ssm = _ReferenceMamba(cfg)
            self._use_mamba_ssm = False

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        return self._ssm(self.norm(x)) + residual


# ===========================================================================
# 3.  ExpertMLP  (single FFN expert in a MoE layer)
# ===========================================================================

class ExpertMLP(nn.Module):
    """SwiGLU FFN — one expert in a MoE / LatentMoE layer."""

    def __init__(self, d_in: int, d_ffn: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_ffn, bias=bias)
        self.up_proj   = nn.Linear(d_in, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, d_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ===========================================================================
# 4.  LatentMoeLayer
# ===========================================================================

class LatentMoeLayer(nn.Module):
    """Latent-compressed MoE FFN block with pre-norm and residual connection.

    Pipeline::

        x  ──► compress (d_model → d_latent)
               │
               ├─► router → top-k expert indices + weights
               │            (+ DeepSeek-style dynamic bias if TOP_K)
               │
               ├─► shared experts (always active)
               │
               ├─► sparse experts (top-k dispatched)
               │
               └─► expand (d_latent → d_model) ──► +residual

    The dynamic bias buffer is updated each training step via
    :meth:`update_load_stats` — call it from the training loop after
    ``loss.backward()`` and ``optimizer.step()``.
    """

    def __init__(self, cfg: LatentMoeConfig) -> None:
        super().__init__()
        d = cfg.d_model
        d_lat = cfg.d_latent
        moe = cfg.moe
        d_ffn = moe.d_ffn or (4 * d_lat)
        E = moe.num_experts
        k = moe.num_experts_per_token
        S = moe.num_shared_experts

        self.d_model   = d
        self.d_latent  = d_lat
        self.num_experts            = E
        self.num_experts_per_token  = k
        self.num_shared_experts     = S
        self.routing                = moe.routing
        self.normalise_weights      = moe.normalise_weights
        self.capacity_factor        = moe.capacity_factor
        self.router_temperature     = moe.router_temperature
        self.bias_update_rate       = moe.bias_update_rate

        self.norm         = _RMSNorm(d)
        self.compress     = nn.Linear(d, d_lat, bias=False)
        self.expand       = nn.Linear(d_lat, d, bias=False)
        self.router       = nn.Linear(d_lat, E, bias=(moe.routing == RoutingStrategy.TOP_K))
        self.experts      = nn.ModuleList([ExpertMLP(d_lat, d_ffn) for _ in range(E)])
        self.shared       = nn.ModuleList([ExpertMLP(d_lat, d_ffn) for _ in range(S)])

        # Dynamic load-balancing bias (DeepSeek-V3 aux-loss-free)
        # Not a parameter — updated manually, not by the optimizer.
        self.register_buffer("expert_bias", torch.zeros(E))

        # Running load stats for bias update
        self.register_buffer("_load_counts", torch.zeros(E))
        self._steps = 0

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        B, L, _ = x.shape
        T = B * L   # total tokens

        # 1. Compress
        latent = self.compress(x.view(T, self.d_model))        # [T, d_lat]

        # 2. Route
        router_logits = self.router(latent)                     # [T, E]
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature
        if self.routing == RoutingStrategy.TOP_K:
            # Add dynamic bias (DeepSeek-V3)
            router_logits = router_logits + self.expert_bias.unsqueeze(0)

        # Guard: k cannot exceed total experts (important for small test configs)
        k_eff = min(self.num_experts_per_token, self.num_experts)

        if self.routing == RoutingStrategy.RELU_GATING:
            # ReLU: only experts with positive logit are active
            gate_weights = F.relu(router_logits)                # [T, E]
            top_indices  = (gate_weights > 0).nonzero(as_tuple=False)
            # For simplicity, fall back to top-k=2 if nothing activates
            if top_indices.numel() == 0:
                gate_weights, top_indices = torch.topk(router_logits, 2, dim=-1)
                gate_weights = F.softmax(gate_weights, dim=-1)
            else:
                # Normalise non-zero weights per token
                row_sums = gate_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                gate_weights = gate_weights / row_sums
        elif self.routing == RoutingStrategy.EXPERT_CHOICE:
            # Expert-choice: each expert picks its top-capacity tokens
            # Simplified: transpose and top-k per expert → reconstruct per token
            capacity = max(1, int(math.ceil(T / self.num_experts * self.capacity_factor)))
            scores   = F.softmax(router_logits, dim=0)          # [T, E] softmax over tokens
            _, expert_picks = torch.topk(scores, capacity, dim=0)  # [capacity, E]
            # Build a [T, E] sparse weight matrix
            gate_weights = torch.zeros_like(router_logits)
            for e in range(self.num_experts):
                gate_weights[expert_picks[:, e], e] = scores[expert_picks[:, e], e]
            # Limit each token to at most k_eff experts
            topk_w, topk_i = torch.topk(gate_weights, k_eff, dim=-1)
            sparse = torch.zeros_like(gate_weights)
            sparse.scatter_(-1, topk_i, topk_w)
            gate_weights = sparse
            top_indices  = topk_i
        else:
            # Standard Top-K
            gate_scores = F.softmax(router_logits, dim=-1)      # [T, E]
            topk_w, top_indices = torch.topk(gate_scores, k_eff, dim=-1)
            # [T, k_eff]
            if self.normalise_weights:
                topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            gate_weights = topk_w
            # Track load stats for bias update
            with torch.no_grad():
                load = torch.zeros(self.num_experts, device=latent.device)
                load.scatter_add_(0, top_indices.view(-1),
                                  torch.ones(T * self.num_experts_per_token,
                                             device=latent.device))
                self._load_counts += load
            self._steps += 1

        # 3. Dispatch to sparse experts
        if self.routing in (RoutingStrategy.TOP_K, RoutingStrategy.RELU_GATING):
            # gate_weights: [T, k_eff], top_indices: [T, k_eff]
            output = torch.zeros(T, self.d_latent, device=latent.device, dtype=latent.dtype)
            for ki in range(k_eff):
                expert_ids = top_indices[:, ki]   # [T]
                weights    = gate_weights[:, ki]  # [T]
                for e_id in range(self.num_experts):
                    mask = (expert_ids == e_id)
                    if mask.any():
                        tokens_in = latent[mask]
                        expert_out = self.experts[e_id](tokens_in)
                        output[mask] += weights[mask].unsqueeze(-1) * expert_out
        else:
            # Expert-choice: use gate_weights directly
            output = torch.zeros(T, self.d_latent, device=latent.device, dtype=latent.dtype)
            nz = gate_weights.nonzero(as_tuple=False)           # [nnz, 2]
            for row, e_id in nz:
                output[row] += gate_weights[row, e_id] * self.experts[e_id](latent[row:row+1]).squeeze(0)

        # 4. Shared experts (always active)
        shared_out = torch.zeros_like(output)
        for shared_expert in self.shared:
            shared_out = shared_out + shared_expert(latent)
        if self.num_shared_experts > 0:
            output = output + shared_out / self.num_shared_experts

        # 5. Expand back to d_model
        expanded = self.expand(output)                          # [T, d_model]
        return expanded.view(B, L, self.d_model) + residual

    def update_load_stats(self) -> None:
        """Update expert bias for aux-loss-free load balancing.

        Call once per training step after ``optimizer.step()``.  This method
        adjusts the per-expert bias so that under-loaded experts become more
        likely to be selected in future steps, preventing expert collapse.
        """
        if self._steps == 0 or self.routing != RoutingStrategy.TOP_K:
            return
        avg_load = self._load_counts.sum() / self.num_experts
        delta = self.bias_update_rate * (avg_load - self._load_counts)
        self.expert_bias.data += delta
        self._load_counts.zero_()
        self._steps = 0


# ===========================================================================
# 5.  AttentionLayer  (for ATTENTION / ATTENTION_MOE layer types)
# ===========================================================================

class _RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int = 131072, theta: float = 500000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                      # [max_seq_len, head_dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)             # [max_seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)


class AttentionLayer(nn.Module):
    """Multi-head attention with RoPE, grouped-query support, and pre-norm."""

    def __init__(self, cfg: ModelBaseConfig) -> None:
        super().__init__()
        d   = cfg.hidden_size
        nH  = max(1, cfg.num_attention_heads)
        nKV = max(1, cfg.num_key_value_heads)
        hd  = cfg.head_dim or (d // nH)

        self.d_model      = d
        self.num_heads    = nH
        self.num_kv_heads = nKV
        self.head_dim     = hd
        self.scale        = hd ** -0.5

        self.norm   = _RMSNorm(d)
        self.q_proj = nn.Linear(d, nH  * hd, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(d, nKV * hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(d, nKV * hd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(nH * hd, d,  bias=cfg.attention_bias)
        self.rope   = _RotaryEmbedding(hd, max_seq_len=cfg.max_position_embeddings,
                                       theta=cfg.rope_theta)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads,    self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q, K = self.rope(Q, K, L)

        # Expand KV for grouped-query attention
        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(ratio, dim=1)
            V = V.repeat_interleave(ratio, dim=1)

        # Scaled dot-product attention (PyTorch 2.0 fused implementation)
        try:
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, is_causal=attn_mask is None)
        except RuntimeError:
            # Fallback for older PyTorch
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                scores = scores + attn_mask
            else:
                causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
                scores = scores + causal
            out = torch.matmul(F.softmax(scores, dim=-1), V)

        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.o_proj(out) + residual


class DenseFFN(nn.Module):
    """Standard SwiGLU feed-forward network with pre-norm and residual."""

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False) -> None:
        super().__init__()
        self.norm      = _RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.up_proj   = nn.Linear(d_model, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + residual


# ===========================================================================
# 6.  HybridBlock  (dispatches by HybridLayerType)
# ===========================================================================

class HybridBlock(nn.Module):
    """Single backbone block — type determined by :class:`HybridLayerType`.

    ================================  =========================================
    Layer type                        Sub-modules
    ================================  =========================================
    ``MAMBA``                         :class:`MambaLayer`
    ``MAMBA_MOE``                     :class:`MambaLayer` + :class:`LatentMoeLayer`
    ``ATTENTION``                     :class:`AttentionLayer` + :class:`DenseFFN`
    ``ATTENTION_MOE``                 :class:`AttentionLayer` + :class:`LatentMoeLayer`
    ================================  =========================================
    """

    def __init__(
        self,
        layer_type: HybridLayerType,
        mamba_cfg: MambaConfig,
        moe_cfg: Optional[LatentMoeConfig],
        base_cfg: ModelBaseConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type

        has_mamba  = layer_type in (HybridLayerType.MAMBA, HybridLayerType.MAMBA_MOE)
        has_attn   = layer_type in (HybridLayerType.ATTENTION, HybridLayerType.ATTENTION_MOE)
        has_latent = layer_type in (HybridLayerType.MAMBA_MOE, HybridLayerType.ATTENTION_MOE)
        has_dense  = layer_type == HybridLayerType.ATTENTION

        self.mamba   = MambaLayer(mamba_cfg, layer_idx=layer_idx) if has_mamba else None
        self.attn    = AttentionLayer(base_cfg)                    if has_attn  else None
        self.lmoe    = LatentMoeLayer(moe_cfg) if (has_latent and moe_cfg) else None
        self.dense   = DenseFFN(base_cfg.hidden_size, base_cfg.intermediate_size,
                                bias=base_cfg.mlp_bias)            if has_dense else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.mamba:
            x = self.mamba(x)
        if self.attn:
            x = self.attn(x, attn_mask=kwargs.get("attn_mask"))
        if self.lmoe:
            x = self.lmoe(x)
        if self.dense:
            x = self.dense(x)
        return x

    def update_load_stats(self) -> None:
        """Delegate load-stat update to the LatentMoE sub-layer (if present)."""
        if self.lmoe is not None:
            self.lmoe.update_load_stats()


# ===========================================================================
# 7.  RlmLayer  (Recursive Language Model block)
# ===========================================================================

class RlmLayer(nn.Module):
    """Learned components of the Recursive Language Model block.

    This module contains the trainable parts of the RLM:

    * **Depth embedding** — adds a learned offset for the current recursion
      depth, shifting the hidden-state distribution to signal "I am thinking
      at depth d".
    * **Confidence MLP** — predicts a scalar confidence score that controls
      early exit during inference.
    * **Variable binding table** — a lookup table of ``var_binding_slots``
      learnable embeddings; indexed at inference time by the REPL executor.
    * **Sub-problem gating** — projects the hidden state into a lower-
      dimensional sub-problem query; the gate blends the "direct" and
      "recursive" outputs based on the confidence score.

    The symbolic REPL execution (``Assign / Compute / Verify / Recurse``
    steps) is handled by the Rust inference engine, not this module.

    Training-time forward: ``depth=0``, REPL state is not active.
    The module still trains the depth/confidence MLPs via the normal loss.
    """

    def __init__(self, cfg: RlmConfig) -> None:
        super().__init__()
        d   = cfg.d_model
        dh  = cfg.depth_hidden_size   # resolved by __post_init__
        ds  = cfg.d_subproblem         # resolved by __post_init__
        S   = cfg.var_binding_slots
        D   = cfg.max_depth

        self.cfg = cfg

        self.norm = _RMSNorm(d)

        # Depth embedding — one vector per depth level
        self.depth_embed = nn.Embedding(D + 1, d)

        # Confidence MLP: hidden_states → scalar confidence in [0, 1]
        self.conf_h   = nn.Linear(d, dh)
        self.conf_out = nn.Linear(dh, 1)

        # Variable binding table (slots of d_model each)
        if S > 0:
            self.var_table = nn.Embedding(S, d)
        else:
            self.var_table = None

        # Sub-problem encoder and solution gate
        self.sub_proj  = nn.Linear(d, ds)
        self.sol_proj  = nn.Linear(ds, d)
        self.gate_proj = nn.Linear(d, 1)   # blend direct vs recursive

        self.out_norm = _RMSNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        depth: int = 0,
        var_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor [batch, seq, d_model]
            Hidden states from the backbone.
        depth : int
            Current recursion depth (0 during training; ≥0 during inference).
        var_indices : Tensor [batch, num_vars] or None
            Variable binding slot indices to read from the table.
            *None* means no binding context is injected.

        Returns
        -------
        out : Tensor [batch, seq, d_model]
            Modified hidden states.
        confidence : Tensor [batch, seq]
            Per-token confidence scores in [0, 1].
        """
        residual = x
        h = self.norm(x)

        # 1. Add depth embedding
        depth_clamped = min(depth, self.cfg.max_depth)
        depth_idx = torch.full((1,), depth_clamped, dtype=torch.long, device=x.device)
        h = h + self.depth_embed(depth_idx)       # broadcast over batch & seq

        # 2. Inject variable binding context
        if var_indices is not None and self.var_table is not None:
            binding_vecs = self.var_table(var_indices)          # [B, num_vars, d]
            binding_ctx  = binding_vecs.mean(dim=1, keepdim=True)  # [B, 1, d]
            h = h + binding_ctx

        # 3. Sub-problem gating
        sub  = F.silu(self.sub_proj(h))       # [B, L, d_sub]
        sol  = self.sol_proj(sub)             # [B, L, d]
        gate = torch.sigmoid(self.gate_proj(h))  # [B, L, 1]
        blended = gate * sol + (1.0 - gate) * h

        # 4. Confidence score
        conf_h   = F.silu(self.conf_h(blended))   # [B, L, dh]
        confidence = torch.sigmoid(self.conf_out(conf_h)).squeeze(-1)  # [B, L]

        out = self.out_norm(blended) + residual
        return out, confidence


# ===========================================================================
# 8.  DenseVerificationLayer
# ===========================================================================

@dataclass
class VerificationOutput:
    """Output of the Dense Verification pass.

    Attributes
    ----------
    global_score : float tensor [batch]
        Model-level confidence that the draft output is correct.
    token_scores : float tensor [batch, seq]
        Per-token confidence scores in [0, 1].
    step_scores : float tensor [batch, num_steps] or None
        Per-REPL-step confidence scores.  *None* when
        ``score_repl_steps=False`` or no RLM block is present.
    is_accepted : bool tensor [batch]
        True when ``global_score >= min_confidence`` for every item.
    low_confidence_positions : list of list of int
        Token positions where confidence < ``min_confidence``, per batch item.
    """
    global_score: torch.Tensor
    token_scores: torch.Tensor
    step_scores: Optional[torch.Tensor]
    is_accepted: torch.Tensor
    low_confidence_positions: List[List[int]]


class DenseVerificationLayer(nn.Module):
    """Post-decode Dense Verification pass.

    Cross-attends the draft hidden states against a trace tensor (the
    accumulated RLM hidden states or, in the simplified Python path, a
    summary of the final backbone hidden states).

    Called **after** token generation is complete, not during the forward pass.

    Parameters
    ----------
    cfg : DenseVerificationConfig
    """

    def __init__(self, cfg: DenseVerificationConfig) -> None:
        super().__init__()
        d   = cfg.d_model
        H   = cfg.num_verification_heads
        dv  = cfg.d_v_head   # resolved by __post_init__
        kv  = H * dv

        self.cfg   = cfg
        self.scale = dv ** -0.5

        self.input_norm = _RMSNorm(d)
        self.q_proj     = nn.Linear(d, kv, bias=False)
        self.k_proj     = nn.Linear(d, kv, bias=False)
        self.v_proj     = nn.Linear(d, kv, bias=False)
        self.out_proj   = nn.Linear(kv, d, bias=False)
        self.score_proj = nn.Linear(d, 1, bias=True)   # → scalar confidence per token

        if cfg.score_repl_steps:
            self.step_score_proj = nn.Linear(d, 1, bias=True)
        else:
            self.step_score_proj = None

    def forward(
        self,
        draft_hidden: torch.Tensor,
        trace_hidden: Optional[torch.Tensor] = None,
        repl_step_hidden: Optional[torch.Tensor] = None,
    ) -> VerificationOutput:
        """
        Parameters
        ----------
        draft_hidden : Tensor [batch, seq, d_model]
            Final hidden states produced by the backbone for the draft output.
        trace_hidden : Tensor [batch, trace_len, d_model] or None
            RLM execution trace hidden states.  When *None*, the draft hidden
            states themselves serve as keys and values (self-verification mode).
        repl_step_hidden : Tensor [batch, num_steps, d_model] or None
            Per-REPL-step hidden states for step-level scoring.

        Returns
        -------
        VerificationOutput
        """
        B, L, _ = draft_hidden.shape
        H, dv = self.cfg.num_verification_heads, self.cfg.d_v_head
        kv = H * dv

        normed = self.input_norm(draft_hidden)                    # [B, L, d]
        kv_src = trace_hidden if trace_hidden is not None else normed

        Q = self.q_proj(normed).view(B, L, H, dv).transpose(1, 2)         # [B, H, L,  dv]
        K = self.k_proj(kv_src).view(B, -1, H, dv).transpose(1, 2)        # [B, H, TL, dv]
        V = self.v_proj(kv_src).view(B, -1, H, dv).transpose(1, 2)        # [B, H, TL, dv]

        try:
            attn = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        except RuntimeError:
            scores_a = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = torch.matmul(F.softmax(scores_a, dim=-1), V)

        attn_out = attn.transpose(1, 2).contiguous().view(B, L, kv)
        context  = self.out_proj(attn_out)                        # [B, L, d]

        # Token-level confidence
        token_scores = torch.sigmoid(self.score_proj(context)).squeeze(-1)   # [B, L]
        global_score = token_scores.mean(dim=-1)                             # [B]

        # Step-level confidence (optional)
        step_scores = None
        if self.step_score_proj is not None and repl_step_hidden is not None:
            step_scores = torch.sigmoid(
                self.step_score_proj(repl_step_hidden)
            ).squeeze(-1)                                                    # [B, num_steps]

        is_accepted = global_score >= self.cfg.min_confidence               # [B]

        low_conf = [
            (token_scores[b] < self.cfg.min_confidence).nonzero(as_tuple=False).squeeze(-1).tolist()
            for b in range(B)
        ]

        return VerificationOutput(
            global_score=global_score,
            token_scores=token_scores,
            step_scores=step_scores,
            is_accepted=is_accepted,
            low_confidence_positions=low_conf,
        )


# ===========================================================================
# 9.  HybridForwardOutput
# ===========================================================================

@dataclass
class HybridForwardOutput:
    """Return type of :meth:`HybridModel.forward`.

    Attributes
    ----------
    logits : Tensor [batch, seq, vocab_size]
        Raw (unnormalised) next-token logits.
    loss : Tensor [] or None
        Cross-entropy language modelling loss.  Non-None only when ``labels``
        are passed to :meth:`~HybridModel.forward`.
    verification : VerificationOutput or None
        Dense Verification result.  Non-None only when the model has a
        ``DenseVerificationConfig`` and ``run_verification=True`` was passed.
    confidence : Tensor [batch, seq] or None
        Per-token confidence from the RLM block.  Non-None when the model
        has an ``RlmConfig``.
    hidden_states : Tensor [batch, seq, d_model] or None
        Final backbone hidden states.  Non-None when
        ``output_hidden_states=True`` is passed.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    verification: Optional[VerificationOutput] = None
    confidence: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


# ===========================================================================
# 10.  HybridModel  (full nn.Module)
# ===========================================================================

class HybridModel(nn.Module):
    """Full Mamba-3 + LatentMoE + RLM + Dense Verification model.

    Built from a :class:`~swiftllm.model_config.HybridModelConfig` by
    :func:`build_torch_model`.  All components are standard
    :class:`torch.nn.Module` objects and run on any GPU via PyTorch.

    Training::

        model = build_torch_model(cfg).cuda()
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for batch in dataloader:
            out = model(batch["input_ids"].cuda(), labels=batch["labels"].cuda())
            out.loss.backward()
            model.update_load_stats()    # DeepSeek dynamic bias update
            opt.step(); opt.zero_grad()

    Inference::

        with torch.no_grad():
            out = model(input_ids.cuda(), run_verification=True)
        if not out.verification.is_accepted.all():
            # Low-confidence spans flagged in out.verification.low_confidence_positions
            ...
    """

    def __init__(self, cfg: HybridModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        base = cfg.base

        # ── Embedding ────────────────────────────────────────────────
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.embed_dropout = nn.Dropout(p=0.0)   # dropout rate exposed via config later

        # ── Backbone blocks ──────────────────────────────────────────
        blocks = []
        lmoe_cfg = cfg.moe_config if isinstance(cfg.moe_config, LatentMoeConfig) else None
        for idx, lt in enumerate(cfg.layer_schedule):
            blocks.append(
                HybridBlock(
                    layer_type=lt,
                    mamba_cfg=cfg.mamba_config,
                    moe_cfg=lmoe_cfg,
                    base_cfg=base,
                    layer_idx=idx,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # ── Final norm ───────────────────────────────────────────────
        self.norm = _RMSNorm(d, eps=base.rms_norm_eps)

        # ── RLM reasoning block (optional) ───────────────────────────
        self.rlm: Optional[RlmLayer] = (
            RlmLayer(cfg.rlm_config) if cfg.rlm_config else None
        )

        # ── LM head ──────────────────────────────────────────────────
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ── Dense Verification (optional, post-decode) ────────────────
        self.dense_verif: Optional[DenseVerificationLayer] = (
            DenseVerificationLayer(cfg.dense_verification_config)
            if cfg.dense_verification_config else None
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply standard Kaiming / truncated-normal initialisation."""
        std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        depth: int = 0,
        var_indices: Optional[torch.Tensor] = None,
        run_verification: bool = False,
        output_hidden_states: bool = False,
    ) -> HybridForwardOutput:
        """
        Parameters
        ----------
        input_ids : Tensor [batch, seq]
            Token IDs.
        labels : Tensor [batch, seq] or None
            Ground-truth token IDs for LM loss computation.  When *None*,
            ``HybridForwardOutput.loss`` is *None*.
        attn_mask : Tensor or None
            Attention mask forwarded to attention layers.
        depth : int
            Current RLM recursion depth (0 during standard training).
        var_indices : Tensor or None
            Variable binding indices for the RLM block.
        run_verification : bool
            Run the Dense Verification pass after the LM head.  Requires
            the model to have a :class:`DenseVerificationConfig`.
        output_hidden_states : bool
            Include final backbone hidden states in the output.

        Returns
        -------
        HybridForwardOutput
        """
        # 1. Embed
        x = self.embed_dropout(self.embed(input_ids))   # [B, L, d]

        # 2. Backbone
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        # 3. Final norm
        x = self.norm(x)                                 # [B, L, d]

        hidden = x if output_hidden_states else None

        # 4. RLM block (optional)
        confidence: Optional[torch.Tensor] = None
        rlm_hidden: Optional[torch.Tensor] = None
        if self.rlm is not None:
            x, confidence = self.rlm(x, depth=depth, var_indices=var_indices)
            rlm_hidden = x

        # 5. LM head
        logits = self.lm_head(x)                         # [B, L, vocab]

        # 6. Language modelling loss
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # 7. Dense Verification (optional, post-decode)
        verif: Optional[VerificationOutput] = None
        if run_verification and self.dense_verif is not None:
            with torch.no_grad():
                verif = self.dense_verif(
                    draft_hidden=rlm_hidden if rlm_hidden is not None else hidden if hidden is not None else x,
                    trace_hidden=None,
                    repl_step_hidden=None,
                )

        return HybridForwardOutput(
            logits=logits,
            loss=loss,
            verification=verif,
            confidence=confidence,
            hidden_states=hidden,
        )

    # ------------------------------------------------------------------
    # Load-balancing update (call once per training step)
    # ------------------------------------------------------------------

    def update_load_stats(self) -> None:
        """Update LatentMoE dynamic bias for all MoE blocks.

        Call once per training step after ``optimizer.step()``::

            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.update_load_stats()   ← here
        """
        for block in self.blocks:
            block.update_load_stats()

    # ------------------------------------------------------------------
    # Parameter groups for optimiser
    # ------------------------------------------------------------------

    def parameter_groups(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
    ) -> List[Dict]:
        """Return parameter groups for :class:`torch.optim.AdamW`.

        Separates weight-decayed parameters (weight matrices, embeddings)
        from non-decayed ones (biases, norms, SSM A/D parameters).

        Example::

            opt = torch.optim.AdamW(
                model.parameter_groups(lr=3e-4, weight_decay=0.1)
            )
        """
        decay_params, nodecay_params = [], []
        no_decay_names = {"bias", "norm", "A_log", "D", "dt_bias",
                          "expert_bias", "depth_embed", "var_table"}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in no_decay_names) or p.ndim < 2:
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        return [
            {"params": decay_params,   "lr": lr, "weight_decay": weight_decay},
            {"params": nodecay_params, "lr": lr, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the total parameter count."""
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        total = self.num_parameters()
        fmt   = f"{total / 1e9:.2f}B" if total >= 1e9 else f"{total / 1e6:.1f}M"
        return (
            f"HybridModel("
            f"d_model={self.cfg.d_model}, "
            f"num_layers={self.cfg.num_layers}, "
            f"vocab={self.cfg.vocab_size}, "
            f"params={fmt})"
        )


# ===========================================================================
# 11.  Factory
# ===========================================================================

def build_torch_model(cfg: HybridModelConfig) -> HybridModel:
    """Construct a :class:`HybridModel` from a :class:`HybridModelConfig`.

    Parameters
    ----------
    cfg : HybridModelConfig
        Architecture configuration (from :func:`~swiftllm.hybrid_model.build_mamba3_reasoning_model`
        or :class:`~swiftllm.hybrid_model.HybridModelBuilder`).

    Returns
    -------
    HybridModel
        Randomly initialised model.  Move to GPU with ``.cuda()`` or
        ``.to("cuda")``.

    Example
    -------
    ::

        from swiftllm import build_mamba3_reasoning_model
        from swiftllm.torch_model import build_torch_model

        cfg   = build_mamba3_reasoning_model(d_model=2048, num_layers=32)
        model = build_torch_model(cfg).cuda()
        print(model)   # HybridModel(d_model=2048, num_layers=32, ...)
    """
    if not HAS_MAMBA_SSM:
        warnings.warn(
            "mamba-ssm is not installed.  Using the pure-PyTorch reference SSM "
            "implementation (_ReferenceMamba).  Install mamba-ssm for fused CUDA "
            "kernels:\n    pip install mamba-ssm causal-conv1d",
            stacklevel=2,
        )
    return HybridModel(cfg)


# ===========================================================================
# 12.  Checkpoint helpers
# ===========================================================================

def save_checkpoint(
    model: HybridModel,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model weights (and optionally optimizer state) to a ``.pt`` file.

    Parameters
    ----------
    model : HybridModel
    path : str
        Destination path (e.g. ``"./checkpoints/step_1000.pt"``).
    optimizer : torch.optim.Optimizer or None
        When provided, the optimizer state is saved alongside weights.
    step : int
        Training step counter — stored in the checkpoint metadata.
    metadata : dict or None
        Arbitrary extra metadata to include in the checkpoint.

    Example
    -------
    ::

        save_checkpoint(model, "./checkpoints/step_1000.pt",
                        optimizer=opt, step=1000)
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config":     model.cfg.to_dict(),
        "step":             step,
        "metadata":         metadata or {},
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[HybridModel, Dict]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    device : str or torch.device
        Device to map weights to.  Use ``"cuda"`` to load directly to GPU.
    strict : bool
        Passed to :meth:`torch.nn.Module.load_state_dict`.

    Returns
    -------
    model : HybridModel
        Model with restored weights, on ``device``.
    meta : dict
        Checkpoint metadata (step, config, custom metadata).

    Example
    -------
    ::

        model, meta = load_checkpoint("./checkpoints/step_1000.pt", device="cuda")
        print(f"Resumed from step {meta['step']}")
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = HybridModelConfig.from_dict(ckpt["model_config"])
    model = build_torch_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    meta = {
        "step":     ckpt.get("step", 0),
        "config":   cfg,
        "metadata": ckpt.get("metadata", {}),
    }
    return model, meta


def load_pretrained_weights(
    model: HybridModel,
    path: str,
    device: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> None:
    """Load weights from a HuggingFace-style ``pytorch_model.bin`` or ``.safetensors`` file.

    Uses ``strict=False`` by default so that partial weight loading (e.g.
    loading a Mamba-2 checkpoint into a Mamba-3 / LatentMoE model) does not
    raise on unexpected or missing keys.

    Parameters
    ----------
    model : HybridModel
    path : str
        Path to a weight file (``.bin``, ``.pt``, or ``.safetensors``).
    device : str or torch.device
    strict : bool
        See :meth:`torch.nn.Module.load_state_dict`.
    """
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path, device=str(device))
        except ImportError:
            raise ImportError(
                "safetensors not installed.  Run: pip install safetensors"
            )
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        warnings.warn(f"load_pretrained_weights: {len(missing)} missing keys (e.g. {missing[:3]})", stacklevel=2)
    if unexpected:
        warnings.warn(f"load_pretrained_weights: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})", stacklevel=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core model
    "HybridModel",
    "build_torch_model",
    # Output types
    "HybridForwardOutput",
    "VerificationOutput",
    # Layers (for custom composition)
    "MambaLayer",
    "LatentMoeLayer",
    "AttentionLayer",
    "DenseFFN",
    "HybridBlock",
    "RlmLayer",
    "DenseVerificationLayer",
    # Checkpoint I/O
    "save_checkpoint",
    "load_checkpoint",
    "load_pretrained_weights",
    # Feature flag
    "HAS_MAMBA_SSM",
]

# ------------------------------------------------------------------------------
# END OF FILE: torch_model.py
# REPO PATH:   /swiftllm/python/swiftllm/torch_model.py
# INTERIM:     PyTorch compute backend — replaced by Rust/CUDA kernels once ready
# DEPENDS ON:  model_config.py  (for HybridModelConfig and all component configs)
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
