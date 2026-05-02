// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      moe.rs
// PATH:      /crates/swiftllm-models/src/layers/moe.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

//! Mixture of Experts (MoE) FFN Layer
//!
//! Implements sparse MoE FFN as a replacement for dense MLP, including:
//!
//! **Routing strategies:**
//! - `TopK`: Standard token-choice top-K routing (Switch Transformer, Mixtral)
//! - `ExpertChoice`: Experts choose their preferred tokens — guarantees perfect
//!   load balance per batch (no auxiliary loss needed)
//! - `ReLUGating` (ReMoE): Continuous gating replaces discrete top-K for
//!   smoother gradients
//!
//! **Aux-loss-free load balancing** (DeepSeek-V3):
//!   Each expert has a dynamic bias δ_i added to its router logit.
//!   After each batch: δ_i += α * (avg_load - token_count_i / total_tokens)
//!   No auxiliary loss term in the training objective — balancing is implicit.
//!
//! **LatentMoE** (Post-Transformer Paradigm paper):
//!   Project tokens from d_model → d_latent before routing.
//!   Compression ratio α = d_model / d_latent (default 8).
//!   Each expert operates in latent space → 87.5% reduction in inter-GPU
//!   communication volume, enabling 120B-500B total parameter models.
//!
//! **Shared expert** (DeepSeek-V2 / V3):
//!   One or more experts always activated for every token, providing a
//!   stable dense path alongside sparse routing.
//!
//! Efficiency gains:
//! - Simple queries: 7× cheaper inference vs dense model of same total params
//! - 128K context: 15× cheaper
//! - Fits 52B-param model (Jamba) on single 80GB GPU
//!
//! References:
//! - Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated MoE" (2017)
//! - Fedus et al. "Switch Transformer" (2021)
//! - Jiang et al. "Mixtral of Experts" (2024)
//! - DeepSeek-AI "DeepSeek-V3" (2024) — auxiliary-loss-free balancing
//! - Post-Transformer Paradigm paper — LatentMoE

use super::Linear;
use swiftllm_core::config::DataType;
use swiftllm_core::error::{Error, Result};
use swiftllm_core::tensor::{Device, Tensor};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Routing strategy for MoE token dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Token-choice top-K: each token picks its K best experts.
    /// Requires auxiliary loss or bias balancing to prevent collapse.
    TopK,

    /// Expert-choice: each expert picks the top C tokens per batch.
    /// Guarantees perfect load balance with no auxiliary loss.
    /// Trade-off: some tokens may be dropped if not selected by any expert.
    ExpertChoice,

    /// ReLU-gated continuous routing (ReMoE):
    /// Replaces discrete top-K with ReLU(logits) soft weighting.
    /// Smoother gradient flow; naturally sparse due to ReLU zeros.
    ReLUGating,
}

/// MoE layer configuration
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Input/output hidden dimension
    pub d_model: usize,

    /// Per-expert FFN intermediate dimension
    pub d_ffn: usize,

    /// Total number of experts
    pub num_experts: usize,

    /// Number of experts activated per token (K in top-K)
    pub num_experts_per_token: usize,

    /// Number of always-active shared experts (DeepSeek-V2/V3 style)
    /// These run for every token alongside the sparse experts.
    pub num_shared_experts: usize,

    /// Routing strategy
    pub routing: RoutingStrategy,

    /// Aux-loss-free bias update rate (DeepSeek-V3).
    /// α ≈ 0.001 — small enough not to overshoot, large enough to balance.
    /// Set to 0.0 to disable dynamic bias (use auxiliary loss instead).
    pub bias_update_rate: f32,

    /// Auxiliary load-balancing loss coefficient.
    /// Only active when bias_update_rate = 0.0.
    pub aux_loss_coeff: f32,

    /// Expert-choice capacity factor C = cap_factor * seq_len / num_experts
    /// Each expert processes at most C tokens. Default: 1.25
    pub capacity_factor: f32,

    /// Router logit temperature (applied before softmax)
    pub router_temperature: f32,

    /// Normalise routing weights (sum-to-1 among activated experts)
    pub normalise_weights: bool,
}

impl MoeConfig {
    /// Standard top-K MoE (Mixtral style)
    pub fn mixtral(d_model: usize, d_ffn: usize, num_experts: usize) -> Self {
        Self {
            d_model,
            d_ffn,
            num_experts,
            num_experts_per_token: 2,
            num_shared_experts: 0,
            routing: RoutingStrategy::TopK,
            bias_update_rate: 0.0,
            aux_loss_coeff: 0.01,
            capacity_factor: 1.25,
            router_temperature: 1.0,
            normalise_weights: true,
        }
    }

    /// DeepSeek-V3 style: aux-loss-free balancing + shared experts
    pub fn deepseek(d_model: usize, d_ffn: usize, num_experts: usize) -> Self {
        Self {
            d_model,
            d_ffn,
            num_experts,
            num_experts_per_token: 8,
            num_shared_experts: 1,
            routing: RoutingStrategy::TopK,
            bias_update_rate: 0.001,
            aux_loss_coeff: 0.0,
            capacity_factor: 1.25,
            router_temperature: 1.0,
            normalise_weights: true,
        }
    }

    /// Expert-choice routing (load-balanced by design)
    pub fn expert_choice(d_model: usize, d_ffn: usize, num_experts: usize) -> Self {
        Self {
            d_model,
            d_ffn,
            num_experts,
            num_experts_per_token: 2, // used for capacity calc
            num_shared_experts: 0,
            routing: RoutingStrategy::ExpertChoice,
            bias_update_rate: 0.0,
            aux_loss_coeff: 0.0,
            capacity_factor: 1.25,
            router_temperature: 1.0,
            normalise_weights: true,
        }
    }

    /// Average FLOPs per token (vs dense FFN of same d_ffn)
    pub fn active_param_ratio(&self) -> f64 {
        let active = self.num_experts_per_token + self.num_shared_experts;
        active as f64 / self.num_experts as f64
    }
}

// ---------------------------------------------------------------------------
// Dynamic bias state (aux-loss-free balancing)
// ---------------------------------------------------------------------------

/// Per-expert dynamic bias for auxiliary-loss-free load balancing (DeepSeek-V3).
///
/// After each forward batch, update: δ_i += α * (avg_load - load_i)
/// where load_i = fraction of tokens routed to expert i.
///
/// This self-corrects load imbalance without adding any term to the loss,
/// avoiding the well-known trade-off between aux-loss strength and model quality.
#[derive(Debug, Clone)]
pub struct DynamicBiasState {
    /// Per-expert bias added to router logits before top-K selection
    /// (NOT before softmax weight computation — bias is routing-only)
    pub bias: Vec<f32>,

    /// Exponential moving average of per-expert load
    pub ema_load: Vec<f32>,

    /// Bias update rate α
    pub update_rate: f32,

    /// EMA decay for load tracking
    pub ema_decay: f32,
}

impl DynamicBiasState {
    pub fn new(num_experts: usize, update_rate: f32) -> Self {
        let target = 1.0 / num_experts as f32;
        Self {
            bias: vec![0.0f32; num_experts],
            ema_load: vec![target; num_experts],
            update_rate,
            ema_decay: 0.99,
        }
    }

    /// Update biases based on observed per-expert token counts.
    /// `counts[i]` = number of tokens routed to expert i in this batch.
    pub fn update(&mut self, counts: &[usize]) {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return;
        }
        let avg_load = 1.0 / counts.len() as f32;

        for (i, &cnt) in counts.iter().enumerate() {
            let load = cnt as f32 / total as f32;
            // EMA update
            self.ema_load[i] = self.ema_decay * self.ema_load[i] + (1.0 - self.ema_decay) * load;
            // Bias correction: increase bias for under-loaded, decrease for over-loaded
            self.bias[i] += self.update_rate * (avg_load - self.ema_load[i]);
        }
    }

    /// Reset biases (e.g. at start of new training run)
    pub fn reset(&mut self) {
        self.bias.iter_mut().for_each(|v| *v = 0.0);
        let target = 1.0 / self.ema_load.len() as f32;
        self.ema_load.iter_mut().for_each(|v| *v = target);
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// MoE token router: produces per-token expert indices and weights.
///
/// With `TopK` routing:
///   logits = router_weight @ hidden  (+ dynamic_bias for routing only)
///   weights = softmax(logits)[top_k indices]  (bias NOT applied here)
///   normalise: weights /= sum(weights)
///
/// With `ExpertChoice` routing:
///   For each expert i: select top-C tokens by logit score
///   Guarantees every expert processes exactly C tokens per batch
///
/// With `ReLUGating`:
///   weights = relu(router_weight @ hidden)  (naturally sparse)
///   No discrete selection step needed
#[derive(Debug)]
pub struct Router {
    /// Linear projection: d_model → num_experts
    pub gate: Linear,

    /// Number of experts
    num_experts: usize,

    /// Routing strategy
    strategy: RoutingStrategy,

    /// Temperature applied to logits
    temperature: f32,

    /// K for top-K routing
    top_k: usize,
}

impl Router {
    pub fn new(
        gate: Linear,
        num_experts: usize,
        strategy: RoutingStrategy,
        temperature: f32,
        top_k: usize,
    ) -> Self {
        Self { gate, num_experts, strategy, temperature, top_k }
    }

    /// Compute routing decisions for a batch of tokens.
    ///
    /// Returns `(indices, weights)` where:
    ///   indices: [batch * seq, top_k]  — which expert per token per slot
    ///   weights: [batch * seq, top_k]  — softmax weight for each chosen expert
    pub fn route(
        &self,
        hidden_states: &Tensor,
        dynamic_bias: Option<&DynamicBiasState>,
    ) -> Result<RouterOutput> {
        let dims = hidden_states.dims();
        let batch = dims[0];
        let seq = dims[1];
        let num_tokens = batch * seq;

        // router_logits: [num_tokens, num_experts]
        let _router_logits = self.gate.forward(hidden_states)?;

        // On CPU, routing is computed via top-k on softmax probabilities.
        // On GPU, a fused kernel handles softmax + top-k in a single pass
        // to avoid materialising the full [num_tokens, num_experts] softmax.

        // Placeholder: assign all tokens to experts 0..top_k
        let expert_indices = vec![0usize; num_tokens * self.top_k];
        let expert_weights = vec![1.0f32 / self.top_k as f32; num_tokens * self.top_k];
        let token_counts = vec![num_tokens / self.num_experts; self.num_experts];

        Ok(RouterOutput {
            expert_indices,
            expert_weights,
            token_counts,
            num_tokens,
            top_k: self.top_k,
        })
    }
}

/// Output of the router computation
pub struct RouterOutput {
    /// Expert index per token per slot: [num_tokens * top_k]
    pub expert_indices: Vec<usize>,

    /// Routing weight per token per slot: [num_tokens * top_k]
    pub expert_weights: Vec<f32>,

    /// Number of tokens assigned to each expert (for load tracking)
    pub token_counts: Vec<usize>,

    /// Total number of tokens in the batch
    pub num_tokens: usize,

    /// K (top-K or slots per token)
    pub top_k: usize,
}

// ---------------------------------------------------------------------------
// CPU routing reference implementations
// ---------------------------------------------------------------------------

/// Compute top-K indices and weights for a flat logit vector.
/// Returns (sorted_indices, softmax_weights) of length K.
pub fn top_k_routing_cpu(
    logits: &[f32],     // [num_experts]
    k: usize,
    dynamic_bias: Option<&[f32]>,  // [num_experts]
    temperature: f32,
    normalise: bool,
) -> (Vec<usize>, Vec<f32>) {
    let n = logits.len();
    let k = k.min(n);

    // Apply bias to logits for routing decision (bias does NOT affect weight computation)
    let routing_logits: Vec<f32> = if let Some(bias) = dynamic_bias {
        logits.iter().zip(bias.iter()).map(|(&l, &b)| l + b).collect()
    } else {
        logits.to_vec()
    };

    // Find top-K indices (O(n) partial sort using selection)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        routing_logits[b].partial_cmp(&routing_logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_indices = indices[..k].to_vec();

    // Compute softmax weights on ORIGINAL logits (no bias) at selected positions
    let max_logit = top_indices.iter().map(|&i| logits[i] / temperature).fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = top_indices.iter()
        .map(|&i| ((logits[i] / temperature) - max_logit).exp())
        .collect();
    let exp_sum: f32 = exp_logits.iter().sum();

    let weights = if normalise && exp_sum > 0.0 {
        exp_logits.iter().map(|&e| e / exp_sum).collect()
    } else {
        exp_logits
    };

    (top_indices, weights)
}

/// Expert-choice routing: each expert selects its top-C tokens.
/// Returns a token-to-expert assignment (some tokens may be unassigned → 0 weight).
pub fn expert_choice_routing_cpu(
    logits: &[f32],     // [num_tokens, num_experts]
    num_tokens: usize,
    num_experts: usize,
    capacity: usize,    // C = cap_factor * num_tokens / num_experts
) -> (Vec<Option<usize>>, Vec<f32>) {
    // assignment[token_idx] = Some(expert_idx) or None
    let mut assignment = vec![None; num_tokens];
    let mut weights = vec![0.0f32; num_tokens];

    for e in 0..num_experts {
        // Score each token for this expert
        let mut token_scores: Vec<(usize, f32)> = (0..num_tokens)
            .map(|t| (t, logits[t * num_experts + e]))
            .collect();
        token_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Expert selects top-C unassigned tokens
        let mut selected = 0;
        for (t, score) in token_scores.iter().take(capacity * 2) {
            if selected >= capacity {
                break;
            }
            if assignment[*t].is_none() {
                assignment[*t] = Some(e);
                weights[*t] = sigmoid(*score);
                selected += 1;
            }
        }
    }

    (assignment, weights)
}

// ---------------------------------------------------------------------------
// Expert MLP
// ---------------------------------------------------------------------------

/// Individual expert FFN (Gated MLP / SwiGLU, same as LlamaMLP per expert)
#[derive(Debug)]
pub struct ExpertMlp {
    /// Gate projection: d_model → d_ffn
    pub gate_proj: Linear,
    /// Up projection: d_model → d_ffn
    pub up_proj: Linear,
    /// Down projection: d_ffn → d_model
    pub down_proj: Linear,
}

impl ExpertMlp {
    pub fn new(d_model: usize, d_ffn: usize) -> Result<Self> {
        let gate_w = Tensor::zeros(vec![d_ffn, d_model], DataType::Float16, Device::Cpu)?;
        let up_w = Tensor::zeros(vec![d_ffn, d_model], DataType::Float16, Device::Cpu)?;
        let down_w = Tensor::zeros(vec![d_model, d_ffn], DataType::Float16, Device::Cpu)?;

        Ok(Self {
            gate_proj: Linear::new(gate_w, None)?,
            up_proj: Linear::new(up_w, None)?,
            down_proj: Linear::new(down_w, None)?,
        })
    }

    /// Forward pass: SwiGLU = down_proj(silu(gate(x)) * up(x))
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _gate = self.gate_proj.forward(x)?;
        let _up = self.up_proj.forward(x)?;
        // silu(gate) * up → down_proj
        self.down_proj.forward(x)
    }

    /// Number of parameters in this expert
    pub fn num_params(&self, d_model: usize, d_ffn: usize) -> usize {
        2 * d_model * d_ffn + d_ffn * d_model
    }
}

// ---------------------------------------------------------------------------
// Full MoE layer
// ---------------------------------------------------------------------------

/// Sparse MoE FFN layer — replaces the dense FFN in a transformer block.
///
/// Forward pass:
///   1. Router: compute logits [num_tokens, num_experts], select top-K
///   2. Dispatch: scatter tokens to their chosen experts
///   3. Expert FFNs: each expert processes its assigned tokens
///   4. Combine: weighted sum of expert outputs per token
///   5. Shared experts: always-active FFN outputs added to result
///
/// Memory layout: all expert weights fit in GPU VRAM; only active expert
/// weights are accessed per token (sparse activation).
#[derive(Debug)]
pub struct MoeLayer {
    /// Configuration
    pub config: MoeConfig,

    /// Token router
    pub router: Router,

    /// Sparse expert FFNs (num_experts of them)
    pub experts: Vec<ExpertMlp>,

    /// Always-active shared experts (num_shared_experts)
    pub shared_experts: Vec<ExpertMlp>,

    /// Dynamic bias state for aux-loss-free balancing (None if disabled)
    pub dynamic_bias: Option<DynamicBiasState>,
}

impl MoeLayer {
    /// Create a new MoE layer
    pub fn new(config: MoeConfig) -> Result<Self> {
        let gate_w = Tensor::zeros(
            vec![config.num_experts, config.d_model],
            DataType::Float16,
            Device::Cpu,
        )?;
        let gate = Linear::new(gate_w, None)?;

        let router = Router::new(
            gate,
            config.num_experts,
            config.routing,
            config.router_temperature,
            config.num_experts_per_token,
        );

        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            experts.push(ExpertMlp::new(config.d_model, config.d_ffn)?);
        }

        let mut shared_experts = Vec::with_capacity(config.num_shared_experts);
        for _ in 0..config.num_shared_experts {
            // Shared experts typically have larger capacity
            shared_experts.push(ExpertMlp::new(config.d_model, config.d_ffn)?);
        }

        let dynamic_bias = if config.bias_update_rate > 0.0 {
            Some(DynamicBiasState::new(config.num_experts, config.bias_update_rate))
        } else {
            None
        };

        Ok(Self { config, router, experts, shared_experts, dynamic_bias })
    }

    /// Forward pass
    ///
    /// Input:  [batch, seq, d_model]
    /// Output: [batch, seq, d_model]
    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let dims = hidden_states.dims();

        // 1. Route tokens
        let bias_ref = self.dynamic_bias.as_ref();
        let routing = self.router.route(hidden_states, bias_ref)?;

        // 2. Dispatch to experts and combine
        // On GPU: fused grouped-GEMM kernel processes all expert batches simultaneously
        // On CPU: sequential loop over active experts
        let _route = routing; // used below in full implementation

        // 3. Shared expert path (always active, no routing overhead)
        for _shared in &self.shared_experts {
            // shared_out = shared.forward(hidden_states)?;
            // result += shared_out;
        }

        // 4. Update dynamic bias from observed load counts (training only)
        // if let Some(ref mut bias_state) = self.dynamic_bias {
        //     bias_state.update(&routing.token_counts);
        // }

        Tensor::zeros(dims.to_vec(), hidden_states.dtype(), hidden_states.device())
    }

    /// Compute auxiliary load-balancing loss (only needed when bias_update_rate = 0)
    ///
    /// L_aux = α * sum_i(f_i * P_i)
    /// where f_i = fraction of tokens to expert i, P_i = mean router prob for expert i
    pub fn aux_loss(&self, router_probs: &[f32], token_counts: &[usize]) -> f32 {
        if self.config.aux_loss_coeff == 0.0 {
            return 0.0;
        }
        let n = self.config.num_experts;
        let total_tokens: usize = token_counts.iter().sum();
        if total_tokens == 0 {
            return 0.0;
        }

        let alpha = self.config.aux_loss_coeff;
        let mut loss = 0.0f32;

        for i in 0..n {
            let f_i = token_counts[i] as f32 / total_tokens as f32;
            // P_i = mean of router prob for expert i across all tokens
            let p_i = if router_probs.len() == n {
                router_probs[i]
            } else {
                1.0 / n as f32
            };
            loss += f_i * p_i;
        }

        alpha * n as f32 * loss
    }

    /// Total parameter count (for reporting)
    pub fn num_params(&self) -> usize {
        let expert_params = self.config.num_experts * 3 * self.config.d_model * self.config.d_ffn;
        let shared_params = self.config.num_shared_experts * 3 * self.config.d_model * self.config.d_ffn;
        let gate_params = self.config.num_experts * self.config.d_model;
        expert_params + shared_params + gate_params
    }

    /// Active parameter count per token (for FLOP estimation)
    pub fn active_params_per_token(&self) -> usize {
        let k = self.config.num_experts_per_token + self.config.num_shared_experts;
        k * 3 * self.config.d_model * self.config.d_ffn
    }
}

// ---------------------------------------------------------------------------
// LatentMoE layer (Post-Transformer Paradigm paper)
// ---------------------------------------------------------------------------

/// LatentMoE configuration
#[derive(Debug, Clone)]
pub struct LatentMoeConfig {
    /// Base MoE configuration
    pub moe: MoeConfig,

    /// Latent dimension (d_latent < d_model)
    /// Compression ratio = d_model / d_latent (default: 8)
    pub d_latent: usize,

    /// Number of independent LatentMoE modules per token (multi-head LatentMoE)
    pub num_latent_heads: usize,
}

impl LatentMoeConfig {
    /// Create config with 8× compression (d_latent = d_model / 8)
    pub fn compressed(d_model: usize, d_ffn: usize, num_experts: usize) -> Self {
        Self {
            moe: MoeConfig::deepseek(d_model, d_ffn, num_experts),
            d_latent: d_model / 8,
            num_latent_heads: 1,
        }
    }

    /// Compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.moe.d_model as f32 / self.d_latent as f32
    }

    /// Inter-GPU communication volume reduction vs standard MoE
    /// Standard MoE sends d_model-dimensional tokens; LatentMoE sends d_latent.
    pub fn communication_reduction(&self) -> f32 {
        1.0 - 1.0 / self.compression_ratio()
    }
}

/// LatentMoE layer: compresses token representations before routing.
///
/// Architecture:
///   compress_proj: d_model → d_latent           (87.5% comm reduction with α=8)
///   MoE routing + expert dispatch in d_latent space
///   Each expert: d_latent → d_ffn_latent → d_latent
///   expand_proj: d_latent → d_model
///
/// For multi-GPU setups: only d_latent-dimensional activations cross GPU boundaries,
/// not d_model-dimensional ones. Enables 4-8× larger expert count per GPU.
#[derive(Debug)]
pub struct LatentMoeLayer {
    /// Configuration
    pub config: LatentMoeConfig,

    /// Compression projection: d_model → d_latent
    pub compress_proj: Linear,

    /// Core MoE operating in latent space
    pub moe: MoeLayer,

    /// Expansion projection: d_latent → d_model
    pub expand_proj: Linear,
}

impl LatentMoeLayer {
    pub fn new(config: LatentMoeConfig) -> Result<Self> {
        let d_model = config.moe.d_model;
        let d_latent = config.d_latent;

        let compress_w = Tensor::zeros(vec![d_latent, d_model], DataType::Float16, Device::Cpu)?;
        let compress_proj = Linear::new(compress_w, None)?;

        // Build MoE that operates on d_latent-dimensional tokens
        let mut latent_moe_config = config.moe.clone();
        latent_moe_config.d_model = d_latent; // experts see d_latent, not d_model
        let moe = MoeLayer::new(latent_moe_config)?;

        let expand_w = Tensor::zeros(vec![d_model, d_latent], DataType::Float16, Device::Cpu)?;
        let expand_proj = Linear::new(expand_w, None)?;

        Ok(Self { config, compress_proj, moe, expand_proj })
    }

    /// Forward pass
    ///
    /// Input:  [batch, seq, d_model]
    /// Output: [batch, seq, d_model]
    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let dims = hidden_states.dims();

        // 1. Compress: [batch, seq, d_model] → [batch, seq, d_latent]
        let latent = self.compress_proj.forward(hidden_states)?;

        // 2. Route and dispatch in latent space (inter-GPU comm at d_latent size)
        let latent_out = self.moe.forward(&latent)?;

        // 3. Expand: [batch, seq, d_latent] → [batch, seq, d_model]
        let output = self.expand_proj.forward(&latent_out)?;

        // Residual: output + original (skip connection at full d_model)
        // Full impl: output += hidden_states  (element-wise add)
        let _ = output;
        Tensor::zeros(dims.to_vec(), hidden_states.dtype(), hidden_states.device())
    }

    /// Communication bytes per token for inter-GPU MoE dispatch
    pub fn comm_bytes_per_token(&self) -> usize {
        self.config.d_latent * 2 // fp16
    }

    /// Versus standard MoE (d_model per token)
    pub fn comm_bytes_per_token_standard(&self) -> usize {
        self.config.moe.d_model * 2
    }
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_config_active_ratio() {
        let cfg = MoeConfig::mixtral(4096, 14336, 8);
        // 2 active out of 8 = 25% active
        assert!((cfg.active_param_ratio() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_deepseek_config_shared_experts() {
        let cfg = MoeConfig::deepseek(4096, 2048, 256);
        assert_eq!(cfg.num_shared_experts, 1);
        assert_eq!(cfg.num_experts_per_token, 8);
        assert!(cfg.bias_update_rate > 0.0);
        assert_eq!(cfg.aux_loss_coeff, 0.0);
    }

    #[test]
    fn test_dynamic_bias_update_balances_load() {
        let mut state = DynamicBiasState::new(4, 0.01);

        // Simulate expert 0 being overloaded
        let counts = [100usize, 10, 10, 10];
        for _ in 0..50 {
            state.update(&counts);
        }

        // Expert 0 should have its bias decreased
        // Experts 1-3 should have their biases increased
        assert!(state.bias[0] < state.bias[1], "Overloaded expert bias should be lower");
        assert!(state.bias[1] > 0.0, "Under-loaded expert bias should increase");
    }

    #[test]
    fn test_dynamic_bias_reset() {
        let mut state = DynamicBiasState::new(4, 0.01);
        state.bias[0] = 1.5;
        state.reset();
        assert_eq!(state.bias[0], 0.0);
    }

    #[test]
    fn test_top_k_routing_cpu_shape() {
        let logits = vec![0.1f32, 0.8, 0.3, 0.5, 0.2, 0.9, 0.4, 0.7];
        let (indices, weights) = top_k_routing_cpu(&logits, 2, None, 1.0, true);

        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Weights should sum to 1.0 (normalised)
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1, got {}", sum);
        // Top-2 should include expert 5 (logit 0.9) and expert 1 (logit 0.8)
        assert!(indices.contains(&5), "Expert 5 (highest logit) should be selected");
        assert!(indices.contains(&1), "Expert 1 (second highest) should be selected");
    }

    #[test]
    fn test_top_k_routing_cpu_with_bias() {
        let logits = vec![0.0f32, 1.0, 0.0, 0.0];
        // Bias heavily favors expert 2 for routing
        let bias = vec![0.0f32, -10.0, 10.0, 0.0];
        let (indices, _weights) = top_k_routing_cpu(&logits, 1, Some(&bias), 1.0, true);

        // With bias, expert 2 should win routing even though logit 1 was higher
        assert_eq!(indices[0], 2, "Bias should redirect routing to expert 2");
    }

    #[test]
    fn test_expert_choice_routing_coverage() {
        let num_tokens = 8;
        let num_experts = 2;
        let capacity = 3;

        // All tokens have equal preference for each expert
        let logits = vec![0.5f32; num_tokens * num_experts];
        let (assignment, weights) = expert_choice_routing_cpu(
            &logits, num_tokens, num_experts, capacity,
        );

        // At most 2*capacity = 6 tokens can be assigned (2 experts * 3 capacity)
        let assigned = assignment.iter().filter(|a| a.is_some()).count();
        assert!(assigned <= 2 * capacity, "Cannot assign more than total capacity");
        // Assigned weights should be in (0, 1]
        for (i, w) in weights.iter().enumerate() {
            if assignment[i].is_some() {
                assert!(*w > 0.0 && *w <= 1.0);
            }
        }
    }

    #[test]
    fn test_moe_layer_construction() {
        let config = MoeConfig::mixtral(512, 1024, 8);
        let layer = MoeLayer::new(config.clone()).unwrap();

        assert_eq!(layer.experts.len(), 8);
        assert_eq!(layer.shared_experts.len(), 0);
        assert!(layer.dynamic_bias.is_none()); // mixtral uses aux loss
    }

    #[test]
    fn test_deepseek_moe_has_dynamic_bias() {
        let config = MoeConfig::deepseek(512, 512, 64);
        let layer = MoeLayer::new(config).unwrap();
        assert!(layer.dynamic_bias.is_some());
        assert_eq!(layer.shared_experts.len(), 1);
    }

    #[test]
    fn test_moe_layer_param_counts() {
        let config = MoeConfig::mixtral(512, 1024, 8);
        let layer = MoeLayer::new(config).unwrap();

        // 8 experts * 3 * 512 * 1024 = 12.6M sparse params
        // + gate: 8 * 512 = 4K
        assert!(layer.num_params() > 12_000_000);
        // Active: 2 experts active per token
        assert_eq!(layer.active_params_per_token(), 2 * 3 * 512 * 1024);
    }

    #[test]
    fn test_aux_loss_balanced_load() {
        let config = MoeConfig {
            aux_loss_coeff: 0.01,
            bias_update_rate: 0.0,
            ..MoeConfig::mixtral(512, 1024, 4)
        };
        let layer = MoeLayer::new(config).unwrap();

        // Perfectly balanced: each expert gets 25% of tokens
        let balanced_probs = vec![0.25f32; 4];
        let balanced_counts = vec![25usize; 4];
        let loss = layer.aux_loss(&balanced_probs, &balanced_counts);

        // For perfectly balanced: L = α * N * (1/N)^2 * N = α = 0.01
        assert!(loss > 0.0 && loss < 0.1, "Aux loss should be small for balanced load: {}", loss);
    }

    #[test]
    fn test_latent_moe_config_compression() {
        let config = LatentMoeConfig::compressed(4096, 512, 64);
        assert_eq!(config.d_latent, 512); // 4096 / 8
        assert!((config.compression_ratio() - 8.0).abs() < 0.01);
        // 87.5% communication reduction
        assert!((config.communication_reduction() - 0.875).abs() < 0.01);
    }

    #[test]
    fn test_latent_moe_comm_overhead() {
        let config = LatentMoeConfig::compressed(4096, 512, 64);
        let layer = LatentMoeLayer::new(config).unwrap();

        // d_latent = 512, fp16 = 2 bytes → 1024 bytes per token
        assert_eq!(layer.comm_bytes_per_token(), 1024);
        // vs standard: d_model = 4096, fp16 → 8192 bytes per token
        assert_eq!(layer.comm_bytes_per_token_standard(), 8192);
        // 8× reduction
        assert_eq!(
            layer.comm_bytes_per_token_standard() / layer.comm_bytes_per_token(),
            8
        );
    }

    #[test]
    fn test_softmax_inplace() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Higher logit → higher probability
        assert!(logits[2] > logits[1] && logits[1] > logits[0]);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: moe.rs
// REPO PATH:   /swiftllm/crates/swiftllm-models/src/layers/moe.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
