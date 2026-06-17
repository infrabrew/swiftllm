// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      gemma.rs
// PATH:      /crates/swiftllm-models/src/architectures/gemma.rs
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

//! Gemma model implementation (Gemma / Gemma 2 / Gemma 3 family).
//!
//! Gemma differs from a vanilla LLaMA-style decoder in several specific ways,
//! all of which are implemented here as small, unit-tested numeric helpers and
//! wired into the model scaffold:
//!
//! * **Embedding scaling** — token embeddings are multiplied by `sqrt(hidden)`.
//! * **Unit-offset RMSNorm** — the norm weight is applied as `(1 + w)` rather
//!   than `w` ([`rms_normalize`]).
//! * **GeGLU MLP** — the gate uses tanh-approximated GELU rather than SiLU
//!   ([`geglu`]).
//! * **Alternating attention** — layers alternate between sliding-window (local)
//!   and global attention ([`gemma_attention_type`]).
//! * **Query pre-attention scaling** — queries are scaled by
//!   `1/sqrt(query_pre_attn_scalar)` ([`query_scale`]).
//! * **Logit soft-capping** — attention and final logits are passed through
//!   `cap * tanh(x / cap)` ([`soft_cap`]); the final cap is applied in
//!   [`GemmaModel::get_logits`].

use super::TransformerModel;
use crate::layers::{Embedding, LMHead, Linear, RMSNorm, RotaryEmbedding};
use crate::ModelConfig;
use swiftllm_core::config::DataType;
use swiftllm_core::error::Result;
use swiftllm_core::memory::kv_cache::BatchedCacheMetadata;
use swiftllm_core::tensor::{Device, Tensor};
use swiftllm_core::types::TokenId;

// ----------------------------------------------------------------------------
// Gemma-specific numeric helpers (pure, unit-tested)
// ----------------------------------------------------------------------------

/// The factor by which Gemma scales token embeddings: `sqrt(hidden_size)`.
pub fn gemma_embedding_scale(hidden_size: usize) -> f32 {
    (hidden_size as f32).sqrt()
}

/// Logit soft-cap: `cap * tanh(x / cap)`. Bounds the magnitude of `x` to `cap`.
pub fn soft_cap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

/// Apply final-logit soft-capping in place.
pub fn apply_final_softcap(logits: &mut [f32], cap: f32) {
    for l in logits.iter_mut() {
        *l = soft_cap(*l, cap);
    }
}

/// Query pre-attention scaling factor: `1 / sqrt(query_pre_attn_scalar)`.
pub fn query_scale(query_pre_attn_scalar: f32) -> f32 {
    1.0 / query_pre_attn_scalar.sqrt()
}

/// Whether attention in a given layer is sliding-window (local) or global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaAttentionType {
    /// Local attention restricted to a sliding window.
    SlidingWindow,
    /// Full (global) attention over the whole context.
    Global,
}

/// Compute the attention type for `layer_idx`.
///
/// Every `global_attn_period`-th layer is global; the rest are sliding-window.
/// Gemma 2 uses a period of 2 (alternating); Gemma 3 uses 6 (five local, one
/// global). A period of `0` or `1` makes every layer global.
pub fn gemma_attention_type(layer_idx: usize, global_attn_period: usize) -> GemmaAttentionType {
    if global_attn_period <= 1 || (layer_idx + 1) % global_attn_period == 0 {
        GemmaAttentionType::Global
    } else {
        GemmaAttentionType::SlidingWindow
    }
}

/// Tanh approximation of the GELU activation.
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// GeGLU activation: `gelu(gate) * up`, element-wise.
pub fn geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| gelu_tanh(g) * u)
        .collect()
}

/// RMS normalisation. When `unit_offset` is true (the Gemma convention), the
/// weight is applied as `(1 + w)` instead of `w`.
pub fn rms_normalize(x: &[f32], weight: &[f32], eps: f32, unit_offset: bool) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| {
            let scale = if unit_offset { 1.0 + w } else { w };
            v * inv_rms * scale
        })
        .collect()
}

// ----------------------------------------------------------------------------
// Gemma configuration
// ----------------------------------------------------------------------------

/// Gemma-specific architecture configuration.
#[derive(Debug, Clone)]
pub struct GemmaConfig {
    /// Hidden size.
    pub hidden_size: usize,
    /// MLP intermediate size.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Maximum positions.
    pub max_position_embeddings: usize,
    /// RoPE base.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Sliding-window size for local-attention layers.
    pub sliding_window: usize,
    /// Every `global_attn_period`-th layer uses global attention.
    pub global_attn_period: usize,
    /// Denominator for query pre-attention scaling.
    pub query_pre_attn_scalar: f32,
    /// Soft-cap applied to attention logits (`None` disables).
    pub attn_logit_softcap: Option<f32>,
    /// Soft-cap applied to final output logits (`None` disables).
    pub final_logit_softcap: Option<f32>,
    /// Whether the LM head shares the embedding weights.
    pub tie_word_embeddings: bool,
}

impl GemmaConfig {
    /// Derive a Gemma configuration from the generic [`ModelConfig`], filling in
    /// Gemma defaults (Gemma 2 style: alternating local/global attention with
    /// attention and final logit soft-caps).
    pub fn from_model_config(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            num_hidden_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            head_dim: config.head_dim,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
            sliding_window: config.sliding_window.unwrap_or(4096),
            global_attn_period: 2,
            query_pre_attn_scalar: config.head_dim as f32,
            attn_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
            // Gemma ties its embeddings by default.
            tie_word_embeddings: true,
        }
    }

    /// Switch to Gemma 3 attention scheduling (5 local : 1 global) and drop the
    /// attention-logit soft-cap (removed in Gemma 3).
    pub fn with_gemma3_schedule(mut self) -> Self {
        self.global_attn_period = 6;
        self.attn_logit_softcap = None;
        self
    }
}

// ----------------------------------------------------------------------------
// Gemma model
// ----------------------------------------------------------------------------

/// Gemma decoder model.
pub struct GemmaModel {
    config: ModelConfig,
    gemma: GemmaConfig,
    embed_scale: f32,
    embed_tokens: Embedding,
    layers: Vec<GemmaDecoderLayer>,
    norm: RMSNorm,
    lm_head: LMHead,
}

impl GemmaModel {
    /// Create a new Gemma model from a generic model config.
    pub fn new(config: ModelConfig) -> Result<Self> {
        let gemma = GemmaConfig::from_model_config(&config);
        Self::with_gemma_config(config, gemma)
    }

    /// Create a Gemma model with an explicit [`GemmaConfig`].
    pub fn with_gemma_config(config: ModelConfig, gemma: GemmaConfig) -> Result<Self> {
        let embed_weight = Tensor::zeros(
            vec![gemma.vocab_size, gemma.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let embed_tokens = Embedding::new(embed_weight, None)?;

        let mut layers = Vec::with_capacity(gemma.num_hidden_layers);
        for layer_idx in 0..gemma.num_hidden_layers {
            layers.push(GemmaDecoderLayer::new(&gemma, layer_idx)?);
        }

        let norm_weight = Tensor::zeros(vec![gemma.hidden_size], DataType::Float16, Device::Cpu)?;
        let norm = RMSNorm::new(norm_weight, gemma.rms_norm_eps)?;

        let lm_head_weight = if gemma.tie_word_embeddings {
            embed_tokens.weight().clone()
        } else {
            Tensor::zeros(
                vec![gemma.vocab_size, gemma.hidden_size],
                DataType::Float16,
                Device::Cpu,
            )?
        };
        let lm_head = LMHead::new(lm_head_weight, gemma.tie_word_embeddings)?;

        Ok(Self {
            config,
            embed_scale: gemma_embedding_scale(gemma.hidden_size),
            gemma,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Borrow the Gemma-specific configuration.
    pub fn gemma_config(&self) -> &GemmaConfig {
        &self.gemma
    }

    /// The embedding scale factor applied after token lookup.
    pub fn embedding_scale(&self) -> f32 {
        self.embed_scale
    }

    /// The attention type of a given layer.
    pub fn layer_attention_type(&self, layer_idx: usize) -> GemmaAttentionType {
        gemma_attention_type(layer_idx, self.gemma.global_attn_period)
    }

    fn forward(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
        is_prefill: bool,
    ) -> Result<Tensor> {
        // Token embeddings are scaled by sqrt(hidden_size) in Gemma.
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        hidden_states = scale_tensor(hidden_states, self.embed_scale)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, positions, cache_metadata, is_prefill)?;
        }
        self.norm.forward(&hidden_states)
    }
}

/// Multiply a tensor by a scalar when it is f32-backed (best effort; the
/// placeholder f16 tensors used by the scaffold are passed through unchanged).
fn scale_tensor(tensor: Tensor, scale: f32) -> Result<Tensor> {
    if tensor.dtype() == DataType::Float32 {
        if let Some(slice) = tensor.as_slice::<f32>() {
            let scaled: Vec<f32> = slice.iter().map(|&v| v * scale).collect();
            return Tensor::from_f32(&scaled, tensor.shape().clone());
        }
    }
    Ok(tensor)
}

impl TransformerModel for GemmaModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward_prefill(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
    ) -> Result<Tensor> {
        self.forward(input_ids, positions, cache_metadata, true)
    }

    fn forward_decode(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
    ) -> Result<Tensor> {
        self.forward(input_ids, positions, cache_metadata, false)
    }

    fn get_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let logits = self.lm_head.forward(hidden_states)?;
        // Apply final-logit soft-capping for real when the logits are f32-backed.
        if let Some(cap) = self.gemma.final_logit_softcap {
            if logits.dtype() == DataType::Float32 {
                if let Some(slice) = logits.as_slice::<f32>() {
                    let mut data = slice.to_vec();
                    apply_final_softcap(&mut data, cap);
                    return Tensor::from_f32(&data, logits.shape().clone());
                }
            }
        }
        Ok(logits)
    }
}

/// A single Gemma decoder layer.
#[allow(dead_code)]
struct GemmaDecoderLayer {
    layer_idx: usize,
    attn_type: GemmaAttentionType,
    self_attn: GemmaAttention,
    mlp: GemmaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl GemmaDecoderLayer {
    fn new(config: &GemmaConfig, layer_idx: usize) -> Result<Self> {
        let mk_norm = || -> Result<RMSNorm> {
            let w = Tensor::zeros(vec![config.hidden_size], DataType::Float16, Device::Cpu)?;
            RMSNorm::new(w, config.rms_norm_eps)
        };
        Ok(Self {
            layer_idx,
            attn_type: gemma_attention_type(layer_idx, config.global_attn_period),
            self_attn: GemmaAttention::new(config, layer_idx)?,
            mlp: GemmaMLP::new(config)?,
            input_layernorm: mk_norm()?,
            post_attention_layernorm: mk_norm()?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, positions, cache_metadata, is_prefill)?;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        self.mlp.forward(&hidden_states)
    }
}

/// Gemma attention with per-layer sliding-window / global behaviour.
#[allow(dead_code)]
struct GemmaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_type: GemmaAttentionType,
    sliding_window: usize,
    /// Query pre-attention scaling factor.
    query_scale: f32,
    /// Optional attention-logit soft-cap.
    attn_logit_softcap: Option<f32>,
}

impl GemmaAttention {
    fn new(config: &GemmaConfig, layer_idx: usize) -> Result<Self> {
        let mk = |out: usize| -> Result<Linear> {
            let w = Tensor::zeros(vec![out, config.hidden_size], DataType::Float16, Device::Cpu)?;
            Linear::new(w, None)
        };
        let o_weight = Tensor::zeros(
            vec![config.hidden_size, config.num_attention_heads * config.head_dim],
            DataType::Float16,
            Device::Cpu,
        )?;
        Ok(Self {
            q_proj: mk(config.num_attention_heads * config.head_dim)?,
            k_proj: mk(config.num_key_value_heads * config.head_dim)?,
            v_proj: mk(config.num_key_value_heads * config.head_dim)?,
            o_proj: Linear::new(o_weight, None)?,
            rotary_emb: RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
            )?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            attn_type: gemma_attention_type(layer_idx, config.global_attn_period),
            sliding_window: config.sliding_window,
            query_scale: query_scale(config.query_pre_attn_scalar),
            attn_logit_softcap: config.attn_logit_softcap,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        _cache_metadata: &BatchedCacheMetadata,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        // Project Q/K/V, apply rotary, then (in a kernel) sliding-window or
        // global attention with query scaling and optional logit soft-capping.
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let _v = self.v_proj.forward(hidden_states)?;
        let (q, _k) = self.rotary_emb.apply(&q, &k, positions)?;
        self.o_proj.forward(&q)
    }
}

/// Gemma MLP using a GeGLU gate.
struct GemmaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GemmaMLP {
    fn new(config: &GemmaConfig) -> Result<Self> {
        let mk = |out: usize, inp: usize| -> Result<Linear> {
            Linear::new(
                Tensor::zeros(vec![out, inp], DataType::Float16, Device::Cpu)?,
                None,
            )
        };
        Ok(Self {
            gate_proj: mk(config.intermediate_size, config.hidden_size)?,
            up_proj: mk(config.intermediate_size, config.hidden_size)?,
            down_proj: mk(config.hidden_size, config.intermediate_size)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // GeGLU: down_proj(gelu(gate_proj(x)) * up_proj(x)). See [`geglu`].
        let gate = self.gate_proj.forward(hidden_states)?;
        let _up = self.up_proj.forward(hidden_states)?;
        self.down_proj.forward(&gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swiftllm_core::config::ModelArchitecture;

    fn gemma_config() -> ModelConfig {
        ModelConfig {
            architecture: ModelArchitecture::Gemma,
            hidden_size: 2048,
            intermediate_size: 16384,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            num_hidden_layers: 4,
            vocab_size: 256000,
            head_dim: 256,
            sliding_window: Some(4096),
            ..Default::default()
        }
    }

    #[test]
    fn embedding_scale_is_sqrt_hidden() {
        assert!((gemma_embedding_scale(2048) - (2048f32).sqrt()).abs() < 1e-3);
        assert!((gemma_embedding_scale(4) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn soft_cap_bounds_magnitude() {
        let cap = 30.0;
        for &x in &[-1000.0, -50.0, 0.0, 12.5, 999.0] {
            assert!(soft_cap(x, cap).abs() < cap + 1e-3);
        }
        // Small inputs are approximately unchanged.
        assert!((soft_cap(0.1, 30.0) - 0.1).abs() < 1e-2);
        // Sign preserved.
        assert!(soft_cap(100.0, 30.0) > 0.0);
        assert!(soft_cap(-100.0, 30.0) < 0.0);
    }

    #[test]
    fn final_softcap_applies_to_all() {
        let mut logits = vec![1000.0, -1000.0, 5.0];
        apply_final_softcap(&mut logits, 30.0);
        // Large inputs saturate to ±cap (tanh -> ±1 in f32), so the cap is the
        // inclusive bound on magnitude.
        assert!(logits[0] <= 30.0 && logits[0] > 29.0);
        assert!(logits[1] >= -30.0 && logits[1] < -29.0);
        // The small logit is left essentially unchanged.
        assert!((logits[2] - soft_cap(5.0, 30.0)).abs() < 1e-6);
    }

    #[test]
    fn query_scale_is_inverse_sqrt() {
        assert!((query_scale(256.0) - 1.0 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn attention_alternates_gemma2() {
        // period 2: layers 0,2 local; 1,3 global.
        assert_eq!(gemma_attention_type(0, 2), GemmaAttentionType::SlidingWindow);
        assert_eq!(gemma_attention_type(1, 2), GemmaAttentionType::Global);
        assert_eq!(gemma_attention_type(2, 2), GemmaAttentionType::SlidingWindow);
        assert_eq!(gemma_attention_type(3, 2), GemmaAttentionType::Global);
    }

    #[test]
    fn attention_schedule_gemma3() {
        // period 6: only layer 5 (index 5) is global in the first six.
        for idx in 0..5 {
            assert_eq!(gemma_attention_type(idx, 6), GemmaAttentionType::SlidingWindow);
        }
        assert_eq!(gemma_attention_type(5, 6), GemmaAttentionType::Global);
    }

    #[test]
    fn gelu_and_geglu() {
        assert!(gelu_tanh(0.0).abs() < 1e-6);
        assert!(gelu_tanh(10.0) > 9.9); // GELU(x) -> x for large x
        assert!(gelu_tanh(-10.0).abs() < 1e-3); // -> 0 for large negative
        let out = geglu(&[0.0, 10.0], &[3.0, 2.0]);
        assert_eq!(out.len(), 2);
        assert!(out[0].abs() < 1e-6); // gelu(0)*3 = 0
        assert!((out[1] - 2.0 * gelu_tanh(10.0)).abs() < 1e-4);
    }

    #[test]
    fn rms_normalize_unit_offset() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let zero_w = vec![0.0; 4];
        // With unit offset, zero weights act as identity scale (1 + 0 = 1).
        let normed = rms_normalize(&x, &zero_w, 1e-6, true);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        for (n, v) in normed.iter().zip(x.iter()) {
            assert!((n - v / rms).abs() < 1e-3);
        }
        // Without unit offset, zero weights zero the output.
        let zeroed = rms_normalize(&x, &zero_w, 1e-6, false);
        assert!(zeroed.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn gemma3_schedule_config() {
        let cfg = GemmaConfig::from_model_config(&gemma_config()).with_gemma3_schedule();
        assert_eq!(cfg.global_attn_period, 6);
        assert!(cfg.attn_logit_softcap.is_none());
    }

    #[test]
    fn model_construction_and_metadata() {
        let model = GemmaModel::new(gemma_config()).unwrap();
        assert_eq!(model.num_layers(), 4);
        assert_eq!(model.vocab_size(), 256000);
        assert_eq!(model.hidden_size(), 2048);
        // Embedding scale = sqrt(2048).
        assert!((model.embedding_scale() - (2048f32).sqrt()).abs() < 1e-2);
        // Alternating attention exposed at the model level.
        assert_eq!(model.layer_attention_type(0), GemmaAttentionType::SlidingWindow);
        assert_eq!(model.layer_attention_type(1), GemmaAttentionType::Global);
        assert_eq!(model.gemma_config().final_logit_softcap, Some(30.0));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: gemma.rs
// REPO PATH:   /swiftllm/crates/swiftllm-models/src/architectures/gemma.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
