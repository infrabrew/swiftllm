//! Attention Layer Implementation
//!
//! This module implements multi-head attention with support for:
//! - Grouped Query Attention (GQA)
//! - Multi-Query Attention (MQA)
//! - Rotary Positional Embeddings (RoPE)
//! - PagedAttention integration

use swiftllm_core::config::DataType;
use swiftllm_core::error::{Error, Result};
use swiftllm_core::memory::paged_attention::PagedAttentionInput;
use swiftllm_core::tensor::{Device, Tensor};
use std::f32::consts::PI;

/// Attention layer configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Hidden size
    pub hidden_size: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Use bias in projections
    pub use_bias: bool,

    /// Rope theta
    pub rope_theta: f32,

    /// Maximum position embeddings
    pub max_position_embeddings: usize,

    /// Sliding window size (optional)
    pub sliding_window: Option<usize>,

    /// Attention scale factor
    pub scale: f32,
}

impl AttentionConfig {
    /// Create a new attention config
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            use_bias: false,
            rope_theta: 10000.0,
            max_position_embeddings: 4096,
            sliding_window: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Check if using grouped query attention
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }
}

/// Multi-head attention layer
#[derive(Debug)]
pub struct Attention {
    /// Configuration
    config: AttentionConfig,

    /// Query projection
    q_proj: Tensor,

    /// Key projection
    k_proj: Tensor,

    /// Value projection
    v_proj: Tensor,

    /// Output projection
    o_proj: Tensor,

    /// Query bias (optional)
    q_bias: Option<Tensor>,

    /// Key bias (optional)
    k_bias: Option<Tensor>,

    /// Value bias (optional)
    v_bias: Option<Tensor>,

    /// Output bias (optional)
    o_bias: Option<Tensor>,

    /// Rotary embeddings
    rotary_emb: RotaryEmbedding,
}

impl Attention {
    /// Create a new attention layer
    pub fn new(
        config: AttentionConfig,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
    ) -> Result<Self> {
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            rotary_emb,
        })
    }

    /// Set biases
    pub fn with_biases(
        mut self,
        q_bias: Tensor,
        k_bias: Tensor,
        v_bias: Tensor,
        o_bias: Tensor,
    ) -> Self {
        self.q_bias = Some(q_bias);
        self.k_bias = Some(k_bias);
        self.v_bias = Some(v_bias);
        self.o_bias = Some(o_bias);
        self
    }

    /// Forward pass for prefill
    pub fn forward_prefill(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        attn_input: &PagedAttentionInput,
    ) -> Result<Tensor> {
        let dims = hidden_states.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        // Project Q, K, V
        // Q: [batch, seq, num_heads * head_dim]
        // K, V: [batch, seq, num_kv_heads * head_dim]

        // Apply rotary embeddings to Q and K

        // Compute attention with causal mask

        // Apply output projection

        // Placeholder output
        Tensor::zeros(
            vec![batch_size, seq_len, self.config.hidden_size],
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }

    /// Forward pass for decode (single token)
    pub fn forward_decode(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        attn_input: &PagedAttentionInput,
    ) -> Result<Tensor> {
        let dims = hidden_states.dims();
        let batch_size = dims[0];

        // Similar to prefill but with PagedAttention for KV cache access

        // Placeholder output
        Tensor::zeros(
            vec![batch_size, 1, self.config.hidden_size],
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }
}

/// Rotary Positional Embeddings
#[derive(Debug)]
pub struct RotaryEmbedding {
    /// Head dimension
    head_dim: usize,

    /// Maximum sequence length
    max_seq_len: usize,

    /// Base theta
    base: f32,

    /// Precomputed cos values
    cos_cache: Vec<f32>,

    /// Precomputed sin values
    sin_cache: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create new rotary embeddings
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0 / base.powf(2.0 * i as f32 / head_dim as f32));
        }

        // Precompute cos and sin for all positions
        let mut cos_cache = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_cache = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_cache.push(angle.cos());
                sin_cache.push(angle.sin());
            }
        }

        Ok(Self {
            head_dim,
            max_seq_len,
            base,
            cos_cache,
            sin_cache,
        })
    }

    /// Apply rotary embeddings to query and key tensors
    pub fn apply(
        &self,
        query: &Tensor,
        key: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // For each position, apply rotation:
        // q_rot = q * cos + rotate_half(q) * sin
        // k_rot = k * cos + rotate_half(k) * sin

        // Placeholder: return clones
        Ok((query.clone(), key.clone()))
    }

    /// Get cos and sin values for given positions
    pub fn get_cos_sin(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let half_dim = self.head_dim / 2;
        let mut cos = Vec::with_capacity(positions.len() * half_dim);
        let mut sin = Vec::with_capacity(positions.len() * half_dim);

        for &pos in positions {
            let pos = pos.min(self.max_seq_len - 1);
            let offset = pos * half_dim;
            cos.extend_from_slice(&self.cos_cache[offset..offset + half_dim]);
            sin.extend_from_slice(&self.sin_cache[offset..offset + half_dim]);
        }

        (cos, sin)
    }
}

/// Apply rotary embedding to a tensor
fn rotate_half(x: &[f32]) -> Vec<f32> {
    // Split into two halves and rotate
    let half = x.len() / 2;
    let mut rotated = Vec::with_capacity(x.len());

    // [-x2, x1] pattern
    for i in 0..half {
        rotated.push(-x[half + i]);
    }
    for i in 0..half {
        rotated.push(x[i]);
    }

    rotated
}

/// YaRN (Yet another RoPE extensioN) scaling
#[derive(Debug)]
pub struct YaRNScaling {
    /// Original max position embeddings
    original_max_position_embeddings: usize,

    /// Scaling factor
    scale: f32,

    /// Attention factor
    attention_factor: f32,

    /// Beta fast
    beta_fast: f32,

    /// Beta slow
    beta_slow: f32,
}

impl YaRNScaling {
    /// Create YaRN scaling
    pub fn new(
        original_max_position_embeddings: usize,
        max_position_embeddings: usize,
        attention_factor: f32,
    ) -> Self {
        let scale = max_position_embeddings as f32 / original_max_position_embeddings as f32;

        Self {
            original_max_position_embeddings,
            scale,
            attention_factor,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// Apply scaling to frequencies
    pub fn apply_scaling(&self, freqs: &mut [f32], dim: usize) {
        let low = (dim as f32 / 2.0 * (self.beta_fast.ln() / self.scale.ln())).floor() as usize;
        let high = (dim as f32 / 2.0 * (self.beta_slow.ln() / self.scale.ln())).ceil() as usize;

        for (i, freq) in freqs.iter_mut().enumerate() {
            if i < low {
                // High frequency - extrapolate
                *freq /= self.scale;
            } else if i > high {
                // Low frequency - no change
            } else {
                // Interpolate
                let t = (i - low) as f32 / (high - low) as f32;
                let smooth = (1.0 - (t * PI).cos()) / 2.0;
                *freq = *freq * (1.0 - smooth) + (*freq / self.scale) * smooth;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(4096, 32, 8);

        assert_eq!(config.head_dim, 128);
        assert!(config.is_gqa());
    }

    #[test]
    fn test_rotary_embedding() {
        let rope = RotaryEmbedding::new(128, 4096, 10000.0).unwrap();

        let (cos, sin) = rope.get_cos_sin(&[0, 1, 2]);

        assert_eq!(cos.len(), 3 * 64); // 3 positions * half_dim
        assert_eq!(sin.len(), 3 * 64);

        // Position 0 should have cos values close to 1
        assert!((cos[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rotate_half() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let rotated = rotate_half(&x);

        assert_eq!(rotated, vec![-3.0, -4.0, 1.0, 2.0]);
    }
}
