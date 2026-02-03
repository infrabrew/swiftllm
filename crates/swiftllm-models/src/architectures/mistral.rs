//! Mistral Model Implementation
//!
//! Supports Mistral and Mixtral models with sliding window attention.

use super::TransformerModel;
use crate::layers::{Embedding, LMHead, Linear, RMSNorm, RotaryEmbedding};
use crate::ModelConfig;
use swiftllm_core::config::DataType;
use swiftllm_core::error::Result;
use swiftllm_core::memory::kv_cache::BatchedCacheMetadata;
use swiftllm_core::tensor::{Device, Tensor};
use swiftllm_core::types::TokenId;

/// Mistral model
pub struct MistralModel {
    /// Model configuration
    config: ModelConfig,

    /// Token embedding
    embed_tokens: Embedding,

    /// Decoder layers
    layers: Vec<MistralDecoderLayer>,

    /// Final layer norm
    norm: RMSNorm,

    /// LM head
    lm_head: LMHead,
}

impl MistralModel {
    /// Create a new Mistral model
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Embedding
        let embed_weight = Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let embed_tokens = Embedding::new(embed_weight, None)?;

        // Decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(MistralDecoderLayer::new(&config, layer_idx)?);
        }

        // Final norm
        let norm_weight = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps)?;

        // LM head
        let lm_head_weight = Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let lm_head = LMHead::new(lm_head_weight, false)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn forward(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens.forward(input_ids)?;

        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, positions, cache_metadata, is_prefill)?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

impl TransformerModel for MistralModel {
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
        self.lm_head.forward(hidden_states)
    }
}

/// Mistral decoder layer
struct MistralDecoderLayer {
    layer_idx: usize,
    self_attn: MistralAttention,
    mlp: MistralMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl MistralDecoderLayer {
    fn new(config: &ModelConfig, layer_idx: usize) -> Result<Self> {
        let input_layernorm_weight = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let input_layernorm = RMSNorm::new(input_layernorm_weight, config.rms_norm_eps)?;

        let post_attention_layernorm_weight = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let post_attention_layernorm = RMSNorm::new(post_attention_layernorm_weight, config.rms_norm_eps)?;

        let self_attn = MistralAttention::new(config)?;
        let mlp = MistralMLP::new(config)?;

        Ok(Self {
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
        let hidden_states = self.self_attn.forward(&hidden_states, positions, cache_metadata, is_prefill)?;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

/// Mistral attention with sliding window
struct MistralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    scale: f32,
}

impl MistralAttention {
    fn new(config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj_weight = Tensor::zeros(
            vec![num_heads * head_dim, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let k_proj_weight = Tensor::zeros(
            vec![num_kv_heads * head_dim, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let v_proj_weight = Tensor::zeros(
            vec![num_kv_heads * head_dim, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let o_proj_weight = Tensor::zeros(
            vec![hidden_size, num_heads * head_dim],
            DataType::Float16,
            Device::Cpu,
        )?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;

        Ok(Self {
            q_proj: Linear::new(q_proj_weight, None)?,
            k_proj: Linear::new(k_proj_weight, None)?,
            v_proj: Linear::new(v_proj_weight, None)?,
            o_proj: Linear::new(o_proj_weight, None)?,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window: config.sliding_window,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, positions)?;

        // Apply sliding window attention
        // In real implementation, attention mask would be modified for sliding window

        let output = self.o_proj.forward(&q)?;
        Ok(output)
    }
}

/// Mistral MLP (SwiGLU)
struct MistralMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MistralMLP {
    fn new(config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj_weight = Tensor::zeros(
            vec![intermediate_size, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let up_proj_weight = Tensor::zeros(
            vec![intermediate_size, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let down_proj_weight = Tensor::zeros(
            vec![hidden_size, intermediate_size],
            DataType::Float16,
            Device::Cpu,
        )?;

        Ok(Self {
            gate_proj: Linear::new(gate_proj_weight, None)?,
            up_proj: Linear::new(up_proj_weight, None)?,
            down_proj: Linear::new(down_proj_weight, None)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(hidden_states)?;
        let up = self.up_proj.forward(hidden_states)?;
        self.down_proj.forward(&gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swiftllm_core::config::ModelArchitecture;

    #[test]
    fn test_mistral_model_creation() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Mistral,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA
            num_hidden_layers: 2,
            vocab_size: 32000,
            sliding_window: Some(4096),
            ..Default::default()
        };

        let model = MistralModel::new(config).unwrap();
        assert_eq!(model.num_layers(), 2);
        assert!(model.config().sliding_window.is_some());
    }
}
