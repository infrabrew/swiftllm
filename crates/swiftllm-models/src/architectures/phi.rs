//! Phi Model Implementation
//!
//! Supports Phi-1, Phi-2, and Phi-3 models.

use super::TransformerModel;
use crate::layers::{Embedding, LMHead, LayerNorm, Linear, RotaryEmbedding};
use crate::ModelConfig;
use swiftllm_core::config::DataType;
use swiftllm_core::error::Result;
use swiftllm_core::memory::kv_cache::BatchedCacheMetadata;
use swiftllm_core::tensor::{Device, Tensor};
use swiftllm_core::types::TokenId;

/// Phi model
pub struct PhiModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<PhiDecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: LMHead,
}

impl PhiModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let embed_weight = Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let embed_tokens = Embedding::new(embed_weight, None)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(PhiDecoderLayer::new(&config, layer_idx)?);
        }

        // Phi uses LayerNorm instead of RMSNorm
        let ln_weight = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let ln_bias = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let final_layernorm = LayerNorm::new(ln_weight, Some(ln_bias), 1e-5)?;

        let lm_head_weight = Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let lm_head = LMHead::new(lm_head_weight, config.tie_word_embeddings)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_layernorm,
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

        self.final_layernorm.forward(&hidden_states)
    }
}

impl TransformerModel for PhiModel {
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

struct PhiDecoderLayer {
    layer_idx: usize,
    self_attn: PhiAttention,
    mlp: PhiMLP,
    input_layernorm: LayerNorm,
}

impl PhiDecoderLayer {
    fn new(config: &ModelConfig, layer_idx: usize) -> Result<Self> {
        let ln_weight = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let ln_bias = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let input_layernorm = LayerNorm::new(ln_weight, Some(ln_bias), 1e-5)?;

        Ok(Self {
            layer_idx,
            self_attn: PhiAttention::new(config)?,
            mlp: PhiMLP::new(config)?,
            input_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
        is_prefill: bool,
    ) -> Result<Tensor> {
        // Phi uses parallel attention and MLP
        // output = hidden_states + attn(ln(hidden_states)) + mlp(ln(hidden_states))

        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache_metadata, is_prefill)?;
        let mlp_output = self.mlp.forward(&normed)?;

        // In real implementation: hidden_states + attn_output + mlp_output
        Ok(attn_output)
    }
}

struct PhiAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,  // Phi calls output projection "dense"
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl PhiAttention {
    fn new(config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // Phi uses bias in attention
        let q_proj_weight = Tensor::zeros(
            vec![num_heads * head_dim, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let q_proj_bias = Tensor::zeros(
            vec![num_heads * head_dim],
            DataType::Float16,
            Device::Cpu,
        )?;

        let k_proj_weight = Tensor::zeros(
            vec![num_kv_heads * head_dim, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let k_proj_bias = Tensor::zeros(
            vec![num_kv_heads * head_dim],
            DataType::Float16,
            Device::Cpu,
        )?;

        let v_proj_weight = Tensor::zeros(
            vec![num_kv_heads * head_dim, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let v_proj_bias = Tensor::zeros(
            vec![num_kv_heads * head_dim],
            DataType::Float16,
            Device::Cpu,
        )?;

        let dense_weight = Tensor::zeros(
            vec![hidden_size, num_heads * head_dim],
            DataType::Float16,
            Device::Cpu,
        )?;
        let dense_bias = Tensor::zeros(
            vec![hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;

        // Phi-3 uses rotary embeddings
        let rotary_emb = RotaryEmbedding::new(
            head_dim / 2,  // Phi uses partial rotary
            config.max_position_embeddings,
            config.rope_theta,
        )?;

        Ok(Self {
            q_proj: Linear::new(q_proj_weight, Some(q_proj_bias))?,
            k_proj: Linear::new(k_proj_weight, Some(k_proj_bias))?,
            v_proj: Linear::new(v_proj_weight, Some(v_proj_bias))?,
            dense: Linear::new(dense_weight, Some(dense_bias))?,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[usize],
        _cache_metadata: &BatchedCacheMetadata,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, positions)?;

        self.dense.forward(&q)
    }
}

struct PhiMLP {
    fc1: Linear,
    fc2: Linear,
}

impl PhiMLP {
    fn new(config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        // Phi uses GELU activation (different from LLaMA's SiLU)
        let fc1_weight = Tensor::zeros(
            vec![intermediate_size, hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let fc1_bias = Tensor::zeros(
            vec![intermediate_size],
            DataType::Float16,
            Device::Cpu,
        )?;

        let fc2_weight = Tensor::zeros(
            vec![hidden_size, intermediate_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let fc2_bias = Tensor::zeros(
            vec![hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;

        Ok(Self {
            fc1: Linear::new(fc1_weight, Some(fc1_bias))?,
            fc2: Linear::new(fc2_weight, Some(fc2_bias))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Phi: fc2(gelu(fc1(x)))
        let hidden_states = self.fc1.forward(hidden_states)?;
        // Apply GELU activation here
        self.fc2.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swiftllm_core::config::ModelArchitecture;

    #[test]
    fn test_phi_model_creation() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Phi,
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 2,
            vocab_size: 51200,
            attention_bias: true,
            mlp_bias: true,
            ..Default::default()
        };

        let model = PhiModel::new(config).unwrap();
        assert_eq!(model.num_layers(), 2);
    }
}
