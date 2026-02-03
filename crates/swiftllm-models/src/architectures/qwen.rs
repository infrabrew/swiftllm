//! Qwen Model Implementation
//!
//! Supports Qwen and Qwen2 models.

use super::TransformerModel;
use crate::layers::{Embedding, LMHead, Linear, RMSNorm, RotaryEmbedding};
use crate::ModelConfig;
use swiftllm_core::config::DataType;
use swiftllm_core::error::Result;
use swiftllm_core::memory::kv_cache::BatchedCacheMetadata;
use swiftllm_core::tensor::{Device, Tensor};
use swiftllm_core::types::TokenId;

/// Qwen model
pub struct QwenModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<QwenDecoderLayer>,
    norm: RMSNorm,
    lm_head: LMHead,
}

impl QwenModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let embed_weight = Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let embed_tokens = Embedding::new(embed_weight, None)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(QwenDecoderLayer::new(&config, layer_idx)?);
        }

        let norm_weight = Tensor::zeros(
            vec![config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps)?;

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

        self.norm.forward(&hidden_states)
    }
}

impl TransformerModel for QwenModel {
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

struct QwenDecoderLayer {
    layer_idx: usize,
    self_attn: QwenAttention,
    mlp: QwenMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl QwenDecoderLayer {
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

        Ok(Self {
            layer_idx,
            self_attn: QwenAttention::new(config)?,
            mlp: QwenMLP::new(config)?,
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
        self.mlp.forward(&hidden_states)
    }
}

struct QwenAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    use_bias: bool,
}

impl QwenAttention {
    fn new(config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let use_bias = config.attention_bias;

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

        // Qwen uses bias in attention
        let bias = if use_bias {
            Some(Tensor::zeros(
                vec![num_heads * head_dim],
                DataType::Float16,
                Device::Cpu,
            )?)
        } else {
            None
        };

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )?;

        Ok(Self {
            q_proj: Linear::new(q_proj_weight, bias.clone())?,
            k_proj: Linear::new(k_proj_weight, bias.clone())?,
            v_proj: Linear::new(v_proj_weight, bias)?,
            o_proj: Linear::new(o_proj_weight, None)?,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            use_bias,
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

        self.o_proj.forward(&q)
    }
}

struct QwenMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl QwenMLP {
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
        let _up = self.up_proj.forward(hidden_states)?;
        self.down_proj.forward(&gate)
    }
}
