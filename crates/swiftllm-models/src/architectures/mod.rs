//! Model Architectures
//!
//! This module provides implementations of popular LLM architectures.

pub mod llama;
pub mod mistral;
pub mod phi;
pub mod qwen;

pub use llama::LlamaModel;
pub use mistral::MistralModel;
pub use phi::PhiModel;
pub use qwen::QwenModel;

use crate::ModelConfig;
use swiftllm_core::config::ModelArchitecture;
use swiftllm_core::error::Result;
use swiftllm_core::memory::kv_cache::BatchedCacheMetadata;
use swiftllm_core::tensor::Tensor;
use swiftllm_core::types::TokenId;

/// Trait for transformer models
pub trait TransformerModel: Send + Sync {
    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Forward pass for prefill (processing full prompt)
    fn forward_prefill(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
    ) -> Result<Tensor>;

    /// Forward pass for decode (generating one token)
    fn forward_decode(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
    ) -> Result<Tensor>;

    /// Get logits for next token prediction
    fn get_logits(&self, hidden_states: &Tensor) -> Result<Tensor>;

    /// Get the vocabulary size
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    /// Get the hidden size
    fn hidden_size(&self) -> usize {
        self.config().hidden_size
    }

    /// Get the number of layers
    fn num_layers(&self) -> usize {
        self.config().num_hidden_layers
    }
}

/// Create a model based on architecture
pub fn create_model(
    architecture: ModelArchitecture,
    config: ModelConfig,
    weights_path: &std::path::Path,
) -> Result<Box<dyn TransformerModel>> {
    match architecture {
        ModelArchitecture::Llama => Ok(Box::new(LlamaModel::new(config)?)),
        ModelArchitecture::Mistral => Ok(Box::new(MistralModel::new(config)?)),
        ModelArchitecture::Qwen | ModelArchitecture::Qwen2 => Ok(Box::new(QwenModel::new(config)?)),
        ModelArchitecture::Phi | ModelArchitecture::Phi3 => Ok(Box::new(PhiModel::new(config)?)),
        _ => Err(swiftllm_core::error::Error::UnsupportedArchitecture(
            format!("{:?}", architecture),
        )),
    }
}

/// Transformer decoder block (common structure)
#[derive(Debug)]
pub struct DecoderLayer {
    /// Layer index
    pub layer_idx: usize,

    /// Self attention
    pub self_attn: crate::layers::Attention,

    /// MLP
    pub mlp: crate::layers::GatedMlp,

    /// Input layer norm
    pub input_layernorm: crate::layers::RMSNorm,

    /// Post-attention layer norm
    pub post_attention_layernorm: crate::layers::RMSNorm,
}
