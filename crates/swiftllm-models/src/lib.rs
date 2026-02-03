//! SwiftLLM Models - Model implementations and loaders
//!
//! This crate provides implementations of popular LLM architectures
//! and model loading utilities for various formats.

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod architectures;
pub mod layers;
pub mod loaders;

use swiftllm_core::config::ModelArchitecture;
use swiftllm_core::error::{Error, Result};
use std::path::Path;

/// Model configuration parsed from model files
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,

    /// Hidden size
    pub hidden_size: usize,

    /// Intermediate size (MLP)
    pub intermediate_size: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA/MQA)
    pub num_key_value_heads: usize,

    /// Number of layers
    pub num_hidden_layers: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    pub rms_norm_eps: f32,

    /// Rope theta (for positional encoding)
    pub rope_theta: f32,

    /// Head dimension
    pub head_dim: usize,

    /// Whether to use bias in attention
    pub attention_bias: bool,

    /// Whether to use bias in MLP
    pub mlp_bias: bool,

    /// Sliding window size (if using sliding window attention)
    pub sliding_window: Option<usize>,

    /// Tie word embeddings
    pub tie_word_embeddings: bool,

    /// Beginning of sequence token ID
    pub bos_token_id: u32,

    /// End of sequence token ID
    pub eos_token_id: u32,

    /// Pad token ID
    pub pad_token_id: Option<u32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 128,
            attention_bias: false,
            mlp_bias: false,
            sliding_window: None,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        }
    }
}

impl ModelConfig {
    /// Get number of KV heads per attention head (for GQA ratio)
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Check if using grouped query attention
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// Check if using multi-query attention
    pub fn is_mqa(&self) -> bool {
        self.num_key_value_heads == 1
    }

    /// Get total KV cache size per token (in bytes, assuming float16)
    pub fn kv_cache_size_per_token(&self) -> usize {
        2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * 2
    }
}

/// Load model configuration from a path
pub fn load_config(path: impl AsRef<Path>) -> Result<ModelConfig> {
    loaders::huggingface::load_config(path)
}

/// Detect model architecture from configuration
pub fn detect_architecture(config_path: impl AsRef<Path>) -> Result<ModelArchitecture> {
    loaders::huggingface::detect_architecture(config_path)
}
