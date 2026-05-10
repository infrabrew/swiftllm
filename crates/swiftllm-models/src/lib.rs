// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      lib.rs
// PATH:      /crates/swiftllm-models/src/lib.rs
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

//! SwiftLLM Models - Model implementations and loaders
//!
//! This crate provides implementations of popular LLM architectures
//! and model loading utilities for various formats.

#![warn(clippy::all)]
// TODO: Re-enable missing_docs once scaffold code is fully documented
#![allow(missing_docs)]

pub mod architectures;
pub mod layers;
pub mod loaders;

use swiftllm_core::config::ModelArchitecture;
use swiftllm_core::error::Result;
use std::path::Path;

pub use architectures::jamba::HybridLayerType;

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

    // -----------------------------------------------------------------------
    // Hybrid / SSM fields (populated for Jamba, Mamba, Nemotron-H, etc.)
    // -----------------------------------------------------------------------

    /// Per-layer type schedule for hybrid models.
    /// Empty = pure transformer (all attention layers).
    /// Length must equal num_hidden_layers when non-empty.
    pub hybrid_schedule: Vec<HybridLayerType>,

    /// Mamba SSM state dimension (N). Default: 64 (Mamba-2), 128 (Mamba-3)
    pub mamba_d_state: usize,

    /// Mamba inner expansion factor. Default: 2 (d_inner = expand * hidden_size)
    pub mamba_expand: usize,

    /// Mamba depthwise-conv kernel width. Default: 4
    pub mamba_d_conv: usize,

    /// Enable Mamba-3 exponential-trapezoidal discretization
    pub mamba_trapezoidal_disc: bool,

    /// Enable Mamba-3 complex-valued SSM states
    pub mamba_complex_states: bool,

    /// Enable Mamba-3 MIMO multi-head formulation
    pub mamba_mimo: bool,

    /// MoE: total number of experts (0 = dense FFN, no MoE)
    pub moe_num_experts: usize,

    /// MoE: number of experts activated per token (top-K)
    pub moe_num_experts_per_token: usize,

    /// MoE: number of always-active shared experts (DeepSeek-V2/V3 style)
    pub moe_num_shared_experts: usize,

    /// MoE: use LatentMoE compression before routing
    /// Reduces inter-GPU communication by d_model/d_latent ratio
    pub moe_use_latent: bool,

    /// MoE: LatentMoE compression ratio (d_latent = hidden_size / ratio)
    pub moe_latent_ratio: usize,

    /// MoE: aux-loss-free bias update rate (0.0 = use auxiliary loss instead)
    pub moe_bias_update_rate: f32,
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
            // Hybrid / SSM defaults (disabled for pure transformer)
            hybrid_schedule: Vec::new(),
            mamba_d_state: 64,
            mamba_expand: 2,
            mamba_d_conv: 4,
            mamba_trapezoidal_disc: false,
            mamba_complex_states: false,
            mamba_mimo: false,
            moe_num_experts: 0,
            moe_num_experts_per_token: 2,
            moe_num_shared_experts: 0,
            moe_use_latent: false,
            moe_latent_ratio: 8,
            moe_bias_update_rate: 0.0,
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

    /// Get total KV cache size per token (in bytes, assuming float16).
    /// For hybrid models, only counts attention layers (not Mamba layers).
    pub fn kv_cache_size_per_token(&self) -> usize {
        let kv_layers = self.kv_cache_layers();
        2 * kv_layers * self.num_key_value_heads * self.head_dim * 2
    }

    /// Number of layers that require KV cache (attention layers only).
    /// For pure transformers: all layers. For hybrid: only attention layers.
    pub fn kv_cache_layers(&self) -> usize {
        if self.hybrid_schedule.is_empty() {
            self.num_hidden_layers
        } else {
            architectures::jamba::count_attention_layers(&self.hybrid_schedule)
        }
    }

    /// Whether this is a hybrid model (Mamba + Attention mix)
    pub fn is_hybrid(&self) -> bool {
        !self.hybrid_schedule.is_empty()
    }

    /// Whether this model uses sparse MoE FFNs
    pub fn is_moe(&self) -> bool {
        self.moe_num_experts > 0
    }

    /// Whether any layer uses Mamba SSM
    pub fn has_mamba_layers(&self) -> bool {
        self.hybrid_schedule.iter().any(|t| t.uses_mamba())
            || matches!(self.architecture, ModelArchitecture::Mamba | ModelArchitecture::Jamba | ModelArchitecture::Zamba | ModelArchitecture::NemotronH)
    }

    /// Mamba inner dimension for this model (expand * hidden_size)
    pub fn mamba_d_inner(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }

    /// KV cache memory reduction factor from hybridisation (1.0 = no reduction)
    pub fn kv_cache_reduction_factor(&self) -> f64 {
        if self.hybrid_schedule.is_empty() {
            1.0
        } else {
            self.kv_cache_layers() as f64 / self.num_hidden_layers as f64
        }
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

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/crates/swiftllm-models/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
