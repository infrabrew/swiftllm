// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      mod.rs
// PATH:      /crates/swiftllm-models/src/architectures/mod.rs
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

//! Model Architectures
//!
//! This module provides implementations of popular LLM architectures.

pub mod jamba;
pub mod llama;
pub mod mistral;
pub mod phi;
pub mod qwen;

pub use jamba::{
    HybridLayerType, HybridRecurrentState, JambaConfig, JambaModel,
    count_attention_layers, jamba_schedule,
};
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
    _weights_path: &std::path::Path,
) -> Result<Box<dyn TransformerModel>> {
    match architecture {
        ModelArchitecture::Llama => Ok(Box::new(LlamaModel::new(config)?)),
        ModelArchitecture::Mistral => Ok(Box::new(MistralModel::new(config)?)),
        ModelArchitecture::Qwen | ModelArchitecture::Qwen2 => Ok(Box::new(QwenModel::new(config)?)),
        ModelArchitecture::Phi | ModelArchitecture::Phi3 => Ok(Box::new(PhiModel::new(config)?)),
        ModelArchitecture::Jamba | ModelArchitecture::NemotronH => {
            // Jamba-style hybrid: build a default schedule, then construct JambaModel
            let jamba_cfg = JambaConfig::small_hybrid(config.hidden_size, config.num_hidden_layers);
            Ok(Box::new(JambaModel::new(jamba_cfg)?))
        }
        ModelArchitecture::Zamba => {
            // Zamba uses weight-shared attention; for now route to Jamba with 1:6 ratio
            let mut jamba_cfg = JambaConfig::small_hybrid(config.hidden_size, config.num_hidden_layers);
            jamba_cfg.layer_schedule = jamba_schedule(config.num_hidden_layers, 6, 0);
            Ok(Box::new(JambaModel::new(jamba_cfg)?))
        }
        ModelArchitecture::Mamba => {
            // Pure Mamba: all layers are SSM, no attention
            let mut jamba_cfg = JambaConfig::small_hybrid(config.hidden_size, config.num_hidden_layers);
            // attn_period = num_layers + 1 → no attention layer ever fires
            jamba_cfg.layer_schedule = jamba_schedule(
                config.num_hidden_layers,
                config.num_hidden_layers + 1,
                0,
            );
            Ok(Box::new(JambaModel::new(jamba_cfg)?))
        }
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

// ------------------------------------------------------------------------------
// END OF FILE: mod.rs
// REPO PATH:   /swiftllm/crates/swiftllm-models/src/architectures/mod.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
