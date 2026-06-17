// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      mtp.rs
// PATH:      /crates/swiftllm-models/src/layers/mtp.rs
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

//! Multi-Token Prediction (MTP) heads.
//!
//! Implements the DeepSeek-V3 style MTP module: a stack of lightweight heads,
//! each of which predicts one additional future token. A head takes the
//! previous depth's hidden state and the embedding of the most recently
//! predicted token, RMS-normalises both, concatenates them, and projects back
//! to `hidden_size` before a (shared) output projection.
//!
//! The token-level drafting/verification logic lives in
//! [`swiftllm_core::execution::speculative::MtpSpeculator`]; this module is the
//! model-side structural component plus the pure [`combine_hidden_and_embedding`]
//! preprocessing step (unit-tested on CPU).

use crate::layers::{Linear, RMSNorm};
use swiftllm_core::config::DataType;
use swiftllm_core::error::Result;
use swiftllm_core::tensor::{Device, Tensor};

/// Configuration for the MTP module.
#[derive(Debug, Clone)]
pub struct MtpConfig {
    /// Model hidden size.
    pub hidden_size: usize,
    /// Vocabulary size (for the output projection).
    pub vocab_size: usize,
    /// Number of MTP heads (additional tokens predicted per step).
    pub num_heads: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            vocab_size: 32000,
            num_heads: 1,
            rms_norm_eps: 1e-5,
        }
    }
}

/// The dimensionality of a single MTP head's projection input: the normalised
/// previous hidden state concatenated with the normalised token embedding.
pub fn projection_input_dim(hidden_size: usize) -> usize {
    2 * hidden_size
}

/// Minimal RMS normalisation used by the MTP combine step.
fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|&v| v * inv).collect()
}

/// Build a single MTP head's projection input from the previous hidden state
/// and the most recently predicted token's embedding.
///
/// Both inputs are RMS-normalised, then concatenated to length
/// `projection_input_dim(hidden)`. Returns an empty vector if the lengths
/// disagree.
pub fn combine_hidden_and_embedding(
    prev_hidden: &[f32],
    token_embedding: &[f32],
    eps: f32,
) -> Vec<f32> {
    if prev_hidden.len() != token_embedding.len() {
        return Vec::new();
    }
    let mut out = rms_norm(prev_hidden, eps);
    out.extend(rms_norm(token_embedding, eps));
    out
}

/// A single MTP head.
#[allow(dead_code)]
struct MtpHead {
    /// Norm applied to the token embedding.
    embedding_norm: RMSNorm,
    /// Norm applied to the previous-depth hidden state.
    hidden_norm: RMSNorm,
    /// Projection of `concat(norm(hidden), norm(embedding))` back to hidden_size.
    projection: Linear,
}

impl MtpHead {
    fn new(config: &MtpConfig) -> Result<Self> {
        let norm = || -> Result<RMSNorm> {
            let w = Tensor::zeros(vec![config.hidden_size], DataType::Float16, Device::Cpu)?;
            RMSNorm::new(w, config.rms_norm_eps)
        };
        let proj_weight = Tensor::zeros(
            vec![config.hidden_size, projection_input_dim(config.hidden_size)],
            DataType::Float16,
            Device::Cpu,
        )?;
        Ok(Self {
            embedding_norm: norm()?,
            hidden_norm: norm()?,
            projection: Linear::new(proj_weight, None)?,
        })
    }
}

/// A stack of MTP heads attached to a base model.
pub struct MtpModule {
    config: MtpConfig,
    heads: Vec<MtpHead>,
    /// Shared output projection to vocabulary logits.
    output_head: Linear,
}

impl MtpModule {
    /// Construct an MTP module with `config.num_heads` heads.
    pub fn new(config: MtpConfig) -> Result<Self> {
        let mut heads = Vec::with_capacity(config.num_heads);
        for _ in 0..config.num_heads {
            heads.push(MtpHead::new(&config)?);
        }
        let output_weight = Tensor::zeros(
            vec![config.vocab_size, config.hidden_size],
            DataType::Float16,
            Device::Cpu,
        )?;
        let output_head = Linear::new(output_weight, None)?;
        Ok(Self {
            config,
            heads,
            output_head,
        })
    }

    /// Number of MTP heads.
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &MtpConfig {
        &self.config
    }

    /// Project a head's hidden state to vocabulary logits (shared output head).
    pub fn project_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.output_head.forward(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_input_dim_is_doubled() {
        assert_eq!(projection_input_dim(4096), 8192);
    }

    #[test]
    fn combine_concatenates_normalised_inputs() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let embedding = vec![4.0, 3.0, 2.0, 1.0];
        let combined = combine_hidden_and_embedding(&hidden, &embedding, 1e-6);
        assert_eq!(combined.len(), projection_input_dim(4));

        // Each half is unit-RMS after normalisation.
        let first_half = &combined[..4];
        let rms = (first_half.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-3);
    }

    #[test]
    fn combine_rejects_mismatched_lengths() {
        assert!(combine_hidden_and_embedding(&[1.0, 2.0], &[1.0], 1e-6).is_empty());
    }

    #[test]
    fn module_builds_requested_heads() {
        let module = MtpModule::new(MtpConfig {
            hidden_size: 512,
            vocab_size: 1000,
            num_heads: 3,
            rms_norm_eps: 1e-5,
        })
        .unwrap();
        assert_eq!(module.num_heads(), 3);
        assert_eq!(module.config().vocab_size, 1000);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: mtp.rs
// REPO PATH:   /swiftllm/crates/swiftllm-models/src/layers/mtp.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
