//! Neural Network Layers
//!
//! This module provides implementations of common transformer layers
//! optimized for inference with PagedAttention.

pub mod attention;
pub mod embedding;
pub mod mlp;
pub mod normalization;

pub use attention::{Attention, AttentionConfig, RotaryEmbedding};
pub use embedding::{Embedding, LMHead};
pub use mlp::{Mlp, MlpConfig, GatedMlp};
pub use normalization::{LayerNorm, RMSNorm};

use swiftllm_core::config::DataType;
use swiftllm_core::error::Result;
use swiftllm_core::tensor::Tensor;

/// Linear layer (matrix multiplication + optional bias)
#[derive(Debug)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Tensor,

    /// Bias vector [out_features] (optional)
    pub bias: Option<Tensor>,

    /// Input features
    pub in_features: usize,

    /// Output features
    pub out_features: usize,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(swiftllm_core::error::Error::Tensor(
                "Linear weight must be 2D".to_string(),
            ));
        }

        let out_features = dims[0];
        let in_features = dims[1];

        if let Some(ref b) = bias {
            if b.dims().len() != 1 || b.dims()[0] != out_features {
                return Err(swiftllm_core::error::Error::Tensor(
                    "Bias dimensions don't match weight".to_string(),
                ));
            }
        }

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // In a real implementation, this would:
        // 1. Compute output = input @ weight.T
        // 2. Add bias if present

        // Placeholder: return zeros with correct shape
        let input_dims = input.dims();
        let batch_dims = &input_dims[..input_dims.len() - 1];
        let mut output_shape = batch_dims.to_vec();
        output_shape.push(self.out_features);

        Tensor::zeros(output_shape, input.dtype(), input.device())
    }

    /// Get the number of parameters
    pub fn num_params(&self) -> usize {
        let mut count = self.weight.numel();
        if let Some(ref b) = self.bias {
            count += b.numel();
        }
        count
    }
}

/// Column-parallel linear layer for tensor parallelism
#[derive(Debug)]
pub struct ColumnParallelLinear {
    /// Linear layer
    pub linear: Linear,

    /// World size
    pub world_size: usize,

    /// Gather output after forward
    pub gather_output: bool,
}

impl ColumnParallelLinear {
    /// Create a new column-parallel linear layer
    pub fn new(linear: Linear, world_size: usize, gather_output: bool) -> Self {
        Self {
            linear,
            world_size,
            gather_output,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Column parallel: each GPU has a shard of the output dimension
        // If gather_output is true, we all-gather to get the full output
        self.linear.forward(input)
    }
}

/// Row-parallel linear layer for tensor parallelism
#[derive(Debug)]
pub struct RowParallelLinear {
    /// Linear layer
    pub linear: Linear,

    /// World size
    pub world_size: usize,

    /// Input is parallel (already sharded)
    pub input_is_parallel: bool,
}

impl RowParallelLinear {
    /// Create a new row-parallel linear layer
    pub fn new(linear: Linear, world_size: usize, input_is_parallel: bool) -> Self {
        Self {
            linear,
            world_size,
            input_is_parallel,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Row parallel: each GPU has a shard of the input dimension
        // We need to all-reduce the output to get the final result
        self.linear.forward(input)
    }
}

/// Merged QKV projection for efficient attention
#[derive(Debug)]
pub struct QKVProjection {
    /// Merged weight [3 * hidden_size, hidden_size] or split
    pub weight: Tensor,

    /// Bias (optional)
    pub bias: Option<Tensor>,

    /// Hidden size
    pub hidden_size: usize,

    /// Number of heads
    pub num_heads: usize,

    /// Number of KV heads
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,
}

impl QKVProjection {
    /// Forward pass - returns (Q, K, V)
    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Project and split into Q, K, V
        let dims = hidden_states.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        // Placeholder
        let q_shape = vec![batch_size, seq_len, self.num_heads, self.head_dim];
        let kv_shape = vec![batch_size, seq_len, self.num_kv_heads, self.head_dim];

        let q = Tensor::zeros(q_shape, hidden_states.dtype(), hidden_states.device())?;
        let k = Tensor::zeros(kv_shape.clone(), hidden_states.dtype(), hidden_states.device())?;
        let v = Tensor::zeros(kv_shape, hidden_states.dtype(), hidden_states.device())?;

        Ok((q, k, v))
    }
}

/// Fused gate-up projection for MLP
#[derive(Debug)]
pub struct GateUpProjection {
    /// Merged weight [2 * intermediate_size, hidden_size]
    pub weight: Tensor,

    /// Hidden size
    pub hidden_size: usize,

    /// Intermediate size
    pub intermediate_size: usize,
}

impl GateUpProjection {
    /// Forward pass - returns (gate, up)
    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = hidden_states.dims();
        let batch_dims = &dims[..dims.len() - 1];
        let mut output_shape = batch_dims.to_vec();
        output_shape.push(self.intermediate_size);

        let gate = Tensor::zeros(
            output_shape.clone(),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        let up = Tensor::zeros(output_shape, hidden_states.dtype(), hidden_states.device())?;

        Ok((gate, up))
    }
}
