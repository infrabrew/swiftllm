//! Tensor Parallelism
//!
//! This module implements tensor parallelism for distributing
//! model inference across multiple GPUs.

use crate::error::{Error, Result};
use crate::tensor::{Device, Tensor};
use std::sync::Arc;

/// Tensor parallel configuration
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// World size (number of GPUs)
    pub world_size: usize,

    /// Current rank
    pub rank: usize,

    /// GPU device IDs
    pub device_ids: Vec<usize>,

    /// Communication backend
    pub backend: CommunicationBackend,
}

impl TensorParallelConfig {
    /// Create config for single GPU
    pub fn single_gpu(device_id: usize) -> Self {
        Self {
            world_size: 1,
            rank: 0,
            device_ids: vec![device_id],
            backend: CommunicationBackend::Nccl,
        }
    }

    /// Create config for multiple GPUs
    pub fn multi_gpu(device_ids: Vec<usize>) -> Self {
        let world_size = device_ids.len();
        Self {
            world_size,
            rank: 0,
            device_ids,
            backend: CommunicationBackend::Nccl,
        }
    }

    /// Check if this is single GPU
    pub fn is_single_gpu(&self) -> bool {
        self.world_size == 1
    }
}

/// Communication backend for tensor parallel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    /// NCCL (NVIDIA Collective Communications Library)
    Nccl,
    /// Gloo (CPU-based)
    Gloo,
    /// Custom implementation
    Custom,
}

/// Tensor parallel executor
pub struct TensorParallelExecutor {
    /// Configuration
    config: TensorParallelConfig,

    /// Communication handle
    comm: Option<CommunicationHandle>,
}

/// Handle for collective communications
#[derive(Debug)]
pub struct CommunicationHandle {
    /// World size
    world_size: usize,

    /// Current rank
    rank: usize,

    /// Backend
    backend: CommunicationBackend,
}

impl TensorParallelExecutor {
    /// Create a new tensor parallel executor
    pub fn new(config: TensorParallelConfig) -> Result<Self> {
        let comm = if config.world_size > 1 {
            Some(CommunicationHandle::new(&config)?)
        } else {
            None
        };

        Ok(Self { config, comm })
    }

    /// Get the world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get the current rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Get the device for this rank
    pub fn device(&self) -> Device {
        Device::Cuda(self.config.device_ids[self.config.rank])
    }

    /// Perform all-reduce operation
    pub fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        if self.config.is_single_gpu() {
            return Ok(());
        }

        // In a real implementation, this would use NCCL
        // For now, this is a placeholder
        Err(Error::not_implemented("NCCL all-reduce"))
    }

    /// Perform all-gather operation
    pub fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        if self.config.is_single_gpu() {
            return Ok(vec![tensor.clone()]);
        }

        Err(Error::not_implemented("NCCL all-gather"))
    }

    /// Perform reduce-scatter operation
    pub fn reduce_scatter(&self, tensor: &Tensor, op: ReduceOp) -> Result<Tensor> {
        if self.config.is_single_gpu() {
            return Ok(tensor.clone());
        }

        Err(Error::not_implemented("NCCL reduce-scatter"))
    }

    /// Broadcast tensor from source rank
    pub fn broadcast(&self, tensor: &mut Tensor, src: usize) -> Result<()> {
        if self.config.is_single_gpu() {
            return Ok(());
        }

        Err(Error::not_implemented("NCCL broadcast"))
    }

    /// Synchronize all ranks
    pub fn barrier(&self) -> Result<()> {
        if self.config.is_single_gpu() {
            return Ok(());
        }

        Err(Error::not_implemented("NCCL barrier"))
    }

    /// Shard a tensor along a dimension for this rank
    pub fn shard_tensor(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        if self.config.is_single_gpu() {
            return Ok(tensor.clone());
        }

        let dims = tensor.dims();
        if dim >= dims.len() {
            return Err(Error::Tensor(format!(
                "Cannot shard along dimension {} for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        let shard_size = dims[dim] / self.world_size();
        if dims[dim] % self.world_size() != 0 {
            return Err(Error::Tensor(format!(
                "Dimension {} size {} is not divisible by world size {}",
                dim,
                dims[dim],
                self.world_size()
            )));
        }

        // Calculate shard for this rank
        let _start = self.rank() * shard_size;
        let _end = (self.rank() + 1) * shard_size;

        // In a real implementation, this would slice the tensor
        Err(Error::not_implemented("Tensor sharding"))
    }

    /// Unshard a tensor (gather from all ranks)
    pub fn unshard_tensor(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        if self.config.is_single_gpu() {
            return Ok(tensor.clone());
        }

        // Gather from all ranks and concatenate
        Err(Error::not_implemented("Tensor unsharding"))
    }
}

impl CommunicationHandle {
    /// Create a new communication handle
    fn new(config: &TensorParallelConfig) -> Result<Self> {
        // In a real implementation, this would initialize NCCL
        Ok(Self {
            world_size: config.world_size,
            rank: config.rank,
            backend: config.backend,
        })
    }
}

/// Reduce operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum
    Sum,
    /// Product
    Product,
    /// Maximum
    Max,
    /// Minimum
    Min,
    /// Average
    Average,
}

/// Parallel linear layer that shards weights across GPUs
pub struct ParallelLinear {
    /// Weight tensor (sharded)
    weight: Tensor,

    /// Bias tensor (optional, replicated)
    bias: Option<Tensor>,

    /// Input dimension
    in_features: usize,

    /// Output dimension (per GPU)
    out_features_per_gpu: usize,

    /// Sharding dimension (0 = row, 1 = column)
    shard_dim: usize,

    /// Tensor parallel config
    tp_config: TensorParallelConfig,
}

impl ParallelLinear {
    /// Create a new parallel linear layer
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        shard_dim: usize,
        tp_config: TensorParallelConfig,
    ) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::Tensor(
                "Linear weight must be 2D".to_string(),
            ));
        }

        let (in_features, out_features_per_gpu) = if shard_dim == 0 {
            (dims[1], dims[0])
        } else {
            (dims[0] / tp_config.world_size, dims[1])
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features_per_gpu,
            shard_dim,
            tp_config,
        })
    }

    /// Check if this is column parallel
    pub fn is_column_parallel(&self) -> bool {
        self.shard_dim == 1
    }

    /// Check if this is row parallel
    pub fn is_row_parallel(&self) -> bool {
        self.shard_dim == 0
    }
}

/// Vocabulary parallel embedding that shards vocabulary across GPUs
pub struct VocabParallelEmbedding {
    /// Embedding weight (sharded by vocabulary)
    weight: Tensor,

    /// Vocabulary size (total)
    vocab_size: usize,

    /// Vocabulary size per GPU
    vocab_size_per_gpu: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Vocabulary start index for this rank
    vocab_start_idx: usize,

    /// Vocabulary end index for this rank
    vocab_end_idx: usize,

    /// Tensor parallel config
    tp_config: TensorParallelConfig,
}

impl VocabParallelEmbedding {
    /// Create a new vocabulary parallel embedding
    pub fn new(
        weight: Tensor,
        vocab_size: usize,
        tp_config: TensorParallelConfig,
    ) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::Tensor(
                "Embedding weight must be 2D".to_string(),
            ));
        }

        let embedding_dim = dims[1];
        let vocab_size_per_gpu = vocab_size / tp_config.world_size;
        let vocab_start_idx = tp_config.rank * vocab_size_per_gpu;
        let vocab_end_idx = (tp_config.rank + 1) * vocab_size_per_gpu;

        Ok(Self {
            weight,
            vocab_size,
            vocab_size_per_gpu,
            embedding_dim,
            vocab_start_idx,
            vocab_end_idx,
            tp_config,
        })
    }

    /// Check if a token is in this rank's vocabulary shard
    pub fn contains_token(&self, token_id: usize) -> bool {
        token_id >= self.vocab_start_idx && token_id < self.vocab_end_idx
    }

    /// Get the local index for a token
    pub fn local_index(&self, token_id: usize) -> Option<usize> {
        if self.contains_token(token_id) {
            Some(token_id - self.vocab_start_idx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_config() {
        let config = TensorParallelConfig::single_gpu(0);

        assert!(config.is_single_gpu());
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
    }

    #[test]
    fn test_multi_gpu_config() {
        let config = TensorParallelConfig::multi_gpu(vec![0, 1, 2, 3]);

        assert!(!config.is_single_gpu());
        assert_eq!(config.world_size, 4);
    }

    #[test]
    fn test_executor_single_gpu() {
        let config = TensorParallelConfig::single_gpu(0);
        let executor = TensorParallelExecutor::new(config).unwrap();

        assert_eq!(executor.world_size(), 1);
        assert_eq!(executor.rank(), 0);
    }
}
