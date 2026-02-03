//! Execution Engine
//!
//! This module provides the execution engine for running model inference,
//! including tensor parallel execution and speculative decoding.

mod speculative;
mod tensor_parallel;

pub use speculative::{SpeculativeDecoder, SpeculativeDecodingConfig};
pub use tensor_parallel::{TensorParallelConfig, TensorParallelExecutor};

use crate::error::{Error, Result};
use crate::memory::kv_cache::BatchedCacheMetadata;
use crate::tensor::Tensor;
use crate::types::{ExecutionBatch, TokenId};

/// Model execution interface
pub trait ModelExecutor: Send + Sync {
    /// Execute prefill (process prompt)
    fn prefill(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
    ) -> Result<Tensor>;

    /// Execute decode (generate one token)
    fn decode(
        &self,
        input_ids: &[TokenId],
        positions: &[usize],
        cache_metadata: &BatchedCacheMetadata,
    ) -> Result<Tensor>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get hidden size
    fn hidden_size(&self) -> usize;

    /// Get number of layers
    fn num_layers(&self) -> usize;
}

/// Execution batch builder
#[derive(Debug, Default)]
pub struct BatchBuilder {
    /// Input tokens
    input_tokens: Vec<TokenId>,

    /// Positions
    positions: Vec<usize>,

    /// Sequence lengths
    seq_lens: Vec<usize>,

    /// Block tables
    block_tables: Vec<Vec<usize>>,

    /// Context lengths
    context_lens: Vec<usize>,

    /// Slot mapping
    slot_mapping: Vec<usize>,

    /// Is prefill
    is_prefill: bool,
}

impl BatchBuilder {
    /// Create a new batch builder
    pub fn new(is_prefill: bool) -> Self {
        Self {
            is_prefill,
            ..Default::default()
        }
    }

    /// Add a sequence to the batch
    pub fn add_sequence(
        &mut self,
        tokens: &[TokenId],
        start_pos: usize,
        block_table: Vec<usize>,
        context_len: usize,
        slot_mapping: Vec<usize>,
    ) {
        let seq_len = tokens.len();

        self.input_tokens.extend_from_slice(tokens);
        self.positions
            .extend((start_pos..start_pos + seq_len).map(|p| p));
        self.seq_lens.push(seq_len);
        self.block_tables.push(block_table);
        self.context_lens.push(context_len);
        self.slot_mapping.extend(slot_mapping);
    }

    /// Build the execution batch
    pub fn build(self) -> ExecutionBatch {
        ExecutionBatch {
            input_tokens: self.input_tokens,
            positions: self.positions,
            seq_lens: self.seq_lens,
            block_tables: self.block_tables,
            context_lens: self.context_lens,
            slot_mapping: self.slot_mapping,
            is_prefill: self.is_prefill,
        }
    }

    /// Get the total number of tokens
    pub fn num_tokens(&self) -> usize {
        self.input_tokens.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.input_tokens.is_empty()
    }

    /// Get the number of sequences
    pub fn num_sequences(&self) -> usize {
        self.seq_lens.len()
    }
}

/// Execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total forward passes
    pub total_forward_passes: usize,

    /// Total prefill tokens
    pub prefill_tokens: usize,

    /// Total decode tokens
    pub decode_tokens: usize,

    /// Total execution time (seconds)
    pub total_time_secs: f64,

    /// Prefill time (seconds)
    pub prefill_time_secs: f64,

    /// Decode time (seconds)
    pub decode_time_secs: f64,

    /// Average tokens per second
    pub tokens_per_second: f64,

    /// Peak GPU memory (bytes)
    pub peak_gpu_memory: usize,
}

impl ExecutionStats {
    /// Update statistics after a forward pass
    pub fn update(&mut self, is_prefill: bool, num_tokens: usize, elapsed_secs: f64) {
        self.total_forward_passes += 1;
        self.total_time_secs += elapsed_secs;

        if is_prefill {
            self.prefill_tokens += num_tokens;
            self.prefill_time_secs += elapsed_secs;
        } else {
            self.decode_tokens += num_tokens;
            self.decode_time_secs += elapsed_secs;
        }

        let total_tokens = self.prefill_tokens + self.decode_tokens;
        if self.total_time_secs > 0.0 {
            self.tokens_per_second = total_tokens as f64 / self.total_time_secs;
        }
    }

    /// Get prefill throughput
    pub fn prefill_throughput(&self) -> f64 {
        if self.prefill_time_secs > 0.0 {
            self.prefill_tokens as f64 / self.prefill_time_secs
        } else {
            0.0
        }
    }

    /// Get decode throughput
    pub fn decode_throughput(&self) -> f64 {
        if self.decode_time_secs > 0.0 {
            self.decode_tokens as f64 / self.decode_time_secs
        } else {
            0.0
        }
    }
}

/// Execution engine configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum batch size
    pub max_batch_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Use CUDA graphs
    pub use_cuda_graphs: bool,

    /// Number of CUDA graphs to cache
    pub num_cuda_graphs: usize,

    /// Tensor parallel size
    pub tensor_parallel_size: usize,

    /// Pipeline parallel size
    pub pipeline_parallel_size: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_seq_len: 4096,
            use_cuda_graphs: true,
            num_cuda_graphs: 10,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
        }
    }
}

/// CUDA graph cache for accelerating inference
#[derive(Debug)]
pub struct CudaGraphCache {
    /// Captured graphs (keyed by batch size)
    graphs: std::collections::HashMap<usize, CapturedGraph>,

    /// Maximum batch size to capture
    max_batch_size: usize,

    /// Maximum sequence length per graph
    max_seq_len: usize,
}

/// A captured CUDA graph
#[derive(Debug)]
pub struct CapturedGraph {
    /// Batch size this graph was captured for
    pub batch_size: usize,

    /// Whether this is a prefill or decode graph
    pub is_prefill: bool,

    // In a real implementation, this would hold the actual CUDA graph handle
}

impl CudaGraphCache {
    /// Create a new CUDA graph cache
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            graphs: std::collections::HashMap::new(),
            max_batch_size,
            max_seq_len,
        }
    }

    /// Check if a graph exists for the given configuration
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Get a graph for the given batch size
    pub fn get(&self, batch_size: usize) -> Option<&CapturedGraph> {
        self.graphs.get(&batch_size)
    }

    /// Cache a captured graph
    pub fn insert(&mut self, batch_size: usize, graph: CapturedGraph) {
        self.graphs.insert(batch_size, graph);
    }

    /// Clear all cached graphs
    pub fn clear(&mut self) {
        self.graphs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_builder() {
        let mut builder = BatchBuilder::new(true);

        builder.add_sequence(
            &[1, 2, 3, 4, 5],
            0,
            vec![0, 1],
            5,
            vec![0, 1, 2, 3, 4],
        );

        assert_eq!(builder.num_tokens(), 5);
        assert_eq!(builder.num_sequences(), 1);
        assert!(builder.is_prefill);

        let batch = builder.build();
        assert_eq!(batch.input_tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(batch.positions, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_execution_stats() {
        let mut stats = ExecutionStats::default();

        stats.update(true, 100, 0.1);
        assert_eq!(stats.prefill_tokens, 100);

        stats.update(false, 50, 0.5);
        assert_eq!(stats.decode_tokens, 50);

        assert!(stats.tokens_per_second > 0.0);
    }

    #[test]
    fn test_cuda_graph_cache() {
        let mut cache = CudaGraphCache::new(256, 4096);

        assert!(!cache.has_graph(16));

        cache.insert(
            16,
            CapturedGraph {
                batch_size: 16,
                is_prefill: false,
            },
        );

        assert!(cache.has_graph(16));
        let graph = cache.get(16).unwrap();
        assert_eq!(graph.batch_size, 16);
    }
}
