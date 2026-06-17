// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      mod.rs
// PATH:      /crates/swiftllm-core/src/execution/mod.rs
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

//! Execution Engine
//!
//! This module provides the execution engine for running model inference,
//! including tensor parallel execution and speculative decoding.

pub mod async_scheduling;
mod speculative;
mod tensor_parallel;

pub use async_scheduling::{
    overlap_speedup, AsyncScheduler, AsyncSchedulerConfig, StepHandle,
};
pub use speculative::{
    MtpConfig, MtpSpeculator, SpeculativeDecoder, SpeculativeDecodingConfig,
};
pub use tensor_parallel::{TensorParallelConfig, TensorParallelExecutor};

use crate::error::Result;
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
            .extend(start_pos..start_pos + seq_len);
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

/// CUDA graph cache for accelerating inference.
///
/// Decode steps are replayed from CUDA graphs captured at a fixed set of batch
/// sizes (the *capture buckets*). A live batch is padded up to the smallest
/// bucket that covers it, so a handful of graphs serve every batch size. Prefill
/// is dynamic-shape and is not graphed.
///
/// Capture/replay of the actual device graph is the GPU seam; the bucketing,
/// padding, eligibility, and hit/miss accounting below are exact and tested.
#[derive(Debug)]
pub struct CudaGraphCache {
    /// Captured graphs (keyed by capture-bucket batch size).
    graphs: std::collections::HashMap<usize, CapturedGraph>,

    /// Batch-size buckets graphs are captured at (ascending).
    capture_sizes: Vec<usize>,

    /// Maximum batch size to capture.
    max_batch_size: usize,

    /// Maximum sequence length per graph.
    max_seq_len: usize,

    /// Number of replays served from a captured graph.
    hits: usize,

    /// Number of steps that fell back to eager execution.
    misses: usize,
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
    /// Create a new CUDA graph cache with default power-of-two capture buckets.
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        let capture_sizes = Self::default_capture_sizes(max_batch_size);
        Self {
            graphs: std::collections::HashMap::new(),
            capture_sizes,
            max_batch_size,
            max_seq_len,
            hits: 0,
            misses: 0,
        }
    }

    /// Default capture buckets: 1, 2, 4, 8, … up to and including `max`.
    pub fn default_capture_sizes(max: usize) -> Vec<usize> {
        if max == 0 {
            return Vec::new();
        }
        let mut sizes = Vec::new();
        let mut s = 1;
        while s < max {
            sizes.push(s);
            s *= 2;
        }
        sizes.push(max);
        sizes
    }

    /// The capture buckets.
    pub fn capture_sizes(&self) -> &[usize] {
        &self.capture_sizes
    }

    /// The smallest capture bucket that covers `batch_size` (the size the live
    /// batch is padded up to), or `None` if the batch exceeds `max_batch_size`.
    pub fn padded_batch_size(&self, batch_size: usize) -> Option<usize> {
        if batch_size == 0 || batch_size > self.max_batch_size {
            return None;
        }
        self.capture_sizes
            .iter()
            .copied()
            .find(|&bucket| bucket >= batch_size)
    }

    /// Whether a decode step at `batch_size` is eligible for graph capture
    /// (prefill is never graphed; oversized batches fall back to eager).
    pub fn should_capture(&self, batch_size: usize, is_prefill: bool) -> bool {
        !is_prefill && self.padded_batch_size(batch_size).is_some()
    }

    /// Check if a graph exists for the given (already-bucketed) batch size.
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Get a graph for the given batch size.
    pub fn get(&self, batch_size: usize) -> Option<&CapturedGraph> {
        self.graphs.get(&batch_size)
    }

    /// Look up the graph for a live `batch_size`, padding up to a capture bucket
    /// and recording the hit/miss. Returns the bucket whose graph should run.
    pub fn acquire(&mut self, batch_size: usize, is_prefill: bool) -> Option<usize> {
        if is_prefill {
            self.misses += 1;
            return None;
        }
        match self.padded_batch_size(batch_size) {
            Some(bucket) if self.graphs.contains_key(&bucket) => {
                self.hits += 1;
                Some(bucket)
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Cache a captured graph.
    pub fn insert(&mut self, batch_size: usize, graph: CapturedGraph) {
        self.graphs.insert(batch_size, graph);
    }

    /// Number of graphs currently captured.
    pub fn num_cached(&self) -> usize {
        self.graphs.len()
    }

    /// Fraction of `acquire` calls served from a captured graph.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Maximum sequence length graphs are captured for.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Clear all cached graphs (and reset stats).
    pub fn clear(&mut self) {
        self.graphs.clear();
        self.hits = 0;
        self.misses = 0;
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

    #[test]
    fn test_cuda_graph_capture_buckets() {
        let cache = CudaGraphCache::new(8, 4096);
        // Powers of two up to and including max.
        assert_eq!(cache.capture_sizes(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_cuda_graph_padding() {
        let cache = CudaGraphCache::new(256, 4096);
        // A batch of 5 is padded up to the next bucket (8).
        assert_eq!(cache.padded_batch_size(5), Some(8));
        assert_eq!(cache.padded_batch_size(1), Some(1));
        assert_eq!(cache.padded_batch_size(256), Some(256));
        // Beyond the max batch size -> no graph (eager fallback).
        assert_eq!(cache.padded_batch_size(300), None);
        assert_eq!(cache.padded_batch_size(0), None);
    }

    #[test]
    fn test_cuda_graph_eligibility() {
        let cache = CudaGraphCache::new(64, 4096);
        // Decode within range is eligible; prefill never is.
        assert!(cache.should_capture(8, false));
        assert!(!cache.should_capture(8, true));
        assert!(!cache.should_capture(1000, false));
    }

    #[test]
    fn test_cuda_graph_acquire_hit_and_miss() {
        let mut cache = CudaGraphCache::new(256, 4096);
        // Capture a graph at bucket 8.
        cache.insert(8, CapturedGraph { batch_size: 8, is_prefill: false });

        // Live batch of 6 pads to 8 and hits.
        assert_eq!(cache.acquire(6, false), Some(8));
        // Live batch of 9 pads to 16, which has no captured graph -> miss.
        assert_eq!(cache.acquire(9, false), None);
        // Prefill always misses.
        assert_eq!(cache.acquire(8, true), None);

        // One hit out of three acquires.
        assert!((cache.hit_rate() - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(cache.num_cached(), 1);
    }

    #[test]
    fn test_cuda_graph_clear_resets_stats() {
        let mut cache = CudaGraphCache::new(256, 4096);
        cache.insert(8, CapturedGraph { batch_size: 8, is_prefill: false });
        let _ = cache.acquire(8, false);
        cache.clear();
        assert_eq!(cache.num_cached(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: mod.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/execution/mod.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
