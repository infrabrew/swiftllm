//! PagedAttention Implementation
//!
//! PagedAttention is a memory management technique that enables efficient
//! KV cache management by:
//! 1. Dividing KV cache into fixed-size blocks
//! 2. Allowing non-contiguous memory allocation
//! 3. Enabling memory sharing between sequences (for beam search)
//! 4. Supporting efficient copy-on-write operations
//!
//! Reference: "Efficient Memory Management for Large Language Model Serving with PagedAttention"

use crate::config::DataType;
use crate::error::{Error, Result};
use crate::memory::block_manager::PhysicalBlockId;
use crate::tensor::{Device, Shape, Tensor};
use std::collections::HashMap;

/// Configuration for PagedAttention
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Number of attention heads (query heads)
    pub num_heads: usize,

    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Scaling factor for attention (usually 1/sqrt(head_dim))
    pub scale: f32,

    /// Block size in tokens
    pub block_size: usize,

    /// Data type
    pub dtype: DataType,

    /// Sliding window size (if using sliding window attention)
    pub sliding_window: Option<usize>,

    /// Whether to use alibi positional encoding
    pub use_alibi: bool,
}

impl PagedAttentionConfig {
    /// Create a new PagedAttention config with default settings
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, block_size: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            block_size,
            dtype: DataType::Float16,
            sliding_window: None,
            use_alibi: false,
        }
    }

    /// Set custom scale
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set sliding window
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Enable alibi
    pub fn with_alibi(mut self) -> Self {
        self.use_alibi = true;
        self
    }

    /// Get the number of query groups (for GQA)
    pub fn num_query_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// PagedAttention operator
#[derive(Debug)]
pub struct PagedAttention {
    /// Configuration
    config: PagedAttentionConfig,

    /// Pre-computed alibi slopes (if using alibi)
    alibi_slopes: Option<Vec<f32>>,
}

impl PagedAttention {
    /// Create a new PagedAttention operator
    pub fn new(config: PagedAttentionConfig) -> Self {
        let alibi_slopes = if config.use_alibi {
            Some(Self::compute_alibi_slopes(config.num_heads))
        } else {
            None
        };

        Self {
            config,
            alibi_slopes,
        }
    }

    /// Compute alibi slopes for given number of heads
    fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
        let closest_power_of_2 = 2usize.pow((num_heads as f32).log2().floor() as u32);
        let base = 2.0f32.powf(-(2.0f32.powf(-((closest_power_of_2 as f32).log2() - 3.0))));

        let mut slopes = Vec::with_capacity(num_heads);

        if num_heads <= closest_power_of_2 {
            for i in 1..=num_heads {
                slopes.push(base.powi(i as i32));
            }
        } else {
            // Handle non-power-of-2 heads
            for i in 1..=closest_power_of_2 {
                slopes.push(base.powi(i as i32));
            }
            let extra_base = 2.0f32.powf(
                -(2.0f32.powf(-((closest_power_of_2 as f32).log2() - 3.0))) / 2.0,
            );
            for i in 1..=(num_heads - closest_power_of_2) {
                slopes.push(extra_base.powi((2 * i) as i32));
            }
        }

        slopes
    }

    /// Get the config
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    /// Prefill forward pass (processes entire prompt)
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch_size, seq_len, num_heads, head_dim]
    /// * `key` - Key tensor [batch_size, seq_len, num_kv_heads, head_dim]
    /// * `value` - Value tensor [batch_size, seq_len, num_kv_heads, head_dim]
    /// * `key_cache` - Key cache tensor for this layer
    /// * `value_cache` - Value cache tensor for this layer
    /// * `slot_mapping` - Mapping from token positions to cache slots
    ///
    /// # Returns
    /// * Output tensor [batch_size, seq_len, num_heads, head_dim]
    pub fn prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_cache: &mut Tensor,
        value_cache: &mut Tensor,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Validate shapes
        let q_dims = query.dims();
        let k_dims = key.dims();

        if q_dims.len() != 4 || k_dims.len() != 4 {
            return Err(Error::ShapeMismatch {
                expected: vec![4],
                actual: vec![q_dims.len()],
            });
        }

        let batch_size = q_dims[0];
        let seq_len = q_dims[1];
        let num_heads = q_dims[2];
        let head_dim = q_dims[3];

        // For now, return a placeholder - actual implementation would:
        // 1. Compute attention scores: Q @ K^T / sqrt(d)
        // 2. Apply causal mask
        // 3. Apply softmax
        // 4. Compute output: attn @ V
        // 5. Store K, V into cache using slot_mapping

        // Placeholder output
        let output_shape = vec![batch_size, seq_len, num_heads, head_dim];
        Tensor::zeros(output_shape, query.dtype(), query.device())
    }

    /// Decode forward pass (processes one token at a time)
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch_size, 1, num_heads, head_dim]
    /// * `key` - Key tensor [batch_size, 1, num_kv_heads, head_dim]
    /// * `value` - Value tensor [batch_size, 1, num_kv_heads, head_dim]
    /// * `key_cache` - Key cache tensor
    /// * `value_cache` - Value cache tensor
    /// * `block_tables` - Block tables for each sequence
    /// * `context_lens` - Context length for each sequence
    /// * `slot_mapping` - Slot mapping for new tokens
    ///
    /// # Returns
    /// * Output tensor [batch_size, 1, num_heads, head_dim]
    pub fn decode(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_cache: &mut Tensor,
        value_cache: &mut Tensor,
        block_tables: &[Vec<PhysicalBlockId>],
        context_lens: &[usize],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let q_dims = query.dims();

        if q_dims.len() != 4 {
            return Err(Error::ShapeMismatch {
                expected: vec![4],
                actual: vec![q_dims.len()],
            });
        }

        let batch_size = q_dims[0];
        let num_heads = q_dims[2];
        let head_dim = q_dims[3];

        // PagedAttention decode would:
        // 1. Store new K, V into cache using slot_mapping
        // 2. For each sequence, gather K, V from cache using block_table
        // 3. Compute attention with gathered K, V
        // 4. Apply causal mask based on context_lens

        // Placeholder output
        let output_shape = vec![batch_size, 1, num_heads, head_dim];
        Tensor::zeros(output_shape, query.dtype(), query.device())
    }

    /// Reshape and cache key-value pairs
    ///
    /// Stores the key and value tensors into the cache at the specified slots
    pub fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &mut Tensor,
        value_cache: &mut Tensor,
        slot_mapping: &[usize],
    ) -> Result<()> {
        // This would copy key/value data into the cache at the specified slots
        // The actual implementation would use CUDA kernels for efficiency

        // Validate slot mapping
        let num_tokens = key.dims()[0] * key.dims()[1];
        if slot_mapping.len() != num_tokens {
            return Err(Error::Tensor(format!(
                "Slot mapping length {} doesn't match number of tokens {}",
                slot_mapping.len(),
                num_tokens
            )));
        }

        Ok(())
    }
}

/// Input metadata for PagedAttention
#[derive(Debug)]
pub struct PagedAttentionInput {
    /// Whether this is a prefill batch
    pub is_prefill: bool,

    /// Slot mapping for all tokens in the batch
    pub slot_mapping: Vec<usize>,

    /// Block tables for each sequence (decode only)
    pub block_tables: Option<Vec<Vec<PhysicalBlockId>>>,

    /// Context lengths for each sequence (decode only)
    pub context_lens: Option<Vec<usize>>,

    /// Maximum context length in the batch
    pub max_context_len: usize,

    /// Sequence start positions (for varlen attention)
    pub seq_start_positions: Vec<usize>,

    /// Query start positions (for varlen attention)
    pub query_start_positions: Vec<usize>,
}

impl PagedAttentionInput {
    /// Create prefill input
    pub fn prefill(slot_mapping: Vec<usize>, seq_lens: Vec<usize>) -> Self {
        let mut seq_start_positions = vec![0];
        let mut query_start_positions = vec![0];

        for &len in &seq_lens {
            seq_start_positions.push(seq_start_positions.last().unwrap() + len);
            query_start_positions.push(query_start_positions.last().unwrap() + len);
        }

        let max_context_len = seq_lens.iter().max().copied().unwrap_or(0);

        Self {
            is_prefill: true,
            slot_mapping,
            block_tables: None,
            context_lens: None,
            max_context_len,
            seq_start_positions,
            query_start_positions,
        }
    }

    /// Create decode input
    pub fn decode(
        slot_mapping: Vec<usize>,
        block_tables: Vec<Vec<PhysicalBlockId>>,
        context_lens: Vec<usize>,
    ) -> Self {
        let max_context_len = context_lens.iter().max().copied().unwrap_or(0);
        let batch_size = context_lens.len();

        let seq_start_positions: Vec<usize> = (0..=batch_size).collect();
        let query_start_positions: Vec<usize> = (0..=batch_size).collect();

        Self {
            is_prefill: false,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            max_context_len,
            seq_start_positions,
            query_start_positions,
        }
    }
}

/// Swap operation for block migration
#[derive(Debug, Clone)]
pub struct SwapOperation {
    /// Source block ID
    pub src_block: PhysicalBlockId,
    /// Destination block ID
    pub dst_block: PhysicalBlockId,
}

/// Copy operation for copy-on-write
#[derive(Debug, Clone)]
pub struct CopyOperation {
    /// Source block ID
    pub src_block: PhysicalBlockId,
    /// Destination block IDs
    pub dst_blocks: Vec<PhysicalBlockId>,
}

/// Batch of memory operations to execute
#[derive(Debug, Default)]
pub struct MemoryOperations {
    /// Blocks to swap from GPU to CPU
    pub swap_out: Vec<SwapOperation>,

    /// Blocks to swap from CPU to GPU
    pub swap_in: Vec<SwapOperation>,

    /// Blocks to copy
    pub copy: Vec<CopyOperation>,
}

impl MemoryOperations {
    /// Check if there are any operations
    pub fn is_empty(&self) -> bool {
        self.swap_out.is_empty() && self.swap_in.is_empty() && self.copy.is_empty()
    }

    /// Add a swap-out operation
    pub fn add_swap_out(&mut self, src: PhysicalBlockId, dst: PhysicalBlockId) {
        self.swap_out.push(SwapOperation {
            src_block: src,
            dst_block: dst,
        });
    }

    /// Add a swap-in operation
    pub fn add_swap_in(&mut self, src: PhysicalBlockId, dst: PhysicalBlockId) {
        self.swap_in.push(SwapOperation {
            src_block: src,
            dst_block: dst,
        });
    }

    /// Add a copy operation
    pub fn add_copy(&mut self, src: PhysicalBlockId, dst: PhysicalBlockId) {
        // Group copies from same source
        if let Some(op) = self.copy.iter_mut().find(|op| op.src_block == src) {
            op.dst_blocks.push(dst);
        } else {
            self.copy.push(CopyOperation {
                src_block: src,
                dst_blocks: vec![dst],
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_attention_config() {
        let config = PagedAttentionConfig::new(32, 8, 128, 16);

        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_query_groups(), 4);
        assert!((config.scale - 1.0 / 128.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_slopes() {
        let slopes = PagedAttention::compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);

        // Slopes should be decreasing powers of 2
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1]);
        }
    }

    #[test]
    fn test_paged_attention_input_prefill() {
        let slot_mapping = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let seq_lens = vec![5, 5];

        let input = PagedAttentionInput::prefill(slot_mapping.clone(), seq_lens);

        assert!(input.is_prefill);
        assert_eq!(input.slot_mapping, slot_mapping);
        assert_eq!(input.max_context_len, 5);
        assert_eq!(input.seq_start_positions, vec![0, 5, 10]);
    }

    #[test]
    fn test_paged_attention_input_decode() {
        let slot_mapping = vec![100, 200];
        let block_tables = vec![vec![0, 1, 2, 3, 4, 5, 6], vec![7, 8, 9, 10, 11, 12]];
        let context_lens = vec![100, 90];

        let input = PagedAttentionInput::decode(
            slot_mapping.clone(),
            block_tables.clone(),
            context_lens.clone(),
        );

        assert!(!input.is_prefill);
        assert_eq!(input.slot_mapping, slot_mapping);
        assert_eq!(input.max_context_len, 100);
        assert_eq!(input.block_tables, Some(block_tables));
        assert_eq!(input.context_lens, Some(context_lens));
    }

    #[test]
    fn test_memory_operations() {
        let mut ops = MemoryOperations::default();

        assert!(ops.is_empty());

        ops.add_swap_out(0, 100);
        ops.add_swap_in(101, 1);
        ops.add_copy(5, 10);
        ops.add_copy(5, 11);

        assert!(!ops.is_empty());
        assert_eq!(ops.swap_out.len(), 1);
        assert_eq!(ops.swap_in.len(), 1);
        assert_eq!(ops.copy.len(), 1);
        assert_eq!(ops.copy[0].dst_blocks, vec![10, 11]);
    }
}
