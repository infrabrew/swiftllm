//! KV Cache implementation for transformer models
//!
//! This module provides the KV cache storage and management that works
//! with PagedAttention for efficient memory usage.

use crate::config::DataType;
use crate::error::{Error, Result};
use crate::memory::block_manager::{BlockManager, PhysicalBlockId};
use crate::tensor::{Device, Shape, Tensor};
use crate::types::SequenceId;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for KV cache
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of layers
    pub num_layers: usize,

    /// Number of KV heads (may differ from attention heads due to GQA/MQA)
    pub num_kv_heads: usize,

    /// Dimension of each head
    pub head_dim: usize,

    /// Block size in tokens
    pub block_size: usize,

    /// Data type for cache
    pub dtype: DataType,

    /// Device for cache storage
    pub device: Device,
}

impl KvCacheConfig {
    /// Calculate the size of one block in bytes
    pub fn block_size_bytes(&self) -> usize {
        // Each block stores:
        // - K cache: [block_size, num_kv_heads, head_dim]
        // - V cache: [block_size, num_kv_heads, head_dim]
        2 * self.block_size * self.num_kv_heads * self.head_dim * self.dtype.size_bytes()
    }

    /// Calculate the size of one token's KV cache in bytes
    pub fn token_size_bytes(&self) -> usize {
        2 * self.num_kv_heads * self.head_dim * self.dtype.size_bytes()
    }

    /// Calculate total cache size for given number of tokens
    pub fn total_size_bytes(&self, num_tokens: usize) -> usize {
        self.num_layers * num_tokens * self.token_size_bytes()
    }
}

/// KV cache for a single layer
#[derive(Debug)]
pub struct LayerKvCache {
    /// Key cache tensor
    /// Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    pub key_cache: Tensor,

    /// Value cache tensor
    /// Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    pub value_cache: Tensor,

    /// Layer index
    pub layer_idx: usize,
}

impl LayerKvCache {
    /// Create a new layer KV cache
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DataType,
        device: Device,
        layer_idx: usize,
    ) -> Result<Self> {
        let shape = vec![num_blocks, block_size, num_kv_heads, head_dim];

        let key_cache = Tensor::zeros(shape.clone(), dtype, device)?;
        let value_cache = Tensor::zeros(shape, dtype, device)?;

        Ok(Self {
            key_cache,
            value_cache,
            layer_idx,
        })
    }

    /// Get the shape of the cache
    pub fn shape(&self) -> &Shape {
        self.key_cache.shape()
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.key_cache.dims()[0]
    }
}

/// Full KV cache for all layers
#[derive(Debug)]
pub struct KvCache {
    /// Configuration
    config: KvCacheConfig,

    /// Layer caches
    layers: Vec<LayerKvCache>,

    /// Block manager for allocation
    block_manager: Arc<BlockManager>,
}

impl KvCache {
    /// Create a new KV cache
    pub fn new(config: KvCacheConfig, block_manager: Arc<BlockManager>) -> Result<Self> {
        let num_blocks = block_manager.num_free_gpu_blocks();
        let mut layers = Vec::with_capacity(config.num_layers);

        for layer_idx in 0..config.num_layers {
            let layer_cache = LayerKvCache::new(
                num_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                config.dtype,
                config.device,
                layer_idx,
            )?;
            layers.push(layer_cache);
        }

        Ok(Self {
            config,
            layers,
            block_manager,
        })
    }

    /// Get the config
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Get layer cache by index
    pub fn get_layer(&self, layer_idx: usize) -> Option<&LayerKvCache> {
        self.layers.get(layer_idx)
    }

    /// Get mutable layer cache by index
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> Option<&mut LayerKvCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Get all layer caches
    pub fn layers(&self) -> &[LayerKvCache] {
        &self.layers
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Allocate cache for a sequence
    pub fn allocate_sequence(&self, seq_id: SequenceId, num_tokens: usize) -> Result<()> {
        self.block_manager.allocate(seq_id, num_tokens)
    }

    /// Free cache for a sequence
    pub fn free_sequence(&self, seq_id: SequenceId) {
        self.block_manager.free(seq_id);
    }

    /// Get slot mapping for PagedAttention
    /// Maps each token position to its physical slot in the cache
    pub fn get_slot_mapping(
        &self,
        seq_id: SequenceId,
        start_pos: usize,
        num_tokens: usize,
    ) -> Result<Vec<usize>> {
        let block_table = self.block_manager.get_block_table(seq_id).ok_or_else(|| {
            Error::KvCache(format!("Block table not found for sequence {:?}", seq_id))
        })?;

        let block_size = self.config.block_size;
        let mut slot_mapping = Vec::with_capacity(num_tokens);

        for pos in start_pos..start_pos + num_tokens {
            let block_idx = pos / block_size;
            let block_offset = pos % block_size;

            let physical_block = block_table.get(block_idx).ok_or_else(|| {
                Error::KvCache(format!(
                    "Block {} not allocated for sequence {:?}",
                    block_idx, seq_id
                ))
            })?;

            // Slot = physical_block * block_size + block_offset
            let slot = physical_block * block_size + block_offset;
            slot_mapping.push(slot);
        }

        Ok(slot_mapping)
    }
}

/// Cache metadata for a sequence
#[derive(Debug, Clone)]
pub struct SequenceCacheMetadata {
    /// Sequence ID
    pub seq_id: SequenceId,

    /// Current cache length (number of tokens cached)
    pub cache_len: usize,

    /// Block table (logical to physical mapping)
    pub block_table: Vec<PhysicalBlockId>,
}

impl SequenceCacheMetadata {
    /// Create new cache metadata
    pub fn new(seq_id: SequenceId) -> Self {
        Self {
            seq_id,
            cache_len: 0,
            block_table: Vec::new(),
        }
    }

    /// Get the number of blocks allocated
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }
}

/// Batched cache metadata for model execution
#[derive(Debug)]
pub struct BatchedCacheMetadata {
    /// Cache metadata for each sequence in the batch
    pub sequences: Vec<SequenceCacheMetadata>,

    /// Slot mapping for the entire batch (flattened)
    pub slot_mapping: Vec<usize>,

    /// Context lengths for each sequence
    pub context_lens: Vec<usize>,

    /// Maximum context length in batch
    pub max_context_len: usize,

    /// Block tables (one per sequence)
    pub block_tables: Vec<Vec<PhysicalBlockId>>,
}

impl BatchedCacheMetadata {
    /// Create new empty batch metadata
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            slot_mapping: Vec::new(),
            context_lens: Vec::new(),
            max_context_len: 0,
            block_tables: Vec::new(),
        }
    }

    /// Add a sequence to the batch
    pub fn add_sequence(&mut self, metadata: SequenceCacheMetadata, slot_mapping: Vec<usize>) {
        self.context_lens.push(metadata.cache_len);
        self.max_context_len = self.max_context_len.max(metadata.cache_len);
        self.block_tables.push(metadata.block_table.clone());
        self.slot_mapping.extend(slot_mapping);
        self.sequences.push(metadata);
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.sequences.len()
    }
}

impl Default for BatchedCacheMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_config() {
        let config = KvCacheConfig {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            dtype: DataType::Float16,
            device: Device::Cpu,
        };

        // Token size = 2 * 8 * 128 * 2 = 4096 bytes
        assert_eq!(config.token_size_bytes(), 4096);

        // Block size = 2 * 16 * 8 * 128 * 2 = 65536 bytes
        assert_eq!(config.block_size_bytes(), 65536);
    }

    #[test]
    fn test_sequence_cache_metadata() {
        let seq_id = SequenceId::new();
        let mut metadata = SequenceCacheMetadata::new(seq_id);

        assert_eq!(metadata.cache_len, 0);
        assert_eq!(metadata.num_blocks(), 0);

        metadata.block_table.push(0);
        metadata.block_table.push(5);
        metadata.cache_len = 32;

        assert_eq!(metadata.num_blocks(), 2);
    }

    #[test]
    fn test_batched_cache_metadata() {
        let mut batch = BatchedCacheMetadata::new();

        let seq1 = SequenceCacheMetadata {
            seq_id: SequenceId::new(),
            cache_len: 100,
            block_table: vec![0, 1, 2, 3, 4, 5, 6],
        };

        let seq2 = SequenceCacheMetadata {
            seq_id: SequenceId::new(),
            cache_len: 50,
            block_table: vec![7, 8, 9, 10],
        };

        batch.add_sequence(seq1, vec![0, 1, 2]);
        batch.add_sequence(seq2, vec![3, 4]);

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.max_context_len, 100);
        assert_eq!(batch.context_lens, vec![100, 50]);
        assert_eq!(batch.slot_mapping.len(), 5);
    }
}
