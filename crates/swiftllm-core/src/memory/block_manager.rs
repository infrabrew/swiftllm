//! Block Manager for PagedAttention
//!
//! This module implements the block allocation and management system
//! that enables PagedAttention's memory-efficient KV cache.
//!
//! Key concepts:
//! - Physical Block: Actual memory allocation on GPU/CPU
//! - Logical Block: Virtual block that sequences reference
//! - Block Table: Mapping from logical to physical blocks

use crate::error::{Error, Result};
use crate::types::SequenceId;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Physical block ID
pub type PhysicalBlockId = usize;

/// Logical block ID
pub type LogicalBlockId = usize;

/// Block table for a sequence (logical to physical mapping)
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Mapping from logical block index to physical block ID
    pub blocks: Vec<PhysicalBlockId>,
}

impl BlockTable {
    /// Create a new empty block table
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Get the number of allocated blocks
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if the block table is empty
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Add a physical block
    pub fn push(&mut self, block_id: PhysicalBlockId) {
        self.blocks.push(block_id);
    }

    /// Get the physical block for a logical index
    pub fn get(&self, logical_idx: usize) -> Option<PhysicalBlockId> {
        self.blocks.get(logical_idx).copied()
    }

    /// Get the last block
    pub fn last(&self) -> Option<PhysicalBlockId> {
        self.blocks.last().copied()
    }

    /// Calculate the number of tokens that can be stored
    pub fn capacity(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }
}

/// Information about a physical block
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// Block ID
    pub id: PhysicalBlockId,

    /// Reference count (for copy-on-write)
    pub ref_count: usize,

    /// Whether this block is on GPU
    pub is_gpu: bool,

    /// Computed hash for prefix caching (if enabled)
    pub hash: Option<u64>,
}

impl PhysicalBlock {
    /// Create a new physical block
    pub fn new(id: PhysicalBlockId, is_gpu: bool) -> Self {
        Self {
            id,
            ref_count: 0,
            is_gpu,
            hash: None,
        }
    }

    /// Increment reference count
    pub fn inc_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count
    pub fn dec_ref(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }
}

/// Block allocator for GPU or CPU memory
#[derive(Debug)]
pub struct BlockAllocator {
    /// Total number of blocks
    num_blocks: usize,

    /// Size of each block in tokens
    block_size: usize,

    /// Free block queue
    free_blocks: VecDeque<PhysicalBlockId>,

    /// Block information
    blocks: Vec<PhysicalBlock>,

    /// Whether this is a GPU allocator
    is_gpu: bool,
}

impl BlockAllocator {
    /// Create a new block allocator
    pub fn new(num_blocks: usize, block_size: usize, is_gpu: bool) -> Self {
        let mut free_blocks = VecDeque::with_capacity(num_blocks);
        let mut blocks = Vec::with_capacity(num_blocks);

        for i in 0..num_blocks {
            free_blocks.push_back(i);
            blocks.push(PhysicalBlock::new(i, is_gpu));
        }

        Self {
            num_blocks,
            block_size,
            free_blocks,
            blocks,
            is_gpu,
        }
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the total number of blocks
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get the number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get the number of used blocks
    pub fn num_used_blocks(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Allocate a block
    pub fn allocate(&mut self) -> Option<PhysicalBlockId> {
        let block_id = self.free_blocks.pop_front()?;
        self.blocks[block_id].inc_ref();
        Some(block_id)
    }

    /// Allocate multiple blocks
    pub fn allocate_many(&mut self, count: usize) -> Option<Vec<PhysicalBlockId>> {
        if self.free_blocks.len() < count {
            return None;
        }

        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(block_id) = self.allocate() {
                allocated.push(block_id);
            } else {
                // Rollback on failure
                for block_id in allocated {
                    self.free(block_id);
                }
                return None;
            }
        }

        Some(allocated)
    }

    /// Free a block
    pub fn free(&mut self, block_id: PhysicalBlockId) {
        if block_id < self.num_blocks {
            if self.blocks[block_id].dec_ref() {
                self.free_blocks.push_back(block_id);
            }
        }
    }

    /// Increment reference count (for copy-on-write)
    pub fn inc_ref(&mut self, block_id: PhysicalBlockId) {
        if block_id < self.num_blocks {
            self.blocks[block_id].inc_ref();
        }
    }

    /// Get the reference count of a block
    pub fn ref_count(&self, block_id: PhysicalBlockId) -> usize {
        if block_id < self.num_blocks {
            self.blocks[block_id].ref_count
        } else {
            0
        }
    }

    /// Check if there are enough free blocks
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.free_blocks.len() >= num_blocks
    }

    /// Get block information
    pub fn get_block(&self, block_id: PhysicalBlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(block_id)
    }

    /// Get utilization (0.0 - 1.0)
    pub fn utilization(&self) -> f32 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        (self.num_blocks - self.free_blocks.len()) as f32 / self.num_blocks as f32
    }
}

/// Block manager that coordinates allocation across GPU and CPU
#[derive(Debug)]
pub struct BlockManager {
    /// Block size in tokens
    block_size: usize,

    /// Number of KV heads
    num_kv_heads: usize,

    /// Head dimension
    head_dim: usize,

    /// Number of layers
    num_layers: usize,

    /// GPU block allocator
    gpu_allocator: Mutex<BlockAllocator>,

    /// CPU block allocator (for swapping)
    cpu_allocator: Mutex<BlockAllocator>,

    /// Sequence to block table mapping
    block_tables: RwLock<HashMap<SequenceId, BlockTable>>,

    /// Enable prefix caching
    enable_prefix_caching: bool,

    /// Prefix cache (hash -> block ID)
    prefix_cache: RwLock<HashMap<u64, PhysicalBlockId>>,

    /// Sliding window size (if enabled)
    sliding_window: Option<usize>,
}

impl BlockManager {
    /// Create a new block manager
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
        enable_prefix_caching: bool,
        sliding_window: Option<usize>,
    ) -> Self {
        Self {
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            gpu_allocator: Mutex::new(BlockAllocator::new(num_gpu_blocks, block_size, true)),
            cpu_allocator: Mutex::new(BlockAllocator::new(num_cpu_blocks, block_size, false)),
            block_tables: RwLock::new(HashMap::new()),
            enable_prefix_caching,
            prefix_cache: RwLock::new(HashMap::new()),
            sliding_window,
        }
    }

    /// Calculate the memory size per block in bytes
    pub fn block_size_bytes(&self) -> usize {
        // Each block stores block_size tokens of KV cache for all layers
        // Memory = 2 (K + V) * num_layers * num_kv_heads * head_dim * block_size * sizeof(dtype)
        // Assuming float16 (2 bytes)
        2 * self.num_layers * self.num_kv_heads * self.head_dim * self.block_size * 2
    }

    /// Calculate number of blocks needed for given number of tokens
    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// Check if we can allocate blocks for a sequence
    pub fn can_allocate(&self, seq_id: SequenceId, num_tokens: usize) -> bool {
        let needed = self.blocks_needed(num_tokens);
        let tables = self.block_tables.read();
        let current = tables
            .get(&seq_id)
            .map(|t| t.len())
            .unwrap_or(0);
        let additional = needed.saturating_sub(current);

        self.gpu_allocator.lock().can_allocate(additional)
    }

    /// Allocate blocks for a sequence
    pub fn allocate(&self, seq_id: SequenceId, num_tokens: usize) -> Result<()> {
        let needed = self.blocks_needed(num_tokens);

        let mut tables = self.block_tables.write();
        let table = tables.entry(seq_id).or_insert_with(BlockTable::new);

        let current = table.len();
        if needed <= current {
            return Ok(());
        }

        let additional = needed - current;
        let mut allocator = self.gpu_allocator.lock();

        for _ in 0..additional {
            let block_id = allocator.allocate().ok_or_else(|| {
                Error::BlockAllocation("No free GPU blocks available".to_string())
            })?;
            table.push(block_id);
        }

        Ok(())
    }

    /// Free blocks for a sequence
    pub fn free(&self, seq_id: SequenceId) {
        let mut tables = self.block_tables.write();
        if let Some(table) = tables.remove(&seq_id) {
            let mut allocator = self.gpu_allocator.lock();
            for block_id in table.blocks {
                allocator.free(block_id);
            }
        }
    }

    /// Free blocks from multiple sequences
    pub fn free_many(&self, seq_ids: &[SequenceId]) {
        let mut tables = self.block_tables.write();
        let mut allocator = self.gpu_allocator.lock();

        for seq_id in seq_ids {
            if let Some(table) = tables.remove(seq_id) {
                for block_id in table.blocks {
                    allocator.free(block_id);
                }
            }
        }
    }

    /// Get the block table for a sequence
    pub fn get_block_table(&self, seq_id: SequenceId) -> Option<BlockTable> {
        self.block_tables.read().get(&seq_id).cloned()
    }

    /// Fork a sequence (for beam search or parallel sampling)
    pub fn fork(&self, source_seq_id: SequenceId, target_seq_id: SequenceId) -> Result<()> {
        let tables = self.block_tables.read();
        let source_table = tables
            .get(&source_seq_id)
            .ok_or_else(|| Error::Internal("Source sequence not found".to_string()))?;

        let mut new_table = BlockTable::new();
        let mut allocator = self.gpu_allocator.lock();

        // Copy-on-write: just increment reference counts
        for &block_id in &source_table.blocks {
            allocator.inc_ref(block_id);
            new_table.push(block_id);
        }

        drop(tables);
        self.block_tables.write().insert(target_seq_id, new_table);

        Ok(())
    }

    /// Copy a block (for copy-on-write)
    pub fn copy_block(
        &self,
        source_block: PhysicalBlockId,
        _seq_id: SequenceId,
    ) -> Result<PhysicalBlockId> {
        let mut allocator = self.gpu_allocator.lock();

        // If ref_count is 1, we can modify in place
        if allocator.ref_count(source_block) == 1 {
            return Ok(source_block);
        }

        // Otherwise, allocate a new block and copy
        let new_block = allocator.allocate().ok_or_else(|| {
            Error::BlockAllocation("No free GPU blocks for copy".to_string())
        })?;

        // Decrement ref count on source
        allocator.free(source_block);

        // TODO: Actually copy the data on GPU
        // This would be done by the CUDA kernel

        Ok(new_block)
    }

    /// Swap blocks from GPU to CPU
    pub fn swap_out(&self, seq_id: SequenceId) -> Result<HashMap<PhysicalBlockId, PhysicalBlockId>> {
        let mut tables = self.block_tables.write();
        let table = tables
            .get_mut(&seq_id)
            .ok_or_else(|| Error::Internal("Sequence not found".to_string()))?;

        let mut gpu_allocator = self.gpu_allocator.lock();
        let mut cpu_allocator = self.cpu_allocator.lock();

        let mut swap_mapping = HashMap::new();

        for block_id in &mut table.blocks {
            let gpu_block = *block_id;

            // Allocate CPU block
            let cpu_block = cpu_allocator.allocate().ok_or_else(|| {
                Error::BlockAllocation("No free CPU blocks for swap".to_string())
            })?;

            swap_mapping.insert(gpu_block, cpu_block);

            // Free GPU block
            gpu_allocator.free(gpu_block);

            // Update block table to point to CPU block
            // Note: We'd need to track that this is a CPU block
            *block_id = cpu_block;
        }

        Ok(swap_mapping)
    }

    /// Swap blocks from CPU to GPU
    pub fn swap_in(&self, seq_id: SequenceId) -> Result<HashMap<PhysicalBlockId, PhysicalBlockId>> {
        let mut tables = self.block_tables.write();
        let table = tables
            .get_mut(&seq_id)
            .ok_or_else(|| Error::Internal("Sequence not found".to_string()))?;

        let mut gpu_allocator = self.gpu_allocator.lock();
        let mut cpu_allocator = self.cpu_allocator.lock();

        let mut swap_mapping = HashMap::new();

        for block_id in &mut table.blocks {
            let cpu_block = *block_id;

            // Allocate GPU block
            let gpu_block = gpu_allocator.allocate().ok_or_else(|| {
                Error::BlockAllocation("No free GPU blocks for swap in".to_string())
            })?;

            swap_mapping.insert(cpu_block, gpu_block);

            // Free CPU block
            cpu_allocator.free(cpu_block);

            // Update block table
            *block_id = gpu_block;
        }

        Ok(swap_mapping)
    }

    /// Get GPU memory utilization
    pub fn gpu_utilization(&self) -> f32 {
        self.gpu_allocator.lock().utilization()
    }

    /// Get CPU swap utilization
    pub fn cpu_utilization(&self) -> f32 {
        self.cpu_allocator.lock().utilization()
    }

    /// Get number of free GPU blocks
    pub fn num_free_gpu_blocks(&self) -> usize {
        self.gpu_allocator.lock().num_free_blocks()
    }

    /// Get number of free CPU blocks
    pub fn num_free_cpu_blocks(&self) -> usize {
        self.cpu_allocator.lock().num_free_blocks()
    }

    /// Get watermark (percentage of blocks to keep free for burstiness)
    pub fn get_watermark(&self, watermark_percent: f32) -> usize {
        let total = self.gpu_allocator.lock().num_blocks();
        (total as f32 * watermark_percent) as usize
    }

    /// Get block manager statistics
    pub fn stats(&self) -> BlockManagerStats {
        let gpu_alloc = self.gpu_allocator.lock();
        let cpu_alloc = self.cpu_allocator.lock();

        BlockManagerStats {
            block_size: self.block_size,
            num_gpu_blocks: gpu_alloc.num_blocks(),
            num_cpu_blocks: cpu_alloc.num_blocks(),
            free_gpu_blocks: gpu_alloc.num_free_blocks(),
            free_cpu_blocks: cpu_alloc.num_free_blocks(),
            gpu_utilization: gpu_alloc.utilization(),
            cpu_utilization: cpu_alloc.utilization(),
            num_sequences: self.block_tables.read().len(),
        }
    }
}

/// Block manager statistics
#[derive(Debug, Clone)]
pub struct BlockManagerStats {
    /// Block size in tokens
    pub block_size: usize,
    /// Total GPU blocks
    pub num_gpu_blocks: usize,
    /// Total CPU blocks
    pub num_cpu_blocks: usize,
    /// Free GPU blocks
    pub free_gpu_blocks: usize,
    /// Free CPU blocks
    pub free_cpu_blocks: usize,
    /// GPU utilization
    pub gpu_utilization: f32,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Number of sequences
    pub num_sequences: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator() {
        let mut allocator = BlockAllocator::new(10, 16, true);

        assert_eq!(allocator.num_free_blocks(), 10);

        let block1 = allocator.allocate().unwrap();
        let block2 = allocator.allocate().unwrap();

        assert_eq!(allocator.num_free_blocks(), 8);
        assert_ne!(block1, block2);

        allocator.free(block1);
        assert_eq!(allocator.num_free_blocks(), 9);
    }

    #[test]
    fn test_block_allocator_ref_count() {
        let mut allocator = BlockAllocator::new(10, 16, true);

        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.ref_count(block), 1);

        allocator.inc_ref(block);
        assert_eq!(allocator.ref_count(block), 2);

        allocator.free(block);
        assert_eq!(allocator.ref_count(block), 1);
        assert_eq!(allocator.num_free_blocks(), 9);

        allocator.free(block);
        assert_eq!(allocator.ref_count(block), 0);
        assert_eq!(allocator.num_free_blocks(), 10);
    }

    #[test]
    fn test_block_manager() {
        let manager = BlockManager::new(
            16,   // block_size
            100,  // num_gpu_blocks
            50,   // num_cpu_blocks
            32,   // num_kv_heads
            128,  // head_dim
            32,   // num_layers
            false, // prefix caching
            None,  // sliding window
        );

        let seq_id = SequenceId::new();

        // Allocate blocks for 100 tokens
        manager.allocate(seq_id, 100).unwrap();

        let table = manager.get_block_table(seq_id).unwrap();
        assert_eq!(table.len(), 7); // ceil(100/16) = 7 blocks

        // Free the sequence
        manager.free(seq_id);
        assert!(manager.get_block_table(seq_id).is_none());
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new();

        table.push(0);
        table.push(5);
        table.push(10);

        assert_eq!(table.len(), 3);
        assert_eq!(table.get(0), Some(0));
        assert_eq!(table.get(1), Some(5));
        assert_eq!(table.get(2), Some(10));
        assert_eq!(table.last(), Some(10));
        assert_eq!(table.capacity(16), 48);
    }
}
