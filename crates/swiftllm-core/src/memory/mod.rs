//! Memory management for SwiftLLM
//!
//! This module provides memory-efficient management of KV cache using
//! PagedAttention, which allows non-contiguous memory allocation and
//! efficient memory sharing between sequences.

pub mod block_manager;
pub mod kv_cache;
pub mod paged_attention;

pub use block_manager::{BlockAllocator, BlockManager, BlockManagerStats, BlockTable};
pub use kv_cache::{BatchedCacheMetadata, KvCache, KvCacheConfig};
pub use paged_attention::{PagedAttention, PagedAttentionConfig};

use crate::config::MemoryConfig;
use crate::error::Result;

/// Memory pool for managing GPU/CPU memory
#[derive(Debug)]
pub struct MemoryPool {
    /// Configuration
    config: MemoryConfig,

    /// GPU memory pool (in bytes)
    gpu_pool_size: usize,

    /// CPU swap pool (in bytes)
    cpu_pool_size: usize,

    /// Currently allocated GPU memory
    gpu_allocated: usize,

    /// Currently allocated CPU memory
    cpu_allocated: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryConfig) -> Self {
        let gpu_pool_size = 0; // Will be set based on available GPU memory
        let cpu_pool_size = (config.swap_space_gib * 1024.0 * 1024.0 * 1024.0) as usize;

        Self {
            config,
            gpu_pool_size,
            cpu_pool_size,
            gpu_allocated: 0,
            cpu_allocated: 0,
        }
    }

    /// Initialize the memory pool with GPU info
    pub fn init(&mut self, total_gpu_memory: usize) -> Result<()> {
        self.gpu_pool_size =
            (total_gpu_memory as f32 * self.config.gpu_memory_utilization) as usize;
        Ok(())
    }

    /// Get available GPU memory
    pub fn available_gpu_memory(&self) -> usize {
        self.gpu_pool_size.saturating_sub(self.gpu_allocated)
    }

    /// Get available CPU swap memory
    pub fn available_cpu_memory(&self) -> usize {
        self.cpu_pool_size.saturating_sub(self.cpu_allocated)
    }

    /// Allocate GPU memory
    pub fn allocate_gpu(&mut self, size: usize) -> Result<()> {
        if self.gpu_allocated + size > self.gpu_pool_size {
            return Err(crate::error::Error::OutOfMemory(
                "GPU memory pool exhausted".to_string(),
            ));
        }
        self.gpu_allocated += size;
        Ok(())
    }

    /// Free GPU memory
    pub fn free_gpu(&mut self, size: usize) {
        self.gpu_allocated = self.gpu_allocated.saturating_sub(size);
    }

    /// Allocate CPU swap memory
    pub fn allocate_cpu(&mut self, size: usize) -> Result<()> {
        if self.cpu_allocated + size > self.cpu_pool_size {
            return Err(crate::error::Error::OutOfMemory(
                "CPU swap pool exhausted".to_string(),
            ));
        }
        self.cpu_allocated += size;
        Ok(())
    }

    /// Free CPU swap memory
    pub fn free_cpu(&mut self, size: usize) {
        self.cpu_allocated = self.cpu_allocated.saturating_sub(size);
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            gpu_pool_size: self.gpu_pool_size,
            gpu_allocated: self.gpu_allocated,
            gpu_available: self.available_gpu_memory(),
            cpu_pool_size: self.cpu_pool_size,
            cpu_allocated: self.cpu_allocated,
            cpu_available: self.available_cpu_memory(),
            utilization: if self.gpu_pool_size > 0 {
                self.gpu_allocated as f32 / self.gpu_pool_size as f32
            } else {
                0.0
            },
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total GPU pool size in bytes
    pub gpu_pool_size: usize,
    /// Allocated GPU memory in bytes
    pub gpu_allocated: usize,
    /// Available GPU memory in bytes
    pub gpu_available: usize,
    /// Total CPU swap pool size in bytes
    pub cpu_pool_size: usize,
    /// Allocated CPU swap memory in bytes
    pub cpu_allocated: usize,
    /// Available CPU swap memory in bytes
    pub cpu_available: usize,
    /// GPU memory utilization (0.0 - 1.0)
    pub utilization: f32,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU: {:.2} GiB / {:.2} GiB ({:.1}%), CPU Swap: {:.2} GiB / {:.2} GiB",
            self.gpu_allocated as f64 / (1024.0 * 1024.0 * 1024.0),
            self.gpu_pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
            self.utilization * 100.0,
            self.cpu_allocated as f64 / (1024.0 * 1024.0 * 1024.0),
            self.cpu_pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    }
}
