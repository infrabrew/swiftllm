//! SwiftLLM Inference Engine
//!
//! This is the main entry point for LLM inference. The engine coordinates
//! all components including the scheduler, memory manager, model executor,
//! and sampling strategies.

use crate::config::{EngineConfig, SamplingConfig};
use crate::error::{Error, Result};
use crate::execution::{ExecutionConfig, ExecutionStats, ModelExecutor};
use crate::memory::{BlockManager, KvCache, KvCacheConfig, MemoryPool, MemoryStats};
use crate::sampling::{SamplingParams, TokenSampler};
use crate::scheduler::{Scheduler, SchedulerStats};
use crate::types::{
    FinishReason, GenerationOutput, Request, RequestId, RequestMetrics, RequestOutput,
    RequestStatus, SequenceGroup, Token, TokenId,
};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Notify};

/// The main inference engine
pub struct Engine {
    /// Engine configuration
    config: EngineConfig,

    /// Scheduler for request management
    scheduler: Arc<Scheduler>,

    /// Block manager for memory allocation
    block_manager: Arc<BlockManager>,

    /// Memory pool
    memory_pool: Mutex<MemoryPool>,

    /// Request outputs (completed requests)
    outputs: RwLock<HashMap<RequestId, RequestOutput>>,

    /// Token samplers per request
    samplers: RwLock<HashMap<RequestId, TokenSampler>>,

    /// Running flag
    running: AtomicBool,

    /// Current step counter
    step_counter: AtomicUsize,

    /// Execution statistics
    exec_stats: Mutex<ExecutionStats>,

    /// Shutdown notification
    shutdown: Arc<Notify>,

    /// Output sender for async streaming
    output_tx: Option<mpsc::UnboundedSender<RequestOutput>>,
}

impl Engine {
    /// Create a new inference engine
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Calculate memory requirements
        let block_size = config.memory.block_size;

        // TODO: Query actual GPU memory
        let total_gpu_memory: usize = 16 * 1024 * 1024 * 1024; // 16 GB placeholder
        let usable_memory =
            (total_gpu_memory as f32 * config.memory.gpu_memory_utilization) as usize;

        // Calculate number of blocks
        // For now, use placeholder model dimensions
        let num_layers = 32;
        let num_kv_heads = 8;
        let head_dim = 128;

        let block_size_bytes = 2 * num_layers * num_kv_heads * head_dim * block_size * 2; // float16
        let num_gpu_blocks = usable_memory / block_size_bytes;
        let num_cpu_blocks = (config.memory.swap_space_gib * 1024.0 * 1024.0 * 1024.0) as usize
            / block_size_bytes;

        tracing::info!(
            "Allocating {} GPU blocks and {} CPU blocks ({} tokens/block)",
            num_gpu_blocks,
            num_cpu_blocks,
            block_size
        );

        // Create block manager
        let block_manager = Arc::new(BlockManager::new(
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            num_kv_heads,
            head_dim,
            num_layers,
            config.memory.enable_prefix_caching,
            config.memory.sliding_window,
        ));

        // Create scheduler
        let scheduler = Arc::new(Scheduler::new(config.scheduler.clone(), block_manager.clone()));

        // Create memory pool
        let memory_pool = MemoryPool::new(config.memory.clone());

        Ok(Self {
            config,
            scheduler,
            block_manager,
            memory_pool: Mutex::new(memory_pool),
            outputs: RwLock::new(HashMap::new()),
            samplers: RwLock::new(HashMap::new()),
            running: AtomicBool::new(false),
            step_counter: AtomicUsize::new(0),
            exec_stats: Mutex::new(ExecutionStats::default()),
            shutdown: Arc::new(Notify::new()),
            output_tx: None,
        })
    }

    /// Add a request to the engine
    pub fn add_request(&self, request: Request) -> Result<RequestId> {
        let request_id = request.id;

        // Create sampler for this request
        let params = SamplingParams::from(&request.sampling_params);
        let sampler = TokenSampler::new(params);
        self.samplers.write().insert(request_id, sampler);

        // Add to scheduler
        self.scheduler.add_request(request)?;

        Ok(request_id)
    }

    /// Add a request with text prompt (requires tokenizer)
    pub fn add_request_text(
        &self,
        prompt: String,
        sampling_params: SamplingConfig,
    ) -> Result<RequestId> {
        // In a real implementation, we would tokenize here
        // For now, return an error
        Err(Error::not_implemented("Text tokenization"))
    }

    /// Abort a request
    pub fn abort_request(&self, request_id: RequestId) -> Result<()> {
        self.scheduler.abort_request(request_id)?;
        self.samplers.write().remove(&request_id);
        Ok(())
    }

    /// Get the output for a completed request
    pub fn get_output(&self, request_id: RequestId) -> Option<RequestOutput> {
        self.outputs.read().get(&request_id).cloned()
    }

    /// Run one step of the engine
    pub fn step(&self) -> Result<Vec<RequestOutput>> {
        // Schedule next batch
        let scheduler_output = self.scheduler.schedule();

        if scheduler_output.scheduled_groups.is_empty()
            && scheduler_output.blocks_to_swap_in.is_empty()
            && scheduler_output.blocks_to_swap_out.is_empty()
        {
            // Nothing to do
            return Ok(Vec::new());
        }

        // Execute memory operations
        // In a real implementation, this would trigger CUDA memory copies
        if !scheduler_output.blocks_to_swap_in.is_empty() {
            tracing::debug!(
                "Swapping in {} blocks",
                scheduler_output.blocks_to_swap_in.len()
            );
        }
        if !scheduler_output.blocks_to_swap_out.is_empty() {
            tracing::debug!(
                "Swapping out {} blocks",
                scheduler_output.blocks_to_swap_out.len()
            );
        }

        // In a real implementation, we would:
        // 1. Build the execution batch
        // 2. Run model forward pass
        // 3. Sample tokens
        // 4. Update sequences

        // For now, increment step counter
        self.step_counter.fetch_add(1, Ordering::Relaxed);

        // Return empty outputs (placeholder)
        Ok(Vec::new())
    }

    /// Run the engine loop asynchronously
    pub async fn run(&self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);

        tracing::info!("Starting inference engine");

        while self.running.load(Ordering::SeqCst) {
            // Check for shutdown
            tokio::select! {
                _ = self.shutdown.notified() => {
                    tracing::info!("Shutdown signal received");
                    break;
                }
                _ = tokio::time::sleep(Duration::from_micros(100)) => {
                    // Run a step
                    match self.step() {
                        Ok(outputs) => {
                            for output in outputs {
                                if let Some(tx) = &self.output_tx {
                                    let _ = tx.send(output.clone());
                                }
                                if output.finished {
                                    self.outputs.write().insert(output.request_id, output);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Engine step error: {}", e);
                        }
                    }

                    // Check if scheduler is empty
                    if self.scheduler.is_empty() {
                        // Wait a bit before checking again
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
            }
        }

        self.running.store(false, Ordering::SeqCst);
        tracing::info!("Inference engine stopped");

        Ok(())
    }

    /// Stop the engine
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.shutdown.notify_waiters();
    }

    /// Check if the engine is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get engine statistics
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            scheduler: self.scheduler.stats(),
            execution: self.exec_stats.lock().clone(),
            memory: self.memory_pool.lock().stats(),
            block_manager: self.block_manager.stats(),
            step_count: self.step_counter.load(Ordering::Relaxed),
        }
    }

    /// Get scheduler statistics
    pub fn scheduler_stats(&self) -> SchedulerStats {
        self.scheduler.stats()
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory_pool.lock().stats()
    }

    /// Get execution statistics
    pub fn execution_stats(&self) -> ExecutionStats {
        self.exec_stats.lock().clone()
    }

    /// Wait for a specific request to complete
    pub async fn wait_for_request(
        &self,
        request_id: RequestId,
        timeout: Duration,
    ) -> Result<RequestOutput> {
        let deadline = Instant::now() + timeout;

        loop {
            // Check if output is available
            if let Some(output) = self.outputs.read().get(&request_id).cloned() {
                return Ok(output);
            }

            // Check timeout
            if Instant::now() >= deadline {
                return Err(Error::RequestTimeout(request_id.to_string()));
            }

            // Wait a bit
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Generate tokens synchronously (blocking)
    pub fn generate_sync(
        &self,
        prompt_tokens: Vec<TokenId>,
        sampling_params: SamplingConfig,
    ) -> Result<GenerationOutput> {
        // Create request
        let request = Request::new(prompt_tokens).with_sampling_params(sampling_params);
        let request_id = request.id;

        // Add request
        self.add_request(request)?;

        // Run until complete
        let timeout = Duration::from_secs(self.config.scheduler.request_timeout_secs);
        let deadline = Instant::now() + timeout;

        while Instant::now() < deadline {
            self.step()?;

            // Check if complete
            if let Some(output) = self.outputs.read().get(&request_id) {
                if output.finished {
                    return output.outputs.first().cloned().ok_or_else(|| {
                        Error::Internal("No output generated".to_string())
                    });
                }
            }
        }

        Err(Error::RequestTimeout(request_id.to_string()))
    }

    /// Get the configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get pending request count
    pub fn pending_requests(&self) -> usize {
        self.scheduler.num_waiting()
    }

    /// Get running request count
    pub fn running_requests(&self) -> usize {
        self.scheduler.num_running()
    }
}

/// Combined engine statistics
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Scheduler statistics
    pub scheduler: SchedulerStats,

    /// Execution statistics
    pub execution: ExecutionStats,

    /// Memory statistics
    pub memory: MemoryStats,

    /// Block manager statistics
    pub block_manager: crate::memory::block_manager::BlockManagerStats,

    /// Total steps executed
    pub step_count: usize,
}

impl std::fmt::Display for EngineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Engine Statistics:")?;
        writeln!(
            f,
            "  Steps: {}, Running: {}, Waiting: {}",
            self.step_count, self.scheduler.running_requests, self.scheduler.waiting_requests
        )?;
        writeln!(
            f,
            "  Throughput: {:.2} tokens/s",
            self.execution.tokens_per_second
        )?;
        writeln!(f, "  Memory: {}", self.memory)?;
        writeln!(
            f,
            "  Blocks: GPU {:.1}% used, CPU {:.1}% used",
            self.block_manager.gpu_utilization * 100.0,
            self.block_manager.cpu_utilization * 100.0
        )?;
        Ok(())
    }
}

/// Builder for creating engine instances
pub struct EngineBuilder {
    config: EngineConfig,
}

impl EngineBuilder {
    /// Create a new engine builder
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    /// Set the model path
    pub fn model(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.config.model.path = path.into();
        self
    }

    /// Set tensor parallel size
    pub fn tensor_parallel(mut self, size: usize) -> Self {
        self.config.device.tensor_parallel_size = size;
        self
    }

    /// Set maximum sequence length
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.config.model.max_seq_len = len;
        self
    }

    /// Set block size
    pub fn block_size(mut self, size: usize) -> Self {
        self.config.memory.block_size = size;
        self
    }

    /// Set GPU memory utilization
    pub fn gpu_memory_utilization(mut self, util: f32) -> Self {
        self.config.memory.gpu_memory_utilization = util;
        self
    }

    /// Enable speculative decoding
    pub fn speculative_decoding(
        mut self,
        draft_model: impl Into<std::path::PathBuf>,
    ) -> Self {
        self.config.speculative = Some(crate::config::SpeculativeConfig {
            draft_model_path: draft_model.into(),
            ..Default::default()
        });
        self
    }

    /// Build the engine
    pub fn build(self) -> Result<Engine> {
        Engine::new(self.config)
    }
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = Engine::new(config).unwrap();

        assert!(!engine.is_running());
        assert_eq!(engine.pending_requests(), 0);
    }

    #[test]
    fn test_add_request() {
        let config = EngineConfig::default();
        let engine = Engine::new(config).unwrap();

        let request = Request::new(vec![1, 2, 3, 4, 5]);
        let request_id = engine.add_request(request).unwrap();

        assert_eq!(engine.pending_requests(), 1);
    }

    #[test]
    fn test_abort_request() {
        let config = EngineConfig::default();
        let engine = Engine::new(config).unwrap();

        let request = Request::new(vec![1, 2, 3, 4, 5]);
        let request_id = engine.add_request(request).unwrap();

        engine.abort_request(request_id).unwrap();
        assert_eq!(engine.pending_requests(), 0);
    }

    #[test]
    fn test_engine_builder() {
        let engine = EngineBuilder::new()
            .model("test-model")
            .tensor_parallel(2)
            .max_seq_len(8192)
            .block_size(32)
            .gpu_memory_utilization(0.85)
            .build()
            .unwrap();

        assert_eq!(engine.config().device.tensor_parallel_size, 2);
        assert_eq!(engine.config().model.max_seq_len, 8192);
    }

    #[test]
    fn test_engine_step() {
        let config = EngineConfig::default();
        let engine = Engine::new(config).unwrap();

        // Empty step should return empty outputs
        let outputs = engine.step().unwrap();
        assert!(outputs.is_empty());
    }
}
