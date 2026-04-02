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
            return Ok(Vec::new());
        }

        // Log memory operations
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

        let step_start = Instant::now();
        let mut finished_outputs = Vec::new();

        // Process each scheduled sequence group
        for scheduled in &scheduler_output.scheduled_groups {
            let seq_group = &scheduled.seq_group;
            let request_id = seq_group.request_id;

            for seq in &seq_group.sequences {
                if seq.is_finished() {
                    continue;
                }

                // Build a synthetic logit vector for sampling
                // In a full implementation, this comes from the model forward pass.
                // Here we simulate logits based on a simple vocabulary distribution.
                let vocab_size = 32000; // Standard LLaMA vocab size
                let mut logits = vec![0.0f32; vocab_size];

                // Create a deterministic but varied distribution based on sequence state
                let seed_val = seq.id.as_u64().wrapping_mul(31)
                    .wrapping_add(seq.len() as u64)
                    .wrapping_add(self.step_counter.load(Ordering::Relaxed) as u64);
                let mut rng_state = seed_val;
                for logit in logits.iter_mut() {
                    // Simple xorshift for deterministic pseudo-random logits
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    *logit = ((rng_state as f32) / (u64::MAX as f32)) * 10.0 - 5.0;
                }

                // Use the sampler to pick a token
                let samplers = self.samplers.read();
                if let Some(sampler) = samplers.get(&request_id) {
                    // We need write access for sampling (RNG state)
                    drop(samplers);
                    let mut samplers = self.samplers.write();
                    if let Some(sampler) = samplers.get_mut(&request_id) {
                        match sampler.sample(&logits) {
                            Ok(token) => {
                                let token_id = token.id;

                                // Check finish conditions
                                let max_tokens = seq_group.sampling_params.max_tokens;
                                let generated = seq.num_generated() + 1;

                                let is_eos = token_id == 2; // Common EOS token ID
                                let hit_max = generated >= max_tokens;

                                let finish_reason = if is_eos {
                                    Some(FinishReason::Stop)
                                } else if hit_max {
                                    Some(FinishReason::Length)
                                } else {
                                    None
                                };

                                // Build output if finished or streaming
                                if finish_reason.is_some() {
                                    let output = RequestOutput {
                                        request_id,
                                        outputs: vec![GenerationOutput {
                                            index: 0,
                                            text: String::new(), // Would need detokenizer
                                            token_ids: seq.output_tokens.iter()
                                                .map(|t| t.id)
                                                .chain(std::iter::once(token_id))
                                                .collect(),
                                            cumulative_logprob: seq.cumulative_logprob
                                                + token.logprob.unwrap_or(0.0),
                                            logprobs: None,
                                            finish_reason,
                                            stop_reason: None,
                                        }],
                                        finished: true,
                                        prompt: None,
                                        prompt_token_ids: None,
                                        prompt_logprobs: None,
                                        metrics: Some(RequestMetrics {
                                            time_to_first_token: 0.0,
                                            total_time: step_start.elapsed().as_secs_f64(),
                                            prompt_tokens: seq.prompt_len,
                                            generated_tokens: generated,
                                            tokens_per_second: 0.0,
                                        }),
                                    };
                                    finished_outputs.push(output);

                                    // Clean up: free blocks and remove sampler
                                    self.block_manager.free(seq.id);
                                    self.scheduler.finish_request(request_id);
                                }
                            }
                            Err(e) => {
                                tracing::error!("Sampling error for request {}: {}", request_id, e);
                                // Clean up on sampling failure
                                self.block_manager.free(seq.id);
                                self.scheduler.finish_request(request_id);
                            }
                        }
                    }
                }
            }
        }

        // Update execution stats
        let elapsed = step_start.elapsed().as_secs_f64();
        let num_tokens = scheduler_output.num_batched_tokens;
        let is_prefill = scheduler_output.scheduled_groups.iter()
            .any(|sg| sg.seq_group.state == crate::types::SequenceGroupState::Prefill);
        self.exec_stats.lock().update(is_prefill, num_tokens, elapsed);

        // Increment step counter
        self.step_counter.fetch_add(1, Ordering::Relaxed);

        // Clean up samplers for finished requests
        {
            let mut samplers = self.samplers.write();
            for output in &finished_outputs {
                samplers.remove(&output.request_id);
            }
        }

        Ok(finished_outputs)
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
