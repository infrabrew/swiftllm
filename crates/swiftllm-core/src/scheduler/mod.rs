//! Scheduler for SwiftLLM
//!
//! This module implements continuous batching and request scheduling,
//! enabling high-throughput LLM inference by dynamically batching
//! requests and managing memory efficiently.

mod continuous_batching;
mod request_queue;

pub use continuous_batching::{ContinuousBatchingScheduler, SchedulerPolicy};
pub use request_queue::{RequestQueue, RequestQueueConfig};

use crate::config::SchedulerConfig;
use crate::error::{Error, Result};
use crate::memory::BlockManager;
use crate::types::{
    Request, RequestId, RequestStatus, ScheduledSequenceGroup, SchedulerOutput, SequenceGroup,
    SequenceGroupState,
};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total number of requests processed
    pub total_requests: usize,

    /// Number of requests currently running
    pub running_requests: usize,

    /// Number of requests waiting in queue
    pub waiting_requests: usize,

    /// Number of preempted requests
    pub preempted_requests: usize,

    /// Number of requests completed
    pub completed_requests: usize,

    /// Number of requests failed
    pub failed_requests: usize,

    /// Average wait time in seconds
    pub avg_wait_time_secs: f64,

    /// Average time to first token in seconds
    pub avg_ttft_secs: f64,

    /// Total tokens processed
    pub total_tokens_processed: usize,

    /// Throughput (tokens per second)
    pub throughput_tps: f64,
}

/// Preemption mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionMode {
    /// Recompute KV cache after preemption
    Recompute,
    /// Swap KV cache to CPU memory
    Swap,
}

impl From<crate::config::PreemptionMode> for PreemptionMode {
    fn from(mode: crate::config::PreemptionMode) -> Self {
        match mode {
            crate::config::PreemptionMode::Recompute => PreemptionMode::Recompute,
            crate::config::PreemptionMode::Swap => PreemptionMode::Swap,
        }
    }
}

/// Main scheduler interface
pub struct Scheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Block manager for memory allocation
    block_manager: Arc<BlockManager>,

    /// Waiting queue (requests not yet started)
    waiting: RwLock<VecDeque<SequenceGroup>>,

    /// Running sequences
    running: RwLock<Vec<SequenceGroup>>,

    /// Swapped out sequences (on CPU)
    swapped: RwLock<Vec<SequenceGroup>>,

    /// Request metadata
    request_metadata: RwLock<HashMap<RequestId, RequestMetadata>>,

    /// Statistics
    stats: Mutex<SchedulerStats>,

    /// Preemption mode
    preemption_mode: PreemptionMode,
}

/// Metadata for tracking request lifecycle
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    /// Request ID
    pub request_id: RequestId,

    /// Arrival time
    pub arrival_time: Instant,

    /// First token time (when generation started)
    pub first_token_time: Option<Instant>,

    /// Completion time
    pub completion_time: Option<Instant>,

    /// Number of prompt tokens
    pub prompt_tokens: usize,

    /// Number of generated tokens
    pub generated_tokens: usize,

    /// Number of times preempted
    pub preemption_count: usize,

    /// Current status
    pub status: RequestStatus,
}

impl RequestMetadata {
    /// Create new request metadata
    pub fn new(request: &Request) -> Self {
        Self {
            request_id: request.id,
            arrival_time: request.arrival_time,
            first_token_time: None,
            completion_time: None,
            prompt_tokens: request.prompt_token_ids.len(),
            generated_tokens: 0,
            preemption_count: 0,
            status: RequestStatus::Pending,
        }
    }

    /// Get time to first token in seconds
    pub fn ttft_secs(&self) -> Option<f64> {
        self.first_token_time
            .map(|t| t.duration_since(self.arrival_time).as_secs_f64())
    }

    /// Get total time in seconds
    pub fn total_time_secs(&self) -> Option<f64> {
        self.completion_time
            .map(|t| t.duration_since(self.arrival_time).as_secs_f64())
    }
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(config: SchedulerConfig, block_manager: Arc<BlockManager>) -> Self {
        let preemption_mode = config.preemption_mode.into();

        Self {
            config,
            block_manager,
            waiting: RwLock::new(VecDeque::new()),
            running: RwLock::new(Vec::new()),
            swapped: RwLock::new(Vec::new()),
            request_metadata: RwLock::new(HashMap::new()),
            stats: Mutex::new(SchedulerStats::default()),
            preemption_mode,
        }
    }

    /// Add a new request to the scheduler
    pub fn add_request(&self, request: Request) -> Result<()> {
        let request_id = request.id;

        // Create metadata
        let metadata = RequestMetadata::new(&request);
        self.request_metadata
            .write()
            .insert(request_id, metadata);

        // Create sequence group
        let seq_group = SequenceGroup::new(&request);

        // Add to waiting queue
        self.waiting.write().push_back(seq_group);

        // Update stats
        let mut stats = self.stats.lock();
        stats.total_requests += 1;
        stats.waiting_requests += 1;

        Ok(())
    }

    /// Abort a request
    pub fn abort_request(&self, request_id: RequestId) -> Result<()> {
        // Remove from waiting
        let mut waiting = self.waiting.write();
        if let Some(idx) = waiting
            .iter()
            .position(|sg| sg.request_id == request_id)
        {
            waiting.remove(idx);
            self.update_request_status(request_id, RequestStatus::Cancelled);
            return Ok(());
        }
        drop(waiting);

        // Remove from running
        let mut running = self.running.write();
        if let Some(idx) = running
            .iter()
            .position(|sg| sg.request_id == request_id)
        {
            let seq_group = running.remove(idx);
            // Free blocks
            for seq in &seq_group.sequences {
                self.block_manager.free(seq.id);
            }
            self.update_request_status(request_id, RequestStatus::Cancelled);
            return Ok(());
        }
        drop(running);

        // Remove from swapped
        let mut swapped = self.swapped.write();
        if let Some(idx) = swapped
            .iter()
            .position(|sg| sg.request_id == request_id)
        {
            swapped.remove(idx);
            self.update_request_status(request_id, RequestStatus::Cancelled);
            return Ok(());
        }

        Err(Error::RequestNotFound(request_id.to_string()))
    }

    /// Schedule the next batch
    pub fn schedule(&self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();

        // First, try to swap in sequences from CPU
        self.schedule_swapped(&mut output);

        // Then, schedule running sequences
        self.schedule_running(&mut output);

        // Finally, schedule waiting sequences
        self.schedule_waiting(&mut output);

        output
    }

    /// Schedule swapped sequences
    fn schedule_swapped(&self, output: &mut SchedulerOutput) {
        let mut swapped = self.swapped.write();
        let mut running = self.running.write();

        let mut to_swap_in = Vec::new();

        for seq_group in swapped.iter() {
            // Check if we can swap in
            let num_tokens = seq_group.total_tokens();
            let blocks_needed = self.block_manager.blocks_needed(num_tokens);

            if self.block_manager.num_free_gpu_blocks() >= blocks_needed {
                to_swap_in.push(seq_group.request_id);
            } else {
                break; // Not enough memory, stop trying
            }
        }

        // Swap in sequences
        for request_id in to_swap_in {
            if let Some(idx) = swapped
                .iter()
                .position(|sg| sg.request_id == request_id)
            {
                let mut seq_group = swapped.remove(idx);

                // Swap in blocks
                for seq in &seq_group.sequences {
                    if let Ok(swap_map) = self.block_manager.swap_in(seq.id) {
                        for (cpu_block, gpu_block) in swap_map {
                            output.blocks_to_swap_in.insert(cpu_block, gpu_block);
                        }
                    }
                }

                seq_group.state = SequenceGroupState::Decode;
                running.push(seq_group);
            }
        }
    }

    /// Schedule running sequences
    fn schedule_running(&self, output: &mut SchedulerOutput) {
        let mut running = self.running.write();
        let mut swapped = self.swapped.write();

        // Check if we need to preempt any sequences
        let free_blocks = self.block_manager.num_free_gpu_blocks();
        let running_count = running.len();

        if free_blocks == 0 && running_count > 0 {
            // Need to preempt - use FCFS (preempt last arrived)
            if self.config.enable_preemption {
                // Sort by arrival time (earliest first)
                running.sort_by(|a, b| a.arrival_time.cmp(&b.arrival_time));

                // Preempt the last one
                if let Some(seq_group) = running.pop() {
                    let request_id = seq_group.request_id;

                    match self.preemption_mode {
                        PreemptionMode::Swap => {
                            // Swap to CPU
                            for seq in &seq_group.sequences {
                                if let Ok(swap_map) = self.block_manager.swap_out(seq.id) {
                                    for (gpu_block, cpu_block) in swap_map {
                                        output.blocks_to_swap_out.insert(gpu_block, cpu_block);
                                    }
                                }
                            }
                            swapped.push(seq_group);
                        }
                        PreemptionMode::Recompute => {
                            // Free blocks and re-queue
                            for seq in &seq_group.sequences {
                                self.block_manager.free(seq.id);
                            }
                            output.preempted.push(seq_group);
                        }
                    }

                    // Update metadata
                    if let Some(metadata) = self.request_metadata.write().get_mut(&request_id) {
                        metadata.preemption_count += 1;
                        metadata.status = RequestStatus::Preempted;
                    }
                }
            }
        }

        // Schedule all running sequences
        for seq_group in running.iter() {
            if !seq_group.is_finished() {
                // Allocate block for next token if needed
                let current_len = seq_group.total_tokens();
                for seq in &seq_group.sequences {
                    if !seq.is_finished() {
                        let _ = self.block_manager.allocate(seq.id, current_len + 1);
                    }
                }
            }
        }
    }

    /// Schedule waiting sequences
    fn schedule_waiting(&self, output: &mut SchedulerOutput) {
        let mut waiting = self.waiting.write();
        let mut running = self.running.write();

        let max_num_seqs = self.config.max_num_seqs;
        let max_num_batched_tokens = self.config.max_num_batched_tokens;

        let mut num_scheduled = running.len();
        let mut num_batched_tokens = running
            .iter()
            .map(|sg| sg.num_unfinished())
            .sum::<usize>();

        let mut to_schedule = Vec::new();

        while let Some(seq_group) = waiting.front() {
            // Check constraints
            if num_scheduled >= max_num_seqs {
                break;
            }

            let prompt_len = seq_group.prompt_len();
            if num_batched_tokens + prompt_len > max_num_batched_tokens {
                break;
            }

            // Check if we can allocate blocks
            let blocks_needed = self.block_manager.blocks_needed(prompt_len);
            if self.block_manager.num_free_gpu_blocks() < blocks_needed {
                break;
            }

            // Schedule this sequence group
            let seq_group = waiting.pop_front().unwrap();
            let request_id = seq_group.request_id;

            // Allocate blocks
            for seq in &seq_group.sequences {
                if let Err(e) = self.block_manager.allocate(seq.id, prompt_len) {
                    // If allocation fails, put back in waiting queue
                    tracing::warn!("Failed to allocate blocks: {}", e);
                    waiting.push_front(seq_group);
                    return;
                }
            }

            to_schedule.push(seq_group);
            num_scheduled += 1;
            num_batched_tokens += prompt_len;

            // Update metadata
            if let Some(metadata) = self.request_metadata.write().get_mut(&request_id) {
                metadata.status = RequestStatus::Running;
                metadata.first_token_time = Some(Instant::now());
            }
        }

        // Move scheduled to running
        for seq_group in to_schedule {
            running.push(seq_group);
        }

        // Update stats
        let mut stats = self.stats.lock();
        stats.running_requests = running.len();
        stats.waiting_requests = waiting.len();

        output.num_batched_tokens = num_batched_tokens;
    }

    /// Mark a sequence group as finished
    pub fn finish_request(&self, request_id: RequestId) {
        let mut running = self.running.write();

        if let Some(idx) = running
            .iter()
            .position(|sg| sg.request_id == request_id)
        {
            let seq_group = running.remove(idx);

            // Free blocks
            for seq in &seq_group.sequences {
                self.block_manager.free(seq.id);
            }

            // Update metadata
            let mut metadata = self.request_metadata.write();
            if let Some(meta) = metadata.get_mut(&request_id) {
                meta.status = RequestStatus::Completed;
                meta.completion_time = Some(Instant::now());
            }

            // Update stats
            let mut stats = self.stats.lock();
            stats.completed_requests += 1;
            stats.running_requests = running.len();
        }
    }

    /// Update request status
    fn update_request_status(&self, request_id: RequestId, status: RequestStatus) {
        if let Some(metadata) = self.request_metadata.write().get_mut(&request_id) {
            metadata.status = status;
            if status.is_finished() {
                metadata.completion_time = Some(Instant::now());
            }
        }
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().clone()
    }

    /// Get the number of running requests
    pub fn num_running(&self) -> usize {
        self.running.read().len()
    }

    /// Get the number of waiting requests
    pub fn num_waiting(&self) -> usize {
        self.waiting.read().len()
    }

    /// Get the number of swapped requests
    pub fn num_swapped(&self) -> usize {
        self.swapped.read().len()
    }

    /// Check if the scheduler is empty
    pub fn is_empty(&self) -> bool {
        self.running.read().is_empty()
            && self.waiting.read().is_empty()
            && self.swapped.read().is_empty()
    }

    /// Get request metadata
    pub fn get_request_metadata(&self, request_id: RequestId) -> Option<RequestMetadata> {
        self.request_metadata.read().get(&request_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;

    fn create_test_scheduler() -> Scheduler {
        let block_manager = Arc::new(BlockManager::new(
            16,    // block_size
            1000,  // num_gpu_blocks
            500,   // num_cpu_blocks
            32,    // num_kv_heads
            128,   // head_dim
            32,    // num_layers
            false, // prefix caching
            None,  // sliding window
        ));

        let config = SchedulerConfig::default();
        Scheduler::new(config, block_manager)
    }

    #[test]
    fn test_add_request() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3, 4, 5]);
        scheduler.add_request(request).unwrap();

        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_schedule() {
        let scheduler = create_test_scheduler();

        // Add a request
        let request = Request::new(vec![1, 2, 3, 4, 5]);
        let request_id = request.id;
        scheduler.add_request(request).unwrap();

        // Schedule
        let output = scheduler.schedule();

        // Request should be running now
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 0);

        // Finish the request
        scheduler.finish_request(request_id);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_abort_request() {
        let scheduler = create_test_scheduler();

        let request = Request::new(vec![1, 2, 3, 4, 5]);
        let request_id = request.id;
        scheduler.add_request(request).unwrap();

        scheduler.abort_request(request_id).unwrap();
        assert_eq!(scheduler.num_waiting(), 0);

        let metadata = scheduler.get_request_metadata(request_id).unwrap();
        assert_eq!(metadata.status, RequestStatus::Cancelled);
    }
}
