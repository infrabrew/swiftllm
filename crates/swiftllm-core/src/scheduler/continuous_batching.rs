//! Continuous Batching Scheduler
//!
//! This module implements iteration-level continuous batching,
//! which allows new requests to join and completed requests to
//! leave the batch at each iteration, maximizing GPU utilization.

use crate::config::SchedulerConfig;
use crate::error::{Error, Result};
use crate::memory::BlockManager;
use crate::types::{
    Request, RequestId, ScheduledSequenceGroup, SchedulerOutput, SequenceGroup,
    SequenceGroupState, SequenceStatus,
};
use parking_lot::RwLock;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// Scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicy {
    /// First Come First Served
    Fcfs,
    /// Shortest Job First (based on max_tokens)
    Sjf,
    /// Priority-based scheduling
    Priority,
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self::Fcfs
    }
}

/// Continuous batching scheduler
pub struct ContinuousBatchingScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Block manager
    block_manager: Arc<BlockManager>,

    /// Scheduling policy
    policy: SchedulerPolicy,

    /// Waiting queue
    waiting: RwLock<WaitingQueue>,

    /// Running sequences
    running: RwLock<Vec<SequenceGroup>>,

    /// Swapped sequences
    swapped: RwLock<VecDeque<SequenceGroup>>,

    /// Request metadata
    request_info: RwLock<HashMap<RequestId, RequestInfo>>,

    /// Enable chunked prefill
    enable_chunked_prefill: bool,

    /// Max tokens per prefill chunk
    max_prefill_tokens: usize,
}

/// Internal request info
#[derive(Debug, Clone)]
struct RequestInfo {
    /// Request ID
    request_id: RequestId,
    /// Expected output length (from max_tokens)
    expected_output_len: usize,
    /// Priority
    priority: i32,
    /// Arrival time
    arrival_time: Instant,
    /// Is in prefill phase
    is_prefill: bool,
    /// Prefill tokens remaining
    prefill_remaining: usize,
}

/// Waiting queue with support for different policies
struct WaitingQueue {
    /// FCFS queue
    fcfs_queue: VecDeque<SequenceGroup>,

    /// Priority queue (for SJF or Priority policy)
    priority_queue: BinaryHeap<PrioritizedSequenceGroup>,

    /// Current policy
    policy: SchedulerPolicy,
}

impl WaitingQueue {
    fn new(policy: SchedulerPolicy) -> Self {
        Self {
            fcfs_queue: VecDeque::new(),
            priority_queue: BinaryHeap::new(),
            policy,
        }
    }

    fn push(&mut self, seq_group: SequenceGroup, priority: i64) {
        match self.policy {
            SchedulerPolicy::Fcfs => {
                self.fcfs_queue.push_back(seq_group);
            }
            SchedulerPolicy::Sjf | SchedulerPolicy::Priority => {
                self.priority_queue.push(PrioritizedSequenceGroup {
                    seq_group,
                    priority,
                });
            }
        }
    }

    fn pop(&mut self) -> Option<SequenceGroup> {
        match self.policy {
            SchedulerPolicy::Fcfs => self.fcfs_queue.pop_front(),
            SchedulerPolicy::Sjf | SchedulerPolicy::Priority => {
                self.priority_queue.pop().map(|p| p.seq_group)
            }
        }
    }

    fn peek(&self) -> Option<&SequenceGroup> {
        match self.policy {
            SchedulerPolicy::Fcfs => self.fcfs_queue.front(),
            SchedulerPolicy::Sjf | SchedulerPolicy::Priority => {
                self.priority_queue.peek().map(|p| &p.seq_group)
            }
        }
    }

    fn len(&self) -> usize {
        match self.policy {
            SchedulerPolicy::Fcfs => self.fcfs_queue.len(),
            SchedulerPolicy::Sjf | SchedulerPolicy::Priority => self.priority_queue.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Wrapper for priority queue ordering
struct PrioritizedSequenceGroup {
    seq_group: SequenceGroup,
    priority: i64,
}

impl PartialEq for PrioritizedSequenceGroup {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedSequenceGroup {}

impl PartialOrd for PrioritizedSequenceGroup {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedSequenceGroup {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first
        self.priority.cmp(&other.priority)
    }
}

impl ContinuousBatchingScheduler {
    /// Create a new continuous batching scheduler
    pub fn new(
        config: SchedulerConfig,
        block_manager: Arc<BlockManager>,
        policy: SchedulerPolicy,
    ) -> Self {
        let enable_chunked_prefill = config.enable_chunked_prefill;
        let max_prefill_tokens = config.max_prefill_tokens;

        Self {
            config,
            block_manager,
            policy,
            waiting: RwLock::new(WaitingQueue::new(policy)),
            running: RwLock::new(Vec::new()),
            swapped: RwLock::new(VecDeque::new()),
            request_info: RwLock::new(HashMap::new()),
            enable_chunked_prefill,
            max_prefill_tokens,
        }
    }

    /// Add a request to the scheduler
    pub fn add_request(&self, request: Request) -> Result<()> {
        let request_id = request.id;
        let prompt_len = request.prompt_token_ids.len();
        let max_tokens = request.sampling_params.max_tokens;

        // Create request info
        let info = RequestInfo {
            request_id,
            expected_output_len: max_tokens,
            priority: request.priority,
            arrival_time: request.arrival_time,
            is_prefill: true,
            prefill_remaining: prompt_len,
        };

        // Create sequence group
        let seq_group = SequenceGroup::new(&request);

        // Calculate priority for queue
        let queue_priority = match self.policy {
            SchedulerPolicy::Fcfs => 0,
            SchedulerPolicy::Sjf => -(max_tokens as i64), // Negative so smaller is higher priority
            SchedulerPolicy::Priority => request.priority as i64,
        };

        // Add to waiting queue
        self.waiting.write().push(seq_group, queue_priority);
        self.request_info.write().insert(request_id, info);

        Ok(())
    }

    /// Schedule the next iteration
    pub fn schedule(&self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();

        // Phase 1: Handle swapped sequences
        self.schedule_swap_in(&mut output);

        // Phase 2: Handle running sequences
        self.schedule_running(&mut output);

        // Phase 3: Schedule new sequences from waiting queue
        self.schedule_waiting(&mut output);

        // Build the scheduled groups
        self.build_scheduled_groups(&mut output);

        output
    }

    /// Try to swap in sequences from CPU
    fn schedule_swap_in(&self, output: &mut SchedulerOutput) {
        let mut swapped = self.swapped.write();
        let mut running = self.running.write();

        while let Some(seq_group) = swapped.front() {
            let tokens_needed = seq_group.total_tokens();
            let blocks_needed = self.block_manager.blocks_needed(tokens_needed);

            if self.block_manager.num_free_gpu_blocks() < blocks_needed {
                break;
            }

            // Swap in
            let mut seq_group = swapped.pop_front().unwrap();

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

    /// Handle running sequences (check for preemption)
    fn schedule_running(&self, output: &mut SchedulerOutput) {
        let mut running = self.running.write();
        let mut swapped = self.swapped.write();

        // Check memory pressure
        let num_running = running.len();
        let free_blocks = self.block_manager.num_free_gpu_blocks();

        // Preempt if no free blocks and we have running sequences
        while free_blocks == 0 && !running.is_empty() {
            if !self.config.enable_preemption {
                break;
            }

            // Preempt the last arrived (LCFS preemption)
            let idx = running
                .iter()
                .enumerate()
                .max_by_key(|(_, sg)| sg.arrival_time)
                .map(|(i, _)| i);

            if let Some(idx) = idx {
                let seq_group = running.remove(idx);

                // Swap out to CPU
                for seq in &seq_group.sequences {
                    if let Ok(swap_map) = self.block_manager.swap_out(seq.id) {
                        for (gpu_block, cpu_block) in swap_map {
                            output.blocks_to_swap_out.insert(gpu_block, cpu_block);
                        }
                    }
                }

                swapped.push_back(seq_group);
            } else {
                break;
            }
        }

        // Ensure all running sequences have blocks for next token
        for seq_group in running.iter() {
            for seq in &seq_group.sequences {
                if !seq.is_finished() {
                    let current_len = seq.len();
                    let _ = self.block_manager.allocate(seq.id, current_len + 1);
                }
            }
        }
    }

    /// Schedule new sequences from waiting queue
    fn schedule_waiting(&self, output: &mut SchedulerOutput) {
        let mut waiting = self.waiting.write();
        let mut running = self.running.write();
        let mut request_info = self.request_info.write();

        let max_num_seqs = self.config.max_num_seqs;
        let max_batched_tokens = self.config.max_num_batched_tokens;

        let mut num_scheduled = running.len();
        let mut batched_tokens: usize = running.iter().map(|sg| sg.num_unfinished()).sum();

        while !waiting.is_empty() {
            // Check sequence limit
            if num_scheduled >= max_num_seqs {
                break;
            }

            let seq_group = waiting.peek().unwrap();
            let request_id = seq_group.request_id;
            let prompt_len = seq_group.prompt_len();

            // Determine tokens to schedule for this request
            let tokens_to_schedule = if self.enable_chunked_prefill {
                let info = request_info.get(&request_id).unwrap();
                std::cmp::min(info.prefill_remaining, self.max_prefill_tokens)
            } else {
                prompt_len
            };

            // Check token budget
            if batched_tokens + tokens_to_schedule > max_batched_tokens {
                break;
            }

            // Check block availability
            let blocks_needed = self.block_manager.blocks_needed(tokens_to_schedule);
            if self.block_manager.num_free_gpu_blocks() < blocks_needed {
                break;
            }

            // Pop from waiting and schedule
            let mut seq_group = waiting.pop().unwrap();

            // Allocate blocks
            for seq in &seq_group.sequences {
                if self.block_manager.allocate(seq.id, tokens_to_schedule).is_err() {
                    // Put back in queue if allocation fails
                    let priority = match self.policy {
                        SchedulerPolicy::Fcfs => 0,
                        SchedulerPolicy::Sjf => {
                            -(request_info.get(&request_id).unwrap().expected_output_len as i64)
                        }
                        SchedulerPolicy::Priority => {
                            request_info.get(&request_id).unwrap().priority as i64
                        }
                    };
                    waiting.push(seq_group, priority);
                    return;
                }
            }

            // Update request info
            if let Some(info) = request_info.get_mut(&request_id) {
                if self.enable_chunked_prefill {
                    info.prefill_remaining -= tokens_to_schedule;
                    if info.prefill_remaining == 0 {
                        info.is_prefill = false;
                        seq_group.state = SequenceGroupState::Decode;
                    }
                } else {
                    info.is_prefill = false;
                    seq_group.state = SequenceGroupState::Decode;
                }
            }

            running.push(seq_group);
            num_scheduled += 1;
            batched_tokens += tokens_to_schedule;
        }

        output.num_batched_tokens = batched_tokens;
    }

    /// Build the final scheduled groups
    fn build_scheduled_groups(&self, output: &mut SchedulerOutput) {
        let running = self.running.read();
        let request_info = self.request_info.read();

        for seq_group in running.iter() {
            if seq_group.is_finished() {
                continue;
            }

            let request_id = seq_group.request_id;
            let info = request_info.get(&request_id);

            // Determine chunk size
            let token_chunk_size = if let Some(info) = info {
                if info.is_prefill && self.enable_chunked_prefill {
                    std::cmp::min(info.prefill_remaining, self.max_prefill_tokens)
                } else if info.is_prefill {
                    seq_group.prompt_len()
                } else {
                    1 // Decode generates one token at a time
                }
            } else {
                1
            };

            // Note: In a real implementation, we'd clone the sequence group
            // For now, we just track the metadata
        }
    }

    /// Mark a request as finished
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
        }

        self.request_info.write().remove(&request_id);
    }

    /// Abort a request
    pub fn abort_request(&self, request_id: RequestId) -> Result<()> {
        // Try to remove from each queue
        let mut waiting = self.waiting.write();

        // For FCFS queue, manually search and remove
        match self.policy {
            SchedulerPolicy::Fcfs => {
                if let Some(idx) = waiting.fcfs_queue
                    .iter()
                    .position(|sg| sg.request_id == request_id)
                {
                    waiting.fcfs_queue.remove(idx);
                    self.request_info.write().remove(&request_id);
                    return Ok(());
                }
            }
            _ => {
                // For priority queues, we'd need to rebuild
                // This is a simplified implementation
            }
        }
        drop(waiting);

        // Try running
        let mut running = self.running.write();
        if let Some(idx) = running
            .iter()
            .position(|sg| sg.request_id == request_id)
        {
            let seq_group = running.remove(idx);
            for seq in &seq_group.sequences {
                self.block_manager.free(seq.id);
            }
            self.request_info.write().remove(&request_id);
            return Ok(());
        }
        drop(running);

        // Try swapped
        let mut swapped = self.swapped.write();
        if let Some(idx) = swapped
            .iter()
            .position(|sg| sg.request_id == request_id)
        {
            swapped.remove(idx);
            self.request_info.write().remove(&request_id);
            return Ok(());
        }

        Err(Error::RequestNotFound(request_id.to_string()))
    }

    /// Get number of waiting requests
    pub fn num_waiting(&self) -> usize {
        self.waiting.read().len()
    }

    /// Get number of running requests
    pub fn num_running(&self) -> usize {
        self.running.read().len()
    }

    /// Get number of swapped requests
    pub fn num_swapped(&self) -> usize {
        self.swapped.read().len()
    }

    /// Check if scheduler is empty
    pub fn is_empty(&self) -> bool {
        self.waiting.read().is_empty()
            && self.running.read().is_empty()
            && self.swapped.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_scheduler(policy: SchedulerPolicy) -> ContinuousBatchingScheduler {
        let block_manager = Arc::new(BlockManager::new(
            16,
            1000,
            500,
            32,
            128,
            32,
            false,
            None,
        ));

        let config = SchedulerConfig::default();
        ContinuousBatchingScheduler::new(config, block_manager, policy)
    }

    #[test]
    fn test_fcfs_scheduling() {
        let scheduler = create_test_scheduler(SchedulerPolicy::Fcfs);

        let r1 = Request::new(vec![1, 2, 3]);
        let r2 = Request::new(vec![4, 5, 6, 7, 8]);

        scheduler.add_request(r1).unwrap();
        scheduler.add_request(r2).unwrap();

        assert_eq!(scheduler.num_waiting(), 2);

        scheduler.schedule();

        assert_eq!(scheduler.num_running(), 2);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn test_abort_request() {
        let scheduler = create_test_scheduler(SchedulerPolicy::Fcfs);

        let request = Request::new(vec![1, 2, 3]);
        let request_id = request.id;

        scheduler.add_request(request).unwrap();
        assert_eq!(scheduler.num_waiting(), 1);

        scheduler.abort_request(request_id).unwrap();
        assert_eq!(scheduler.num_waiting(), 0);
    }
}
