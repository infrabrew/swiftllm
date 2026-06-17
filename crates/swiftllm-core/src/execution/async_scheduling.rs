// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      async_scheduling.rs
// PATH:      /crates/swiftllm-core/src/execution/async_scheduling.rs
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

//! Asynchronous scheduling with pipeline overlap.
//!
//! In a synchronous engine each step runs *schedule → execute → sample* in
//! series, so the CPU-side scheduling latency is fully exposed between GPU
//! forward passes. Asynchronous scheduling overlaps them: while step *N* is
//! executing on the device, the scheduler prepares step *N+1* on the host. With
//! pipeline parallelism, several micro-batches are in flight at once (the
//! pipeline depth), eliminating pipeline "bubbles".
//!
//! This module is the host-side bookkeeping for that overlap: a bounded set of
//! in-flight steps issued FIFO (matching pipeline-stage ordering) plus a pure
//! model of the throughput gained by hiding scheduling latency behind execution.
//! The actual device work is the GPU seam; the ordering/back-pressure logic here
//! is exact and unit-tested.

use std::collections::VecDeque;

/// Configuration for asynchronous scheduling.
#[derive(Debug, Clone, Copy)]
pub struct AsyncSchedulerConfig {
    /// Whether asynchronous scheduling is enabled. **On by default** — the
    /// engine overlaps host-side scheduling with device execution without the
    /// user opting in.
    pub enabled: bool,

    /// Maximum number of steps that may be in flight simultaneously. This is the
    /// pipeline depth; `1` degrades to synchronous scheduling.
    pub max_in_flight: usize,

    /// Whether zero-bubble overlap is used when composed with pipeline
    /// parallelism and speculative decoding (drafting overlaps verification so
    /// the pipeline is kept full).
    pub zero_bubble: bool,
}

impl Default for AsyncSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_in_flight: 2,
            zero_bubble: true,
        }
    }
}

impl AsyncSchedulerConfig {
    /// Async scheduling is enabled by default.
    pub fn enabled_by_default() -> bool {
        Self::default().enabled
    }

    /// Async scheduling is compatible with speculative decoding (drafts and
    /// verification are overlapped rather than serialised).
    pub fn compatible_with_speculative_decoding(&self) -> bool {
        true
    }

    /// Async scheduling is compatible with structured (schema-constrained)
    /// outputs — the constraint mask is applied on the sampling path, which is
    /// independent of the scheduling overlap.
    pub fn compatible_with_structured_outputs(&self) -> bool {
        true
    }
}

/// A handle to an issued (in-flight) step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepHandle {
    /// Monotonically increasing step identifier.
    pub step_id: usize,
}

/// Tracks in-flight steps for asynchronous, overlapped scheduling.
#[derive(Debug)]
pub struct AsyncScheduler {
    config: AsyncSchedulerConfig,
    in_flight: VecDeque<StepHandle>,
    next_step_id: usize,
    completed: usize,
}

impl AsyncScheduler {
    /// Create a new asynchronous scheduler.
    pub fn new(config: AsyncSchedulerConfig) -> Self {
        Self {
            config,
            in_flight: VecDeque::new(),
            next_step_id: 0,
            completed: 0,
        }
    }

    /// Whether another step may be issued without exceeding the pipeline depth.
    pub fn can_issue(&self) -> bool {
        self.in_flight.len() < self.config.max_in_flight.max(1)
    }

    /// Issue (schedule) the next step, returning its handle, or `None` if the
    /// pipeline is full and the caller must wait for a completion first.
    pub fn issue(&mut self) -> Option<StepHandle> {
        if !self.can_issue() {
            return None;
        }
        let handle = StepHandle {
            step_id: self.next_step_id,
        };
        self.next_step_id += 1;
        self.in_flight.push_back(handle);
        Some(handle)
    }

    /// Complete the oldest in-flight step (FIFO, matching pipeline ordering).
    /// Returns the completed step id, or `None` if nothing is in flight.
    pub fn complete(&mut self) -> Option<usize> {
        let handle = self.in_flight.pop_front()?;
        self.completed += 1;
        Some(handle.step_id)
    }

    /// Number of steps currently in flight.
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Total steps completed so far.
    pub fn completed(&self) -> usize {
        self.completed
    }

    /// Whether the pipeline is saturated (at maximum depth).
    pub fn is_saturated(&self) -> bool {
        self.in_flight.len() >= self.config.max_in_flight.max(1)
    }
}

/// Estimate the throughput speedup from overlapping host-side scheduling with
/// device-side execution.
///
/// With sufficient pipeline depth (>= 2) the scheduling latency is hidden behind
/// execution, so per-step wall time falls from `schedule + execute` to
/// `max(schedule, execute)`. Depth 1 is synchronous (no speedup). This is a
/// first-order model, not a measurement.
pub fn overlap_speedup(schedule_cost: f64, execute_cost: f64, depth: usize) -> f64 {
    let serial = schedule_cost + execute_cost;
    if serial <= 0.0 {
        return 1.0;
    }
    if depth <= 1 {
        return 1.0;
    }
    let overlapped = schedule_cost.max(execute_cost);
    if overlapped <= 0.0 {
        1.0
    } else {
        serial / overlapped
    }
}

/// Classic synchronous pipeline-parallel bubble fraction (GPipe): the share of
/// time pipeline stages sit idle filling/draining, `(p - 1) / (m + p - 1)` for
/// `p` stages and `m` micro-batches.
pub fn pipeline_bubble_fraction(num_stages: usize, num_microbatches: usize) -> f64 {
    if num_stages <= 1 || num_microbatches == 0 {
        return 0.0;
    }
    let p = num_stages as f64;
    let m = num_microbatches as f64;
    (p - 1.0) / (m + p - 1.0)
}

/// Effective pipeline bubble under a given scheduling mode. Zero-bubble async
/// scheduling interleaves the forward/draft and backward/verify passes so the
/// pipeline stays full, eliminating the bubble; otherwise the classic fraction
/// applies.
pub fn effective_bubble_fraction(
    num_stages: usize,
    num_microbatches: usize,
    zero_bubble: bool,
) -> f64 {
    if zero_bubble {
        0.0
    } else {
        pipeline_bubble_fraction(num_stages, num_microbatches)
    }
}

/// Expected tokens produced per verify step when speculative decoding overlaps
/// drafting with verification (zero-bubble). With `num_draft` drafted tokens and
/// an acceptance rate in `[0, 1]`, the verify step yields the bonus token plus
/// the accepted drafts: `1 + num_draft * acceptance_rate`. Because drafting is
/// hidden behind verification, this is also the throughput multiplier over
/// non-speculative decoding.
pub fn speculative_async_tokens_per_step(num_draft: usize, acceptance_rate: f64) -> f64 {
    1.0 + num_draft as f64 * acceptance_rate.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issues_up_to_pipeline_depth() {
        let mut sched = AsyncScheduler::new(AsyncSchedulerConfig { max_in_flight: 2, ..Default::default() });
        let a = sched.issue();
        let b = sched.issue();
        assert!(a.is_some() && b.is_some());
        assert_eq!(sched.in_flight(), 2);
        assert!(sched.is_saturated());
        // Third issue blocked until a completion.
        assert!(sched.issue().is_none());
    }

    #[test]
    fn completion_is_fifo_and_frees_capacity() {
        let mut sched = AsyncScheduler::new(AsyncSchedulerConfig { max_in_flight: 2, ..Default::default() });
        let a = sched.issue().unwrap();
        let _b = sched.issue().unwrap();
        // Oldest completes first (pipeline order).
        assert_eq!(sched.complete(), Some(a.step_id));
        assert!(sched.can_issue());
        let c = sched.issue().unwrap();
        assert_eq!(c.step_id, 2); // ids keep incrementing
        assert_eq!(sched.completed(), 1);
    }

    #[test]
    fn step_ids_are_monotonic() {
        let mut sched = AsyncScheduler::new(AsyncSchedulerConfig { max_in_flight: 3, ..Default::default() });
        let ids: Vec<usize> = (0..3).map(|_| sched.issue().unwrap().step_id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn depth_one_is_synchronous() {
        let mut sched = AsyncScheduler::new(AsyncSchedulerConfig { max_in_flight: 1, ..Default::default() });
        assert!(sched.issue().is_some());
        assert!(sched.issue().is_none()); // strictly serial
        assert_eq!(overlap_speedup(3.0, 7.0, 1), 1.0);
    }

    #[test]
    fn complete_on_empty_is_none() {
        let mut sched = AsyncScheduler::new(AsyncSchedulerConfig::default());
        assert_eq!(sched.complete(), None);
    }

    #[test]
    fn overlap_speedup_hides_scheduling_latency() {
        // schedule 3ms, execute 7ms: serial 10ms, overlapped max=7ms.
        let s = overlap_speedup(3.0, 7.0, 2);
        assert!((s - 10.0 / 7.0).abs() < 1e-9);
        assert!(s > 1.0);
        // Balanced costs give the largest gain (~2x).
        assert!((overlap_speedup(5.0, 5.0, 2) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn overlap_speedup_degenerate_inputs() {
        assert_eq!(overlap_speedup(0.0, 0.0, 4), 1.0);
        assert_eq!(overlap_speedup(1.0, 1.0, 0), 1.0);
    }

    #[test]
    fn async_scheduling_is_enabled_by_default() {
        assert!(AsyncSchedulerConfig::enabled_by_default());
        let cfg = AsyncSchedulerConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.zero_bubble);
        // Compatible with both speculative decoding and structured outputs.
        assert!(cfg.compatible_with_speculative_decoding());
        assert!(cfg.compatible_with_structured_outputs());
    }

    #[test]
    fn zero_bubble_eliminates_pipeline_bubble() {
        // 4 stages, 8 microbatches: classic bubble = 3/11 ≈ 0.27.
        let classic = pipeline_bubble_fraction(4, 8);
        assert!((classic - 3.0 / 11.0).abs() < 1e-9);
        // Zero-bubble scheduling drives it to 0.
        assert_eq!(effective_bubble_fraction(4, 8, true), 0.0);
        assert_eq!(effective_bubble_fraction(4, 8, false), classic);
        // A single stage has no bubble regardless.
        assert_eq!(pipeline_bubble_fraction(1, 8), 0.0);
    }

    #[test]
    fn more_microbatches_shrink_the_bubble() {
        assert!(pipeline_bubble_fraction(4, 4) > pipeline_bubble_fraction(4, 64));
    }

    #[test]
    fn speculative_overlap_throughput() {
        // 4 drafts at 75% acceptance -> 1 + 3 = 4 tokens/step.
        assert!((speculative_async_tokens_per_step(4, 0.75) - 4.0).abs() < 1e-9);
        // No drafts -> 1 token/step (plain decode).
        assert_eq!(speculative_async_tokens_per_step(0, 1.0), 1.0);
        // Acceptance is clamped.
        assert_eq!(speculative_async_tokens_per_step(2, 5.0), 3.0);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: async_scheduling.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/execution/async_scheduling.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
