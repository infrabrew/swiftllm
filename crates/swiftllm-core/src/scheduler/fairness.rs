// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      fairness.rs
// PATH:      /crates/swiftllm-core/src/scheduler/fairness.rs
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

//! Head-of-line-blocking mitigation for the waiting queue.
//!
//! A strict FCFS scheduler stalls the entire batch whenever the request at the
//! head needs more KV blocks than are currently free: every later (possibly
//! tiny) request waits behind it. This module lets the scheduler *skip* a head
//! request that cannot be allocated and serve a later one that fits — while
//! bounding skips and wait time so the skipped request cannot starve. Once a
//! request has been skipped too many times (or waited too many steps) it becomes
//! "aged out" and is held at the head (to be served next or to trigger
//! preemption) instead of being skipped again.

/// Per-request aging counters tracked while a request waits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestAging {
    /// Number of times this request has been skipped at the head of the queue.
    pub skips: usize,
    /// Number of scheduling steps this request has waited.
    pub waiting_steps: usize,
}

impl RequestAging {
    /// Record that the request was skipped this step.
    pub fn record_skip(&mut self) {
        self.skips += 1;
        self.waiting_steps += 1;
    }

    /// Record that the request waited (but was not the skip target) this step.
    pub fn record_wait(&mut self) {
        self.waiting_steps += 1;
    }
}

/// Configuration for the fairness policy.
#[derive(Debug, Clone, Copy)]
pub struct FairnessConfig {
    /// Maximum times a head request may be skipped before it is aged out.
    pub max_skips: usize,
    /// Maximum steps a request may wait before it is aged out.
    pub max_waiting_steps: usize,
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            max_skips: 8,
            max_waiting_steps: 64,
        }
    }
}

/// A waiting request as seen by the fairness policy.
#[derive(Debug, Clone, Copy)]
pub struct WaitingRequest {
    /// KV blocks this request needs to be admitted now.
    pub blocks_needed: usize,
    /// Aging counters.
    pub aging: RequestAging,
}

/// The scheduler's decision for the current step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleDecision {
    /// Schedule the request at this index in the waiting list.
    Schedule(usize),
    /// The head request is aged out but does not fit; hold it (wait or preempt)
    /// rather than skipping past it again.
    HoldHead,
    /// Nothing can be scheduled right now.
    Idle,
}

/// Fairness policy applying bounded skip-ahead with anti-starvation aging.
#[derive(Debug, Clone, Copy, Default)]
pub struct FairnessPolicy {
    config: FairnessConfig,
}

impl FairnessPolicy {
    /// Create a policy with the given configuration.
    pub fn new(config: FairnessConfig) -> Self {
        Self { config }
    }

    /// Whether a request has aged out and must no longer be skipped.
    pub fn is_aged_out(&self, aging: &RequestAging) -> bool {
        aging.skips >= self.config.max_skips || aging.waiting_steps >= self.config.max_waiting_steps
    }

    /// Decide which waiting request to schedule given the free-block budget.
    ///
    /// The list is in queue (priority/FCFS) order. The head is preferred; it is
    /// only skipped when it does not fit *and* has not aged out, in which case
    /// the first later request that fits is chosen.
    pub fn decide(&self, requests: &[WaitingRequest], free_blocks: usize) -> ScheduleDecision {
        let head = match requests.first() {
            Some(h) => h,
            None => return ScheduleDecision::Idle,
        };

        if head.blocks_needed <= free_blocks {
            return ScheduleDecision::Schedule(0);
        }

        // Head does not fit. If it has aged out, do not skip it again.
        if self.is_aged_out(&head.aging) {
            return ScheduleDecision::HoldHead;
        }

        // Otherwise skip ahead to the first later request that fits.
        for (idx, req) in requests.iter().enumerate().skip(1) {
            if req.blocks_needed <= free_blocks {
                return ScheduleDecision::Schedule(idx);
            }
        }

        // Nothing later fits either; keep the head (make no progress this step).
        ScheduleDecision::HoldHead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(blocks_needed: usize) -> WaitingRequest {
        WaitingRequest {
            blocks_needed,
            aging: RequestAging::default(),
        }
    }

    #[test]
    fn schedules_head_when_it_fits() {
        let policy = FairnessPolicy::default();
        let reqs = [req(4), req(2)];
        assert_eq!(policy.decide(&reqs, 8), ScheduleDecision::Schedule(0));
    }

    #[test]
    fn skips_blocked_head_to_serve_smaller_request() {
        let policy = FairnessPolicy::default();
        // Head needs 100 blocks (won't fit), a later request needs 2.
        let reqs = [req(100), req(50), req(2)];
        assert_eq!(policy.decide(&reqs, 4), ScheduleDecision::Schedule(2));
    }

    #[test]
    fn aged_out_head_is_held_not_skipped() {
        let policy = FairnessPolicy::new(FairnessConfig {
            max_skips: 3,
            max_waiting_steps: 100,
        });
        let mut head = req(100);
        head.aging.skips = 3; // already at the skip limit
        let reqs = [head, req(2)];
        // Even though req(2) fits, the aged-out head must be held (anti-starvation).
        assert_eq!(policy.decide(&reqs, 4), ScheduleDecision::HoldHead);
    }

    #[test]
    fn holds_head_when_nothing_fits() {
        let policy = FairnessPolicy::default();
        let reqs = [req(100), req(80)];
        assert_eq!(policy.decide(&reqs, 4), ScheduleDecision::HoldHead);
    }

    #[test]
    fn idle_when_empty() {
        let policy = FairnessPolicy::default();
        assert_eq!(policy.decide(&[], 100), ScheduleDecision::Idle);
    }

    #[test]
    fn aging_counters_advance() {
        let mut aging = RequestAging::default();
        aging.record_skip();
        aging.record_wait();
        assert_eq!(aging.skips, 1);
        assert_eq!(aging.waiting_steps, 2);
    }

    #[test]
    fn aged_out_by_wait_time() {
        let policy = FairnessPolicy::new(FairnessConfig {
            max_skips: 1000,
            max_waiting_steps: 10,
        });
        let aging = RequestAging {
            skips: 0,
            waiting_steps: 10,
        };
        assert!(policy.is_aged_out(&aging));
    }

    #[test]
    fn starvation_is_bounded_under_repeated_skips() {
        // A blocked head is skipped each step; after max_skips it ages out and is
        // held, guaranteeing it is no longer bypassed.
        let policy = FairnessPolicy::new(FairnessConfig {
            max_skips: 3,
            max_waiting_steps: 1000,
        });
        let mut head = req(100);
        let mut skipped = 0;
        for _ in 0..10 {
            let reqs = [head, req(1)];
            match policy.decide(&reqs, 4) {
                ScheduleDecision::Schedule(0) => unreachable!("head does not fit"),
                ScheduleDecision::Schedule(_) => {
                    head.aging.record_skip();
                    skipped += 1;
                }
                ScheduleDecision::HoldHead => break,
                ScheduleDecision::Idle => unreachable!(),
            }
        }
        // It is skipped at most max_skips times before being held.
        assert_eq!(skipped, 3);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: fairness.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/scheduler/fairness.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
