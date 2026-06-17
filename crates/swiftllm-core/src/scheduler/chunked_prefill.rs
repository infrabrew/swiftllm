// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      chunked_prefill.rs
// PATH:      /crates/swiftllm-core/src/scheduler/chunked_prefill.rs
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

//! Chunked-prefill token budgeting.
//!
//! Large prompts are split into fixed-size micro-batches so that a long prefill
//! cannot monopolise a scheduling step and starve in-flight decode requests
//! (which would inflate their time-per-output-token). The policy is:
//!
//! 1. Reserve the per-step token budget consumed by running decodes first.
//! 2. Fill the *remaining* budget with prefill, capped at `max_prefill_tokens`
//!    per request.
//! 3. If the budget is exhausted, the chunk size is `0` — the scheduler defers
//!    that prefill to a later step rather than blocking the batch.
//!
//! Capping (rather than all-or-nothing admission) is what removes the
//! head-of-line stall: a prompt that does not fit whole still advances by a
//! partial chunk, and the step is packed up to the token budget.

/// A per-step chunked-prefill token budget.
#[derive(Debug, Clone, Copy)]
pub struct ChunkedPrefillBudget {
    /// Total token budget for one scheduling step (prefill + decode).
    pub max_batched_tokens: usize,
    /// Maximum prefill tokens admitted from a single request per step.
    pub max_prefill_tokens: usize,
}

impl ChunkedPrefillBudget {
    /// Create a new budget.
    pub fn new(max_batched_tokens: usize, max_prefill_tokens: usize) -> Self {
        Self {
            max_batched_tokens,
            max_prefill_tokens,
        }
    }

    /// Tokens left in the step's budget after `tokens_used` are already taken
    /// (decodes plus prefill chunks admitted earlier in this step).
    pub fn remaining_budget(&self, tokens_used: usize) -> usize {
        self.max_batched_tokens.saturating_sub(tokens_used)
    }

    /// Compute the prefill chunk size for one request.
    ///
    /// `prefill_remaining` is how many prompt tokens that request still needs to
    /// prefill; `tokens_used` is the step budget already consumed. The result is
    /// `min(prefill_remaining, max_prefill_tokens, remaining_budget)` and is `0`
    /// only when the step budget is exhausted.
    pub fn chunk_size(&self, prefill_remaining: usize, tokens_used: usize) -> usize {
        prefill_remaining
            .min(self.max_prefill_tokens)
            .min(self.remaining_budget(tokens_used))
    }

    /// Whether admitting `chunk` more tokens stays within the step budget.
    pub fn fits(&self, tokens_used: usize, chunk: usize) -> bool {
        tokens_used + chunk <= self.max_batched_tokens
    }

    /// Number of chunks a prompt of `prompt_len` tokens would be split into.
    pub fn num_chunks(&self, prompt_len: usize) -> usize {
        if self.max_prefill_tokens == 0 {
            return 0;
        }
        prompt_len.div_ceil(self.max_prefill_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_decode_budget_before_prefill() {
        let budget = ChunkedPrefillBudget::new(512, 256);
        // 100 decodes already consumed 100 tokens this step.
        let chunk = budget.chunk_size(1000, 100);
        // Prefill is capped at max_prefill_tokens (256), well within remaining.
        assert_eq!(chunk, 256);
    }

    #[test]
    fn caps_chunk_to_remaining_budget() {
        let budget = ChunkedPrefillBudget::new(512, 256);
        // 400 tokens already used -> only 112 left despite a 256 cap.
        assert_eq!(budget.chunk_size(1000, 400), 112);
    }

    #[test]
    fn zero_when_budget_exhausted() {
        let budget = ChunkedPrefillBudget::new(512, 256);
        assert_eq!(budget.chunk_size(1000, 512), 0);
        assert_eq!(budget.chunk_size(1000, 600), 0); // saturating
    }

    #[test]
    fn chunk_never_exceeds_prefill_remaining() {
        let budget = ChunkedPrefillBudget::new(512, 256);
        // Only 10 tokens left to prefill.
        assert_eq!(budget.chunk_size(10, 0), 10);
    }

    #[test]
    fn fits_checks_budget() {
        let budget = ChunkedPrefillBudget::new(512, 256);
        assert!(budget.fits(256, 256));
        assert!(!budget.fits(256, 257));
    }

    #[test]
    fn num_chunks_rounds_up() {
        let budget = ChunkedPrefillBudget::new(512, 256);
        assert_eq!(budget.num_chunks(256), 1);
        assert_eq!(budget.num_chunks(257), 2);
        assert_eq!(budget.num_chunks(512), 2);
        assert_eq!(budget.num_chunks(513), 3);
    }

    #[test]
    fn packing_a_step_advances_all_until_budget_spent() {
        // Two long prefills sharing a 512-token step after 12 decodes.
        let budget = ChunkedPrefillBudget::new(512, 256);
        let mut used = 12; // decodes
        let a = budget.chunk_size(1000, used);
        assert_eq!(a, 256);
        used += a;
        let b = budget.chunk_size(1000, used);
        // 512 - 12 - 256 = 244 remaining for the second prefill.
        assert_eq!(b, 244);
        used += b;
        assert_eq!(used, 512);
        // Budget now exhausted: a third prefill gets nothing this step.
        assert_eq!(budget.chunk_size(1000, used), 0);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: chunked_prefill.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/scheduler/chunked_prefill.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
