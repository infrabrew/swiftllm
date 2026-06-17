// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      self_learning.rs
// PATH:      /crates/swiftllm-training/src/self_learning.rs
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

//! Self-learning (self-training / rejection-sampling fine-tuning).
//!
//! The model improves from *its own* outputs: it samples candidate completions,
//! a reward or confidence signal **self-labels** the good ones, and those become
//! new supervised-fine-tuning (SFT) data — repeated for several iterations. This
//! is the STaR / ReST / RFT loop (and the data engine behind self-rewarding
//! LMs). Unlike [`crate::realtime_rl`] (online policy gradient), self-learning
//! does plain SFT on a *filtered* self-generated dataset, which is simpler and
//! very stable.
//!
//! The two model-bound steps — generating candidates and fine-tuning on the
//! accepted set — are the seam. Everything here (self-labeling / acceptance,
//! deduplication, yield accounting, and the iteration controller with its
//! stopping rule) is pure and unit-tested. Candidates carry a `reward` (use a
//! verifier or [`crate::grpo::RewardFunction`]) and an `avg_logprob` (for the
//! confidence policy).

use std::collections::HashMap;

/// A model-generated candidate awaiting self-labeling.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The prompt this answers.
    pub prompt: String,
    /// The generated completion.
    pub output: String,
    /// Quality/correctness score from a verifier or reward model (higher better).
    pub reward: f32,
    /// Mean per-token log-probability of the output under the policy.
    pub avg_logprob: f32,
}

impl Candidate {
    /// Geometric-mean token probability in `[0, 1]` — a calibrated confidence.
    pub fn confidence(&self) -> f32 {
        self.avg_logprob.exp().clamp(0.0, 1.0)
    }
}

/// How candidates are accepted as self-labeled SFT data.
#[derive(Debug, Clone)]
pub enum AcceptancePolicy {
    /// Accept candidates whose reward is at least `min_reward`.
    RewardThreshold {
        /// Minimum reward to accept.
        min_reward: f32,
    },
    /// Accept candidates whose confidence is at least `min_confidence`.
    ConfidenceThreshold {
        /// Minimum confidence (`exp(avg_logprob)`) to accept.
        min_confidence: f32,
    },
    /// Keep only the single highest-reward candidate per prompt.
    BestOfN,
    /// Keep the top-`k` highest-reward candidates per prompt.
    TopK {
        /// Number kept per prompt.
        k: usize,
    },
}

/// A self-labeled supervised example.
#[derive(Debug, Clone, PartialEq)]
pub struct SftExample {
    /// The prompt.
    pub prompt: String,
    /// The accepted completion.
    pub completion: String,
}

/// Configuration for the self-training loop.
#[derive(Debug, Clone)]
pub struct SelfTrainingConfig {
    /// Acceptance policy.
    pub policy: AcceptancePolicy,
    /// Drop duplicate `(prompt, completion)` pairs from the accepted set.
    pub dedup: bool,
    /// Maximum self-training iterations.
    pub max_iterations: usize,
    /// Stop once a round's acceptance rate drops below this (improvement plateau).
    pub min_acceptance_rate: f32,
}

impl Default for SelfTrainingConfig {
    fn default() -> Self {
        Self {
            policy: AcceptancePolicy::RewardThreshold { min_reward: 0.5 },
            dedup: true,
            max_iterations: 5,
            min_acceptance_rate: 0.02,
        }
    }
}

/// Outcome of self-labeling one round of candidates.
#[derive(Debug, Clone)]
pub struct RoundResult {
    /// Accepted SFT examples (after dedup, if enabled).
    pub accepted: Vec<SftExample>,
    /// How many candidates were considered.
    pub total_candidates: usize,
    /// Accepted / total — the data yield.
    pub acceptance_rate: f32,
    /// Mean reward over accepted candidates.
    pub mean_accepted_reward: f32,
}

/// Self-label a batch of candidates into accepted SFT examples.
pub fn filter_candidates(
    candidates: &[Candidate],
    policy: &AcceptancePolicy,
    dedup: bool,
) -> RoundResult {
    let total = candidates.len();
    let accepted_refs: Vec<&Candidate> = match policy {
        AcceptancePolicy::RewardThreshold { min_reward } => {
            candidates.iter().filter(|c| c.reward >= *min_reward).collect()
        }
        AcceptancePolicy::ConfidenceThreshold { min_confidence } => candidates
            .iter()
            .filter(|c| c.confidence() >= *min_confidence)
            .collect(),
        AcceptancePolicy::BestOfN => per_prompt_top(candidates, 1),
        AcceptancePolicy::TopK { k } => per_prompt_top(candidates, *k),
    };

    let sum_reward: f32 = accepted_refs.iter().map(|c| c.reward).sum();
    let mean_accepted_reward = if accepted_refs.is_empty() {
        0.0
    } else {
        sum_reward / accepted_refs.len() as f32
    };

    let mut accepted: Vec<SftExample> = accepted_refs
        .iter()
        .map(|c| SftExample {
            prompt: c.prompt.clone(),
            completion: c.output.clone(),
        })
        .collect();

    if dedup {
        let mut seen = std::collections::HashSet::new();
        accepted.retain(|e| seen.insert((e.prompt.clone(), e.completion.clone())));
    }

    let acceptance_rate = if total == 0 {
        0.0
    } else {
        accepted.len() as f32 / total as f32
    };

    RoundResult {
        accepted,
        total_candidates: total,
        acceptance_rate,
        mean_accepted_reward,
    }
}

/// Keep the top-`k` candidates per prompt by reward (first-seen prompt order;
/// stable by input order on reward ties).
fn per_prompt_top(candidates: &[Candidate], k: usize) -> Vec<&Candidate> {
    // Group indices by prompt, preserving first-seen order.
    let mut order: Vec<&str> = Vec::new();
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, c) in candidates.iter().enumerate() {
        groups
            .entry(c.prompt.as_str())
            .or_insert_with(|| {
                order.push(c.prompt.as_str());
                Vec::new()
            })
            .push(i);
    }
    let mut out = Vec::new();
    for prompt in order {
        let mut idxs = groups.remove(prompt).unwrap();
        // Sort by reward descending; stable so ties keep input order.
        idxs.sort_by(|&a, &b| {
            candidates[b]
                .reward
                .partial_cmp(&candidates[a].reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in idxs.iter().take(k) {
            out.push(&candidates[i]);
        }
    }
    out
}

/// The self-training iteration controller.
#[derive(Debug)]
pub struct SelfTrainingLoop {
    config: SelfTrainingConfig,
    iteration: usize,
    total_accepted: usize,
    last_acceptance_rate: f32,
}

impl SelfTrainingLoop {
    /// Create a loop from configuration.
    pub fn new(config: SelfTrainingConfig) -> Self {
        Self {
            config,
            iteration: 0,
            total_accepted: 0,
            last_acceptance_rate: 1.0,
        }
    }

    /// Iterations completed so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Total accepted SFT examples across all rounds.
    pub fn total_accepted(&self) -> usize {
        self.total_accepted
    }

    /// Self-label one round of candidates, advancing the iteration counter. The
    /// returned accepted examples are what the (seam) SFT step trains on.
    pub fn run_round(&mut self, candidates: &[Candidate]) -> RoundResult {
        let result = filter_candidates(candidates, &self.config.policy, self.config.dedup);
        self.iteration += 1;
        self.total_accepted += result.accepted.len();
        self.last_acceptance_rate = result.acceptance_rate;
        result
    }

    /// Whether to run another round: under the iteration cap and the last round's
    /// yield has not collapsed below `min_acceptance_rate` (the model is still
    /// producing new accepted data).
    pub fn should_continue(&self) -> bool {
        if self.iteration >= self.config.max_iterations {
            return false;
        }
        // The first round always runs.
        self.iteration == 0 || self.last_acceptance_rate >= self.config.min_acceptance_rate
    }
}

/// Score a candidate output with a [`crate::grpo::RewardFunction`] — the bridge
/// to verifiable / rule-based rewards for self-labeling.
pub fn score_candidate(
    output: &str,
    reference: Option<&str>,
    reward_fn: &crate::grpo::RewardFunction,
) -> f32 {
    reward_fn.score(output, reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(prompt: &str, output: &str, reward: f32, avg_logprob: f32) -> Candidate {
        Candidate {
            prompt: prompt.to_string(),
            output: output.to_string(),
            reward,
            avg_logprob,
        }
    }

    #[test]
    fn reward_threshold_accepts_above_tau() {
        let cands = vec![
            cand("p", "good", 0.9, -0.2),
            cand("p", "ok", 0.5, -0.5),
            cand("p", "bad", 0.1, -2.0),
        ];
        let r = filter_candidates(&cands, &AcceptancePolicy::RewardThreshold { min_reward: 0.5 }, true);
        assert_eq!(r.accepted.len(), 2); // 0.9 and 0.5 pass
        assert_eq!(r.total_candidates, 3);
        assert!((r.acceptance_rate - 2.0 / 3.0).abs() < 1e-6);
        assert!((r.mean_accepted_reward - 0.7).abs() < 1e-6);
    }

    #[test]
    fn confidence_threshold_uses_logprob() {
        // confidence = exp(avg_logprob). -0.1 -> ~0.90, -2.0 -> ~0.14.
        let cands = vec![
            cand("p", "sure", -0.1, -0.1),
            cand("p", "unsure", -2.0, -2.0),
        ];
        let r = filter_candidates(
            &cands,
            &AcceptancePolicy::ConfidenceThreshold { min_confidence: 0.5 },
            true,
        );
        assert_eq!(r.accepted.len(), 1);
        assert_eq!(r.accepted[0].completion, "sure");
    }

    #[test]
    fn best_of_n_keeps_one_per_prompt() {
        let cands = vec![
            cand("p1", "a", 0.3, -0.5),
            cand("p1", "b", 0.9, -0.5), // best for p1
            cand("p2", "c", 0.7, -0.5), // best for p2
            cand("p2", "d", 0.2, -0.5),
        ];
        let r = filter_candidates(&cands, &AcceptancePolicy::BestOfN, true);
        assert_eq!(r.accepted.len(), 2);
        assert_eq!(r.accepted[0].completion, "b");
        assert_eq!(r.accepted[1].completion, "c");
    }

    #[test]
    fn top_k_keeps_k_per_prompt() {
        let cands = vec![
            cand("p", "a", 0.9, -0.5),
            cand("p", "b", 0.8, -0.5),
            cand("p", "c", 0.1, -0.5),
        ];
        let r = filter_candidates(&cands, &AcceptancePolicy::TopK { k: 2 }, true);
        assert_eq!(r.accepted.len(), 2);
        assert_eq!(r.accepted[0].completion, "a");
        assert_eq!(r.accepted[1].completion, "b");
    }

    #[test]
    fn dedup_removes_identical_examples() {
        let cands = vec![
            cand("p", "same", 0.9, -0.5),
            cand("p", "same", 0.8, -0.5),
        ];
        let with = filter_candidates(&cands, &AcceptancePolicy::RewardThreshold { min_reward: 0.0 }, true);
        let without = filter_candidates(&cands, &AcceptancePolicy::RewardThreshold { min_reward: 0.0 }, false);
        assert_eq!(with.accepted.len(), 1);
        assert_eq!(without.accepted.len(), 2);
    }

    #[test]
    fn loop_stops_at_max_iterations() {
        let mut lp = SelfTrainingLoop::new(SelfTrainingConfig {
            policy: AcceptancePolicy::RewardThreshold { min_reward: 0.0 },
            max_iterations: 2,
            min_acceptance_rate: 0.0,
            dedup: false,
        });
        let cands = vec![cand("p", "x", 1.0, -0.5)];
        assert!(lp.should_continue());
        lp.run_round(&cands);
        assert!(lp.should_continue());
        lp.run_round(&cands);
        assert!(!lp.should_continue()); // hit max_iterations
        assert_eq!(lp.iteration(), 2);
        assert_eq!(lp.total_accepted(), 2);
    }

    #[test]
    fn loop_stops_on_acceptance_plateau() {
        let mut lp = SelfTrainingLoop::new(SelfTrainingConfig {
            policy: AcceptancePolicy::RewardThreshold { min_reward: 0.9 },
            max_iterations: 10,
            min_acceptance_rate: 0.5,
            dedup: false,
        });
        // First round: most rejected → low acceptance rate → should stop after.
        let cands = vec![
            cand("p", "a", 0.1, -0.5),
            cand("p", "b", 0.2, -0.5),
            cand("p", "c", 0.95, -0.5),
        ];
        lp.run_round(&cands); // acceptance_rate = 1/3 < 0.5
        assert!(!lp.should_continue());
    }

    #[test]
    fn score_candidate_bridges_reward_function() {
        use crate::grpo::RewardFunction;
        // Correctness reward: matches reference → 1.0.
        let s = score_candidate("42", Some("42"), &RewardFunction::Correctness);
        assert!(s > 0.5);
        let s2 = score_candidate("99", Some("42"), &RewardFunction::Correctness);
        assert!(s2 < 0.5);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: self_learning.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/self_learning.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
