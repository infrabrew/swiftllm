// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      refinement.rs
// PATH:      /crates/swiftllm-core/src/inference/refinement.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   (no intra-crate imports — generate_fn closure wires to engine at call site)
// USED BY:
//   - swiftllm-core/src/inference/mod.rs    re-exports RefinementPipeline, RefinementConfig, etc.
//   - swiftllm-core/src/lib.rs              indirectly via inference module
// SEE ALSO:
//   - swiftllm-core/src/engine.rs               Engine::generate() is the generate_fn in prod
//   - swiftllm-core/src/inference/verification.rs  verifier can score each refined draft
//   - swiftllm-core/src/sampling/self_consistency.rs  run self-consistency on the final draft
//   - swiftllm-training/src/grpo.rs             GRPO fine-tunes the model that self-refines
//   - swiftllm-training/src/process_reward.rs   PRM can evaluate intermediate refinement rounds
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

//! Multi-round Self-Refinement — iterative critique and revision.
//!
//! Self-Refine (Madaan et al., 2023) shows that LLMs can substantially improve
//! their own outputs by alternating between three operations on the *same*
//! model, with no additional training:
//!
//! ```text
//! output₀  = Generate(prompt)
//! for i in 1..=max_rounds:
//!     critique_i = Critique(prompt, output_{i-1})
//!     output_i   = Revise(prompt, output_{i-1}, critique_i)
//!     if stop_criterion(output_{i-1}, output_i): break
//! return output_last
//! ```
//!
//! ## References
//! - "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)
//!   https://arxiv.org/abs/2303.17651
//! - DeepSeek-R1 multi-round verification pipeline

use serde::{Deserialize, Serialize};

/// When to stop iterating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoppingCriterion {
    /// Stop after a fixed number of rounds regardless of improvement.
    MaxRounds,
    /// Stop when the improvement score falls below `threshold`.
    MinImprovement { threshold: f32 },
    /// Stop when both max rounds is reached OR improvement < threshold.
    Either { max_rounds: usize, threshold: f32 },
}

/// How to score the improvement between two consecutive outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementMetric {
    /// Character-level edit distance (normalised to [0, 1]).
    EditDistance,
    /// Check if the output changed at all.
    AnyChange,
    /// Custom similarity score supplied externally; refinement pipeline
    /// calls a user-provided closure (not serialisable — use ExternalScore).
    ExternalScore,
}

/// One round of refinement: generate, critique, then revise.
#[derive(Debug, Clone)]
pub struct RefinementRound {
    /// 0-based round index (0 = initial generation, no critique).
    pub round_idx: usize,
    /// The output produced at this round.
    pub output: String,
    /// Critique produced for this round's output (None for round 0).
    pub critique: Option<String>,
    /// Improvement score relative to the previous round (None for round 0).
    pub improvement_score: Option<f32>,
    /// Whether this round was accepted as better than the previous.
    pub accepted: bool,
}

/// Full result of the refinement pipeline.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// All rounds including the initial generation.
    pub rounds: Vec<RefinementRound>,
    /// The final accepted output.
    pub final_output: String,
    /// Number of refinement rounds executed (not counting initial generation).
    pub num_rounds_used: usize,
    /// True if the pipeline stopped due to the stopping criterion (converged),
    /// false if it ran to max_rounds without convergence.
    pub converged: bool,
}

/// Configuration for the multi-round self-refinement pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementConfig {
    /// Maximum number of critique-revision cycles (not counting initial gen).
    pub max_rounds: usize,
    /// Stopping criterion.
    pub stopping: StoppingCriterion,
    /// Improvement scoring metric.
    pub metric: ImprovementMetric,
    /// Template for the critique prompt.
    /// Use `{output}` as a placeholder for the previous output.
    pub critique_template: String,
    /// Template for the revision prompt.
    /// Use `{output}` for the previous output and `{critique}` for the critique.
    pub revision_template: String,
    /// Always keep the best-scoring output, even if later rounds regress.
    pub keep_best: bool,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            stopping: StoppingCriterion::Either {
                max_rounds: 3,
                threshold: 0.01,
            },
            metric: ImprovementMetric::EditDistance,
            critique_template: "Review the following output and identify any errors or areas for improvement:\n\n{output}\n\nCritique:".to_string(),
            revision_template: "Original output:\n{output}\n\nCritique:\n{critique}\n\nRevised output:".to_string(),
            keep_best: true,
        }
    }
}

impl RefinementConfig {
    /// Render the critique prompt for a given output.
    pub fn render_critique_prompt(&self, output: &str) -> String {
        self.critique_template.replace("{output}", output)
    }

    /// Render the revision prompt for a given output and critique.
    pub fn render_revision_prompt(&self, output: &str, critique: &str) -> String {
        self.revision_template
            .replace("{output}", output)
            .replace("{critique}", critique)
    }
}

/// Compute an improvement score between two consecutive outputs.
pub fn improvement_score(prev: &str, curr: &str, metric: &ImprovementMetric) -> f32 {
    match metric {
        ImprovementMetric::AnyChange => {
            if prev == curr { 0.0 } else { 1.0 }
        }
        ImprovementMetric::EditDistance => {
            normalised_edit_distance(prev, curr)
        }
        ImprovementMetric::ExternalScore => {
            // Caller is responsible for setting the score externally.
            // Return 1.0 (always improve) as a safe default.
            1.0
        }
    }
}

/// Normalised Levenshtein distance on characters ∈ [0, 1].
///
/// 0.0 = identical, 1.0 = completely different.
/// Uses O(min(m,n)) space via the two-row DP approach.
pub fn normalised_edit_distance(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 && n == 0 {
        return 0.0;
    }
    if m == 0 {
        return 1.0;
    }
    if n == 0 {
        return 1.0;
    }

    let (a, b, m, n) = if m > n { (&b, &a, n, m) } else { (&a, &b, m, n) };

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];

    for j in 1..=n {
        curr[0] = j;
        for i in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[i] = (prev[i] + 1)
                .min(curr[i - 1] + 1)
                .min(prev[i - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m] as f32 / n.max(m) as f32
}

/// Check whether the stopping criterion is met.
fn should_stop(
    config: &RefinementConfig,
    round_idx: usize,
    improvement: f32,
) -> bool {
    match &config.stopping {
        StoppingCriterion::MaxRounds => round_idx >= config.max_rounds,
        StoppingCriterion::MinImprovement { threshold } => improvement < *threshold,
        StoppingCriterion::Either { max_rounds, threshold } => {
            round_idx >= *max_rounds || improvement < *threshold
        }
    }
}

/// Orchestrates multi-round self-refinement over simulated generation calls.
///
/// In a full implementation the `generate_fn` closure calls the inference
/// engine. Here it is generic so callers can wire in any text-generation
/// backend (or a mock for tests).
pub struct RefinementPipeline {
    /// Configuration.
    pub config: RefinementConfig,
}

impl RefinementPipeline {
    /// Create a new pipeline with the given config.
    pub fn new(config: RefinementConfig) -> Self {
        Self { config }
    }

    /// Run refinement starting from an already-generated initial output.
    ///
    /// `generate_fn` is called with a prompt string and returns the model's
    /// response. It is invoked once per round (critique pass + revision pass).
    pub fn refine<F>(&self, initial_output: String, mut generate_fn: F) -> RefinementResult
    where
        F: FnMut(&str) -> String,
    {
        let mut rounds: Vec<RefinementRound> = vec![RefinementRound {
            round_idx: 0,
            output: initial_output.clone(),
            critique: None,
            improvement_score: None,
            accepted: true,
        }];

        let mut best_output = initial_output;
        let mut best_score = f32::NEG_INFINITY; // higher edit distance = more change
        let mut converged = false;
        let mut num_rounds_used = 0;

        for round in 1..=self.config.max_rounds {
            let prev_output = &rounds.last().unwrap().output;

            // --- Critique pass ---
            let critique_prompt = self.config.render_critique_prompt(prev_output);
            let critique = generate_fn(&critique_prompt);

            // --- Revision pass ---
            let revision_prompt = self.config.render_revision_prompt(prev_output, &critique);
            let revised = generate_fn(&revision_prompt);

            let score = improvement_score(prev_output, &revised, &self.config.metric);
            num_rounds_used += 1;

            let accepted = score > 0.0;
            if accepted && score > best_score {
                best_score = score;
                best_output = revised.clone();
            }

            rounds.push(RefinementRound {
                round_idx: round,
                output: revised,
                critique: Some(critique),
                improvement_score: Some(score),
                accepted,
            });

            if should_stop(&self.config, round, score) {
                converged = true;
                break;
            }
        }

        let final_output = if self.config.keep_best {
            best_output
        } else {
            rounds.last().unwrap().output.clone()
        };

        RefinementResult {
            rounds,
            final_output,
            num_rounds_used,
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_generate(prompt: &str) -> String {
        // Simulates a model that always "improves" by appending " (revised)".
        format!("{} (revised)", prompt.lines().last().unwrap_or(""))
    }

    fn no_change_generate(_prompt: &str) -> String {
        "unchanged output".to_string()
    }

    // ── normalised_edit_distance ─────────────────────────────────────────────

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(normalised_edit_distance("hello", "hello"), 0.0);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(normalised_edit_distance("", ""), 0.0);
        assert_eq!(normalised_edit_distance("", "a"), 1.0);
        assert_eq!(normalised_edit_distance("a", ""), 1.0);
    }

    #[test]
    fn test_edit_distance_single_sub() {
        // "cat" → "bat": 1 substitution out of 3 chars = 0.333
        let d = normalised_edit_distance("cat", "bat");
        assert!((d - 1.0 / 3.0).abs() < 1e-5, "got {d}");
    }

    #[test]
    fn test_edit_distance_range() {
        let d = normalised_edit_distance("abc", "xyz");
        assert!(d > 0.0 && d <= 1.0);
    }

    // ── improvement_score ────────────────────────────────────────────────────

    #[test]
    fn test_improvement_any_change_same() {
        assert_eq!(improvement_score("a", "a", &ImprovementMetric::AnyChange), 0.0);
    }

    #[test]
    fn test_improvement_any_change_different() {
        assert_eq!(improvement_score("a", "b", &ImprovementMetric::AnyChange), 1.0);
    }

    #[test]
    fn test_improvement_edit_distance() {
        let s = improvement_score("hello", "hello world", &ImprovementMetric::EditDistance);
        assert!(s > 0.0);
    }

    // ── should_stop ──────────────────────────────────────────────────────────

    #[test]
    fn test_stop_max_rounds() {
        let cfg = RefinementConfig {
            stopping: StoppingCriterion::MaxRounds,
            max_rounds: 3,
            ..Default::default()
        };
        assert!(!should_stop(&cfg, 2, 0.5));
        assert!(should_stop(&cfg, 3, 0.5));
    }

    #[test]
    fn test_stop_min_improvement() {
        let cfg = RefinementConfig {
            stopping: StoppingCriterion::MinImprovement { threshold: 0.05 },
            ..Default::default()
        };
        assert!(!should_stop(&cfg, 1, 0.1));
        assert!(should_stop(&cfg, 1, 0.01));
    }

    #[test]
    fn test_stop_either() {
        let cfg = RefinementConfig {
            stopping: StoppingCriterion::Either { max_rounds: 3, threshold: 0.05 },
            ..Default::default()
        };
        assert!(should_stop(&cfg, 3, 0.5)); // max_rounds hit
        assert!(should_stop(&cfg, 1, 0.01)); // threshold hit
        assert!(!should_stop(&cfg, 1, 0.1)); // neither hit
    }

    // ── RefinementConfig templates ───────────────────────────────────────────

    #[test]
    fn test_critique_template_substitution() {
        let cfg = RefinementConfig::default();
        let prompt = cfg.render_critique_prompt("my output");
        assert!(prompt.contains("my output"));
    }

    #[test]
    fn test_revision_template_substitution() {
        let cfg = RefinementConfig::default();
        let prompt = cfg.render_revision_prompt("output text", "critique text");
        assert!(prompt.contains("output text"));
        assert!(prompt.contains("critique text"));
    }

    // ── RefinementPipeline ───────────────────────────────────────────────────

    #[test]
    fn test_pipeline_runs_max_rounds() {
        let config = RefinementConfig {
            max_rounds: 2,
            stopping: StoppingCriterion::MaxRounds,
            ..Default::default()
        };
        let pipeline = RefinementPipeline::new(config);
        let result = pipeline.refine("initial".to_string(), echo_generate);
        // round 0 (initial) + 2 refinement rounds
        assert_eq!(result.rounds.len(), 3);
        assert_eq!(result.num_rounds_used, 2);
    }

    #[test]
    fn test_pipeline_converges_on_no_change() {
        let config = RefinementConfig {
            max_rounds: 5,
            stopping: StoppingCriterion::MinImprovement { threshold: 0.01 },
            metric: ImprovementMetric::AnyChange,
            ..Default::default()
        };
        let pipeline = RefinementPipeline::new(config);
        // no_change_generate always returns "unchanged output".
        // Round 1: "initial" → "unchanged output" (different → score=1, no stop)
        // Round 2: "unchanged output" → "unchanged output" (same → score=0, stop)
        let result = pipeline.refine("initial".to_string(), no_change_generate);
        assert!(result.converged);
        assert_eq!(result.num_rounds_used, 2, "stops on second no-change round");
    }

    #[test]
    fn test_pipeline_initial_round_no_critique() {
        let pipeline = RefinementPipeline::new(RefinementConfig::default());
        let result = pipeline.refine("initial".to_string(), echo_generate);
        assert!(result.rounds[0].critique.is_none());
        assert!(result.rounds[0].improvement_score.is_none());
        assert!(result.rounds[0].accepted);
    }

    #[test]
    fn test_pipeline_subsequent_rounds_have_critique() {
        let pipeline = RefinementPipeline::new(RefinementConfig::default());
        let result = pipeline.refine("initial".to_string(), echo_generate);
        for r in &result.rounds[1..] {
            assert!(r.critique.is_some());
            assert!(r.improvement_score.is_some());
        }
    }

    #[test]
    fn test_pipeline_final_output_non_empty() {
        let pipeline = RefinementPipeline::new(RefinementConfig::default());
        let result = pipeline.refine("start".to_string(), echo_generate);
        assert!(!result.final_output.is_empty());
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: refinement.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/inference/refinement.rs
// INTEGRATES:  engine.rs · inference/verification.rs · sampling/self_consistency.rs · process_reward.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
