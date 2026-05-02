// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      process_reward.rs
// PATH:      /crates/swiftllm-training/src/process_reward.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   (no intra-crate imports — self-contained scoring logic)
// USED BY:
//   - swiftllm-training/src/grpo.rs       blend_prm_with_outcome() merges PRM + outcome reward
//   - swiftllm-training/src/config.rs     PrmConfig embedded in TrainingConfig.prm
//   - swiftllm-training/src/lib.rs        re-exports RulePrm, NeuralPrm, PrmConfig, etc.
// SEE ALSO:
//   - swiftllm-training/src/long_reward.rs           complementary token-level dense reward
//   - swiftllm-training/src/grpo.rs                  PRM scores feed GrpoGroup::rewards
//   - swiftllm-core/src/inference/verification.rs    PRM scores can replace neural verifier
//   - swiftllm-core/src/inference/refinement.rs      PRM can evaluate each refined draft
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

//! Process Reward Model (PRM) — step-level feedback on reasoning chains.
//!
//! PRMs evaluate the quality of intermediate reasoning steps rather than just
//! the final answer. This enables denser training signal for multi-step problems.
//!
//! ## References
//! - "Let's Verify Step by Step" (Lightman et al., 2023)
//! - "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"
//!   (Wang et al., 2023)
//! - DeepSeek-R1 process supervision approach
//!
//! ## Architecture
//! ```text
//! Reasoning chain → [Step 1] → [Step 2] → ... → [Step N] → Final answer
//!                        ↓           ↓                  ↓
//!                   PRM score   PRM score          PRM score
//!                        ↓           ↓                  ↓
//!                   Aggregate → Sequence-level reward
//! ```

use serde::{Deserialize, Serialize};

/// How to detect step boundaries in generated text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepBoundary {
    /// Split on double-newline paragraphs.
    DoubleNewline,
    /// Split on explicit step markers like "Step 1:", "Step 2:".
    StepMarker,
    /// Split on a custom delimiter string.
    Custom(String),
    /// Fixed number of tokens per step.
    FixedTokens(usize),
}

/// A single reasoning step parsed from a chain-of-thought response.
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Zero-indexed step number within the sequence.
    pub index: usize,
    /// Raw text content of this step.
    pub text: String,
    /// Token index where this step begins in the full sequence.
    pub token_start: usize,
    /// Token index where this step ends (exclusive).
    pub token_end: usize,
    /// Whether this is the final answer step.
    pub is_final: bool,
}

/// Scored result for a single reasoning step.
#[derive(Debug, Clone)]
pub struct StepScore {
    /// Step index (matches `ReasoningStep::index`).
    pub step_index: usize,
    /// Reward in `[-1, 1]`; 1.0 = clearly correct, -1.0 = clearly wrong.
    pub score: f32,
    /// Confidence in the score (0.0–1.0).
    pub confidence: f32,
    /// Optional label for interpretability.
    pub label: Option<String>,
}

/// Aggregation strategy for combining per-step scores into a sequence reward.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrmAggregation {
    /// Product of all step scores (strict: any bad step tanks reward).
    Product,
    /// Arithmetic mean of all step scores.
    Mean,
    /// Minimum step score (identifies the weakest link).
    Min,
    /// Score of the final step only (useful when earlier steps are noisy).
    LastStep,
    /// Weighted mean with geometric decay (recent steps weighted higher).
    WeightedMean { decay: f32 },
}

/// Full PRM output for one generated sequence.
#[derive(Debug, Clone)]
pub struct PrmResult {
    /// Per-step scores.
    pub step_scores: Vec<StepScore>,
    /// Aggregated sequence-level reward.
    pub sequence_reward: f32,
    /// Number of steps evaluated.
    pub num_steps: usize,
    /// Whether the final answer was detected as correct.
    pub final_correct: Option<bool>,
}

/// Configuration for the Process Reward Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrmConfig {
    /// How to split generated text into steps.
    pub boundary: StepBoundary,
    /// How to aggregate step scores.
    pub aggregation: PrmAggregation,
    /// Reward for a clearly correct step.
    pub correct_reward: f32,
    /// Reward for a clearly incorrect step.
    pub incorrect_reward: f32,
    /// Minimum number of steps to consider a chain valid.
    pub min_steps: usize,
    /// Penalise overly-long chains (None = no penalty).
    pub length_penalty: Option<f32>,
    /// Weight for PRM signal vs terminal reward (0..=1).
    pub prm_weight: f32,
}

impl Default for PrmConfig {
    fn default() -> Self {
        Self {
            boundary: StepBoundary::DoubleNewline,
            aggregation: PrmAggregation::Min,
            correct_reward: 1.0,
            incorrect_reward: -1.0,
            min_steps: 1,
            length_penalty: None,
            prm_weight: 0.3,
        }
    }
}

/// Parse a generated response into discrete reasoning steps.
pub fn parse_steps(text: &str, boundary: &StepBoundary) -> Vec<ReasoningStep> {
    let raw_parts: Vec<&str> = match boundary {
        StepBoundary::DoubleNewline => text.split("\n\n").collect(),
        StepBoundary::StepMarker => split_on_step_markers(text),
        StepBoundary::Custom(delim) => text.split(delim.as_str()).collect(),
        StepBoundary::FixedTokens(n) => {
            // Approximate: split by whitespace tokens of size n.
            let words: Vec<&str> = text.split_whitespace().collect();
            return words
                .chunks(*n)
                .enumerate()
                .map(|(i, chunk)| {
                    let joined = chunk.join(" ");
                    let token_start = i * n;
                    let token_end = token_start + chunk.len();
                    ReasoningStep {
                        index: i,
                        text: joined,
                        token_start,
                        token_end,
                        is_final: false,
                    }
                })
                .collect();
        }
    };

    // Track approximate token position by character offset / 4 (rough tokenisation).
    let mut steps: Vec<ReasoningStep> = raw_parts
        .iter()
        .enumerate()
        .filter_map(|(i, part)| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return None;
            }
            let char_offset: usize = raw_parts[..i]
                .iter()
                .map(|p| p.len() + 2) // +2 for the delimiter
                .sum();
            let token_start = char_offset / 4;
            let token_end = token_start + trimmed.len() / 4 + 1;
            Some(ReasoningStep {
                index: i,
                text: trimmed.to_string(),
                token_start,
                token_end,
                is_final: false,
            })
        })
        .collect();

    // Re-index and mark final step.
    let n = steps.len();
    for (i, step) in steps.iter_mut().enumerate() {
        step.index = i;
        step.is_final = i == n.saturating_sub(1);
    }
    steps
}

/// Split text on "Step N:" markers, returning slices.
fn split_on_step_markers(text: &str) -> Vec<&str> {
    // Simple heuristic: find lines that start with "Step " followed by digits.
    let mut parts: Vec<&str> = Vec::new();
    let mut last = 0usize;
    for (i, _) in text.match_indices('\n') {
        let rest = &text[i + 1..];
        if rest.starts_with("Step ") {
            parts.push(&text[last..i]);
            last = i + 1;
        }
    }
    parts.push(&text[last..]);
    parts
}

/// Aggregate step scores into a single sequence-level reward.
pub fn aggregate_step_scores(scores: &[StepScore], strategy: PrmAggregation) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    match strategy {
        PrmAggregation::Product => {
            // Map scores from [-1,1] to [0,1] before multiplying, then back.
            let product: f32 = scores.iter().map(|s| (s.score + 1.0) / 2.0).product();
            product * 2.0 - 1.0
        }
        PrmAggregation::Mean => {
            let sum: f32 = scores.iter().map(|s| s.score).sum();
            sum / scores.len() as f32
        }
        PrmAggregation::Min => {
            scores.iter().map(|s| s.score).fold(f32::INFINITY, f32::min)
        }
        PrmAggregation::LastStep => scores.last().map(|s| s.score).unwrap_or(0.0),
        PrmAggregation::WeightedMean { decay } => {
            let n = scores.len() as f32;
            let mut total_weight = 0.0f32;
            let mut weighted_sum = 0.0f32;
            for (i, s) in scores.iter().enumerate() {
                // Higher weight for steps closer to the end.
                let weight = decay.powi((n as i32) - 1 - (i as i32));
                weighted_sum += s.score * weight;
                total_weight += weight;
            }
            if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                0.0
            }
        }
    }
}

/// Rule-based PRM that scores steps using heuristics (no neural network).
///
/// Useful for math tasks where correctness can be checked symbolically.
pub struct RulePrm {
    /// PRM configuration.
    config: PrmConfig,
    /// Gold answer string for final-step verification.
    gold_answer: Option<String>,
}

impl RulePrm {
    /// Create a new rule-based PRM.
    pub fn new(config: PrmConfig, gold_answer: Option<String>) -> Self {
        Self { config, gold_answer }
    }

    /// Score one step using heuristics.
    fn score_step(&self, step: &ReasoningStep) -> StepScore {
        let text_lower = step.text.to_lowercase();

        // Heuristic: contradiction indicators lower the score.
        let contradiction_signals = [
            "this is wrong",
            "contradiction",
            "impossible",
            "undefined",
            "error",
        ];
        let has_contradiction = contradiction_signals
            .iter()
            .any(|s| text_lower.contains(s));

        // Heuristic: positive reasoning signals.
        let positive_signals = [
            "therefore",
            "thus",
            "hence",
            "so we have",
            "we get",
            "equals",
            "simplify",
        ];
        let positive_count = positive_signals
            .iter()
            .filter(|s| text_lower.contains(*s))
            .count();

        let base_score = if has_contradiction {
            self.config.incorrect_reward
        } else if positive_count >= 2 {
            self.config.correct_reward * 0.8
        } else {
            0.0 // neutral
        };

        // Final step: check against gold answer if available.
        let (score, confidence, label) = if step.is_final {
            if let Some(ref gold) = self.gold_answer {
                let gold_norm = gold.trim().to_lowercase();
                let text_norm = step.text.trim().to_lowercase();
                if text_norm.contains(&gold_norm) {
                    (
                        self.config.correct_reward,
                        0.95,
                        Some("final_correct".to_string()),
                    )
                } else {
                    (
                        self.config.incorrect_reward,
                        0.90,
                        Some("final_wrong".to_string()),
                    )
                }
            } else {
                (base_score, 0.5, None)
            }
        } else {
            (base_score, 0.6, None)
        };

        StepScore {
            step_index: step.index,
            score,
            confidence,
            label,
        }
    }

    /// Score a full reasoning chain.
    pub fn score(&self, response: &str) -> PrmResult {
        let steps = parse_steps(response, &self.config.boundary);
        if steps.len() < self.config.min_steps {
            return PrmResult {
                step_scores: Vec::new(),
                sequence_reward: self.config.incorrect_reward,
                num_steps: steps.len(),
                final_correct: None,
            };
        }

        let step_scores: Vec<StepScore> =
            steps.iter().map(|s| self.score_step(s)).collect();

        let final_correct = step_scores.last().and_then(|s| {
            s.label.as_deref().map(|l| l == "final_correct")
        });

        let mut seq_reward = aggregate_step_scores(&step_scores, self.config.aggregation);

        // Optional length penalty: penalise chains that are too long.
        if let Some(penalty) = self.config.length_penalty {
            let excess = (steps.len() as f32 - self.config.min_steps as f32).max(0.0);
            seq_reward -= penalty * excess * 0.01;
            seq_reward = seq_reward.clamp(-1.0, 1.0);
        }

        PrmResult {
            num_steps: step_scores.len(),
            step_scores,
            sequence_reward: seq_reward,
            final_correct,
        }
    }
}

/// Neural PRM placeholder — wraps a model that scores each reasoning step.
///
/// In a full implementation this calls the scoring head of a reward model
/// trained on human-labelled step correctness annotations.
pub struct NeuralPrm {
    /// PRM configuration.
    pub config: PrmConfig,
    /// Model identifier / path used for loading weights.
    pub model_id: String,
}

impl NeuralPrm {
    /// Create a new neural PRM.
    pub fn new(config: PrmConfig, model_id: impl Into<String>) -> Self {
        Self { config, model_id: model_id.into() }
    }

    /// Score steps (simulated — returns 0.5 confidence placeholder).
    pub fn score(&self, response: &str) -> PrmResult {
        let steps = parse_steps(response, &self.config.boundary);
        // Placeholder: uniform neutral scores until GPU inference is wired.
        let step_scores: Vec<StepScore> = steps
            .iter()
            .map(|s| StepScore {
                step_index: s.index,
                score: 0.0,
                confidence: 0.5,
                label: None,
            })
            .collect();
        let sequence_reward = aggregate_step_scores(&step_scores, self.config.aggregation);
        PrmResult {
            num_steps: step_scores.len(),
            step_scores,
            sequence_reward,
            final_correct: None,
        }
    }
}

/// Blend a terminal (outcome) reward with PRM step-level rewards.
///
/// `prm_weight` is taken from `PrmConfig::prm_weight`.
pub fn blend_prm_with_outcome(
    outcome_reward: f32,
    prm_result: &PrmResult,
    prm_weight: f32,
) -> f32 {
    (1.0 - prm_weight) * outcome_reward + prm_weight * prm_result.sequence_reward
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_math_response() -> &'static str {
        "We need to find x such that 2x + 4 = 10.\n\nSubtract 4 from both sides: 2x = 6.\n\nDivide both sides by 2: x = 3.\n\nTherefore the answer is 3."
    }

    #[test]
    fn test_parse_steps_double_newline() {
        let steps = parse_steps(make_math_response(), &StepBoundary::DoubleNewline);
        assert_eq!(steps.len(), 4, "Expected 4 steps separated by double newlines");
        assert!(steps.last().unwrap().is_final);
        assert!(!steps[0].is_final);
    }

    #[test]
    fn test_parse_steps_re_indexed() {
        let steps = parse_steps(make_math_response(), &StepBoundary::DoubleNewline);
        for (i, s) in steps.iter().enumerate() {
            assert_eq!(s.index, i);
        }
    }

    #[test]
    fn test_aggregate_mean() {
        let scores = vec![
            StepScore { step_index: 0, score: 1.0, confidence: 1.0, label: None },
            StepScore { step_index: 1, score: 0.0, confidence: 1.0, label: None },
            StepScore { step_index: 2, score: -1.0, confidence: 1.0, label: None },
        ];
        let agg = aggregate_step_scores(&scores, PrmAggregation::Mean);
        assert!((agg - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_min() {
        let scores = vec![
            StepScore { step_index: 0, score: 0.8, confidence: 1.0, label: None },
            StepScore { step_index: 1, score: -0.5, confidence: 1.0, label: None },
            StepScore { step_index: 2, score: 0.9, confidence: 1.0, label: None },
        ];
        let agg = aggregate_step_scores(&scores, PrmAggregation::Min);
        assert!((agg - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_last_step() {
        let scores = vec![
            StepScore { step_index: 0, score: -1.0, confidence: 1.0, label: None },
            StepScore { step_index: 1, score: 0.7, confidence: 1.0, label: None },
        ];
        let agg = aggregate_step_scores(&scores, PrmAggregation::LastStep);
        assert!((agg - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_product() {
        // All correct → near +1.
        let scores = vec![
            StepScore { step_index: 0, score: 1.0, confidence: 1.0, label: None },
            StepScore { step_index: 1, score: 1.0, confidence: 1.0, label: None },
        ];
        let agg = aggregate_step_scores(&scores, PrmAggregation::Product);
        assert!(agg > 0.9, "product of all-correct should be near 1");
    }

    #[test]
    fn test_aggregate_weighted_mean() {
        let scores = vec![
            StepScore { step_index: 0, score: -1.0, confidence: 1.0, label: None },
            StepScore { step_index: 1, score: 1.0, confidence: 1.0, label: None },
        ];
        // decay=0.5 → weights [0.5, 1.0]; weighted mean = (-0.5 + 1.0) / 1.5 = 0.333
        let agg = aggregate_step_scores(&scores, PrmAggregation::WeightedMean { decay: 0.5 });
        assert!(agg > 0.0, "later correct step should dominate with decay<1");
    }

    #[test]
    fn test_rule_prm_correct_answer() {
        // Use LastStep aggregation so only the final step score matters.
        let config = PrmConfig {
            aggregation: PrmAggregation::LastStep,
            ..PrmConfig::default()
        };
        let prm = RulePrm::new(config, Some("3".to_string()));
        let result = prm.score(make_math_response());
        assert!(result.sequence_reward > 0.0, "correct final answer should yield positive reward");
        assert_eq!(result.final_correct, Some(true));
    }

    #[test]
    fn test_rule_prm_wrong_answer() {
        let config = PrmConfig::default();
        let prm = RulePrm::new(config, Some("42".to_string()));
        let result = prm.score(make_math_response());
        // Min aggregation: final step is wrong → sequence reward is negative.
        assert!(result.sequence_reward < 0.0);
        assert_eq!(result.final_correct, Some(false));
    }

    #[test]
    fn test_rule_prm_no_gold() {
        let config = PrmConfig::default();
        let prm = RulePrm::new(config, None);
        let result = prm.score(make_math_response());
        assert_eq!(result.num_steps, 4);
        assert!(result.final_correct.is_none());
    }

    #[test]
    fn test_min_steps_gate() {
        let config = PrmConfig {
            min_steps: 10, // impossible to satisfy
            ..PrmConfig::default()
        };
        let prm = RulePrm::new(config.clone(), None);
        let result = prm.score(make_math_response());
        assert_eq!(result.step_scores.len(), 0);
        assert_eq!(result.sequence_reward, config.incorrect_reward);
    }

    #[test]
    fn test_neural_prm_placeholder() {
        let prm = NeuralPrm::new(PrmConfig::default(), "test-prm");
        let result = prm.score(make_math_response());
        assert_eq!(result.num_steps, 4);
        assert_eq!(result.sequence_reward, 0.0); // neutral placeholder
    }

    #[test]
    fn test_blend_prm_with_outcome() {
        let prm_result = PrmResult {
            step_scores: Vec::new(),
            sequence_reward: 1.0,
            num_steps: 0,
            final_correct: None,
        };
        // prm_weight=0.3, outcome=-1, prm=+1 → 0.7*(-1) + 0.3*1 = -0.4
        let blended = blend_prm_with_outcome(-1.0, &prm_result, 0.3);
        assert!((blended - (-0.4)).abs() < 1e-5);
    }

    #[test]
    fn test_parse_empty() {
        let steps = parse_steps("", &StepBoundary::DoubleNewline);
        assert_eq!(steps.len(), 0);
    }

    #[test]
    fn test_parse_fixed_tokens() {
        let text = "one two three four five six";
        let steps = parse_steps(text, &StepBoundary::FixedTokens(2));
        assert_eq!(steps.len(), 3);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: process_reward.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/process_reward.rs
// INTEGRATES:  grpo.rs · config.rs · long_reward.rs · inference/verification.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
