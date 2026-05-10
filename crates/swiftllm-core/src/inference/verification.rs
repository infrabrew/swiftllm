// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      verification.rs
// PATH:      /crates/swiftllm-core/src/inference/verification.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   (no intra-crate imports — scores caller-supplied (text, logprob) candidates)
// USED BY:
//   - swiftllm-core/src/inference/mod.rs    re-exports verify_and_rank, ScoringStrategy, etc.
//   - swiftllm-core/src/lib.rs              indirectly via inference module
// SEE ALSO:
//   - swiftllm-core/src/sampling/mod.rs             TokenSampler generates the N candidates
//   - swiftllm-core/src/sampling/self_consistency.rs  alternative test-time scaling (voting)
//   - swiftllm-core/src/inference/refinement.rs       verifier scores each refined draft
//   - swiftllm-training/src/process_reward.rs         PRM scores slot into neural_score field
//   - swiftllm-training/src/long_reward.rs            dense rewards can augment rule_score
//   - swiftllm-core/src/engine.rs                     Engine produces the candidate outputs
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

//! Dense Verification Layer — Best-of-N reranking with verifier scoring.
//!
//! Test-time compute scaling generates N candidate outputs then scores each
//! with a verifier. The highest-scoring candidate is returned. Combinable
//! with PRM step-level scores (see `swiftllm-training::process_reward`).
//!
//! ## Scoring strategies
//! 1. **Rule-based** — heuristics: format compliance, length, answer
//!    extractability. No additional model calls. O(1) per candidate.
//! 2. **Neural** — a reward model scores each candidate. Placeholder here;
//!    wired to the actual model forward pass in production.
//! 3. **Ensemble** — weighted combination of multiple scores.
//! 4. **SequenceLogProb** — rank purely by the model's own log-probability.
//!    Equivalent to best-of-N with no external verifier.
//!
//! ## References
//! - DeepSeek-R1: best-of-N + PRM reranking at inference time
//! - "Scaling LLM Test-Time Compute Optimally" (Snell et al., 2024)
//! - "Let's Verify Step by Step" (Lightman et al., 2023)

use serde::{Deserialize, Serialize};

/// How to score each candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringStrategy {
    /// Heuristic rule-based scoring (no model calls).
    RuleBased,
    /// Neural verifier (placeholder — returns 0.0 until wired to GPU model).
    Neural,
    /// Weighted linear combination: `w_rule * rule + w_neural * neural + w_lp * logprob`.
    Ensemble {
        /// Weight for rule-based scoring
        rule_weight: f32,
        /// Weight for neural scoring
        neural_weight: f32,
        /// Weight for log-probability scoring
        logprob_weight: f32,
    },
    /// Rank by the model's own sequence log-probability (no verifier call).
    SequenceLogProb,
}

/// A single scored candidate.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// Full generated text.
    pub text: String,
    /// Cumulative log-probability (lower magnitude = more likely).
    pub sequence_logprob: f32,
    /// Rule-based heuristic score ∈ [0, 1].
    pub rule_score: f32,
    /// Neural verifier score ∈ [0, 1] (0.0 when not computed).
    pub neural_score: f32,
    /// Final aggregate score used for ranking.
    pub final_score: f32,
    /// Rank (1 = best).
    pub rank: usize,
    /// Original candidate index.
    pub candidate_idx: usize,
}

/// Full verification result.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// The best-scoring candidate.
    pub best: ScoredCandidate,
    /// All candidates in rank order (best first).
    pub all_candidates: Vec<ScoredCandidate>,
    /// Scoring strategy used.
    pub strategy: String,
}

/// Configuration for the verification layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Number of candidates to generate (N in Best-of-N).
    pub num_candidates: usize,
    /// Scoring strategy.
    pub strategy: ScoringStrategy,
    /// Expected answer string for exact-match bonus (optional).
    pub gold_answer: Option<String>,
    /// Minimum text length for a candidate to be considered valid.
    pub min_length: usize,
    /// Sentinel strings that mark a well-formed final answer.
    pub answer_sentinels: Vec<String>,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            num_candidates: 8,
            strategy: ScoringStrategy::RuleBased,
            gold_answer: None,
            min_length: 10,
            answer_sentinels: vec![
                "the answer is".to_string(),
                "####".to_string(),
                "<answer>".to_string(),
            ],
        }
    }
}

/// Rule-based heuristic scorer for one candidate.
///
/// Scores based on:
/// - Length (longer is generally better up to a cap)
/// - Format compliance (contains expected answer markers)
/// - Exact match with gold answer if provided
/// - Penalise degenerate outputs (repetition, empty)
pub fn rule_score(text: &str, config: &VerificationConfig) -> f32 {
    if text.trim().len() < config.min_length {
        return 0.0;
    }

    let text_lower = text.to_lowercase();

    // Format: presence of answer sentinels
    let sentinel_hits = config
        .answer_sentinels
        .iter()
        .filter(|s| text_lower.contains(s.as_str()))
        .count();
    let format_score = (sentinel_hits as f32 / config.answer_sentinels.len().max(1) as f32)
        .min(1.0);

    // Length score: normalise to [0, 1] over [min_length, 2000] chars
    let len = text.len().min(2000);
    let length_score = (len.saturating_sub(config.min_length) as f32 / 2000.0_f32).min(1.0);

    // Degenerate penalty: excessive repetition
    let words: Vec<&str> = text.split_whitespace().collect();
    let unique_words = {
        let mut s: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for &w in &words { s.insert(w); }
        s.len()
    };
    let diversity = if words.is_empty() {
        0.0
    } else {
        unique_words as f32 / words.len() as f32
    };

    // Gold answer bonus
    let gold_bonus = if let Some(ref gold) = config.gold_answer {
        if text_lower.contains(&gold.to_lowercase()) {
            0.3
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Weighted combination
    let base = 0.4 * format_score + 0.2 * length_score + 0.4 * diversity;
    (base + gold_bonus).min(1.0)
}

/// Compute the final aggregate score for a candidate.
fn aggregate_score(
    rule: f32,
    neural: f32,
    logprob: f32,
    strategy: &ScoringStrategy,
) -> f32 {
    match strategy {
        ScoringStrategy::RuleBased => rule,
        ScoringStrategy::Neural => neural,
        ScoringStrategy::SequenceLogProb => {
            // Convert log-prob to a score: less negative = better → map to [0,1]
            // Typical log-probs ∈ [-500, 0]; use sigmoid-like mapping.
            let clamped = logprob.clamp(-100.0, 0.0);
            1.0 + clamped / 100.0 // maps [-100,0] → [0,1]
        }
        ScoringStrategy::Ensemble {
            rule_weight,
            neural_weight,
            logprob_weight,
        } => {
            let lp_score = 1.0 + logprob.clamp(-100.0, 0.0) / 100.0;
            rule_weight * rule + neural_weight * neural + logprob_weight * lp_score
        }
    }
}

/// Score and rank a set of candidate texts, returning the best.
///
/// `candidates` is a list of `(text, sequence_logprob)` pairs — the same
/// format as the output of the sampling engine.
pub fn verify_and_rank(
    candidates: Vec<(String, f32)>,
    config: &VerificationConfig,
) -> Option<VerificationResult> {
    if candidates.is_empty() {
        return None;
    }

    let strategy_name = match &config.strategy {
        ScoringStrategy::RuleBased => "rule_based",
        ScoringStrategy::Neural => "neural",
        ScoringStrategy::Ensemble { .. } => "ensemble",
        ScoringStrategy::SequenceLogProb => "sequence_logprob",
    }
    .to_string();

    let mut scored: Vec<ScoredCandidate> = candidates
        .into_iter()
        .enumerate()
        .map(|(idx, (text, logprob))| {
            let rule = rule_score(&text, config);
            // Neural placeholder: always 0.0 until GPU inference is wired.
            let neural = 0.0f32;
            let final_score = aggregate_score(rule, neural, logprob, &config.strategy);
            ScoredCandidate {
                text,
                sequence_logprob: logprob,
                rule_score: rule,
                neural_score: neural,
                final_score,
                rank: 0, // filled below
                candidate_idx: idx,
            }
        })
        .collect();

    // Sort descending by final_score.
    scored.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign ranks (1-based).
    for (i, c) in scored.iter_mut().enumerate() {
        c.rank = i + 1;
    }

    let best = scored[0].clone();
    Some(VerificationResult {
        best,
        all_candidates: scored,
        strategy: strategy_name,
    })
}

/// Convenience: Best-of-N with sequence log-probability ranking only.
///
/// Equivalent to `verify_and_rank` with `ScoringStrategy::SequenceLogProb`.
pub fn best_of_n_by_logprob(candidates: Vec<(String, f32)>) -> Option<(String, f32)> {
    candidates
        .into_iter()
        .max_by(|(_, lp_a), (_, lp_b)| {
            lp_a.partial_cmp(lp_b).unwrap_or(std::cmp::Ordering::Equal)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates(texts: &[&str]) -> Vec<(String, f32)> {
        texts
            .iter()
            .enumerate()
            .map(|(i, &t)| (t.to_string(), -(i as f32) * 5.0))
            .collect()
    }

    // ── rule_score ────────────────────────────────────────────────────────────

    #[test]
    fn test_rule_score_empty() {
        let cfg = VerificationConfig::default();
        assert_eq!(rule_score("", &cfg), 0.0);
    }

    #[test]
    fn test_rule_score_too_short() {
        let cfg = VerificationConfig { min_length: 20, ..Default::default() };
        assert_eq!(rule_score("short", &cfg), 0.0);
    }

    #[test]
    fn test_rule_score_with_sentinel() {
        let cfg = VerificationConfig::default();
        let text = "We computed everything carefully. The answer is 42. This is a good answer.";
        let score = rule_score(text, &cfg);
        assert!(score > 0.0);
    }

    #[test]
    fn test_rule_score_gold_bonus() {
        let cfg = VerificationConfig {
            gold_answer: Some("42".to_string()),
            ..Default::default()
        };
        let with_gold = rule_score(
            "The answer is 42, verified by all steps above.",
            &cfg,
        );
        let without_gold = rule_score(
            "The answer is 99, verified by all steps above.",
            &cfg,
        );
        assert!(with_gold > without_gold, "gold bonus should raise the score");
    }

    #[test]
    fn test_rule_score_repetitive_penalty() {
        let cfg = VerificationConfig::default();
        let repetitive = "the the the the the the the the the the the the the the";
        let diverse = "we computed the answer by working through each step carefully";
        let rep_score = rule_score(repetitive, &cfg);
        let div_score = rule_score(diverse, &cfg);
        assert!(div_score > rep_score, "diverse text should score higher");
    }

    // ── aggregate_score ──────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_rule_based() {
        let score = aggregate_score(0.8, 0.0, -10.0, &ScoringStrategy::RuleBased);
        assert!((score - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_logprob_zero() {
        let score = aggregate_score(0.0, 0.0, 0.0, &ScoringStrategy::SequenceLogProb);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_aggregate_logprob_worst() {
        let score = aggregate_score(0.0, 0.0, -100.0, &ScoringStrategy::SequenceLogProb);
        assert!(score.abs() < 1e-4);
    }

    #[test]
    fn test_aggregate_ensemble() {
        let strategy = ScoringStrategy::Ensemble {
            rule_weight: 0.5,
            neural_weight: 0.0,
            logprob_weight: 0.5,
        };
        let score = aggregate_score(0.8, 0.0, 0.0, &strategy);
        // 0.5*0.8 + 0.0 + 0.5*1.0 = 0.9
        assert!((score - 0.9).abs() < 1e-5);
    }

    // ── verify_and_rank ───────────────────────────────────────────────────────

    #[test]
    fn test_verify_empty() {
        let cfg = VerificationConfig::default();
        assert!(verify_and_rank(vec![], &cfg).is_none());
    }

    #[test]
    fn test_verify_ranks_all_candidates() {
        let texts = &[
            "The answer is 42. We worked through every step carefully here.",
            "short",
            "The answer is 7. Another detailed explanation follows to pad length.",
        ];
        let candidates = make_candidates(texts);
        let cfg = VerificationConfig::default();
        let result = verify_and_rank(candidates, &cfg).unwrap();
        assert_eq!(result.all_candidates.len(), 3);
        assert_eq!(result.all_candidates[0].rank, 1);
        assert_eq!(result.all_candidates.last().unwrap().rank, 3);
    }

    #[test]
    fn test_verify_best_has_rank_1() {
        let candidates = make_candidates(&[
            "The answer is 42. Detailed reasoning across many words follows here for length.",
            "short",
        ]);
        let cfg = VerificationConfig::default();
        let result = verify_and_rank(candidates, &cfg).unwrap();
        assert_eq!(result.best.rank, 1);
    }

    #[test]
    fn test_verify_ranks_sorted_descending() {
        let candidates = make_candidates(&[
            "The answer is 42. Long and detailed.",
            "medium length answer is here with some detail.",
            "ok",
        ]);
        let cfg = VerificationConfig::default();
        let result = verify_and_rank(candidates, &cfg).unwrap();
        for i in 1..result.all_candidates.len() {
            assert!(
                result.all_candidates[i - 1].final_score
                    >= result.all_candidates[i].final_score
            );
        }
    }

    #[test]
    fn test_verify_strategy_name_rule_based() {
        let cfg = VerificationConfig::default();
        let result = verify_and_rank(make_candidates(&["some text here"]), &cfg).unwrap();
        assert_eq!(result.strategy, "rule_based");
    }

    #[test]
    fn test_verify_logprob_strategy() {
        // Best-of-N: candidate 0 has highest logprob (0.0), candidate 1 = -5.0
        let candidates = vec![
            ("answer A with plenty of text".to_string(), 0.0f32),
            ("answer B with plenty of text".to_string(), -5.0),
        ];
        let cfg = VerificationConfig {
            strategy: ScoringStrategy::SequenceLogProb,
            ..Default::default()
        };
        let result = verify_and_rank(candidates, &cfg).unwrap();
        assert_eq!(result.best.candidate_idx, 0);
    }

    // ── best_of_n_by_logprob ─────────────────────────────────────────────────

    #[test]
    fn test_best_of_n_returns_highest_logprob() {
        let candidates = vec![
            ("a".to_string(), -10.0f32),
            ("b".to_string(), -1.0),
            ("c".to_string(), -5.0),
        ];
        let best = best_of_n_by_logprob(candidates).unwrap();
        assert_eq!(best.0, "b");
        assert!((best.1 - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_best_of_n_empty() {
        assert!(best_of_n_by_logprob(vec![]).is_none());
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: verification.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/inference/verification.rs
// INTEGRATES:  engine.rs · sampling/mod.rs · sampling/self_consistency.rs · process_reward.rs · refinement.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
