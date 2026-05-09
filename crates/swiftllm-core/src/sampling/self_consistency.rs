// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      self_consistency.rs
// PATH:      /crates/swiftllm-core/src/sampling/self_consistency.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   (no intra-crate imports — operates on caller-supplied (text, logprob) pairs)
// USED BY:
//   - swiftllm-core/src/sampling/mod.rs        pub mod self_consistency; re-exports all types
//   - swiftllm-core/src/lib.rs                 indirectly via sampling::self_consistency
// SEE ALSO:
//   - swiftllm-core/src/sampling/strategies.rs  SamplerChain produces the raw candidate texts
//   - swiftllm-core/src/sampling/mod.rs         TokenSampler.sample() generates each chain
//   - swiftllm-core/src/inference/verification.rs  alternative test-time scaling (Best-of-N)
//   - swiftllm-core/src/inference/refinement.rs    self-refine can wrap self-consistency
//   - swiftllm-training/src/grpo.rs               same majority-vote idea used during training
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

//! Self-Consistency Sampling — majority-vote over diverse reasoning paths.
//!
//! Rather than returning a single greedy generation, self-consistency samples
//! `num_samples` independent reasoning chains with temperature > 0, extracts
//! the final answer from each, then returns the answer that wins a plurality
//! vote. This reliably improves accuracy on reasoning tasks without any
//! additional training.
//!
//! ## Algorithm (Wang et al., 2022)
//! ```text
//! for i in 1..=num_samples:
//!     chain_i = generate(prompt, temperature=T)   # diverse paths
//!     answer_i = extract_answer(chain_i)
//! winner = majority_vote({answer_i})
//! ```
//!
//! ## Gains reported
//! - GSM8K: +17.9 pp over greedy CoT
//! - MATH: +11.0 pp
//! - StrategyQA: +6.4 pp
//!
//! ## References
//! - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
//!   (Wang et al., 2022) — https://arxiv.org/abs/2203.11171
//! - DeepSeek-R1 best-of-N inference pipeline

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Strategy used to extract the final answer from a reasoning chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnswerExtractor {
    /// Look for the last occurrence of "The answer is X" / "= X" patterns.
    Heuristic,
    /// Return everything after the last occurrence of a sentinel string (e.g. "####").
    AfterSentinel(String),
    /// Use the last non-empty line of the text.
    LastLine,
    /// Capture everything inside `<answer>…</answer>` tags (DeepSeek-R1 style).
    XmlTag(String),
}

impl AnswerExtractor {
    /// Extract the candidate answer from generated text.
    pub fn extract<'a>(&self, text: &'a str) -> Option<&'a str> {
        match self {
            AnswerExtractor::Heuristic => extract_heuristic(text),
            AnswerExtractor::AfterSentinel(sentinel) => {
                text.rfind(sentinel.as_str())
                    .map(|pos| text[pos + sentinel.len()..].trim())
            }
            AnswerExtractor::LastLine => {
                text.lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| l.trim())
            }
            AnswerExtractor::XmlTag(tag) => {
                let open = format!("<{}>", tag);
                let close = format!("</{}>", tag);
                let start = text.rfind(&open)?.checked_add(open.len())?;
                let end = text[start..].find(&close)?;
                Some(text[start..start + end].trim())
            }
        }
    }
}

/// Heuristic answer extractor: looks for "= X", "answer is X", or the last number.
fn extract_heuristic(text: &str) -> Option<&str> {
    // Check for "The answer is X" / "answer: X" pattern.
    let patterns = ["the answer is ", "answer: ", "answer is ", "= "];
    let text_lower = text.to_lowercase();
    for pat in &patterns {
        if let Some(pos) = text_lower.rfind(pat) {
            let after = text[pos + pat.len()..].trim();
            // Take up to first whitespace or punctuation.
            let end = after.find(|c: char| c.is_whitespace() || c == ',' || c == '.')
                .unwrap_or(after.len());
            if end > 0 {
                return Some(&after[..end]);
            }
        }
    }
    // Last resort: last non-empty line.
    text.lines().rev().find(|l| !l.trim().is_empty()).map(|l| l.trim())
}

/// Normalise an extracted answer for voting: lowercase, strip punctuation/whitespace.
pub fn normalise_answer(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '-')
        .collect::<String>()
        .trim()
        .to_string()
}

/// A single sampled candidate from one generation.
#[derive(Debug, Clone)]
pub struct ConsistencyCandidate {
    /// Full generated text (reasoning chain + answer).
    pub text: String,
    /// Extracted and normalised answer (None if extraction failed).
    pub extracted_answer: Option<String>,
    /// Cumulative log-probability of the sequence (lower = more likely).
    pub sequence_logprob: f32,
    /// Sample index (0-based).
    pub sample_idx: usize,
}

/// Result of a self-consistency vote.
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    /// The winning candidate (full text).
    pub winner: ConsistencyCandidate,
    /// The winning answer string (normalised).
    pub answer: String,
    /// Number of votes for the winning answer.
    pub vote_count: usize,
    /// Total number of valid (answer-extractable) candidates.
    pub total_valid: usize,
    /// Agreement fraction = `vote_count / total_valid` ∈ (0, 1].
    pub agreement_fraction: f32,
    /// All vote tallies, sorted descending.
    pub vote_tallies: Vec<(String, usize)>,
}

/// Configuration for self-consistency sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfConsistencyConfig {
    /// Number of independent samples to generate.
    pub num_samples: usize,
    /// Sampling temperature for diversity (0.0 = greedy is pointless here).
    pub temperature: f32,
    /// Strategy used to extract answers from generated chains.
    pub extractor: AnswerExtractor,
    /// Return None if the winning answer has fewer votes than this fraction.
    /// Set to 0.0 to always return a result.
    pub min_agreement: f32,
    /// Whether to break ties using sequence log-probability (higher → better).
    pub tiebreak_by_logprob: bool,
}

impl Default for SelfConsistencyConfig {
    fn default() -> Self {
        Self {
            num_samples: 32,
            temperature: 0.7,
            extractor: AnswerExtractor::Heuristic,
            min_agreement: 0.0,
            tiebreak_by_logprob: true,
        }
    }
}

/// Perform majority-vote over a set of candidates.
///
/// Returns `None` if no candidates have an extractable answer, or if the
/// winning agreement fraction falls below `min_agreement`.
pub fn majority_vote(
    candidates: Vec<ConsistencyCandidate>,
    config: &SelfConsistencyConfig,
) -> Option<ConsistencyResult> {
    // Partition into valid (answer extracted) and invalid.
    let valid: Vec<ConsistencyCandidate> = candidates
        .into_iter()
        .filter(|c| c.extracted_answer.is_some())
        .collect();

    if valid.is_empty() {
        return None;
    }

    // Count votes per normalised answer.
    let mut vote_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, c) in valid.iter().enumerate() {
        let key = c.extracted_answer.clone().unwrap();
        vote_map.entry(key).or_default().push(idx);
    }

    // Sort by vote count desc, then by min-sum-logprob (highest probability) for ties.
    let mut tallies: Vec<(String, Vec<usize>)> = vote_map.into_iter().collect();
    tallies.sort_by(|(_ans_a, idxs_a), (_ans_b, idxs_b)| {
        let cmp = idxs_b.len().cmp(&idxs_a.len());
        if cmp != std::cmp::Ordering::Equal || !config.tiebreak_by_logprob {
            return cmp;
        }
        // Tiebreak: pick the answer whose candidates have the best avg log-prob.
        let avg_lp = |idxs: &[usize]| -> f32 {
            let sum: f32 = idxs.iter().map(|&i| valid[i].sequence_logprob).sum();
            sum / idxs.len() as f32
        };
        avg_lp(idxs_b).partial_cmp(&avg_lp(idxs_a)).unwrap_or(std::cmp::Ordering::Equal)
    });

    let (winning_answer, winning_idxs) = tallies.first()?;
    let vote_count = winning_idxs.len();
    let total_valid = valid.len();
    let agreement_fraction = vote_count as f32 / total_valid as f32;

    if agreement_fraction < config.min_agreement {
        return None;
    }

    // Among all candidates with the winning answer, pick the one with the best
    // log-probability as the representative "winner" to return.
    let best_idx = if config.tiebreak_by_logprob {
        *winning_idxs
            .iter()
            .max_by(|&&a, &&b| {
                valid[a].sequence_logprob
                    .partial_cmp(&valid[b].sequence_logprob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    } else {
        winning_idxs[0]
    };

    let vote_tallies: Vec<(String, usize)> = tallies
        .iter()
        .map(|(ans, idxs)| (ans.clone(), idxs.len()))
        .collect();

    Some(ConsistencyResult {
        winner: valid[best_idx].clone(),
        answer: winning_answer.clone(),
        vote_count,
        total_valid,
        agreement_fraction,
        vote_tallies,
    })
}

/// Build `ConsistencyCandidate` objects from raw generation outputs.
///
/// In a full system, `texts` comes from `num_samples` parallel forward passes
/// with the same prompt and temperature > 0. Each element is
/// `(generated_text, cumulative_logprob)`.
pub fn build_candidates(
    texts: Vec<(String, f32)>,
    extractor: &AnswerExtractor,
) -> Vec<ConsistencyCandidate> {
    texts
        .into_iter()
        .enumerate()
        .map(|(idx, (text, logprob))| {
            let extracted_answer = extractor
                .extract(&text)
                .map(normalise_answer)
                .filter(|s| !s.is_empty());
            ConsistencyCandidate {
                text,
                extracted_answer,
                sequence_logprob: logprob,
                sample_idx: idx,
            }
        })
        .collect()
}

/// Simulate self-consistency over pre-generated texts (integration helper).
///
/// In production this is replaced by actual model sampling. Used in tests
/// and for offline reranking of already-generated candidates.
pub fn self_consistency_vote(
    generated: Vec<(String, f32)>,
    config: &SelfConsistencyConfig,
) -> Option<ConsistencyResult> {
    let candidates = build_candidates(generated, &config.extractor);
    majority_vote(candidates, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_texts(answers: &[&str]) -> Vec<(String, f32)> {
        answers
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let text = format!("Step 1: some work.\n\nThe answer is {a}.");
                let lp = -(i as f32 * 0.1); // decreasing logprob
                (text, lp)
            })
            .collect()
    }

    // ── AnswerExtractor tests ────────────────────────────────────────────────

    #[test]
    fn test_heuristic_extractor_answer_is() {
        let ex = AnswerExtractor::Heuristic;
        // The heuristic strips trailing punctuation (`.`) from the answer.
        let text = "We compute 2+2. The answer is 4.";
        assert_eq!(ex.extract(text), Some("4"));
    }

    #[test]
    fn test_heuristic_extractor_equals() {
        let ex = AnswerExtractor::Heuristic;
        let text = "x + 1 = 5, so x = 4";
        // rfind for "= " should find the last one
        let ans = ex.extract(text);
        assert!(ans.is_some());
        assert!(ans.unwrap().starts_with('4'));
    }

    #[test]
    fn test_after_sentinel_extractor() {
        let ex = AnswerExtractor::AfterSentinel("####".to_string());
        let text = "Work here\n#### 42";
        assert_eq!(ex.extract(text), Some("42"));
    }

    #[test]
    fn test_last_line_extractor() {
        let ex = AnswerExtractor::LastLine;
        let text = "Step 1: blah\nStep 2: blah\n42";
        assert_eq!(ex.extract(text), Some("42"));
    }

    #[test]
    fn test_xml_tag_extractor() {
        let ex = AnswerExtractor::XmlTag("answer".to_string());
        let text = "<think>reasoning</think><answer>42</answer>";
        assert_eq!(ex.extract(text), Some("42"));
    }

    #[test]
    fn test_xml_tag_missing() {
        let ex = AnswerExtractor::XmlTag("answer".to_string());
        assert_eq!(ex.extract("no tags here"), None);
    }

    // ── normalise_answer ─────────────────────────────────────────────────────

    #[test]
    fn test_normalise_strips_punctuation() {
        assert_eq!(normalise_answer("  Paris,  "), "paris");
    }

    #[test]
    fn test_normalise_preserves_decimal() {
        assert_eq!(normalise_answer("3.14"), "3.14");
    }

    // ── majority_vote ────────────────────────────────────────────────────────

    #[test]
    fn test_majority_vote_clear_winner() {
        // 3 × "42", 1 × "43", 1 × "44"
        let texts = make_texts(&["42", "42", "42", "43", "44"]);
        let cfg = SelfConsistencyConfig::default();
        let result = self_consistency_vote(texts, &cfg).unwrap();
        assert_eq!(result.answer, "42");
        assert_eq!(result.vote_count, 3);
        assert_eq!(result.total_valid, 5);
        assert!((result.agreement_fraction - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_majority_vote_tiebreak_logprob() {
        // 2 × "42" (logprob 0 and -0.1), 2 × "43" (logprob -0.2 and -0.3)
        // "42" wins on logprob tiebreak (better average)
        let texts = vec![
            ("The answer is 42.".to_string(), 0.0f32),
            ("The answer is 43.".to_string(), -0.2),
            ("The answer is 42.".to_string(), -0.1),
            ("The answer is 43.".to_string(), -0.3),
        ];
        let cfg = SelfConsistencyConfig { tiebreak_by_logprob: true, ..Default::default() };
        let result = self_consistency_vote(texts, &cfg).unwrap();
        assert_eq!(result.answer, "42");
    }

    #[test]
    fn test_min_agreement_gate() {
        // 2/5 = 40% agreement — below threshold 0.5
        let texts = make_texts(&["42", "42", "43", "44", "45"]);
        let cfg = SelfConsistencyConfig {
            min_agreement: 0.5,
            ..Default::default()
        };
        assert!(self_consistency_vote(texts, &cfg).is_none());
    }

    #[test]
    fn test_all_same_answer() {
        let texts = make_texts(&["7", "7", "7"]);
        let cfg = SelfConsistencyConfig::default();
        let result = self_consistency_vote(texts, &cfg).unwrap();
        assert_eq!(result.vote_count, 3);
        assert!((result.agreement_fraction - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_no_extractable_answers() {
        // Texts with no recognisable pattern
        let texts = vec![
            ("lorem ipsum".to_string(), -1.0f32),
            ("dolor sit amet".to_string(), -1.0),
        ];
        let cfg = SelfConsistencyConfig {
            extractor: AnswerExtractor::XmlTag("answer".to_string()),
            ..Default::default()
        };
        assert!(self_consistency_vote(texts, &cfg).is_none());
    }

    #[test]
    fn test_vote_tallies_ordered() {
        let texts = make_texts(&["42", "42", "43", "43", "43", "44"]);
        let cfg = SelfConsistencyConfig::default();
        let result = self_consistency_vote(texts, &cfg).unwrap();
        // First tally should be the winner
        assert_eq!(result.vote_tallies[0].0, result.answer);
        // Tallies should be in descending order
        for i in 1..result.vote_tallies.len() {
            assert!(result.vote_tallies[i - 1].1 >= result.vote_tallies[i].1);
        }
    }

    #[test]
    fn test_build_candidates_count() {
        let texts = vec![
            ("The answer is 1.".to_string(), -0.5),
            ("The answer is 2.".to_string(), -1.0),
        ];
        let ex = AnswerExtractor::Heuristic;
        let cands = build_candidates(texts, &ex);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].sample_idx, 0);
        assert_eq!(cands[1].sample_idx, 1);
    }

    #[test]
    fn test_after_sentinel_with_gsm8k_format() {
        // GSM8K uses "#### <answer>" format
        let ex = AnswerExtractor::AfterSentinel("#### ".to_string());
        let text = "Step 1: 2+2=4.\nStep 2: 4+3=7.\n#### 7";
        assert_eq!(ex.extract(text), Some("7"));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: self_consistency.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/sampling/self_consistency.rs
// INTEGRATES:  sampling/mod.rs · sampling/strategies.rs · inference/verification.rs · grpo.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
