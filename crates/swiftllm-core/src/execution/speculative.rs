// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      speculative.rs
// PATH:      /crates/swiftllm-core/src/execution/speculative.rs
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

//! Speculative Decoding
//!
//! This module implements speculative decoding, a technique that uses
//! a smaller draft model to propose tokens that are then verified by
//! the target model, potentially accelerating inference.

use crate::error::{Error, Result};
use crate::types::TokenId;
use rand::Rng;

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Number of speculative tokens to generate
    pub num_speculative_tokens: usize,

    /// Minimum acceptance probability threshold
    pub min_acceptance_prob: f32,

    /// Enable adaptive speculation (adjust num tokens based on acceptance rate)
    pub adaptive: bool,

    /// Minimum speculative tokens when adaptive
    pub min_speculative_tokens: usize,

    /// Maximum speculative tokens when adaptive
    pub max_speculative_tokens: usize,

    /// Target acceptance rate for adaptive mode
    pub target_acceptance_rate: f32,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 5,
            min_acceptance_prob: 0.0,
            adaptive: true,
            min_speculative_tokens: 1,
            max_speculative_tokens: 10,
            target_acceptance_rate: 0.8,
        }
    }
}

/// Speculative decoder
pub struct SpeculativeDecoder {
    /// Configuration
    config: SpeculativeDecodingConfig,

    /// Current number of speculative tokens
    current_num_tokens: usize,

    /// Rolling acceptance rate
    acceptance_rate: f32,

    /// Number of accepted tokens (for statistics)
    total_accepted: usize,

    /// Number of proposed tokens (for statistics)
    total_proposed: usize,

    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    pub fn new(config: SpeculativeDecodingConfig) -> Self {
        let current_num_tokens = config.num_speculative_tokens;
        Self {
            config,
            current_num_tokens,
            acceptance_rate: 1.0,
            total_accepted: 0,
            total_proposed: 0,
            rng: rand::thread_rng(),
        }
    }

    /// Get the number of speculative tokens to generate
    pub fn num_speculative_tokens(&self) -> usize {
        self.current_num_tokens
    }

    /// Verify speculative tokens and determine accepted tokens
    ///
    /// # Arguments
    /// * `draft_probs` - Probabilities from draft model for each proposed token
    /// * `target_probs` - Probabilities from target model for each position
    /// * `draft_tokens` - Tokens proposed by draft model
    ///
    /// # Returns
    /// * (accepted_tokens, next_token) - Accepted tokens and the next token to generate
    pub fn verify(
        &mut self,
        draft_probs: &[Vec<f32>],
        target_probs: &[Vec<f32>],
        draft_tokens: &[TokenId],
    ) -> Result<(Vec<TokenId>, Option<TokenId>)> {
        if draft_tokens.is_empty() {
            return Ok((Vec::new(), None));
        }

        if draft_probs.len() != draft_tokens.len() || target_probs.len() != draft_tokens.len() + 1 {
            return Err(Error::SpeculativeDecoding(
                "Probability array size mismatch".to_string(),
            ));
        }

        let mut accepted = Vec::new();
        let mut all_accepted = true;

        // Verify each proposed token
        for (i, &token) in draft_tokens.iter().enumerate() {
            let draft_prob = draft_probs[i][token as usize];
            let target_prob = target_probs[i][token as usize];

            // Rejection sampling: accept with probability min(1, p_target / p_draft)
            let accept_prob = if draft_prob > 0.0 {
                (target_prob / draft_prob).min(1.0)
            } else if target_prob > 0.0 {
                1.0
            } else {
                0.0
            };

            if accept_prob >= self.config.min_acceptance_prob {
                let r: f32 = self.rng.gen();
                if r <= accept_prob {
                    accepted.push(token);
                } else {
                    all_accepted = false;
                    break;
                }
            } else {
                all_accepted = false;
                break;
            }
        }

        // Update statistics
        self.total_proposed += draft_tokens.len();
        self.total_accepted += accepted.len();

        // Sample the next token
        let next_token = if all_accepted {
            // All tokens accepted, sample from target distribution at last position
            let last_target = &target_probs[target_probs.len() - 1];
            Some(self.sample_from_probs(last_target))
        } else {
            // Some tokens rejected, sample from adjusted distribution
            let reject_pos = accepted.len();
            let adjusted_probs = self.compute_adjusted_distribution(
                &draft_probs[reject_pos],
                &target_probs[reject_pos],
            );
            Some(self.sample_from_probs(&adjusted_probs))
        };

        // Update acceptance rate and adaptive speculation
        self.update_acceptance_rate();

        Ok((accepted, next_token))
    }

    /// Compute adjusted distribution for rejected position
    fn compute_adjusted_distribution(&self, draft: &[f32], target: &[f32]) -> Vec<f32> {
        let mut adjusted = vec![0.0; target.len()];

        for i in 0..target.len() {
            // p' = max(0, p_target - p_draft)
            adjusted[i] = (target[i] - draft[i]).max(0.0);
        }

        // Normalize
        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            for p in &mut adjusted {
                *p /= sum;
            }
        } else {
            // Fallback to target distribution
            adjusted = target.to_vec();
        }

        adjusted
    }

    /// Sample a token from probabilities
    fn sample_from_probs(&mut self, probs: &[f32]) -> TokenId {
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= r {
                return idx as TokenId;
            }
        }

        // Fallback to argmax
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as TokenId)
            .unwrap_or(0)
    }

    /// Update the rolling acceptance rate
    fn update_acceptance_rate(&mut self) {
        if self.total_proposed > 0 {
            self.acceptance_rate = self.total_accepted as f32 / self.total_proposed as f32;
        }

        // Adaptive speculation: adjust number of tokens based on acceptance rate
        if self.config.adaptive && self.total_proposed >= 100 {
            if self.acceptance_rate > self.config.target_acceptance_rate + 0.1 {
                // High acceptance rate, try more tokens
                self.current_num_tokens = (self.current_num_tokens + 1)
                    .min(self.config.max_speculative_tokens);
            } else if self.acceptance_rate < self.config.target_acceptance_rate - 0.1 {
                // Low acceptance rate, try fewer tokens
                self.current_num_tokens = (self.current_num_tokens.saturating_sub(1))
                    .max(self.config.min_speculative_tokens);
            }
        }
    }

    /// Get the current acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        self.acceptance_rate
    }

    /// Get statistics
    pub fn stats(&self) -> SpeculativeStats {
        SpeculativeStats {
            total_accepted: self.total_accepted,
            total_proposed: self.total_proposed,
            acceptance_rate: self.acceptance_rate,
            current_num_tokens: self.current_num_tokens,
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_accepted = 0;
        self.total_proposed = 0;
        self.acceptance_rate = 1.0;
        self.current_num_tokens = self.config.num_speculative_tokens;
    }
}

/// Statistics for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    /// Total accepted tokens
    pub total_accepted: usize,

    /// Total proposed tokens
    pub total_proposed: usize,

    /// Current acceptance rate
    pub acceptance_rate: f32,

    /// Current number of speculative tokens
    pub current_num_tokens: usize,
}

impl SpeculativeStats {
    /// Calculate speedup from speculative decoding
    /// Assumes draft model is N times faster than target model
    pub fn speedup(&self, draft_speedup: f32) -> f32 {
        if self.total_proposed == 0 {
            return 1.0;
        }

        // Average accepted tokens per iteration
        let avg_accepted = self.acceptance_rate * self.current_num_tokens as f32;

        // Cost: 1 target forward + k draft forwards
        // Benefit: avg_accepted + 1 tokens
        let cost = 1.0 + (self.current_num_tokens as f32 / draft_speedup);
        let benefit = avg_accepted + 1.0;

        benefit / cost
    }
}

/// Ngram-based speculation (no draft model required)
#[derive(Debug)]
#[allow(dead_code)]
pub struct NgramSpeculator {
    /// Ngram lookup table
    ngram_table: std::collections::HashMap<Vec<TokenId>, Vec<(TokenId, usize)>>,

    /// Maximum ngram size
    max_ngram_size: usize,

    /// Maximum number of candidates per ngram
    max_candidates: usize,
}

#[allow(dead_code)]
impl NgramSpeculator {
    /// Create a new ngram speculator
    pub fn new(max_ngram_size: usize, max_candidates: usize) -> Self {
        Self {
            ngram_table: std::collections::HashMap::new(),
            max_ngram_size,
            max_candidates,
        }
    }

    /// Update ngram table with new tokens
    pub fn update(&mut self, context: &[TokenId], next_token: TokenId) {
        for n in 1..=self.max_ngram_size.min(context.len()) {
            let start = context.len() - n;
            let ngram: Vec<TokenId> = context[start..].to_vec();

            let entry = self.ngram_table.entry(ngram).or_default();

            // Update count for this token
            if let Some(pos) = entry.iter().position(|(t, _)| *t == next_token) {
                entry[pos].1 += 1;
            } else if entry.len() < self.max_candidates {
                entry.push((next_token, 1));
            }

            // Sort by count
            entry.sort_by(|a, b| b.1.cmp(&a.1));
        }
    }

    /// Get speculation candidates based on context
    pub fn speculate(&self, context: &[TokenId], num_tokens: usize) -> Vec<TokenId> {
        let mut result = Vec::with_capacity(num_tokens);
        let mut current_context = context.to_vec();

        for _ in 0..num_tokens {
            // Try ngrams from largest to smallest
            let mut found = None;
            for n in (1..=self.max_ngram_size.min(current_context.len())).rev() {
                let start = current_context.len() - n;
                let ngram: Vec<TokenId> = current_context[start..].to_vec();

                if let Some(candidates) = self.ngram_table.get(&ngram) {
                    if let Some((token, _)) = candidates.first() {
                        found = Some(*token);
                        break;
                    }
                }
            }

            match found {
                Some(token) => {
                    result.push(token);
                    current_context.push(token);
                }
                None => break,
            }
        }

        result
    }

    /// Clear the ngram table
    pub fn clear(&mut self) {
        self.ngram_table.clear();
    }
}

/// Configuration for Multi-Token Prediction (MTP) drafting.
///
/// MTP attaches lightweight prediction heads to the main model that, in a
/// single forward pass, propose the next `num_predict_heads` tokens. Those
/// drafts are then verified by the same [`SpeculativeDecoder`] rejection-
/// sampling machinery as any other speculative method, so MTP needs no separate
/// draft model and no ngram table.
#[derive(Debug, Clone)]
pub struct MtpConfig {
    /// Number of MTP heads (i.e. how many future tokens are drafted per step).
    pub num_predict_heads: usize,

    /// Draft greedily (argmax per head) instead of sampling from the head.
    pub greedy: bool,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_predict_heads: 1,
            greedy: true,
        }
    }
}

/// Multi-Token Prediction speculator.
///
/// Turns the per-head logit vectors produced by a model's MTP heads into draft
/// tokens (and their probabilities) for verification.
pub struct MtpSpeculator {
    config: MtpConfig,
    rng: rand::rngs::ThreadRng,
    total_proposed: usize,
    total_accepted: usize,
}

impl MtpSpeculator {
    /// Create a new MTP speculator.
    pub fn new(config: MtpConfig) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
            total_proposed: 0,
            total_accepted: 0,
        }
    }

    /// Number of tokens drafted per step (= number of MTP heads).
    pub fn num_draft_tokens(&self) -> usize {
        self.config.num_predict_heads
    }

    /// Convert per-head logit vectors into draft probability distributions.
    pub fn draft_probs(head_logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        head_logits
            .iter()
            .map(|logits| crate::sampling::softmax(logits))
            .collect()
    }

    /// Produce draft tokens from per-head logit vectors.
    ///
    /// Each head proposes one token: greedily (argmax) or by sampling from the
    /// head's softmax distribution, per [`MtpConfig::greedy`].
    pub fn draft_tokens(&mut self, head_logits: &[Vec<f32>]) -> Vec<TokenId> {
        let n = head_logits.len().min(self.config.num_predict_heads);
        let mut tokens = Vec::with_capacity(n);
        for logits in head_logits.iter().take(n) {
            if logits.is_empty() {
                break;
            }
            let token = if self.config.greedy {
                argmax(logits)
            } else {
                self.sample(logits)
            };
            tokens.push(token);
        }
        self.total_proposed += tokens.len();
        tokens
    }

    /// Record how many drafted tokens a verifier accepted (for statistics).
    pub fn record_accepted(&mut self, accepted: usize) {
        self.total_accepted += accepted;
    }

    /// Rolling acceptance rate across all proposed MTP drafts.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_proposed == 0 {
            1.0
        } else {
            self.total_accepted as f32 / self.total_proposed as f32
        }
    }

    fn sample(&mut self, logits: &[f32]) -> TokenId {
        let probs = crate::sampling::softmax(logits);
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return idx as TokenId;
            }
        }
        (probs.len().saturating_sub(1)) as TokenId
    }
}

/// Argmax over a logit slice.
fn argmax(logits: &[f32]) -> TokenId {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as TokenId)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config() {
        let config = SpeculativeDecodingConfig::default();

        assert_eq!(config.num_speculative_tokens, 5);
        assert!(config.adaptive);
    }

    #[test]
    fn test_speculative_decoder() {
        let config = SpeculativeDecodingConfig {
            num_speculative_tokens: 3,
            adaptive: false,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        // Simple test with uniform distributions
        let draft_probs = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let target_probs = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25], // +1 for next token
        ];
        let draft_tokens = vec![0, 1, 2];

        let (accepted, next) = decoder.verify(&draft_probs, &target_probs, &draft_tokens).unwrap();

        // With uniform distributions, all should be accepted
        assert!(!accepted.is_empty());
        assert!(next.is_some());
    }

    #[test]
    fn test_ngram_speculator() {
        let mut speculator = NgramSpeculator::new(3, 5);

        // Build some ngram history
        speculator.update(&[1, 2, 3], 4);
        speculator.update(&[2, 3, 4], 5);
        speculator.update(&[1, 2, 3], 4); // Reinforce

        // Speculate
        let candidates = speculator.speculate(&[1, 2, 3], 2);

        // Should predict 4 as most likely
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], 4);
    }

    #[test]
    fn test_speculative_stats() {
        let stats = SpeculativeStats {
            total_accepted: 80,
            total_proposed: 100,
            acceptance_rate: 0.8,
            current_num_tokens: 5,
        };

        // With draft model 10x faster
        let speedup = stats.speedup(10.0);

        // Expected: (0.8 * 5 + 1) / (1 + 5/10) = 5 / 1.5 ≈ 3.33
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_mtp_config_default() {
        let cfg = MtpConfig::default();
        assert_eq!(cfg.num_predict_heads, 1);
        assert!(cfg.greedy);
    }

    #[test]
    fn test_mtp_greedy_drafting() {
        let mut mtp = MtpSpeculator::new(MtpConfig {
            num_predict_heads: 3,
            greedy: true,
        });
        assert_eq!(mtp.num_draft_tokens(), 3);

        // Three heads, each argmax at a distinct vocab index.
        let head_logits = vec![
            vec![0.1, 5.0, 0.2, 0.3], // -> 1
            vec![3.0, 0.1, 0.1, 0.1], // -> 0
            vec![0.1, 0.1, 0.1, 9.0], // -> 3
        ];
        let drafts = mtp.draft_tokens(&head_logits);
        assert_eq!(drafts, vec![1, 0, 3]);
    }

    #[test]
    fn test_mtp_respects_head_budget() {
        let mut mtp = MtpSpeculator::new(MtpConfig {
            num_predict_heads: 2,
            greedy: true,
        });
        // Four heads available but only two requested.
        let head_logits = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(mtp.draft_tokens(&head_logits).len(), 2);
    }

    #[test]
    fn test_mtp_drafts_verified_by_decoder() {
        // End-to-end: MTP proposes tokens, the rejection-sampling verifier
        // accepts them when the target distribution agrees.
        let mut mtp = MtpSpeculator::new(MtpConfig {
            num_predict_heads: 2,
            greedy: true,
        });
        let head_logits = vec![
            vec![0.0, 10.0, 0.0], // head 0 -> token 1
            vec![10.0, 0.0, 0.0], // head 1 -> token 0
        ];
        let drafts = mtp.draft_tokens(&head_logits);
        assert_eq!(drafts, vec![1, 0]);

        let draft_probs = MtpSpeculator::draft_probs(&head_logits);
        // Target strongly agrees with the drafted tokens at each position, plus
        // one extra distribution for the bonus token.
        let target_probs = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.34, 0.33, 0.33],
        ];

        let mut decoder = SpeculativeDecoder::new(SpeculativeDecodingConfig {
            num_speculative_tokens: 2,
            adaptive: false,
            ..Default::default()
        });
        let (accepted, next) = decoder.verify(&draft_probs, &target_probs, &drafts).unwrap();
        mtp.record_accepted(accepted.len());

        // Both drafts should be accepted, and a bonus token sampled.
        assert_eq!(accepted, vec![1, 0]);
        assert!(next.is_some());
        assert!(mtp.acceptance_rate() > 0.0);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: speculative.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/execution/speculative.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
