//! Speculative Decoding
//!
//! This module implements speculative decoding, a technique that uses
//! a smaller draft model to propose tokens that are then verified by
//! the target model, potentially accelerating inference.

use crate::error::{Error, Result};
use crate::sampling::TokenSampler;
use crate::types::{Token, TokenId};
use rand::Rng;
use std::sync::Arc;

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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
pub struct NgramSpeculator {
    /// Ngram lookup table
    ngram_table: std::collections::HashMap<Vec<TokenId>, Vec<(TokenId, usize)>>,

    /// Maximum ngram size
    max_ngram_size: usize,

    /// Maximum number of candidates per ngram
    max_candidates: usize,
}

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

            let entry = self.ngram_table.entry(ngram).or_insert_with(Vec::new);

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
}
