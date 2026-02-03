//! Token Sampling Strategies
//!
//! This module implements various token sampling strategies for
//! text generation, including temperature, top-k, top-p (nucleus),
//! and other advanced techniques.

mod strategies;

pub use strategies::{
    BeamSearchSampler, GreedySampler, MinPSampler, RepetitionPenalty, Sampler, SamplerChain,
    TemperatureSampler, TopKSampler, TopPSampler,
};

use crate::config::SamplingConfig;
use crate::error::{Error, Result};
use crate::types::{Token, TokenId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Sampling parameters
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature (0.0 = greedy, higher = more random)
    pub temperature: f32,

    /// Top-k sampling (keep only top k tokens)
    pub top_k: i32,

    /// Top-p (nucleus) sampling (keep tokens with cumulative prob <= p)
    pub top_p: f32,

    /// Min-p sampling (filter tokens with prob < p * max_prob)
    pub min_p: f32,

    /// Repetition penalty
    pub repetition_penalty: f32,

    /// Frequency penalty (OpenAI style)
    pub frequency_penalty: f32,

    /// Presence penalty (OpenAI style)
    pub presence_penalty: f32,

    /// Random seed (None for random)
    pub seed: Option<u64>,

    /// Logit bias
    pub logit_bias: HashMap<TokenId, f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            logit_bias: HashMap::new(),
        }
    }
}

impl From<&SamplingConfig> for SamplingParams {
    fn from(config: &SamplingConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            repetition_penalty: config.repetition_penalty,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            seed: config.seed,
            logit_bias: config.logit_bias.clone().unwrap_or_default(),
        }
    }
}

impl SamplingParams {
    /// Create greedy sampling parameters
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Create default sampling with temperature
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Check if sampling is greedy
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }
}

/// Sampling strategy (combining multiple techniques)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Pure greedy (argmax)
    Greedy,
    /// Temperature sampling only
    Temperature,
    /// Top-k sampling
    TopK,
    /// Top-p (nucleus) sampling
    TopP,
    /// Combined top-k and top-p
    TopKTopP,
    /// Min-p sampling
    MinP,
    /// Beam search
    BeamSearch,
}

/// Token sampler
pub struct TokenSampler {
    /// Sampling parameters
    params: SamplingParams,

    /// Random number generator
    rng: StdRng,

    /// Token frequency counts (for frequency/presence penalty)
    token_counts: HashMap<TokenId, usize>,
}

impl TokenSampler {
    /// Create a new token sampler
    pub fn new(params: SamplingParams) -> Self {
        let rng = match params.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            params,
            rng,
            token_counts: HashMap::new(),
        }
    }

    /// Sample a token from logits
    pub fn sample(&mut self, logits: &[f32]) -> Result<Token> {
        if logits.is_empty() {
            return Err(Error::Sampling("Empty logits".to_string()));
        }

        // Apply modifications in order:
        // 1. Logit bias
        // 2. Repetition penalty
        // 3. Frequency/presence penalty
        // 4. Temperature
        // 5. Top-k / Top-p / Min-p filtering
        // 6. Sample

        let mut logits = logits.to_vec();

        // Apply logit bias
        for (&token_id, &bias) in &self.params.logit_bias {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] += bias;
            }
        }

        // Apply repetition penalty
        if self.params.repetition_penalty != 1.0 {
            for (&token_id, _) in &self.token_counts {
                if (token_id as usize) < logits.len() {
                    let logit = logits[token_id as usize];
                    if logit > 0.0 {
                        logits[token_id as usize] = logit / self.params.repetition_penalty;
                    } else {
                        logits[token_id as usize] = logit * self.params.repetition_penalty;
                    }
                }
            }
        }

        // Apply frequency and presence penalties
        if self.params.frequency_penalty != 0.0 || self.params.presence_penalty != 0.0 {
            for (&token_id, &count) in &self.token_counts {
                if (token_id as usize) < logits.len() {
                    logits[token_id as usize] -= self.params.frequency_penalty * count as f32;
                    if count > 0 {
                        logits[token_id as usize] -= self.params.presence_penalty;
                    }
                }
            }
        }

        // Sample based on strategy
        let token_id = if self.params.is_greedy() {
            self.sample_greedy(&logits)
        } else {
            self.sample_with_filtering(&mut logits)?
        };

        // Update token counts
        *self.token_counts.entry(token_id).or_insert(0) += 1;

        // Get logprob
        let logprob = self.compute_logprob(&logits, token_id);

        Ok(Token {
            id: token_id,
            text: None,
            logprob: Some(logprob),
            top_logprobs: None,
        })
    }

    /// Greedy sampling (argmax)
    fn sample_greedy(&self, logits: &[f32]) -> TokenId {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as TokenId)
            .unwrap_or(0)
    }

    /// Sample with filtering (temperature, top-k, top-p, min-p)
    fn sample_with_filtering(&mut self, logits: &mut [f32]) -> Result<TokenId> {
        let vocab_size = logits.len();

        // Apply temperature
        if self.params.temperature != 1.0 && self.params.temperature > 0.0 {
            for logit in logits.iter_mut() {
                *logit /= self.params.temperature;
            }
        }

        // Convert to probabilities (softmax)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }
        for prob in logits.iter_mut() {
            *prob /= sum;
        }

        // Create sorted indices
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

        // Apply min-p filter
        if self.params.min_p > 0.0 {
            let max_prob = logits[indices[0]];
            let threshold = max_prob * self.params.min_p;
            for i in 0..vocab_size {
                if logits[i] < threshold {
                    logits[i] = 0.0;
                }
            }
        }

        // Apply top-k filter
        if self.params.top_k > 0 && (self.params.top_k as usize) < vocab_size {
            let k = self.params.top_k as usize;
            for &idx in &indices[k..] {
                logits[idx] = 0.0;
            }
        }

        // Apply top-p filter
        if self.params.top_p < 1.0 && self.params.top_p > 0.0 {
            let mut cumsum = 0.0;
            let mut cutoff_idx = vocab_size;

            for (i, &idx) in indices.iter().enumerate() {
                cumsum += logits[idx];
                if cumsum > self.params.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            for &idx in &indices[cutoff_idx..] {
                logits[idx] = 0.0;
            }
        }

        // Renormalize
        sum = logits.iter().sum();
        if sum <= 0.0 {
            // Fallback to greedy if all probabilities are zero
            return Ok(indices[0] as TokenId);
        }

        for prob in logits.iter_mut() {
            *prob /= sum;
        }

        // Sample from the distribution
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;

        for (idx, &prob) in logits.iter().enumerate() {
            cumsum += prob;
            if cumsum >= r {
                return Ok(idx as TokenId);
            }
        }

        // Fallback to last token
        Ok((vocab_size - 1) as TokenId)
    }

    /// Compute log probability for a token
    fn compute_logprob(&self, logits: &[f32], token_id: TokenId) -> f32 {
        if (token_id as usize) >= logits.len() {
            return f32::NEG_INFINITY;
        }

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

        (logits[token_id as usize] - max_logit) - sum.ln()
    }

    /// Get top log probabilities
    pub fn get_top_logprobs(&self, logits: &[f32], top_n: usize) -> Vec<(TokenId, f32)> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_sum = sum.ln();

        let mut indexed: Vec<(TokenId, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &logit)| {
                let logprob = (logit - max_logit) - log_sum;
                (idx as TokenId, logprob)
            })
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_n);
        indexed
    }

    /// Reset token counts (for new sequence)
    pub fn reset(&mut self) {
        self.token_counts.clear();
    }

    /// Set the seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

/// Sample multiple tokens (for beam search)
pub fn sample_top_n(logits: &[f32], n: usize) -> Vec<(TokenId, f32)> {
    let mut indexed: Vec<(TokenId, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| (idx as TokenId, logit))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(n);
    indexed
}

/// Apply softmax to logits
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

/// Apply log softmax to logits
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = sum.ln();
    logits.iter().map(|&x| (x - max) - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let params = SamplingParams::greedy();
        let mut sampler = TokenSampler::new(params);

        let logits = vec![1.0, 5.0, 2.0, 3.0, 4.0];
        let token = sampler.sample(&logits).unwrap();

        assert_eq!(token.id, 1); // Index of highest value
    }

    #[test]
    fn test_temperature_sampling() {
        let params = SamplingParams {
            temperature: 0.5,
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params);

        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // With low temperature, should favor higher logits
        let mut counts = vec![0; 5];
        for _ in 0..1000 {
            sampler.reset();
            let token = sampler.sample(&logits).unwrap();
            counts[token.id as usize] += 1;
        }

        // Token 4 (highest logit) should be sampled most often
        assert!(counts[4] > counts[0]);
    }

    #[test]
    fn test_top_k_sampling() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params);

        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];

        for _ in 0..100 {
            sampler.reset();
            let token = sampler.sample(&logits).unwrap();
            // Should only sample from top 2 (indices 1 and 3)
            assert!(token.id == 1 || token.id == 3);
        }
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);

        // Log probs should be negative
        for &lp in &log_probs {
            assert!(lp <= 0.0);
        }

        // exp(log_softmax) should equal softmax
        let probs: Vec<f32> = log_probs.iter().map(|&x| x.exp()).collect();
        let expected = softmax(&logits);

        for (a, b) in probs.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
