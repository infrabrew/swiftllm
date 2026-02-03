//! Composable Sampling Strategies
//!
//! This module provides composable sampling strategies that can be
//! chained together to create complex sampling pipelines.

use crate::error::Result;
use crate::types::TokenId;

/// Trait for sampling strategies
pub trait Sampler: Send + Sync {
    /// Apply the sampling strategy to logits
    fn apply(&self, logits: &mut [f32], context: &SamplingContext);

    /// Get the name of this sampler
    fn name(&self) -> &str;
}

/// Context for sampling (previous tokens, etc.)
#[derive(Debug, Clone, Default)]
pub struct SamplingContext {
    /// Previously generated tokens
    pub previous_tokens: Vec<TokenId>,

    /// Token frequency counts
    pub token_counts: std::collections::HashMap<TokenId, usize>,

    /// Current position
    pub position: usize,
}

impl SamplingContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a token to the context
    pub fn add_token(&mut self, token_id: TokenId) {
        self.previous_tokens.push(token_id);
        *self.token_counts.entry(token_id).or_insert(0) += 1;
        self.position += 1;
    }

    /// Clear the context
    pub fn clear(&mut self) {
        self.previous_tokens.clear();
        self.token_counts.clear();
        self.position = 0;
    }
}

/// Temperature sampler
#[derive(Debug, Clone)]
pub struct TemperatureSampler {
    /// Temperature value
    pub temperature: f32,
}

impl TemperatureSampler {
    /// Create a new temperature sampler
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl Sampler for TemperatureSampler {
    fn apply(&self, logits: &mut [f32], _context: &SamplingContext) {
        if self.temperature != 1.0 && self.temperature > 0.0 {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }

    fn name(&self) -> &str {
        "temperature"
    }
}

/// Top-K sampler
#[derive(Debug, Clone)]
pub struct TopKSampler {
    /// K value (number of top tokens to keep)
    pub k: usize,
}

impl TopKSampler {
    /// Create a new top-k sampler
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Sampler for TopKSampler {
    fn apply(&self, logits: &mut [f32], _context: &SamplingContext) {
        if self.k == 0 || self.k >= logits.len() {
            return;
        }

        // Find the k-th largest value
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[self.k - 1];

        // Set all values below threshold to -inf
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

/// Top-P (nucleus) sampler
#[derive(Debug, Clone)]
pub struct TopPSampler {
    /// P value (cumulative probability threshold)
    pub p: f32,
}

impl TopPSampler {
    /// Create a new top-p sampler
    pub fn new(p: f32) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }
}

impl Sampler for TopPSampler {
    fn apply(&self, logits: &mut [f32], _context: &SamplingContext) {
        if self.p >= 1.0 {
            return;
        }

        // Convert to probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();

        // Sort indices by probability (descending)
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = logits.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum > self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Set excluded tokens to -inf
        for &idx in &indices[cutoff_idx..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

/// Min-P sampler
#[derive(Debug, Clone)]
pub struct MinPSampler {
    /// P value (minimum probability as fraction of max)
    pub p: f32,
}

impl MinPSampler {
    /// Create a new min-p sampler
    pub fn new(p: f32) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }
}

impl Sampler for MinPSampler {
    fn apply(&self, logits: &mut [f32], _context: &SamplingContext) {
        if self.p <= 0.0 {
            return;
        }

        // Convert to probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();

        // Find max probability
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = max_prob * self.p;

        // Filter tokens below threshold
        for (logit, &prob) in logits.iter_mut().zip(probs.iter()) {
            if prob < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "min_p"
    }
}

/// Repetition penalty sampler
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    /// Penalty value (> 1.0 penalizes, < 1.0 encourages)
    pub penalty: f32,
}

impl RepetitionPenalty {
    /// Create a new repetition penalty
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl Sampler for RepetitionPenalty {
    fn apply(&self, logits: &mut [f32], context: &SamplingContext) {
        if self.penalty == 1.0 {
            return;
        }

        for &token_id in &context.previous_tokens {
            if (token_id as usize) < logits.len() {
                let logit = logits[token_id as usize];
                if logit > 0.0 {
                    logits[token_id as usize] = logit / self.penalty;
                } else {
                    logits[token_id as usize] = logit * self.penalty;
                }
            }
        }
    }

    fn name(&self) -> &str {
        "repetition_penalty"
    }
}

/// Greedy sampler (always picks the highest logit)
#[derive(Debug, Clone, Default)]
pub struct GreedySampler;

impl GreedySampler {
    /// Create a new greedy sampler
    pub fn new() -> Self {
        Self
    }

    /// Sample a token
    pub fn sample(&self, logits: &[f32]) -> TokenId {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as TokenId)
            .unwrap_or(0)
    }
}

impl Sampler for GreedySampler {
    fn apply(&self, logits: &mut [f32], _context: &SamplingContext) {
        // Find max and set all others to -inf
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        for (i, logit) in logits.iter_mut().enumerate() {
            if i != max_idx {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

/// Beam search sampler
#[derive(Debug, Clone)]
pub struct BeamSearchSampler {
    /// Number of beams
    pub num_beams: usize,

    /// Length penalty
    pub length_penalty: f32,

    /// Early stopping
    pub early_stopping: bool,
}

impl BeamSearchSampler {
    /// Create a new beam search sampler
    pub fn new(num_beams: usize) -> Self {
        Self {
            num_beams,
            length_penalty: 1.0,
            early_stopping: false,
        }
    }

    /// Get top-k candidates for beam search
    pub fn get_top_candidates(&self, logits: &[f32]) -> Vec<(TokenId, f32)> {
        let mut indexed: Vec<(TokenId, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &logit)| (idx as TokenId, logit))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(self.num_beams);
        indexed
    }
}

impl Sampler for BeamSearchSampler {
    fn apply(&self, logits: &mut [f32], _context: &SamplingContext) {
        // For beam search, we keep top-k candidates
        let mut sorted: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let threshold = if sorted.len() > self.num_beams {
            sorted[self.num_beams - 1].1
        } else {
            f32::NEG_INFINITY
        };

        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "beam_search"
    }
}

/// Chain of samplers
pub struct SamplerChain {
    /// Samplers in order
    samplers: Vec<Box<dyn Sampler>>,
}

impl SamplerChain {
    /// Create an empty sampler chain
    pub fn new() -> Self {
        Self {
            samplers: Vec::new(),
        }
    }

    /// Add a sampler to the chain
    pub fn add<S: Sampler + 'static>(mut self, sampler: S) -> Self {
        self.samplers.push(Box::new(sampler));
        self
    }

    /// Apply all samplers in order
    pub fn apply(&self, logits: &mut [f32], context: &SamplingContext) {
        for sampler in &self.samplers {
            sampler.apply(logits, context);
        }
    }

    /// Get the names of all samplers
    pub fn names(&self) -> Vec<&str> {
        self.samplers.iter().map(|s| s.name()).collect()
    }
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a standard sampling chain from parameters
pub fn create_sampler_chain(
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
) -> SamplerChain {
    let mut chain = SamplerChain::new();

    // Add repetition penalty first
    if repetition_penalty != 1.0 {
        chain = chain.add(RepetitionPenalty::new(repetition_penalty));
    }

    // Temperature scaling
    if temperature > 0.0 && temperature != 1.0 {
        chain = chain.add(TemperatureSampler::new(temperature));
    }

    // Min-p filtering (before top-k/top-p)
    if min_p > 0.0 {
        chain = chain.add(MinPSampler::new(min_p));
    }

    // Top-k filtering
    if top_k > 0 {
        chain = chain.add(TopKSampler::new(top_k as usize));
    }

    // Top-p filtering
    if top_p < 1.0 && top_p > 0.0 {
        chain = chain.add(TopPSampler::new(top_p));
    }

    // Greedy if temperature is 0
    if temperature == 0.0 {
        chain = chain.add(GreedySampler::new());
    }

    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_sampler() {
        let sampler = TemperatureSampler::new(2.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let context = SamplingContext::new();

        sampler.apply(&mut logits, &context);

        assert_eq!(logits, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_top_k_sampler() {
        let sampler = TopKSampler::new(2);
        let mut logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let context = SamplingContext::new();

        sampler.apply(&mut logits, &context);

        // Only top 2 (indices 1 and 3) should remain
        assert!(logits[1].is_finite());
        assert!(logits[3].is_finite());
        assert!(logits[0].is_neg_infinity());
        assert!(logits[2].is_neg_infinity());
        assert!(logits[4].is_neg_infinity());
    }

    #[test]
    fn test_greedy_sampler() {
        let sampler = GreedySampler::new();
        let logits = vec![1.0, 5.0, 2.0, 3.0, 4.0];

        let token = sampler.sample(&logits);
        assert_eq!(token, 1); // Index of max value
    }

    #[test]
    fn test_repetition_penalty() {
        let sampler = RepetitionPenalty::new(2.0);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut context = SamplingContext::new();
        context.add_token(2);
        context.add_token(4);

        sampler.apply(&mut logits, &context);

        // Tokens 2 and 4 should be penalized
        assert_eq!(logits[2], 1.5); // 3.0 / 2.0
        assert_eq!(logits[4], 2.5); // 5.0 / 2.0
        assert_eq!(logits[0], 1.0); // Unchanged
    }

    #[test]
    fn test_sampler_chain() {
        let chain = SamplerChain::new()
            .add(RepetitionPenalty::new(1.5))
            .add(TemperatureSampler::new(0.8))
            .add(TopKSampler::new(3));

        let names = chain.names();
        assert_eq!(names, vec!["repetition_penalty", "temperature", "top_k"]);
    }

    #[test]
    fn test_create_sampler_chain() {
        let chain = create_sampler_chain(0.7, 50, 0.9, 0.1, 1.2);
        let names = chain.names();

        assert!(names.contains(&"repetition_penalty"));
        assert!(names.contains(&"temperature"));
        assert!(names.contains(&"min_p"));
        assert!(names.contains(&"top_k"));
        assert!(names.contains(&"top_p"));
    }
}
