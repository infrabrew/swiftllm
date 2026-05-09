// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      grpo.rs
// PATH:      /crates/swiftllm-training/src/grpo.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   - swiftllm-training/src/config.rs        GrpoConfig field lives on TrainingConfig
//   - swiftllm-training/src/optimizer.rs     base Optimizer trait; AdamW drives param updates
//   - swiftllm-training/src/process_reward.rs blend_prm_with_outcome() combines PRM + GRPO reward
//   - swiftllm-training/src/long_reward.rs   dense LongR rewards populate GrpoGroup::rewards
// USED BY:
//   - swiftllm-training/src/lib.rs           re-exports all public types
//   - swiftllm-training/src/config.rs        GrpoConfig embedded in TrainingConfig.grpo
//   - swiftllm-training/src/trainer.rs       training loop reads GrpoConfig to gate RL mode
// SEE ALSO:
//   - swiftllm-training/src/curriculum.rs    CGAR curriculum and phased-spec run in parallel
//   - swiftllm-core/src/sampling/self_consistency.rs  same majority-vote idea at inference time
//   - swiftllm-models/src/architectures/jamba.rs      Jamba is the primary target model
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

//! Group Relative Policy Optimization (GRPO)
//!
//! GRPO is the RL training algorithm used in DeepSeek-R1 to achieve reasoning
//! without supervised fine-tuning. Key innovations over PPO:
//!
//! 1. **No critic model** — eliminates the separate value network. Instead,
//!    a group of G outputs is sampled per prompt; the group mean reward becomes
//!    the baseline. This cuts memory and compute roughly in half vs PPO.
//!
//! 2. **Group-relative advantage** — for each output i in group of size G:
//!    A_i = (r_i - mean(r_1..G)) / std(r_1..G)
//!    This normalises across the group, producing zero-mean advantages.
//!
//! 3. **Rule-based rewards** — DeepSeek-R1-Zero uses only format/correctness
//!    rewards (no learned reward model), enabling "aha moment" emergence.
//!    Supported: correctness, format, length, process-level (PRM) rewards.
//!
//! 4. **KL divergence penalty** — prevents the policy from drifting too far
//!    from the reference (original frozen) model:
//!    L_KL = β * KL(π_θ || π_ref)
//!
//! **Algorithm (one GRPO update step):**
//!   For each prompt x in batch:
//!     1. Sample G outputs {y_1, ..., y_G} from current policy π_θ
//!     2. Score each output: r_i = reward(x, y_i)
//!     3. Compute group-relative advantages: A_i = (r_i - μ) / (σ + ε)
//!     4. Compute policy gradient loss with PPO-style clipping:
//!        L_i = min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i)
//!        where ρ_i = π_θ(y_i|x) / π_θ_old(y_i|x)
//!     5. Add KL penalty: L_total = -mean(L_i) + β * KL
//!     6. Backprop and update θ
//!
//! **DeepSeek-R1 results:**
//!   - AIME 2024 pass@1: 15.6% → 71.0% via RL alone (no SFT)
//!   - MATH-500: 97.3% accuracy
//!   - Emergent "aha moments": self-verification, backtracking, reflection
//!
//! References:
//! - DeepSeek-AI "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
//!   via Reinforcement Learning" (2025)
//! - Shao et al. "DeepSeekMath" (2024) — original GRPO paper
//! - Schulman et al. "Proximal Policy Optimization" (2017)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// GRPO training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Number of output samples per prompt (G). Typical: 8-16.
    /// More samples → more stable advantage estimate, but G× more generation cost.
    pub group_size: usize,

    /// PPO clipping ratio ε. Typical: 0.1-0.2.
    /// Constrains the probability ratio ρ = π_new/π_old to [1-ε, 1+ε].
    pub clip_epsilon: f32,

    /// KL divergence penalty coefficient β. Typical: 0.01-0.1.
    /// Set to 0.0 to disable KL penalty (pure policy gradient).
    pub kl_coeff: f32,

    /// Advantage normalisation epsilon (prevents division by zero)
    pub adv_epsilon: f32,

    /// Reward function(s) to use
    pub reward_fns: Vec<RewardFunction>,

    /// Maximum number of PPO epochs per batch of experience
    pub ppo_epochs: usize,

    /// Mini-batch size within each PPO epoch
    pub mini_batch_size: usize,

    /// Maximum token length for generated outputs
    pub max_gen_len: usize,

    /// Temperature for sampling policy outputs
    pub gen_temperature: f32,

    /// Discard samples with advantage magnitude < this threshold
    /// (avoids training on near-zero-advantage outputs)
    pub advantage_filter_threshold: f32,

    /// Use token-level KL (more accurate) vs sequence-level KL
    pub token_level_kl: bool,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            clip_epsilon: 0.2,
            kl_coeff: 0.01,
            adv_epsilon: 1e-8,
            reward_fns: vec![RewardFunction::Correctness],
            ppo_epochs: 1,
            mini_batch_size: 4,
            max_gen_len: 2048,
            gen_temperature: 0.7,
            advantage_filter_threshold: 1e-6,
            token_level_kl: true,
        }
    }
}

impl GrpoConfig {
    /// Configuration matching DeepSeek-R1-Zero (pure rule-based, no SFT)
    pub fn deepseek_r1_zero() -> Self {
        Self {
            group_size: 8,
            clip_epsilon: 0.2,
            kl_coeff: 0.01,
            reward_fns: vec![
                RewardFunction::Correctness,
                RewardFunction::Format,
            ],
            ppo_epochs: 1,
            gen_temperature: 0.6,
            ..Default::default()
        }
    }

    /// Configuration for math/coding tasks (higher group size, strict correctness)
    pub fn math_reasoning() -> Self {
        Self {
            group_size: 16,
            clip_epsilon: 0.1,
            kl_coeff: 0.005,
            reward_fns: vec![
                RewardFunction::Correctness,
                RewardFunction::ProcessLevel { weight: 0.3 },
            ],
            gen_temperature: 0.8,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Reward functions
// ---------------------------------------------------------------------------

/// Available reward functions for GRPO training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardFunction {
    /// Binary correctness: 1.0 if output matches reference answer, else 0.0.
    /// Requires a reference answer in the dataset.
    Correctness,

    /// Format compliance: checks output follows requested format.
    /// E.g. presence of <think> tags, JSON structure, etc.
    Format,

    /// Length penalty: reward shorter correct answers (prevents length hacking)
    LengthPenalty {
        /// Max tokens before penalty kicks in
        max_tokens: usize,
        /// Penalty per token beyond max_tokens
        penalty_per_token: f32,
    },

    /// Process-level reward using a trained PRM (see process_reward.rs)
    ProcessLevel {
        /// Weight of PRM score in total reward (0..1)
        weight: f32,
    },

    /// Composite: weighted sum of multiple rewards
    Composite {
        /// Sub-rewards and their weights, as (name, weight) pairs
        components: Vec<(String, f32)>,
    },
}

impl RewardFunction {
    /// Name for logging
    pub fn name(&self) -> &str {
        match self {
            RewardFunction::Correctness => "correctness",
            RewardFunction::Format => "format",
            RewardFunction::LengthPenalty { .. } => "length_penalty",
            RewardFunction::ProcessLevel { .. } => "process_level",
            RewardFunction::Composite { .. } => "composite",
        }
    }

    /// Compute reward score given output text and optional reference.
    /// Returns a value in [0.0, 1.0] (higher is better).
    pub fn score(&self, output: &str, reference: Option<&str>) -> f32 {
        match self {
            RewardFunction::Correctness => {
                match reference {
                    Some(ref_ans) => {
                        let clean_out = normalize_answer(output);
                        let clean_ref = normalize_answer(ref_ans);
                        // Exact match first; fall back to containment for
                        // outputs like "The answer is 42" when gold is "42".
                        if clean_out == clean_ref || clean_out.contains(&clean_ref) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    None => 0.5, // No reference: neutral
                }
            }
            RewardFunction::Format => {
                // Check for expected structure markers
                let has_think = output.contains("<think>") && output.contains("</think>");
                let has_answer = output.contains("<answer>") || output.len() > 10;
                if has_think && has_answer { 1.0 }
                else if has_answer { 0.5 }
                else { 0.0 }
            }
            RewardFunction::LengthPenalty { max_tokens, penalty_per_token } => {
                let token_count = output.split_whitespace().count();
                if token_count <= *max_tokens {
                    1.0
                } else {
                    let excess = (token_count - max_tokens) as f32;
                    (1.0 - excess * penalty_per_token).max(0.0)
                }
            }
            RewardFunction::ProcessLevel { weight } => {
                // PRM score would be computed by the external process reward model.
                // Here we return a placeholder; real integration queries PrmScorer.
                *weight * 0.5
            }
            RewardFunction::Composite { components } => {
                let total_weight: f32 = components.iter().map(|(_, w)| w).sum();
                if total_weight == 0.0 {
                    return 0.0;
                }
                // In full implementation: look up each component by name and aggregate
                0.5 // placeholder
            }
        }
    }
}

/// Normalise an answer string for comparison (lowercase, strip whitespace/punct)
fn normalize_answer(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Experience buffer
// ---------------------------------------------------------------------------

/// A single GRPO experience sample
#[derive(Debug, Clone)]
pub struct GrpoSample {
    /// Prompt text
    pub prompt: String,

    /// Generated output
    pub output: String,

    /// Log-probabilities of each output token under the policy at sampling time
    /// (π_θ_old). Length = output token count.
    pub log_probs_old: Vec<f32>,

    /// Log-probabilities under the reference model (for KL)
    pub log_probs_ref: Vec<f32>,

    /// Raw reward score
    pub reward: f32,

    /// Group-relative advantage (computed after group is complete)
    pub advantage: f32,
}

impl GrpoSample {
    /// Sequence-level log probability (sum of token log-probs)
    pub fn seq_log_prob(&self) -> f32 {
        self.log_probs_old.iter().sum()
    }

    /// KL divergence estimate for this sequence (token-level sum)
    pub fn kl_divergence(&self) -> f32 {
        self.log_probs_old
            .iter()
            .zip(self.log_probs_ref.iter())
            .map(|(&p, &q)| p.exp() * (p - q)) // p * log(p/q)
            .sum::<f32>()
            .max(0.0) // KL is non-negative
    }
}

/// A group of G outputs for one prompt (all sampled before advantage computation)
#[derive(Debug, Clone)]
pub struct GrpoGroup {
    /// Prompt shared by all outputs in this group
    pub prompt: String,

    /// G samples (outputs + metadata)
    pub samples: Vec<GrpoSample>,

    /// Group mean reward
    pub mean_reward: f32,

    /// Group reward standard deviation
    pub std_reward: f32,
}

impl GrpoGroup {
    /// Build a group and compute group-relative advantages
    pub fn new(prompt: String, mut samples: Vec<GrpoSample>, adv_epsilon: f32) -> Self {
        assert!(!samples.is_empty(), "GRPO group must have at least 1 sample");

        let rewards: Vec<f32> = samples.iter().map(|s| s.reward).collect();
        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let var_reward = rewards.iter().map(|r| (r - mean_reward).powi(2)).sum::<f32>()
            / rewards.len() as f32;
        let std_reward = var_reward.sqrt();

        // Assign normalised advantages
        for sample in &mut samples {
            sample.advantage = (sample.reward - mean_reward) / (std_reward + adv_epsilon);
        }

        Self { prompt, samples, mean_reward, std_reward }
    }

    /// Whether this group has any learning signal (non-zero advantage variance)
    pub fn has_learning_signal(&self) -> bool {
        self.std_reward > 1e-6
    }

    /// All-same-reward: no learning signal (e.g. all correct or all wrong)
    pub fn is_degenerate(&self) -> bool {
        self.std_reward < 1e-6
    }
}

// ---------------------------------------------------------------------------
// GRPO loss computation
// ---------------------------------------------------------------------------

/// Result of computing the GRPO policy gradient loss
#[derive(Debug, Clone)]
pub struct GrpoLossResult {
    /// Policy gradient loss (negative = better)
    pub pg_loss: f32,

    /// KL divergence penalty
    pub kl_loss: f32,

    /// Total loss: pg_loss + kl_coeff * kl_loss
    pub total_loss: f32,

    /// Mean probability ratio ρ (diagnostic: should stay near 1.0)
    pub mean_ratio: f32,

    /// Fraction of samples where clipping was active
    pub clip_fraction: f32,

    /// Mean reward across all groups in the batch
    pub mean_reward: f32,

    /// Mean advantage magnitude (diagnostic for advantage estimation quality)
    pub mean_abs_advantage: f32,
}

/// Compute the GRPO objective for a batch of experience groups.
///
/// Per token update for sample i with advantage A_i:
///   ρ_i(t) = exp(log_π_new(token t | context) - log_π_old(token t | context))
///   L_clip = min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i)
///   L_kl   = β * (log_π_new - log_π_ref)     [token-level KL]
///   L_total = -mean(L_clip) + mean(L_kl)
///
/// In practice: gradient of L_total w.r.t. model params drives the update.
/// This function computes the scalar losses for logging; actual autograd
/// is handled by the training loop in trainer.rs.
pub fn compute_grpo_loss(
    groups: &[GrpoGroup],
    log_probs_new: &[Vec<f32>],   // Current policy log-probs, one Vec per sample
    config: &GrpoConfig,
    filter_threshold: f32,
) -> GrpoLossResult {
    assert_eq!(
        groups.iter().map(|g| g.samples.len()).sum::<usize>(),
        log_probs_new.len(),
        "log_probs_new must have one entry per sample across all groups"
    );

    let mut total_pg = 0.0f32;
    let mut total_kl = 0.0f32;
    let mut n_tokens = 0usize;
    let mut n_samples = 0usize;
    let mut n_clipped = 0usize;
    let mut sum_ratio = 0.0f32;
    let mut sum_abs_adv = 0.0f32;
    let mut sum_reward = 0.0f32;

    let mut sample_idx = 0;
    for group in groups {
        sum_reward += group.mean_reward;

        if group.is_degenerate() {
            // All same reward → zero advantage everywhere → skip (no signal)
            sample_idx += group.samples.len();
            continue;
        }

        for sample in &group.samples {
            let adv = sample.advantage;

            // Filter near-zero advantages
            if adv.abs() < filter_threshold {
                sample_idx += 1;
                continue;
            }

            sum_abs_adv += adv.abs();
            let new_lps = &log_probs_new[sample_idx];
            let old_lps = &sample.log_probs_old;
            let ref_lps = &sample.log_probs_ref;

            let token_count = new_lps.len().min(old_lps.len());

            for t in 0..token_count {
                let log_ratio = new_lps[t] - old_lps[t];
                let ratio = log_ratio.exp().clamp(0.0, 10.0); // numerical stability
                sum_ratio += ratio;

                // Clipped policy gradient
                let pg_unclipped = ratio * adv;
                let ratio_clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon);
                let pg_clipped = ratio_clipped * adv;
                let pg = pg_unclipped.min(pg_clipped);

                if (ratio - ratio_clipped).abs() > 1e-6 {
                    n_clipped += 1;
                }

                total_pg += pg;

                // KL penalty: π_new * log(π_new / π_ref) = π_new * (log_π_new - log_π_ref)
                if config.kl_coeff > 0.0 {
                    let kl_t = ratio * (new_lps[t] - ref_lps[t].min(new_lps[t] + 10.0));
                    total_kl += kl_t.max(0.0);
                }

                n_tokens += 1;
            }

            n_samples += 1;
            sample_idx += 1;
        }
    }

    let n_t = n_tokens.max(1) as f32;
    let n_s = n_samples.max(1) as f32;
    let n_g = groups.len().max(1) as f32;

    let pg_loss = -total_pg / n_t;     // Negate: we minimise loss
    let kl_loss = total_kl / n_t * config.kl_coeff;
    let total_loss = pg_loss + kl_loss;

    GrpoLossResult {
        pg_loss,
        kl_loss,
        total_loss,
        mean_ratio: sum_ratio / n_t,
        clip_fraction: n_clipped as f32 / n_t,
        mean_reward: sum_reward / n_g,
        mean_abs_advantage: sum_abs_adv / n_s,
    }
}

// ---------------------------------------------------------------------------
// Reward scorer
// ---------------------------------------------------------------------------

/// Multi-function reward scorer: aggregates multiple RewardFunction scores
/// into a single scalar reward with optional weighting.
#[derive(Debug, Clone)]
pub struct RewardScorer {
    /// Reward functions and their weights
    pub functions: Vec<(RewardFunction, f32)>,
}

impl RewardScorer {
    /// Create a scorer with uniform weights
    pub fn new(fns: Vec<RewardFunction>) -> Self {
        let n = fns.len() as f32;
        let functions = fns.into_iter().map(|f| (f, 1.0 / n)).collect();
        Self { functions }
    }

    /// Create with explicit weights
    pub fn weighted(fns: Vec<(RewardFunction, f32)>) -> Self {
        Self { functions: fns }
    }

    /// Score an output given optional reference answer
    pub fn score(&self, output: &str, reference: Option<&str>) -> f32 {
        let total_weight: f32 = self.functions.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        self.functions.iter()
            .map(|(f, w)| f.score(output, reference) * w)
            .sum::<f32>() / total_weight
    }

    /// Score a batch of outputs, returning per-output rewards
    pub fn score_batch(&self, outputs: &[&str], reference: Option<&str>) -> Vec<f32> {
        outputs.iter().map(|o| self.score(o, reference)).collect()
    }
}

// ---------------------------------------------------------------------------
// GRPO trainer (orchestrates generation + scoring + loss)
// ---------------------------------------------------------------------------

/// Statistics collected per GRPO training step
#[derive(Debug, Clone, Default)]
pub struct GrpoStepStats {
    /// Number of prompts processed
    pub num_prompts: usize,
    /// Total samples generated (prompts × group_size)
    pub num_samples: usize,
    /// Mean reward across all groups
    pub mean_reward: f32,
    /// Fraction of groups that were degenerate (all same reward)
    pub degenerate_fraction: f32,
    /// Policy gradient loss
    pub pg_loss: f32,
    /// KL penalty
    pub kl_loss: f32,
    /// Total loss
    pub total_loss: f32,
    /// Mean probability ratio
    pub mean_ratio: f32,
    /// PPO clip fraction
    pub clip_fraction: f32,
}

/// GRPO trainer orchestrator.
///
/// Usage flow (per batch):
///   1. `score_groups()` — generate G outputs per prompt, score them
///   2. `compute_loss()` — compute GRPO objective given current policy log-probs
///   3. Update model via backprop (handled by outer training loop)
///   4. `record_stats()` — accumulate metrics
pub struct GrpoTrainer {
    /// Configuration
    pub config: GrpoConfig,

    /// Reward scorer
    pub scorer: RewardScorer,

    /// Running statistics across steps
    stats_history: Vec<GrpoStepStats>,
}

impl GrpoTrainer {
    /// Create a GRPO trainer from config
    pub fn new(config: GrpoConfig) -> Self {
        let scorer = RewardScorer::new(config.reward_fns.clone());
        Self { config, scorer, stats_history: Vec::new() }
    }

    /// Score a set of prompts and their candidate outputs, building GRPO groups.
    ///
    /// `prompts`: batch of prompt strings
    /// `candidate_outputs`: for each prompt, a Vec of G candidate strings
    /// `references`: optional reference answers per prompt
    pub fn build_groups(
        &self,
        prompts: &[&str],
        candidate_outputs: &[Vec<String>],
        references: Option<&[Option<&str>]>,
    ) -> Vec<GrpoGroup> {
        assert_eq!(prompts.len(), candidate_outputs.len());

        prompts
            .iter()
            .zip(candidate_outputs.iter())
            .enumerate()
            .map(|(i, (&prompt, outputs))| {
                let ref_ans = references.and_then(|r| r[i]);

                let samples = outputs
                    .iter()
                    .map(|out| {
                        let reward = self.scorer.score(out, ref_ans);
                        GrpoSample {
                            prompt: prompt.to_string(),
                            output: out.clone(),
                            log_probs_old: Vec::new(), // filled by caller
                            log_probs_ref: Vec::new(), // filled by caller
                            reward,
                            advantage: 0.0, // computed in GrpoGroup::new
                        }
                    })
                    .collect();

                GrpoGroup::new(
                    prompt.to_string(),
                    samples,
                    self.config.adv_epsilon,
                )
            })
            .collect()
    }

    /// Compute GRPO loss for the current policy
    pub fn compute_loss(
        &self,
        groups: &[GrpoGroup],
        log_probs_new: &[Vec<f32>],
    ) -> GrpoLossResult {
        compute_grpo_loss(
            groups,
            log_probs_new,
            &self.config,
            self.config.advantage_filter_threshold,
        )
    }

    /// Record per-step statistics
    pub fn record_stats(&mut self, stats: GrpoStepStats) {
        self.stats_history.push(stats);
    }

    /// Return the N most recent stats
    pub fn recent_stats(&self, n: usize) -> &[GrpoStepStats] {
        let start = self.stats_history.len().saturating_sub(n);
        &self.stats_history[start..]
    }

    /// Mean reward over recent history
    pub fn mean_recent_reward(&self, n: usize) -> f32 {
        let recent = self.recent_stats(n);
        if recent.is_empty() {
            return 0.0;
        }
        recent.iter().map(|s| s.mean_reward).sum::<f32>() / recent.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpo_config_defaults() {
        let cfg = GrpoConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.clip_epsilon - 0.2).abs() < 1e-6);
        assert!(!cfg.reward_fns.is_empty());
    }

    #[test]
    fn test_reward_correctness_match() {
        let r = RewardFunction::Correctness;
        assert_eq!(r.score("The answer is 42", Some("42")), 1.0);
        assert_eq!(r.score("The answer is 42", Some("43")), 0.0);
    }

    #[test]
    fn test_reward_correctness_normalisation() {
        let r = RewardFunction::Correctness;
        // Case-insensitive, punctuation-insensitive
        assert_eq!(r.score("  PARIS  ", Some("paris")), 1.0);
    }

    #[test]
    fn test_reward_format() {
        let r = RewardFunction::Format;
        let good = "<think>Reasoning here</think><answer>42</answer>";
        let bad = "just some text";
        assert!(r.score(good, None) > r.score(bad, None));
    }

    #[test]
    fn test_reward_length_penalty() {
        let r = RewardFunction::LengthPenalty { max_tokens: 5, penalty_per_token: 0.1 };
        // Short: no penalty
        let short = "one two three";
        assert!((r.score(short, None) - 1.0).abs() < 1e-5);
        // Long: penalty applied
        let long = "one two three four five six seven eight nine ten eleven";
        assert!(r.score(long, None) < 1.0);
    }

    #[test]
    fn test_normalize_answer() {
        assert_eq!(normalize_answer("  Hello, World! "), "hello world");
        assert_eq!(normalize_answer("42"), "42");
    }

    #[test]
    fn test_grpo_group_advantages() {
        let samples = vec![
            GrpoSample { reward: 1.0, prompt: "q".into(), output: "a".into(),
                log_probs_old: vec![], log_probs_ref: vec![], advantage: 0.0 },
            GrpoSample { reward: 0.0, prompt: "q".into(), output: "b".into(),
                log_probs_old: vec![], log_probs_ref: vec![], advantage: 0.0 },
            GrpoSample { reward: 1.0, prompt: "q".into(), output: "c".into(),
                log_probs_old: vec![], log_probs_ref: vec![], advantage: 0.0 },
            GrpoSample { reward: 0.0, prompt: "q".into(), output: "d".into(),
                log_probs_old: vec![], log_probs_ref: vec![], advantage: 0.0 },
        ];
        let group = GrpoGroup::new("q".into(), samples, 1e-8);

        // Mean reward = 0.5
        assert!((group.mean_reward - 0.5).abs() < 1e-5);

        // Advantages: reward=1 → positive, reward=0 → negative
        assert!(group.samples[0].advantage > 0.0, "High reward → positive advantage");
        assert!(group.samples[1].advantage < 0.0, "Low reward → negative advantage");

        // Advantages should sum to ~0 (zero-mean by construction)
        let adv_sum: f32 = group.samples.iter().map(|s| s.advantage).sum();
        assert!(adv_sum.abs() < 1e-4, "Advantages should sum to zero: {}", adv_sum);
    }

    #[test]
    fn test_grpo_group_degenerate() {
        // All same reward → degenerate group (no learning signal)
        let samples = vec![
            GrpoSample { reward: 1.0, prompt: "q".into(), output: "a".into(),
                log_probs_old: vec![], log_probs_ref: vec![], advantage: 0.0 },
            GrpoSample { reward: 1.0, prompt: "q".into(), output: "b".into(),
                log_probs_old: vec![], log_probs_ref: vec![], advantage: 0.0 },
        ];
        let group = GrpoGroup::new("q".into(), samples, 1e-8);
        assert!(group.is_degenerate(), "All-same-reward group should be degenerate");
        assert!(!group.has_learning_signal());
    }

    #[test]
    fn test_grpo_sample_kl() {
        let sample = GrpoSample {
            prompt: "q".into(),
            output: "a".into(),
            log_probs_old: vec![-1.0, -2.0],
            log_probs_ref: vec![-1.5, -2.5],
            reward: 1.0,
            advantage: 0.5,
        };
        let kl = sample.kl_divergence();
        assert!(kl >= 0.0, "KL divergence should be non-negative: {}", kl);
    }

    #[test]
    fn test_compute_grpo_loss_shapes() {
        let config = GrpoConfig::default();
        let samples = vec![
            GrpoSample { reward: 1.0, prompt: "q".into(), output: "ans".into(),
                log_probs_old: vec![-0.5, -0.3], log_probs_ref: vec![-0.6, -0.4], advantage: 0.0 },
            GrpoSample { reward: 0.0, prompt: "q".into(), output: "wrong".into(),
                log_probs_old: vec![-1.5, -1.2], log_probs_ref: vec![-1.6, -1.3], advantage: 0.0 },
        ];
        let group = GrpoGroup::new("q".into(), samples, 1e-8);
        let groups = vec![group];

        // Current policy slightly better for sample 0
        let log_probs_new = vec![vec![-0.4f32, -0.2], vec![-1.6, -1.3]];
        let result = compute_grpo_loss(&groups, &log_probs_new, &config, 0.0);

        assert!(result.total_loss.is_finite(), "Loss should be finite");
        assert!(result.mean_ratio > 0.0, "Mean ratio should be positive");
        assert!(result.clip_fraction >= 0.0 && result.clip_fraction <= 1.0);
    }

    #[test]
    fn test_reward_scorer_uniform() {
        let scorer = RewardScorer::new(vec![
            RewardFunction::Correctness,
            RewardFunction::Format,
        ]);
        let score = scorer.score("correct", Some("correct"));
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_grpo_trainer_build_groups() {
        let config = GrpoConfig::default();
        let trainer = GrpoTrainer::new(config);

        let prompts = ["What is 1+1?", "Capital of France?"];
        let outputs = vec![
            vec!["2".to_string(), "Three".to_string()],
            vec!["Paris".to_string(), "London".to_string()],
        ];
        let refs: Vec<Option<&str>> = vec![Some("2"), Some("Paris")];

        let groups = trainer.build_groups(&prompts, &outputs, Some(&refs));
        assert_eq!(groups.len(), 2);
        // First group: "2" correct, "Three" wrong
        assert!(groups[0].samples[0].reward > groups[0].samples[1].reward);
        // Second group: "Paris" correct, "London" wrong
        assert!(groups[1].samples[0].reward > groups[1].samples[1].reward);
    }

    #[test]
    fn test_grpo_trainer_stats_tracking() {
        let config = GrpoConfig::default();
        let mut trainer = GrpoTrainer::new(config);
        trainer.record_stats(GrpoStepStats { mean_reward: 0.3, ..Default::default() });
        trainer.record_stats(GrpoStepStats { mean_reward: 0.6, ..Default::default() });

        let mean = trainer.mean_recent_reward(10);
        assert!((mean - 0.45).abs() < 1e-5);
    }

    #[test]
    fn test_deepseek_r1_zero_config() {
        let cfg = GrpoConfig::deepseek_r1_zero();
        assert_eq!(cfg.group_size, 8);
        assert!(cfg.reward_fns.iter().any(|r| r.name() == "correctness"));
        assert!(cfg.reward_fns.iter().any(|r| r.name() == "format"));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: grpo.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/grpo.rs
// INTEGRATES:  config.rs · optimizer.rs · process_reward.rs · long_reward.rs · trainer.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
