// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      curriculum.rs
// PATH:      /crates/swiftllm-training/src/curriculum.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   - swiftllm-models/src/architectures/jamba.rs  HybridLayerType schedule drives phased-spec
//   - swiftllm-models/src/layers/mamba.rs          Mamba SSM layers are the SSM-lead targets
//   - swiftllm-models/src/layers/moe.rs            MoE FFN is frozen/activated by CGAR depth
// USED BY:
//   - swiftllm-training/src/trainer.rs    CurriculumState::step() called every training step
//   - swiftllm-training/src/config.rs     CgarConfig + PhasedSpecialisationConfig on TrainingConfig
//   - swiftllm-training/src/lib.rs        re-exports CgarScheduler, CurriculumTick, etc.
// SEE ALSO:
//   - swiftllm-training/src/grpo.rs       GRPO rewards run inside the same training step
//   - swiftllm-training/src/trainer.rs    apply_curriculum_lr() scales LR from CurriculumTick
//   - swiftllm-core/src/serving/disaggregated.rs  serving-side complement (phase-aware batching)
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

//! Curriculum-Guided Adaptive Recursion (CGAR) and Phased Training Schedules
//!
//! This module implements two complementary training efficiency techniques
//! from the research papers:
//!
//! ## CGAR (Curriculum-Guided Adaptive Recursion)
//!
//! For hybrid Mamba+Attention models, training with full depth from step 0
//! causes conflicting gradients between SSM and attention layers. CGAR
//! addresses this by progressively increasing training depth:
//!
//! ```text
//! Phase 1 (0%–30% of training steps): shallow
//!   — Only the first K layers active; rest frozen
//!   — Low computational cost, rapid initial convergence
//!
//! Phase 2 (30%–60%): medium
//!   — First 2K layers active; deeper layers still frozen
//!   — Intermediate representations stabilise
//!
//! Phase 3 (60%–100%): full
//!   — All layers active; standard training
//!   — Fine-grained adjustment of all parameters
//! ```
//!
//! **Results:** 1.71× training speedup (10.93h → 6.38h) with only 0.63%
//! accuracy drop on downstream benchmarks.
//!
//! ## Phased Specialisation (Hybrid Models)
//!
//! For hybrid Transformer + Mamba architectures, gradient conflicts between
//! attention and SSM layers degrade convergence. Phased specialisation
//! bifurcates training into two phases:
//!
//! ```text
//! Phase 1 (Attention Lead):
//!   — Attention/MoE layers: full learning rate
//!   — Mamba SSM layers: minimal learning rate (lr × 0.1)
//!   — Duration: first 40% of training
//!
//! Phase 2 (SSM Specialisation):
//!   — Attention/MoE layers: minimal learning rate (lr × 0.1)
//!   — Mamba SSM layers: full learning rate
//!   — Duration: remaining 60% of training
//! ```
//!
//! **Results:** 70% variance reduction in loss curves, 53% reduction in
//! gradient conflict rate between SSM and attention parameters.
//!
//! ## Hierarchical Supervision Weighting
//!
//! In multi-exit or recursive models, later representations should receive
//! exponentially more supervision weight. For a model with N exits:
//!   weight(exit k) = exp(λ * k / N) / Z   where Z is the normalisation constant
//!
//! This ensures early exits learn quickly without dominating the gradient signal.
//!
//! References:
//! - Post-Transformer Paradigm paper (2026): CGAR, LongR, dense verification
//! - Hybrid Architectures paper (2026): phased specialisation, conflict analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CGAR Configuration
// ---------------------------------------------------------------------------

/// Configuration for Curriculum-Guided Adaptive Recursion (CGAR)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgarConfig {
    /// Fraction of total steps spent in shallow phase (0.0–1.0)
    pub shallow_fraction: f32,

    /// Fraction of total steps spent in medium phase (0.0–1.0)
    /// deep_fraction = 1.0 - shallow - medium
    pub medium_fraction: f32,

    /// Fraction of layers active during shallow phase (0.0–1.0)
    pub shallow_layer_fraction: f32,

    /// Fraction of layers active during medium phase (0.0–1.0)
    pub medium_layer_fraction: f32,

    /// Whether to freeze inactive layers (True) or just zero their gradients
    pub freeze_inactive: bool,

    /// Minimum number of active layers (always at least this many)
    pub min_active_layers: usize,

    /// Smoothly ramp layer count at phase transitions instead of step-change
    pub smooth_transitions: bool,

    /// Transition ramp width as fraction of phase length
    pub ramp_fraction: f32,
}

impl Default for CgarConfig {
    fn default() -> Self {
        Self {
            shallow_fraction: 0.30,
            medium_fraction: 0.30,
            shallow_layer_fraction: 0.33,
            medium_layer_fraction: 0.67,
            freeze_inactive: true,
            min_active_layers: 1,
            smooth_transitions: true,
            ramp_fraction: 0.05,
        }
    }
}

impl CgarConfig {
    /// Validate that fractions sum correctly
    pub fn validate(&self) -> Result<(), String> {
        if self.shallow_fraction + self.medium_fraction > 1.0 {
            return Err(format!(
                "shallow ({}) + medium ({}) fractions must be ≤ 1.0",
                self.shallow_fraction, self.medium_fraction
            ));
        }
        if self.shallow_layer_fraction > self.medium_layer_fraction {
            return Err("shallow_layer_fraction must be ≤ medium_layer_fraction".into());
        }
        if self.medium_layer_fraction > 1.0 {
            return Err("medium_layer_fraction must be ≤ 1.0".into());
        }
        Ok(())
    }

    /// Fraction of steps in full/deep phase
    pub fn deep_fraction(&self) -> f32 {
        1.0 - self.shallow_fraction - self.medium_fraction
    }
}

// ---------------------------------------------------------------------------
// Training phase tracking
// ---------------------------------------------------------------------------

/// Current phase of CGAR training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CgarPhase {
    /// Shallow phase: first layers only
    Shallow,
    /// Medium phase: first two-thirds of layers
    Medium,
    /// Full/deep phase: all layers active
    Full,
}

impl CgarPhase {
    /// Human-readable label
    pub fn label(&self) -> &str {
        match self {
            CgarPhase::Shallow => "shallow",
            CgarPhase::Medium => "medium",
            CgarPhase::Full => "full",
        }
    }
}

/// Phased specialisation phase (attention-lead vs SSM-lead)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecialisationPhase {
    /// Attention layers at full LR; Mamba at reduced LR
    AttentionLead,
    /// Mamba layers at full LR; attention at reduced LR
    SsmLead,
}

// ---------------------------------------------------------------------------
// CGAR curriculum scheduler
// ---------------------------------------------------------------------------

/// CGAR curriculum scheduler: controls which layers are active at each step.
///
/// Maintains the current phase and exposes the number of active layers,
/// which the training loop uses to decide which parameters receive gradients.
pub struct CgarScheduler {
    /// Configuration
    pub config: CgarConfig,

    /// Total training steps
    total_steps: usize,

    /// Total number of model layers
    num_layers: usize,

    /// Current global step
    current_step: usize,
}

impl CgarScheduler {
    pub fn new(config: CgarConfig, total_steps: usize, num_layers: usize) -> Self {
        let _ = config.validate().expect("Invalid CGAR config");
        Self { config, total_steps, num_layers, current_step: 0 }
    }

    /// Advance one step; returns the active layer count for this step
    pub fn step(&mut self) -> usize {
        self.current_step += 1;
        self.active_layers()
    }

    /// Current training phase
    pub fn phase(&self) -> CgarPhase {
        let progress = self.current_step as f32 / self.total_steps.max(1) as f32;
        if progress < self.config.shallow_fraction {
            CgarPhase::Shallow
        } else if progress < self.config.shallow_fraction + self.config.medium_fraction {
            CgarPhase::Medium
        } else {
            CgarPhase::Full
        }
    }

    /// Number of active layers for the current step.
    ///
    /// With smooth_transitions enabled, the count ramps linearly over
    /// a fraction of the phase length rather than stepping instantly.
    pub fn active_layers(&self) -> usize {
        let n = self.num_layers;
        let progress = self.current_step as f32 / self.total_steps.max(1) as f32;

        let target_fraction = if !self.config.smooth_transitions {
            match self.phase() {
                CgarPhase::Shallow => self.config.shallow_layer_fraction,
                CgarPhase::Medium => self.config.medium_layer_fraction,
                CgarPhase::Full => 1.0,
            }
        } else {
            self.smooth_layer_fraction(progress)
        };

        let active = (n as f32 * target_fraction).ceil() as usize;
        active.max(self.config.min_active_layers).min(n)
    }

    /// Smoothly interpolated layer fraction at a given training progress (0..1)
    fn smooth_layer_fraction(&self, progress: f32) -> f32 {
        let cfg = &self.config;
        let p_shallow_end = cfg.shallow_fraction;
        let p_medium_end = cfg.shallow_fraction + cfg.medium_fraction;
        let ramp = cfg.ramp_fraction;

        // Shallow → Medium transition
        if progress >= p_shallow_end - ramp && progress < p_shallow_end + ramp {
            let t = (progress - (p_shallow_end - ramp)) / (2.0 * ramp);
            let t = t.clamp(0.0, 1.0);
            return lerp(cfg.shallow_layer_fraction, cfg.medium_layer_fraction, smooth_step(t));
        }

        // Medium → Full transition
        if progress >= p_medium_end - ramp && progress < p_medium_end + ramp {
            let t = (progress - (p_medium_end - ramp)) / (2.0 * ramp);
            let t = t.clamp(0.0, 1.0);
            return lerp(cfg.medium_layer_fraction, 1.0, smooth_step(t));
        }

        match self.phase() {
            CgarPhase::Shallow => cfg.shallow_layer_fraction,
            CgarPhase::Medium => cfg.medium_layer_fraction,
            CgarPhase::Full => 1.0,
        }
    }

    /// Whether a given layer index is currently active
    pub fn is_layer_active(&self, layer_idx: usize) -> bool {
        layer_idx < self.active_layers()
    }

    /// Progress within the current phase (0.0 = just entered, 1.0 = about to exit)
    pub fn phase_progress(&self) -> f32 {
        let progress = self.current_step as f32 / self.total_steps.max(1) as f32;
        match self.phase() {
            CgarPhase::Shallow => progress / self.config.shallow_fraction,
            CgarPhase::Medium => {
                (progress - self.config.shallow_fraction) / self.config.medium_fraction
            }
            CgarPhase::Full => {
                let start = self.config.shallow_fraction + self.config.medium_fraction;
                (progress - start) / self.config.deep_fraction().max(1e-8)
            }
        }
        .clamp(0.0, 1.0)
    }

    /// Estimated remaining steps in the current phase
    pub fn steps_remaining_in_phase(&self) -> usize {
        let progress = self.current_step as f32 / self.total_steps.max(1) as f32;
        let phase_end = match self.phase() {
            CgarPhase::Shallow => self.config.shallow_fraction,
            CgarPhase::Medium => self.config.shallow_fraction + self.config.medium_fraction,
            CgarPhase::Full => 1.0,
        };
        let steps_at_end = (phase_end * self.total_steps as f32) as usize;
        steps_at_end.saturating_sub(self.current_step)
    }
}

// ---------------------------------------------------------------------------
// Phased specialisation scheduler (hybrid models)
// ---------------------------------------------------------------------------

/// Configuration for hybrid model phased specialisation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasedSpecialisationConfig {
    /// Fraction of training steps in attention-lead phase
    pub attention_lead_fraction: f32,

    /// LR multiplier for the low-priority component in each phase
    /// (default: 0.1 — 10× lower than the high-priority component)
    pub low_priority_lr_scale: f32,

    /// Whether to linearly ramp between phases rather than hard switch
    pub smooth_transition: bool,

    /// Transition ramp width as fraction of total steps
    pub transition_width: f32,
}

impl Default for PhasedSpecialisationConfig {
    fn default() -> Self {
        Self {
            attention_lead_fraction: 0.4,
            low_priority_lr_scale: 0.1,
            smooth_transition: true,
            transition_width: 0.05,
        }
    }
}

/// Phased specialisation scheduler for hybrid Mamba+Attention models.
///
/// Returns per-component LR multipliers at each step so the training loop
/// can scale the effective learning rate for attention vs SSM parameters.
pub struct PhasedSpecialisationScheduler {
    pub config: PhasedSpecialisationConfig,
    total_steps: usize,
    current_step: usize,
}

impl PhasedSpecialisationScheduler {
    pub fn new(config: PhasedSpecialisationConfig, total_steps: usize) -> Self {
        Self { config, total_steps, current_step: 0 }
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Current specialisation phase
    pub fn phase(&self) -> SpecialisationPhase {
        let progress = self.current_step as f32 / self.total_steps.max(1) as f32;
        if progress < self.config.attention_lead_fraction {
            SpecialisationPhase::AttentionLead
        } else {
            SpecialisationPhase::SsmLead
        }
    }

    /// LR multiplier for attention layers at the current step
    pub fn attention_lr_scale(&self) -> f32 {
        let progress = self.current_step as f32 / self.total_steps.max(1) as f32;
        let boundary = self.config.attention_lead_fraction;
        let low = self.config.low_priority_lr_scale;

        if !self.config.smooth_transition {
            return if progress < boundary { 1.0 } else { low };
        }

        let half_width = self.config.transition_width / 2.0;
        if (progress - boundary).abs() < half_width {
            let t = (progress - (boundary - half_width)) / self.config.transition_width;
            lerp(1.0, low, smooth_step(t.clamp(0.0, 1.0)))
        } else if progress < boundary {
            1.0
        } else {
            low
        }
    }

    /// LR multiplier for Mamba SSM layers at the current step (inverse of attention)
    pub fn ssm_lr_scale(&self) -> f32 {
        let attn_scale = self.attention_lr_scale();
        let low = self.config.low_priority_lr_scale;
        // When attention is high (1.0), SSM is low (low_scale), and vice versa
        if attn_scale > 0.5 {
            low + (1.0 - low) * (1.0 - attn_scale) / (1.0 - low).max(1e-8)
        } else {
            low + (1.0 - low) * (1.0 - attn_scale) / (1.0 - low).max(1e-8)
        }
    }

    /// LR multipliers for attention and SSM respectively
    pub fn lr_scales(&self) -> (f32, f32) {
        (self.attention_lr_scale(), self.ssm_lr_scale())
    }
}

// ---------------------------------------------------------------------------
// Hierarchical supervision weighting
// ---------------------------------------------------------------------------

/// Compute exponentially-weighted supervision weights for multi-exit models.
///
/// For a model with `num_exits` exits and exponential scale `lambda`:
///   weight(exit k) = exp(λ * k / N) / Z
/// where Z normalises to sum = 1.0.
///
/// Later exits receive more weight, ensuring the primary output is trained
/// hardest while earlier exits still receive gradient signal.
pub fn hierarchical_supervision_weights(num_exits: usize, lambda: f32) -> Vec<f32> {
    assert!(num_exits > 0);
    let n = num_exits as f32;
    let unnorm: Vec<f32> = (0..num_exits)
        .map(|k| (lambda * k as f32 / n).exp())
        .collect();
    let total: f32 = unnorm.iter().sum();
    unnorm.into_iter().map(|w| w / total).collect()
}

/// Exponential decay supervision weights (emphasises LATER exits)
/// λ = 1.0 is a reasonable default; higher λ concentrates weight on final exit.
pub fn supervision_weights_default(num_exits: usize) -> Vec<f32> {
    hierarchical_supervision_weights(num_exits, 1.0)
}

// ---------------------------------------------------------------------------
// Combined curriculum state
// ---------------------------------------------------------------------------

/// Full curriculum state for a training run.
/// Encapsulates both CGAR depth scheduling and phased specialisation.
pub struct CurriculumState {
    /// CGAR depth scheduler (None = not used / pure transformer)
    pub cgar: Option<CgarScheduler>,

    /// Phased specialisation (None = not used / single-type model)
    pub phased: Option<PhasedSpecialisationScheduler>,

    /// Current global step (kept in sync)
    current_step: usize,
}

impl CurriculumState {
    /// Create with both CGAR and phased specialisation
    pub fn hybrid(
        cgar_cfg: CgarConfig,
        phased_cfg: PhasedSpecialisationConfig,
        total_steps: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            cgar: Some(CgarScheduler::new(cgar_cfg, total_steps, num_layers)),
            phased: Some(PhasedSpecialisationScheduler::new(phased_cfg, total_steps)),
            current_step: 0,
        }
    }

    /// CGAR only (for Mamba-only or single-architecture models)
    pub fn cgar_only(cfg: CgarConfig, total_steps: usize, num_layers: usize) -> Self {
        Self {
            cgar: Some(CgarScheduler::new(cfg, total_steps, num_layers)),
            phased: None,
            current_step: 0,
        }
    }

    /// No curriculum (standard training)
    pub fn none() -> Self {
        Self { cgar: None, phased: None, current_step: 0 }
    }

    /// Advance one step; returns (active_layers, attn_lr_scale, ssm_lr_scale)
    pub fn step(&mut self, num_layers: usize) -> CurriculumTick {
        self.current_step += 1;

        let active_layers = self.cgar
            .as_mut()
            .map(|c| c.step())
            .unwrap_or(num_layers);

        let (attn_scale, ssm_scale) = self.phased
            .as_mut()
            .map(|p| { p.step(); p.lr_scales() })
            .unwrap_or((1.0, 1.0));

        let cgar_phase = self.cgar.as_ref().map(|c| c.phase());
        let spec_phase = self.phased.as_ref().map(|p| p.phase());

        CurriculumTick {
            step: self.current_step,
            active_layers,
            attn_lr_scale: attn_scale,
            ssm_lr_scale: ssm_scale,
            cgar_phase,
            spec_phase,
        }
    }
}

/// Output of a single curriculum step
#[derive(Debug, Clone)]
pub struct CurriculumTick {
    /// Global step number
    pub step: usize,

    /// Number of layers receiving gradients this step
    pub active_layers: usize,

    /// LR multiplier for attention parameters
    pub attn_lr_scale: f32,

    /// LR multiplier for SSM (Mamba) parameters
    pub ssm_lr_scale: f32,

    /// CGAR phase (if active)
    pub cgar_phase: Option<CgarPhase>,

    /// Phased specialisation phase (if active)
    pub spec_phase: Option<SpecialisationPhase>,
}

impl CurriculumTick {
    /// Log-friendly summary line
    pub fn summary(&self) -> String {
        let cgar_str = self.cgar_phase
            .map(|p| p.label().to_string())
            .unwrap_or_else(|| "full".into());
        let spec_str = self.spec_phase
            .map(|p| match p {
                SpecialisationPhase::AttentionLead => "attn-lead",
                SpecialisationPhase::SsmLead => "ssm-lead",
            })
            .unwrap_or("uniform");
        format!(
            "step={} active_layers={} cgar={} spec={} attn_lr_scale={:.3} ssm_lr_scale={:.3}",
            self.step, self.active_layers, cgar_str, spec_str,
            self.attn_lr_scale, self.ssm_lr_scale
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smooth Hermite interpolation (no discontinuous derivative at endpoints)
#[inline]
fn smooth_step(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgar_config_validation() {
        let good = CgarConfig::default();
        assert!(good.validate().is_ok());

        let bad = CgarConfig { shallow_fraction: 0.7, medium_fraction: 0.7, ..Default::default() };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_cgar_phases_progress_correctly() {
        let cfg = CgarConfig::default(); // 0.30 / 0.30 / 0.40
        let total = 100;
        let mut sched = CgarScheduler::new(cfg, total, 24);

        // Step 0: shallow
        assert_eq!(sched.phase(), CgarPhase::Shallow);

        // Step 30: transition to medium
        for _ in 0..30 { sched.step(); }
        assert_eq!(sched.phase(), CgarPhase::Medium);

        // Step 60: transition to full
        for _ in 0..30 { sched.step(); }
        assert_eq!(sched.phase(), CgarPhase::Full);
    }

    #[test]
    fn test_cgar_active_layers_monotone() {
        let cfg = CgarConfig { smooth_transitions: false, ..Default::default() };
        let total = 100;
        let num_layers = 32;
        let mut sched = CgarScheduler::new(cfg, total, num_layers);

        let shallow_layers = sched.active_layers();

        for _ in 0..30 { sched.step(); }
        let medium_layers = sched.active_layers();

        for _ in 0..30 { sched.step(); }
        let full_layers = sched.active_layers();

        assert!(shallow_layers <= medium_layers, "Active layers should be non-decreasing");
        assert!(medium_layers <= full_layers);
        assert_eq!(full_layers, num_layers);
    }

    #[test]
    fn test_cgar_min_active_layers() {
        let cfg = CgarConfig {
            shallow_layer_fraction: 0.0,
            min_active_layers: 2,
            ..Default::default()
        };
        let sched = CgarScheduler::new(cfg, 100, 32);
        assert!(sched.active_layers() >= 2, "Should respect min_active_layers");
    }

    #[test]
    fn test_cgar_smooth_transition_continuity() {
        let cfg = CgarConfig { smooth_transitions: true, ramp_fraction: 0.1, ..Default::default() };
        let total = 1000;
        let num_layers = 32;
        let mut sched = CgarScheduler::new(cfg, total, num_layers);

        let mut prev = sched.active_layers();
        let mut max_jump = 0usize;
        for _ in 0..total {
            let curr = sched.step();
            let jump = curr.abs_diff(prev);
            max_jump = max_jump.max(jump);
            prev = curr;
        }
        // With smooth transitions, no single step should jump by more than ~10% of layers
        assert!(max_jump <= 4, "Smooth transition should not jump >4 layers at once: max={}", max_jump);
    }

    #[test]
    fn test_cgar_phase_progress() {
        let cfg = CgarConfig::default();
        let total = 100;
        let mut sched = CgarScheduler::new(cfg, total, 32);

        // At step 15 (halfway through shallow phase): progress should be ~0.5
        for _ in 0..15 { sched.step(); }
        let progress = sched.phase_progress();
        assert!((progress - 0.5).abs() < 0.1, "Phase progress should be ~0.5 at midpoint: {}", progress);
    }

    #[test]
    fn test_phased_spec_attention_lead() {
        let cfg = PhasedSpecialisationConfig::default(); // 40% attention-lead
        let total = 100;
        let mut sched = PhasedSpecialisationScheduler::new(cfg, total);

        // First 40 steps: attention-lead → attn_scale high, ssm_scale low
        assert_eq!(sched.phase(), SpecialisationPhase::AttentionLead);
        let (attn, _ssm) = sched.lr_scales();
        assert!(attn > 0.5, "Attention should have high LR in attention-lead phase");

        for _ in 0..40 { sched.step(); }
        assert_eq!(sched.phase(), SpecialisationPhase::SsmLead);
    }

    #[test]
    fn test_phased_spec_hard_switch() {
        let cfg = PhasedSpecialisationConfig {
            smooth_transition: false,
            attention_lead_fraction: 0.5,
            low_priority_lr_scale: 0.1,
            ..Default::default()
        };
        let total = 100;
        let mut sched = PhasedSpecialisationScheduler::new(cfg, total);

        // Before switch: attn = 1.0
        assert!((sched.attention_lr_scale() - 1.0).abs() < 1e-6);

        // After switch: attn = 0.1
        for _ in 0..50 { sched.step(); }
        assert!((sched.attention_lr_scale() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_hierarchical_supervision_weights_sum() {
        let weights = hierarchical_supervision_weights(4, 1.0);
        assert_eq!(weights.len(), 4);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1: {}", sum);
    }

    #[test]
    fn test_hierarchical_supervision_weights_monotone() {
        let weights = hierarchical_supervision_weights(5, 2.0);
        // Each later exit should have higher weight
        for i in 1..weights.len() {
            assert!(
                weights[i] > weights[i - 1],
                "Later exit should have higher weight: w[{}]={} < w[{}]={}",
                i, weights[i], i - 1, weights[i - 1]
            );
        }
    }

    #[test]
    fn test_curriculum_state_hybrid() {
        let total = 200;
        let num_layers = 32;
        let mut state = CurriculumState::hybrid(
            CgarConfig::default(),
            PhasedSpecialisationConfig::default(),
            total,
            num_layers,
        );

        let tick = state.step(num_layers);
        assert_eq!(tick.step, 1);
        assert!(tick.active_layers < num_layers, "Should start with fewer than all layers");
        assert!(tick.cgar_phase.is_some());
        assert!(tick.spec_phase.is_some());
    }

    #[test]
    fn test_curriculum_state_none() {
        let num_layers = 16;
        let mut state = CurriculumState::none();
        let tick = state.step(num_layers);
        assert_eq!(tick.active_layers, num_layers, "No curriculum → all layers active");
        assert!((tick.attn_lr_scale - 1.0).abs() < 1e-6);
        assert!((tick.ssm_lr_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tick_summary_format() {
        let tick = CurriculumTick {
            step: 42,
            active_layers: 16,
            attn_lr_scale: 0.8,
            ssm_lr_scale: 0.2,
            cgar_phase: Some(CgarPhase::Medium),
            spec_phase: Some(SpecialisationPhase::AttentionLead),
        };
        let summary = tick.summary();
        assert!(summary.contains("step=42"));
        assert!(summary.contains("active_layers=16"));
        assert!(summary.contains("medium"));
    }

    #[test]
    fn test_smooth_step() {
        // smooth_step(0) = 0, smooth_step(1) = 1, smooth_step(0.5) = 0.5
        assert!((smooth_step(0.0) - 0.0).abs() < 1e-6);
        assert!((smooth_step(1.0) - 1.0).abs() < 1e-6);
        assert!((smooth_step(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_steps_remaining_in_phase() {
        let cfg = CgarConfig { smooth_transitions: false, ..Default::default() };
        let total = 100;
        let mut sched = CgarScheduler::new(cfg, total, 32);

        // At step 0, remaining in shallow phase ≈ 30 steps
        let remaining = sched.steps_remaining_in_phase();
        assert!(remaining > 25 && remaining <= 30, "Expected ~30 remaining, got {}", remaining);

        for _ in 0..30 { sched.step(); }
        // Now in medium phase, ~30 steps remaining
        let remaining = sched.steps_remaining_in_phase();
        assert!(remaining <= 30);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: curriculum.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/curriculum.rs
// INTEGRATES:  trainer.rs · config.rs · jamba.rs · mamba.rs · moe.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
