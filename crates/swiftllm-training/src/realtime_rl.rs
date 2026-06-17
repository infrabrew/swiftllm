// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      realtime_rl.rs
// PATH:      /crates/swiftllm-training/src/realtime_rl.rs
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

//! Realtime (online) reinforcement learning.
//!
//! Batch RLHF freezes the policy, generates a dataset, scores it, trains, and
//! redeploys. **Realtime RL collapses that loop**: generation (serving) and
//! learning run concurrently and continuously, so the policy improves from live
//! interaction without a redeploy step.
//!
//! This module is the *online orchestration layer* that the offline
//! [`crate::grpo`] machinery lacks. It reuses GRPO for the actual objective
//! (group-relative advantage + clipped importance-sampling loss with a KL
//! anchor) and adds the four things that make RL "realtime":
//!
//! 1. [`ExperienceBuffer`] — a bounded replay buffer with **staleness bounds**.
//!    Because the actor keeps generating while the learner trains, experiences
//!    are off-policy by the time they are used; the buffer drops any experience
//!    older than `max_staleness` policy versions.
//! 2. [`RewardJoin`] — joins **delayed reward** (human feedback, or a verifier
//!    that returns later) back to the response that earned it, keyed by request
//!    id, then computes group-relative advantages.
//! 3. [`ActorLearner`] — the async **actor↔learner** controller: replay-ratio
//!    back-pressure (an off-policy budget), staleness gating, and weight-sync
//!    scheduling. This is the IMPALA / Ape-X / async-RLHF pattern.
//! 4. [`AdapterRegistry`] — **LoRA adapter versioning** with hot-swap, a
//!    last-known-good checkpoint, and rollback (the safe, cheap update surface).
//!
//! The actual forward pass (π_new log-probs) and the gradient application are
//! the GPU/model seam: they are expressed through the [`LearnerBackend`] trait
//! so the orchestration is fully unit-testable on CPU with a mock backend, and
//! a real backend can wire in the engine + autograd later.

use std::collections::{HashMap, VecDeque};

use crate::grpo::{compute_grpo_loss, GrpoConfig, GrpoGroup, GrpoLossResult, GrpoSample};

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------

/// Configuration for realtime/online RL.
#[derive(Debug, Clone)]
pub struct RealtimeRlConfig {
    /// Underlying GRPO objective configuration (group size, clip ε, KL β, …).
    pub grpo: GrpoConfig,

    /// Maximum experiences retained in the replay buffer.
    pub buffer_capacity: usize,

    /// Drop experiences generated more than this many policy versions ago. This
    /// bounds off-policy-ness; `0` means strictly on-policy (current version
    /// only).
    pub max_staleness: u64,

    /// Off-policy budget: maximum learner updates per collected experience. A
    /// value of `1.0` keeps updates and fresh experiences in balance; higher
    /// values reuse data more aggressively (and more off-policy).
    pub max_replay_ratio: f32,

    /// Minimum buffered experiences before a learner step may run.
    pub min_groups_per_step: usize,

    /// Number of experience groups sampled per learner step.
    pub batch_groups: usize,

    /// Learner updates between LoRA adapter hot-swaps (weight syncs).
    pub sync_every: u64,

    /// Clamp every reward to `[-reward_clip, reward_clip]` before use — a
    /// guardrail against reward-hacking / poisoned live feedback.
    pub reward_clip: f32,
}

impl Default for RealtimeRlConfig {
    fn default() -> Self {
        Self {
            grpo: GrpoConfig::default(),
            buffer_capacity: 4096,
            max_staleness: 4,
            max_replay_ratio: 2.0,
            min_groups_per_step: 1,
            batch_groups: 8,
            sync_every: 4,
            reward_clip: 10.0,
        }
    }
}

// ----------------------------------------------------------------------------
// Experience + replay buffer
// ----------------------------------------------------------------------------

/// A single completed experience: one GRPO group (a prompt + its G scored,
/// advantage-assigned completions) tagged with the policy version that
/// generated it.
#[derive(Debug, Clone)]
pub struct Experience {
    /// The originating request id.
    pub request_id: u64,
    /// Policy version (= adapter version) that generated the completions.
    pub policy_version: u64,
    /// The GRPO group (advantages already computed).
    pub group: GrpoGroup,
}

/// A bounded, staleness-aware experience replay buffer.
#[derive(Debug)]
pub struct ExperienceBuffer {
    capacity: usize,
    items: VecDeque<Experience>,
}

impl ExperienceBuffer {
    /// Create a buffer holding at most `capacity` experiences.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            items: VecDeque::new(),
        }
    }

    /// Push an experience, evicting the oldest if at capacity.
    pub fn push(&mut self, exp: Experience) {
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }
        self.items.push_back(exp);
    }

    /// Number of buffered experiences.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Whether an experience is fresh enough to train on at `current_version`.
    fn is_fresh(exp: &Experience, current_version: u64, max_staleness: u64) -> bool {
        current_version.saturating_sub(exp.policy_version) <= max_staleness
    }

    /// Count experiences too stale to use at `current_version`.
    pub fn stale_count(&self, current_version: u64, max_staleness: u64) -> usize {
        self.items
            .iter()
            .filter(|e| !Self::is_fresh(e, current_version, max_staleness))
            .count()
    }

    /// Drop experiences older than `max_staleness` versions; returns how many
    /// were removed.
    pub fn prune_stale(&mut self, current_version: u64, max_staleness: u64) -> usize {
        let before = self.items.len();
        self.items
            .retain(|e| Self::is_fresh(e, current_version, max_staleness));
        before - self.items.len()
    }

    /// Sample up to `n` fresh groups (most recent first) for a learner step.
    pub fn sample_fresh(
        &self,
        n: usize,
        current_version: u64,
        max_staleness: u64,
    ) -> Vec<GrpoGroup> {
        self.items
            .iter()
            .rev()
            .filter(|e| Self::is_fresh(e, current_version, max_staleness))
            .take(n)
            .map(|e| e.group.clone())
            .collect()
    }
}

// ----------------------------------------------------------------------------
// Delayed-reward join
// ----------------------------------------------------------------------------

/// A response awaiting its (possibly delayed) reward, keyed by request id.
#[derive(Debug, Clone)]
pub struct PendingResponse {
    /// Request id this response answered.
    pub request_id: u64,
    /// Policy version that generated it.
    pub policy_version: u64,
    /// Prompt shared by the group.
    pub prompt: String,
    /// The G samples, with `reward`/`advantage` left at `0.0` until joined.
    pub samples: Vec<GrpoSample>,
}

/// Joins delayed rewards back to the responses that earned them.
///
/// With a verifiable reward the join is immediate; with human feedback it may
/// arrive seconds or minutes later. Either way the response is held by request
/// id until its reward vector arrives, then turned into an [`Experience`].
#[derive(Debug)]
pub struct RewardJoin {
    pending: HashMap<u64, PendingResponse>,
    adv_epsilon: f32,
}

impl RewardJoin {
    /// Create a join table using `adv_epsilon` for advantage normalisation.
    pub fn new(adv_epsilon: f32) -> Self {
        Self {
            pending: HashMap::new(),
            adv_epsilon,
        }
    }

    /// Register a response awaiting reward.
    pub fn register(&mut self, response: PendingResponse) {
        self.pending.insert(response.request_id, response);
    }

    /// Number of responses still awaiting reward.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Submit the reward vector (one reward per sample) for `request_id`.
    ///
    /// Rewards are clamped to `[-reward_clip, reward_clip]`; the group's
    /// advantages are then computed and the joined [`Experience`] returned.
    /// Returns `None` if the id is unknown or the reward count mismatches.
    pub fn submit(
        &mut self,
        request_id: u64,
        rewards: Vec<f32>,
        reward_clip: f32,
    ) -> Option<Experience> {
        let pending = self.pending.get(&request_id)?;
        if rewards.len() != pending.samples.len() {
            return None;
        }
        let mut response = self.pending.remove(&request_id)?;
        for (sample, r) in response.samples.iter_mut().zip(rewards) {
            sample.reward = r.clamp(-reward_clip, reward_clip);
        }
        let group = GrpoGroup::new(response.prompt, response.samples, self.adv_epsilon);
        Some(Experience {
            request_id,
            policy_version: response.policy_version,
            group,
        })
    }
}

// ----------------------------------------------------------------------------
// Adapter versioning (LoRA hot-swap + rollback)
// ----------------------------------------------------------------------------

/// Tracks the live LoRA adapter version, the last validated ("known good")
/// version, and supports rollback.
#[derive(Debug, Clone)]
pub struct AdapterRegistry {
    current_version: u64,
    last_known_good: u64,
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterRegistry {
    /// Start at version 0 (the base policy), which is also the known-good base.
    pub fn new() -> Self {
        Self {
            current_version: 0,
            last_known_good: 0,
        }
    }

    /// The currently-serving adapter version.
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// The last version that passed validation.
    pub fn last_known_good(&self) -> u64 {
        self.last_known_good
    }

    /// Promote a freshly-trained adapter to live, returning the new version.
    pub fn promote(&mut self) -> u64 {
        self.current_version += 1;
        self.current_version
    }

    /// Mark the current live version as validated (a safe rollback target).
    pub fn mark_good(&mut self) {
        self.last_known_good = self.current_version;
    }

    /// Roll the live adapter back to the last known-good version.
    pub fn rollback(&mut self) -> u64 {
        self.current_version = self.last_known_good;
        self.current_version
    }
}

// ----------------------------------------------------------------------------
// Actor ↔ Learner controller
// ----------------------------------------------------------------------------

/// Snapshot of the controller's counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActorLearnerStats {
    /// Total experiences recorded by actors.
    pub total_experiences: u64,
    /// Total learner updates applied.
    pub total_updates: u64,
    /// Live policy/adapter version.
    pub policy_version: u64,
    /// Experiences currently buffered.
    pub buffered: usize,
}

/// The realtime RL controller: back-pressures the learner against the actor's
/// experience stream, gates on staleness, and schedules weight syncs.
#[derive(Debug)]
pub struct ActorLearner {
    config: RealtimeRlConfig,
    buffer: ExperienceBuffer,
    adapters: AdapterRegistry,
    total_experiences: u64,
    total_updates: u64,
    updates_since_sync: u64,
}

impl ActorLearner {
    /// Create a controller from configuration.
    pub fn new(config: RealtimeRlConfig) -> Self {
        let buffer = ExperienceBuffer::new(config.buffer_capacity);
        Self {
            config,
            buffer,
            adapters: AdapterRegistry::new(),
            total_experiences: 0,
            total_updates: 0,
            updates_since_sync: 0,
        }
    }

    /// The live policy (adapter) version.
    pub fn policy_version(&self) -> u64 {
        self.adapters.current_version()
    }

    /// Actor side: record a completed, reward-joined experience.
    pub fn record(&mut self, exp: Experience) {
        self.buffer.push(exp);
        self.total_experiences += 1;
    }

    /// Whether a learner update may run now: enough fresh data buffered, and the
    /// replay-ratio (off-policy) budget is not exhausted.
    pub fn can_learn(&self) -> bool {
        let fresh = self.buffer.len() - self.buffer.stale_count(
            self.policy_version(),
            self.config.max_staleness,
        );
        if fresh < self.config.min_groups_per_step {
            return false;
        }
        // Off-policy budget: updates must not outrun experiences × replay ratio.
        let budget = self.total_experiences as f32 * self.config.max_replay_ratio;
        (self.total_updates as f32) < budget
    }

    /// Sample a fresh batch of groups for the learner.
    pub fn sample_batch(&self) -> Vec<GrpoGroup> {
        self.buffer.sample_fresh(
            self.config.batch_groups,
            self.policy_version(),
            self.config.max_staleness,
        )
    }

    /// Record that one learner update was applied.
    pub fn on_update(&mut self) {
        self.total_updates += 1;
        self.updates_since_sync += 1;
    }

    /// Whether enough updates have accrued to hot-swap the adapter.
    pub fn should_sync(&self) -> bool {
        self.updates_since_sync >= self.config.sync_every
    }

    /// Hot-swap: promote the trained adapter to a new live version and reset the
    /// sync counter. Returns the new policy version.
    pub fn sync_weights(&mut self) -> u64 {
        self.updates_since_sync = 0;
        self.adapters.promote()
    }

    /// Mark the live version validated (rollback target).
    pub fn mark_good(&mut self) {
        self.adapters.mark_good();
    }

    /// Roll back to the last known-good adapter (e.g. after a failed eval).
    pub fn rollback(&mut self) -> u64 {
        self.updates_since_sync = 0;
        self.adapters.rollback()
    }

    /// Prune stale experiences from the buffer; returns how many were dropped.
    pub fn prune(&mut self) -> usize {
        self.buffer
            .prune_stale(self.policy_version(), self.config.max_staleness)
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &RealtimeRlConfig {
        &self.config
    }

    /// Current counters.
    pub fn stats(&self) -> ActorLearnerStats {
        ActorLearnerStats {
            total_experiences: self.total_experiences,
            total_updates: self.total_updates,
            policy_version: self.policy_version(),
            buffered: self.buffer.len(),
        }
    }
}

// ----------------------------------------------------------------------------
// Learner backend seam + high-level trainer
// ----------------------------------------------------------------------------

/// The GPU/model seam for realtime RL.
///
/// A real implementation runs the current policy's forward pass to obtain
/// `π_new` log-probs (for the importance ratio) and applies the gradient update.
/// The orchestration above is independent of how those two steps are realised,
/// so it can be tested with a CPU mock.
pub trait LearnerBackend {
    /// Re-score each sampled group's samples under the **current** policy,
    /// returning new log-probs per sample (one `Vec<f32>` per sample, in group
    /// then sample order — matching [`compute_grpo_loss`]'s expectation).
    fn policy_log_probs(&self, groups: &[GrpoGroup]) -> Vec<Vec<f32>>;

    /// Apply the gradient update for the computed loss. Returns whether the step
    /// succeeded (a real backend may reject e.g. a non-finite loss).
    fn apply_update(&mut self, loss: &GrpoLossResult) -> bool;
}

/// Drives realtime RL end to end over a [`LearnerBackend`].
#[derive(Debug)]
pub struct RealtimeRlTrainer<B: LearnerBackend> {
    controller: ActorLearner,
    backend: B,
}

impl<B: LearnerBackend> RealtimeRlTrainer<B> {
    /// Create a trainer from configuration and a backend.
    pub fn new(config: RealtimeRlConfig, backend: B) -> Self {
        Self {
            controller: ActorLearner::new(config),
            backend,
        }
    }

    /// Actor side: record a reward-joined experience.
    pub fn record(&mut self, exp: Experience) {
        self.controller.record(exp);
    }

    /// Whether a learner step may run now.
    pub fn can_learn(&self) -> bool {
        self.controller.can_learn()
    }

    /// The live policy version.
    pub fn policy_version(&self) -> u64 {
        self.controller.policy_version()
    }

    /// Controller counters.
    pub fn stats(&self) -> ActorLearnerStats {
        self.controller.stats()
    }

    /// Borrow the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Run one learner iteration if allowed: sample a fresh batch, score it under
    /// the current policy (backend), compute the GRPO loss, apply the update, and
    /// hot-swap the adapter when enough updates have accrued.
    ///
    /// Returns the loss for this step, or `None` if no step was taken (insufficient
    /// fresh data or replay-budget exhausted).
    pub fn learner_step(&mut self) -> Option<GrpoLossResult> {
        if !self.controller.can_learn() {
            return None;
        }
        let groups = self.controller.sample_batch();
        if groups.is_empty() {
            return None;
        }

        let log_probs_new = self.backend.policy_log_probs(&groups);
        let cfg = self.controller.config();
        let loss = compute_grpo_loss(
            &groups,
            &log_probs_new,
            &cfg.grpo,
            cfg.grpo.advantage_filter_threshold,
        );

        if !self.backend.apply_update(&loss) {
            return None;
        }
        self.controller.on_update();
        if self.controller.should_sync() {
            self.controller.sync_weights();
        }
        Some(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sample with the given old/ref log-probs (ratio is exp(new-old)).
    fn sample(reward: f32, n_tokens: usize) -> GrpoSample {
        GrpoSample {
            prompt: "p".to_string(),
            output: "o".to_string(),
            log_probs_old: vec![-1.0; n_tokens],
            log_probs_ref: vec![-1.0; n_tokens],
            reward,
            advantage: 0.0,
        }
    }

    fn group(rewards: &[f32]) -> GrpoGroup {
        let samples: Vec<GrpoSample> = rewards.iter().map(|&r| sample(r, 3)).collect();
        GrpoGroup::new("p".to_string(), samples, 1e-8)
    }

    fn exp(request_id: u64, version: u64, rewards: &[f32]) -> Experience {
        Experience {
            request_id,
            policy_version: version,
            group: group(rewards),
        }
    }

    #[test]
    fn buffer_evicts_oldest_at_capacity() {
        let mut buf = ExperienceBuffer::new(2);
        buf.push(exp(1, 0, &[1.0, 0.0]));
        buf.push(exp(2, 0, &[1.0, 0.0]));
        buf.push(exp(3, 0, &[1.0, 0.0]));
        assert_eq!(buf.len(), 2); // first was evicted
    }

    #[test]
    fn buffer_filters_and_prunes_stale() {
        let mut buf = ExperienceBuffer::new(10);
        buf.push(exp(1, 0, &[1.0, 0.0])); // version 0
        buf.push(exp(2, 3, &[1.0, 0.0])); // version 3
        // At current version 5 with max_staleness 2: version 0 is stale (5-0>2),
        // version 3 is fresh (5-3<=2).
        assert_eq!(buf.stale_count(5, 2), 1);
        let fresh = buf.sample_fresh(10, 5, 2);
        assert_eq!(fresh.len(), 1);
        assert_eq!(buf.prune_stale(5, 2), 1);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn reward_join_delays_then_builds_experience() {
        let mut join = RewardJoin::new(1e-8);
        join.register(PendingResponse {
            request_id: 42,
            policy_version: 7,
            prompt: "p".to_string(),
            samples: vec![sample(0.0, 3), sample(0.0, 3)],
        });
        assert_eq!(join.pending_len(), 1);

        // Unknown id → None; wrong reward count → None.
        assert!(join.submit(99, vec![1.0, 0.0], 10.0).is_none());
        assert!(join.submit(42, vec![1.0], 10.0).is_none());
        assert_eq!(join.pending_len(), 1); // still pending after bad submits

        let experience = join.submit(42, vec![1.0, 0.0], 10.0).unwrap();
        assert_eq!(experience.request_id, 42);
        assert_eq!(experience.policy_version, 7);
        // Group-relative advantage: reward 1.0 is above the group mean, 0.0 below.
        assert!(experience.group.samples[0].advantage > 0.0);
        assert!(experience.group.samples[1].advantage < 0.0);
        assert_eq!(join.pending_len(), 0);
    }

    #[test]
    fn reward_clipping_bounds_poisoned_feedback() {
        let mut join = RewardJoin::new(1e-8);
        join.register(PendingResponse {
            request_id: 1,
            policy_version: 0,
            prompt: "p".to_string(),
            samples: vec![sample(0.0, 2), sample(0.0, 2)],
        });
        // A wildly out-of-range reward is clamped to ±reward_clip.
        let e = join.submit(1, vec![1e9, -1e9], 5.0).unwrap();
        assert_eq!(e.group.samples[0].reward, 5.0);
        assert_eq!(e.group.samples[1].reward, -5.0);
    }

    #[test]
    fn adapter_registry_promote_mark_rollback() {
        let mut reg = AdapterRegistry::new();
        assert_eq!(reg.current_version(), 0);
        assert_eq!(reg.promote(), 1);
        assert_eq!(reg.promote(), 2);
        reg.mark_good(); // version 2 is validated
        assert_eq!(reg.promote(), 3); // a bad update
        assert_eq!(reg.rollback(), 2); // roll back to known-good
        assert_eq!(reg.current_version(), 2);
        assert_eq!(reg.last_known_good(), 2);
    }

    #[test]
    fn controller_replay_ratio_back_pressure() {
        let config = RealtimeRlConfig {
            min_groups_per_step: 1,
            max_replay_ratio: 2.0,
            max_staleness: 100,
            ..Default::default()
        };
        let mut ctrl = ActorLearner::new(config);
        ctrl.record(exp(1, 0, &[1.0, 0.0])); // 1 experience → budget = 2 updates
        assert!(ctrl.can_learn());
        ctrl.on_update();
        assert!(ctrl.can_learn()); // 1 < 2
        ctrl.on_update();
        assert!(!ctrl.can_learn()); // 2 == budget → exhausted until more experience
        ctrl.record(exp(2, 0, &[1.0, 0.0])); // budget now 4
        assert!(ctrl.can_learn());
    }

    #[test]
    fn controller_syncs_every_n_updates() {
        let config = RealtimeRlConfig {
            sync_every: 3,
            max_replay_ratio: 100.0,
            min_groups_per_step: 1,
            max_staleness: 100,
            ..Default::default()
        };
        let mut ctrl = ActorLearner::new(config);
        ctrl.record(exp(1, 0, &[1.0, 0.0]));
        assert_eq!(ctrl.policy_version(), 0);
        for _ in 0..3 {
            assert!(!ctrl.should_sync() || ctrl.policy_version() > 0);
            ctrl.on_update();
        }
        assert!(ctrl.should_sync());
        assert_eq!(ctrl.sync_weights(), 1); // hot-swapped to version 1
        assert!(!ctrl.should_sync()); // counter reset
    }

    /// Mock backend: ratio = 1 (new log-probs == old), counts applied updates.
    struct MockBackend {
        updates: usize,
    }
    impl LearnerBackend for MockBackend {
        fn policy_log_probs(&self, groups: &[GrpoGroup]) -> Vec<Vec<f32>> {
            groups
                .iter()
                .flat_map(|g| g.samples.iter().map(|s| s.log_probs_old.clone()))
                .collect()
        }
        fn apply_update(&mut self, _loss: &GrpoLossResult) -> bool {
            self.updates += 1;
            true
        }
    }

    #[test]
    fn end_to_end_learner_loop_advances_policy() {
        let config = RealtimeRlConfig {
            min_groups_per_step: 1,
            batch_groups: 4,
            sync_every: 2,
            max_replay_ratio: 100.0,
            max_staleness: 100,
            ..Default::default()
        };
        let mut trainer = RealtimeRlTrainer::new(config, MockBackend { updates: 0 });

        // Actors stream in experiences with a learning signal (mixed rewards).
        trainer.record(exp(1, 0, &[1.0, 0.0, 1.0, 0.0]));
        trainer.record(exp(2, 0, &[0.0, 1.0, 0.0, 1.0]));

        // Two learner steps → one weight sync (sync_every = 2).
        let l1 = trainer.learner_step().expect("step 1 runs");
        assert!(l1.total_loss.is_finite());
        let _l2 = trainer.learner_step().expect("step 2 runs");

        assert_eq!(trainer.policy_version(), 1); // hot-swapped after 2 updates
        assert_eq!(trainer.backend().updates, 2);
        assert_eq!(trainer.stats().total_updates, 2);
    }

    #[test]
    fn stale_experience_stops_learning_after_policy_advances() {
        // Strictly on-policy (max_staleness 0), sync after every update.
        let config = RealtimeRlConfig {
            min_groups_per_step: 1,
            batch_groups: 4,
            sync_every: 1,
            max_replay_ratio: 100.0,
            max_staleness: 0,
            ..Default::default()
        };
        let mut trainer = RealtimeRlTrainer::new(config, MockBackend { updates: 0 });
        // Fresh experience at version 0 → one step runs and hot-swaps to v1.
        trainer.record(exp(1, 0, &[1.0, 0.0]));
        assert!(trainer.learner_step().is_some());
        assert_eq!(trainer.policy_version(), 1);
        // That experience is now stale (generated at v0, policy at v1) → idle.
        assert!(trainer.learner_step().is_none());
    }

    #[test]
    fn no_step_below_min_groups() {
        let mut trainer = RealtimeRlTrainer::new(
            RealtimeRlConfig { max_staleness: 100, min_groups_per_step: 2, ..Default::default() },
            MockBackend { updates: 0 },
        );
        trainer.record(exp(1, 0, &[1.0, 0.0]));
        assert!(trainer.learner_step().is_none()); // need 2 groups, have 1
        trainer.record(exp(2, 0, &[0.0, 1.0]));
        assert!(trainer.learner_step().is_some()); // now 2 → runs
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: realtime_rl.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/realtime_rl.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
