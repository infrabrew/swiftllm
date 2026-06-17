// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      lib.rs
// PATH:      /crates/swiftllm-training/src/lib.rs
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

//! SwiftLLM Training — Training and fine-tuning for LLM models
//!
//! This crate provides:
//! - Full model training from scratch
//! - Fine-tuning (full, LoRA, QLoRA)
//! - Data loading and preprocessing
//! - Optimizers (AdamW, SGD) and learning rate schedulers
//! - Checkpoint saving/loading
//! - Training metrics and logging

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod config;
pub mod curriculum;
pub mod data;
pub mod fine_tuning;
pub mod grpo;
pub mod long_reward;
pub mod metrics;
pub mod muon;
pub mod optimizer;
pub mod process_reward;
pub mod realtime_rl;
pub mod self_learning;
pub mod self_supervised;
pub mod trainer;

pub use config::{DataConfig, FineTuningConfig, TrainingConfig};
pub use curriculum::{
    CgarConfig, CgarPhase, CgarScheduler, CurriculumTick, PhasedSpecialisationConfig,
    PhasedSpecialisationScheduler, SpecialisationPhase,
};
pub use data::{DataLoader, Dataset, InstructionDataset, TextDataset};
pub use fine_tuning::{FineTuningMethod, FullFineTuning, LoRAConfig, LoRAFineTuning, QLoRAFineTuning};
pub use grpo::{
    compute_grpo_loss, GrpoConfig, GrpoGroup, GrpoLossResult, GrpoSample, GrpoStepStats,
    GrpoTrainer, RewardFunction,
};
pub use long_reward::{
    aggregate_dense_rewards, compute_dense_rewards, normalise_batch_rewards, DenseAggregation,
    DenseRewardResult, LongRewardConfig, TokenReward,
};
pub use metrics::TrainingMetrics;
pub use muon::{Muon, MuonConfig};
pub use optimizer::{clip_grad_norm, AdamW, LearningRateScheduler, Optimizer, SGD};
pub use process_reward::{
    aggregate_step_scores, blend_prm_with_outcome, parse_steps, NeuralPrm, PrmAggregation,
    PrmConfig, PrmResult, ReasoningStep, RulePrm, StepBoundary, StepScore,
};
pub use realtime_rl::{
    ActorLearner, ActorLearnerStats, AdapterRegistry, Experience, ExperienceBuffer, LearnerBackend,
    PendingResponse, RealtimeRlConfig, RealtimeRlTrainer, RewardJoin,
};
pub use self_learning::{
    filter_candidates, score_candidate, AcceptancePolicy, Candidate, RoundResult, SelfTrainingConfig,
    SelfTrainingLoop, SftExample,
};
pub use self_supervised::{
    build_causal_lm, build_example, build_masked_lm, build_span_corruption, MlmConfig, SpanConfig,
    SslExample, SslObjective,
};
pub use trainer::Trainer;

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
