// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      mod.rs
// PATH:      /crates/swiftllm-core/src/inference/mod.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   - swiftllm-core/src/inference/refinement.rs    multi-round self-refinement pipeline
//   - swiftllm-core/src/inference/verification.rs  Best-of-N dense verification layer
// USED BY:
//   - swiftllm-core/src/lib.rs    pub mod inference; exposes this module crate-wide
// SEE ALSO:
//   - swiftllm-core/src/sampling/mod.rs             upstream: sampling feeds inference
//   - swiftllm-core/src/serving/mod.rs              downstream: inference feeds serving
//   - swiftllm-core/src/engine.rs                   Engine ties sampling → inference → serving
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

//! Test-time inference enhancements.
//!
//! - [`refinement`] — Multi-round self-refinement (Self-Refine, 2023)
//! - [`verification`] — Dense verification + Best-of-N PRM reranking

pub mod refinement;
pub mod verification;

pub use refinement::{
    improvement_score, normalised_edit_distance, ImprovementMetric, RefinementConfig,
    RefinementPipeline, RefinementResult, RefinementRound, StoppingCriterion,
};
pub use verification::{
    best_of_n_by_logprob, rule_score, verify_and_rank, ScoredCandidate, ScoringStrategy,
    VerificationConfig, VerificationResult,
};

// ------------------------------------------------------------------------------
// END OF FILE: mod.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/inference/mod.rs
// INTEGRATES:  inference/refinement.rs · inference/verification.rs · engine.rs · serving/mod.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
