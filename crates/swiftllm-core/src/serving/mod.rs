// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      mod.rs
// PATH:      /crates/swiftllm-core/src/serving/mod.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   - swiftllm-core/src/serving/disaggregated.rs  disaggregated prefill/decode scheduler
// USED BY:
//   - swiftllm-core/src/lib.rs    pub mod serving; exposes this module crate-wide
// SEE ALSO:
//   - swiftllm-core/src/scheduler/mod.rs              Scheduler instantiated per worker role
//   - swiftllm-core/src/memory/block_manager.rs       block allocation coordinated with scheduler
//   - swiftllm-core/src/inference/mod.rs              inference pipeline feeds into serving
//   - swiftllm-server/src/api/openai.rs               HTTP layer routes requests to workers
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

//! Serving infrastructure.
//!
//! - [`disaggregated`] — Disaggregated prefill/decode serving (Splitwise/DistServe)

pub mod disaggregated;

pub use disaggregated::{
    optimal_worker_ratio, DisaggregatedConfig, DisaggregatedScheduler, KvTransferMetadata,
    SchedulingPolicy, WorkerAssignment, WorkerLoad, WorkerRole, WorkerSpec,
};

// ------------------------------------------------------------------------------
// END OF FILE: mod.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/serving/mod.rs
// INTEGRATES:  serving/disaggregated.rs · scheduler/mod.rs · memory/block_manager.rs · openai.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
