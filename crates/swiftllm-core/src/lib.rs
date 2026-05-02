// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      lib.rs
// PATH:      /crates/swiftllm-core/src/lib.rs
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

//! SwiftLLM Core - High-performance LLM inference engine
//!
//! This crate provides the core components for efficient LLM inference:
//! - PagedAttention for memory-efficient KV cache management
//! - Continuous batching for high-throughput serving
//! - Speculative decoding for faster generation
//! - Tensor parallelism for multi-GPU inference

#![warn(clippy::all)]
#![warn(missing_docs)]

pub mod config;
pub mod engine;
pub mod execution;
pub mod inference;
pub mod memory;
pub mod sampling;
pub mod scheduler;
pub mod serving;

pub mod error;
pub mod tensor;
pub mod types;

pub use config::{EngineConfig, ModelConfig, SchedulerConfig, SamplingConfig};
pub use engine::Engine;
pub use error::{Error, Result};
pub use types::{
    GenerationOutput, Request, RequestId, RequestOutput, RequestStatus, SequenceGroup,
    SequenceId, Token, TokenId,
};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::config::*;
    pub use crate::engine::Engine;
    pub use crate::error::{Error, Result};
    pub use crate::sampling::{SamplingParams, SamplingStrategy};
    pub use crate::types::*;
}

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
