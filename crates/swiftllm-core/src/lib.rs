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
pub mod memory;
pub mod sampling;
pub mod scheduler;

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
