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
pub mod data;
pub mod fine_tuning;
pub mod metrics;
pub mod optimizer;
pub mod trainer;

pub use config::{DataConfig, FineTuningConfig, TrainingConfig};
pub use data::{DataLoader, Dataset, InstructionDataset, TextDataset};
pub use fine_tuning::{FineTuningMethod, FullFineTuning, LoRAConfig, LoRAFineTuning, QLoRAFineTuning};
pub use metrics::TrainingMetrics;
pub use optimizer::{clip_grad_norm, AdamW, LearningRateScheduler, Optimizer, SGD};
pub use trainer::Trainer;
