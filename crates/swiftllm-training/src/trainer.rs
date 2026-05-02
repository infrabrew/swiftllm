// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      trainer.rs
// PATH:      /crates/swiftllm-training/src/trainer.rs
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

//! Core training loop

use crate::config::{TrainingConfig, WarmupConfig};
use crate::curriculum::{CurriculumState, CurriculumTick};
use crate::data::{DataLoader, Dataset, TrainingSample};
use crate::metrics::TrainingMetrics;
use crate::optimizer::{LearningRateScheduler, Optimizer, SchedulerType};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Training errors
#[derive(Error, Debug)]
pub enum TrainingError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    /// Training error
    #[error("Training error: {0}")]
    Training(String),
    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
}

type Result<T> = std::result::Result<T, TrainingError>;

/// Checkpoint data
#[derive(Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Epoch number
    pub epoch: usize,
    /// Step number
    pub step: usize,
    /// Training loss at checkpoint
    pub train_loss: f64,
    /// Evaluation loss at checkpoint
    pub eval_loss: Option<f64>,
    /// Learning rate at checkpoint
    pub learning_rate: f64,
    /// Configuration
    pub config: TrainingConfig,
}

/// Training state
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current global step
    pub global_step: usize,
    /// Best evaluation loss seen so far
    pub best_eval_loss: f64,
    /// Gradient accumulation buffer step
    pub accumulation_step: usize,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            best_eval_loss: f64::INFINITY,
            accumulation_step: 0,
        }
    }
}

/// The main Trainer
pub struct Trainer {
    /// Training configuration
    config: TrainingConfig,

    /// Training state
    state: TrainingState,

    /// Metrics tracker
    metrics: TrainingMetrics,

    /// Checkpoint paths (for save limit management)
    checkpoint_paths: Vec<PathBuf>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Result<Self> {
        // Validate config
        if config.model_path.as_os_str().is_empty() {
            return Err(TrainingError::Config("model_path must not be empty".to_string()));
        }
        if config.per_device_batch_size == 0 {
            return Err(TrainingError::Config("batch_size must be > 0".to_string()));
        }
        if config.learning_rate <= 0.0 {
            return Err(TrainingError::Config("learning_rate must be > 0".to_string()));
        }

        Ok(Self {
            config,
            state: TrainingState::default(),
            metrics: TrainingMetrics::new(100),
            checkpoint_paths: Vec::new(),
        })
    }

    /// Calculate total training steps
    pub fn total_steps(&self, dataset_len: usize) -> usize {
        let steps_per_epoch = (dataset_len + self.config.effective_batch_size() - 1)
            / self.config.effective_batch_size();
        steps_per_epoch * self.config.num_epochs
    }

    /// Calculate warmup steps
    pub fn warmup_steps(&self, total_steps: usize) -> usize {
        match &self.config.warmup_steps {
            WarmupConfig::Steps(s) => *s,
            WarmupConfig::Ratio(r) => (total_steps as f64 * r) as usize,
        }
    }

    /// Run the training loop
    pub fn train<D: Dataset>(&mut self, train_data: &mut DataLoader<D>, mut eval_data: Option<&mut DataLoader<D>>) -> Result<()> {
        let total_steps = self.total_steps(train_data.len());
        let warmup_steps = self.warmup_steps(total_steps);

        tracing::info!("Starting training");
        tracing::info!("  Epochs: {}", self.config.num_epochs);
        tracing::info!("  Effective batch size: {}", self.config.effective_batch_size());
        tracing::info!("  Total steps: {}", total_steps);
        tracing::info!("  Warmup steps: {}", warmup_steps);
        tracing::info!("  Learning rate: {:.2e}", self.config.learning_rate);

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Build curriculum state now that total_steps is known.
        let num_layers = self.config.num_layers;
        let mut curriculum = match (&self.config.cgar, &self.config.phased_spec) {
            (Some(cgar_cfg), Some(spec_cfg)) => CurriculumState::hybrid(
                cgar_cfg.clone(),
                spec_cfg.clone(),
                total_steps,
                num_layers,
            ),
            (Some(cgar_cfg), None) => {
                CurriculumState::cgar_only(cgar_cfg.clone(), total_steps, num_layers)
            }
            _ => CurriculumState::none(),
        };

        // Create LR scheduler
        let scheduler_type = match self.config.lr_scheduler {
            crate::config::LrSchedulerType::Linear => SchedulerType::Linear,
            crate::config::LrSchedulerType::Cosine => SchedulerType::Cosine,
            _ => SchedulerType::Cosine,
        };
        let mut lr_scheduler = LearningRateScheduler::new(
            self.config.learning_rate,
            total_steps,
            warmup_steps,
            scheduler_type,
        );

        for epoch in 0..self.config.num_epochs {
            self.state.epoch = epoch;
            self.metrics.next_epoch();
            train_data.reset();

            tracing::info!("Epoch {}/{}", epoch + 1, self.config.num_epochs);

            // Training loop
            while let Some(batch) = train_data.next_batch() {
                let lr = lr_scheduler.step();
                self.state.global_step += 1;

                // Advance curriculum and apply per-component LR scaling.
                let tick = curriculum.step(num_layers);
                let effective_lr = apply_curriculum_lr(lr, &tick);

                // Simulate training step (actual implementation would run model forward/backward)
                let loss = self.train_step(&batch, effective_lr);

                let num_tokens: usize = batch.iter().map(|s| s.token_ids.len().max(1)).sum();
                self.metrics.record_step(loss, lr, num_tokens);

                // Logging
                if self.config.logging_steps > 0
                    && self.state.global_step % self.config.logging_steps == 0
                {
                    tracing::info!("{}", self.metrics.log_line());
                }

                // Evaluation
                if self.config.eval_steps > 0
                    && self.state.global_step % self.config.eval_steps == 0
                {
                    if let Some(ref mut eval) = eval_data {
                        let eval_loss = self.evaluate(eval);
                        self.metrics.record_eval(eval_loss);
                        tracing::info!("Eval loss: {:.4} | Eval ppl: {:.2}", eval_loss, eval_loss.exp());

                        if eval_loss < self.state.best_eval_loss {
                            self.state.best_eval_loss = eval_loss;
                            tracing::info!("New best eval loss: {:.4}", eval_loss);
                        }
                    }
                }

                // Checkpoint
                if self.config.save_steps > 0
                    && self.state.global_step % self.config.save_steps == 0
                {
                    self.save_checkpoint()?;
                }
            }

            // End of epoch checkpoint
            self.save_checkpoint()?;
        }

        tracing::info!("Training complete!");
        tracing::info!("{}", self.metrics.log_line());

        // Save final model
        self.save_checkpoint()?;

        Ok(())
    }

    /// Run a single training step (simulated)
    fn train_step(&self, batch: &[&TrainingSample], _lr: f64) -> f64 {
        // In a full implementation, this would:
        // 1. Forward pass through the model
        // 2. Compute cross-entropy loss
        // 3. Backward pass to compute gradients
        // 4. Gradient accumulation
        // 5. Optimizer step + gradient clipping

        // Simulate a decreasing loss curve
        let base_loss = 3.0;
        let decay = (-(self.state.global_step as f64) * 0.001).exp();
        let noise = (self.state.global_step as f64 * 0.1).sin() * 0.05;
        base_loss * decay + 0.5 + noise
    }

    /// Run evaluation
    fn evaluate<D: Dataset>(&self, eval_data: &mut DataLoader<D>) -> f64 {
        eval_data.reset();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some(batch) = eval_data.next_batch() {
            // Simulate eval loss (slightly higher than train)
            let loss = self.train_step(&batch, 0.0) * 1.1;
            total_loss += loss;
            num_batches += 1;
        }

        if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        }
    }

    /// Save a checkpoint
    pub fn save_checkpoint(&mut self) -> Result<()> {
        let checkpoint_name = format!("checkpoint-{}", self.state.global_step);
        let checkpoint_dir = self.config.output_dir.join(&checkpoint_name);
        std::fs::create_dir_all(&checkpoint_dir)?;

        let checkpoint = Checkpoint {
            epoch: self.state.epoch,
            step: self.state.global_step,
            train_loss: self.metrics.last_train_loss(),
            eval_loss: self.metrics.last_eval_loss(),
            learning_rate: self.metrics.current_lr(),
            config: self.config.clone(),
        };

        let checkpoint_path = checkpoint_dir.join("trainer_state.json");
        let json = serde_json::to_string_pretty(&checkpoint)?;
        std::fs::write(&checkpoint_path, json)?;

        // Save metrics summary
        let metrics_path = checkpoint_dir.join("metrics.json");
        let metrics_json = serde_json::to_string_pretty(&self.metrics.summary())?;
        std::fs::write(&metrics_path, metrics_json)?;

        self.checkpoint_paths.push(checkpoint_dir.clone());
        tracing::info!("Saved checkpoint: {}", checkpoint_dir.display());

        // Enforce save limit
        if let Some(limit) = self.config.save_total_limit {
            while self.checkpoint_paths.len() > limit {
                if let Some(old_path) = self.checkpoint_paths.first().cloned() {
                    if old_path.exists() {
                        let _ = std::fs::remove_dir_all(&old_path);
                        tracing::debug!("Removed old checkpoint: {}", old_path.display());
                    }
                    self.checkpoint_paths.remove(0);
                }
            }
        }

        Ok(())
    }

    /// Load from a checkpoint
    pub fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
        let state_path = path.join("trainer_state.json");
        let json = std::fs::read_to_string(&state_path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&json)?;
        Ok(checkpoint)
    }

    /// Get the current metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get the training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }
}

/// Apply curriculum LR scaling, choosing the lower of attn/ssm scale for
/// parameters that don't distinguish between the two (e.g. embeddings).
fn apply_curriculum_lr(base_lr: f64, tick: &CurriculumTick) -> f64 {
    let scale = tick.attn_lr_scale.min(tick.ssm_lr_scale) as f64;
    base_lr * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TextDataset;
    use std::io::Write;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig {
            model_path: "test-model".into(),
            ..Default::default()
        };
        let trainer = Trainer::new(config).unwrap();
        assert_eq!(trainer.state.epoch, 0);
    }

    #[test]
    fn test_trainer_validation() {
        let config = TrainingConfig::default(); // empty model_path
        assert!(Trainer::new(config).is_err());
    }

    #[test]
    fn test_total_steps_calculation() {
        let config = TrainingConfig {
            model_path: "test".into(),
            num_epochs: 3,
            per_device_batch_size: 4,
            gradient_accumulation_steps: 2,
            ..Default::default()
        };
        let trainer = Trainer::new(config).unwrap();

        // 100 samples, effective batch size 8 -> 13 steps/epoch * 3 epochs = 39
        assert_eq!(trainer.total_steps(100), 39);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: trainer.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/trainer.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
