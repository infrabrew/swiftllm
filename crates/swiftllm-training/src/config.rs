//! Training configuration types

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model path to train or fine-tune
    pub model_path: PathBuf,

    /// Output directory for checkpoints and logs
    pub output_dir: PathBuf,

    /// Number of training epochs
    pub num_epochs: usize,

    /// Batch size per device
    pub per_device_batch_size: usize,

    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Weight decay
    pub weight_decay: f64,

    /// Warmup steps (or fraction of total steps)
    pub warmup_steps: WarmupConfig,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,

    /// Learning rate scheduler type
    pub lr_scheduler: LrSchedulerType,

    /// Mixed precision training mode
    pub mixed_precision: MixedPrecision,

    /// Logging interval (in steps)
    pub logging_steps: usize,

    /// Checkpoint save interval (in steps, 0 = save per epoch only)
    pub save_steps: usize,

    /// Maximum number of checkpoints to keep
    pub save_total_limit: Option<usize>,

    /// Evaluation interval (in steps, 0 = per epoch only)
    pub eval_steps: usize,

    /// Random seed
    pub seed: u64,

    /// Data configuration
    pub data: DataConfig,

    /// Resume from checkpoint path (optional)
    pub resume_from_checkpoint: Option<PathBuf>,

    /// Number of data loading workers
    pub dataloader_num_workers: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            output_dir: PathBuf::from("./output"),
            num_epochs: 3,
            per_device_batch_size: 4,
            gradient_accumulation_steps: 1,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            warmup_steps: WarmupConfig::Steps(100),
            max_grad_norm: 1.0,
            lr_scheduler: LrSchedulerType::Cosine,
            mixed_precision: MixedPrecision::Fp16,
            logging_steps: 10,
            save_steps: 500,
            save_total_limit: Some(3),
            eval_steps: 500,
            seed: 42,
            data: DataConfig::default(),
            resume_from_checkpoint: None,
            dataloader_num_workers: 4,
        }
    }
}

impl TrainingConfig {
    /// Effective batch size (per_device * gradient_accumulation)
    pub fn effective_batch_size(&self) -> usize {
        self.per_device_batch_size * self.gradient_accumulation_steps
    }
}

/// Fine-tuning configuration (extends TrainingConfig)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Base training configuration
    #[serde(flatten)]
    pub training: TrainingConfig,

    /// Fine-tuning method
    pub method: FineTuningMethod,

    /// LoRA-specific configuration
    pub lora: Option<LoRAParams>,

    /// Freeze embedding layers
    pub freeze_embeddings: bool,

    /// Freeze specific layer indices (0-indexed)
    pub freeze_layers: Vec<usize>,

    /// Target modules for LoRA (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            training: TrainingConfig {
                learning_rate: 2e-4,
                num_epochs: 1,
                ..Default::default()
            },
            method: FineTuningMethod::LoRA,
            lora: Some(LoRAParams::default()),
            freeze_embeddings: true,
            freeze_layers: Vec::new(),
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

/// Fine-tuning method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FineTuningMethod {
    /// Full parameter fine-tuning
    Full,
    /// Low-Rank Adaptation
    LoRA,
    /// Quantized LoRA
    QLoRA,
}

/// LoRA hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAParams {
    /// LoRA rank
    pub r: usize,

    /// LoRA alpha (scaling factor)
    pub alpha: f64,

    /// LoRA dropout
    pub dropout: f64,

    /// Use RSLoRA scaling
    pub use_rslora: bool,
}

impl Default for LoRAParams {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 32.0,
            dropout: 0.05,
            use_rslora: false,
        }
    }
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data path (file or directory)
    pub train_path: PathBuf,

    /// Validation data path (optional)
    pub eval_path: Option<PathBuf>,

    /// Data format
    pub format: DataFormat,

    /// Maximum sequence length (tokens)
    pub max_seq_len: usize,

    /// Shuffle training data
    pub shuffle: bool,

    /// Instruction template (for instruction datasets)
    pub instruction_template: Option<String>,

    /// Column mapping for structured data
    pub column_mapping: Option<ColumnMapping>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_path: PathBuf::new(),
            eval_path: None,
            format: DataFormat::Jsonl,
            max_seq_len: 2048,
            shuffle: true,
            instruction_template: None,
            column_mapping: None,
        }
    }
}

/// Supported data formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataFormat {
    /// JSON Lines (one JSON object per line)
    Jsonl,
    /// CSV with headers
    Csv,
    /// Plain text (one sample per line)
    Text,
    /// Parquet
    Parquet,
}

/// Column mapping for structured data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMapping {
    /// Column name for instruction/prompt
    pub instruction: String,
    /// Column name for input (optional context)
    pub input: Option<String>,
    /// Column name for output/response
    pub output: String,
}

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WarmupConfig {
    /// Number of warmup steps
    Steps(usize),
    /// Fraction of total steps (0.0 - 1.0)
    Ratio(f64),
}

/// Learning rate scheduler type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LrSchedulerType {
    /// Linear decay
    Linear,
    /// Cosine annealing
    Cosine,
    /// Cosine with restarts
    CosineWithRestarts,
    /// Constant learning rate
    Constant,
    /// Constant with warmup
    ConstantWithWarmup,
    /// Polynomial decay
    Polynomial,
}

/// Mixed precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MixedPrecision {
    /// No mixed precision
    No,
    /// FP16 mixed precision
    Fp16,
    /// BF16 mixed precision
    Bf16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.effective_batch_size(), 4);
    }

    #[test]
    fn test_effective_batch_size() {
        let config = TrainingConfig {
            per_device_batch_size: 8,
            gradient_accumulation_steps: 4,
            ..Default::default()
        };
        assert_eq!(config.effective_batch_size(), 32);
    }

    #[test]
    fn test_fine_tuning_config_defaults() {
        let config = FineTuningConfig::default();
        assert_eq!(config.method, FineTuningMethod::LoRA);
        assert!(config.lora.is_some());
        assert_eq!(config.lora.unwrap().r, 16);
    }

    #[test]
    fn test_serialization() {
        let config = TrainingConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.num_epochs, deserialized.num_epochs);
    }
}
