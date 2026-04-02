//! Fine-tuning methods: Full, LoRA, QLoRA

use crate::config::{FineTuningConfig, LoRAParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for fine-tuning methods
pub trait FineTuningMethod: Send + Sync {
    /// Get the method name
    fn name(&self) -> &str;

    /// Get the number of trainable parameters
    fn num_trainable_params(&self) -> usize;

    /// Get the total number of parameters
    fn num_total_params(&self) -> usize;

    /// Get the percentage of trainable parameters
    fn trainable_percentage(&self) -> f64 {
        if self.num_total_params() == 0 {
            return 0.0;
        }
        (self.num_trainable_params() as f64 / self.num_total_params() as f64) * 100.0
    }

    /// Check if a parameter should be trained
    fn is_trainable(&self, param_name: &str) -> bool;

    /// Apply method-specific parameter transformation
    fn transform_grad(&self, param_name: &str, grad: &[f32]) -> Vec<f32>;
}

/// Full parameter fine-tuning
pub struct FullFineTuning {
    total_params: usize,
    frozen_params: Vec<String>,
}

impl FullFineTuning {
    /// Create a new full fine-tuning method
    pub fn new(total_params: usize) -> Self {
        Self {
            total_params,
            frozen_params: Vec::new(),
        }
    }

    /// Freeze specific parameters
    pub fn freeze(&mut self, param_names: Vec<String>) {
        self.frozen_params = param_names;
    }
}

impl FineTuningMethod for FullFineTuning {
    fn name(&self) -> &str { "full" }

    fn num_trainable_params(&self) -> usize {
        self.total_params // Approximate; actual count requires param sizes
    }

    fn num_total_params(&self) -> usize { self.total_params }

    fn is_trainable(&self, param_name: &str) -> bool {
        !self.frozen_params.iter().any(|f| param_name.contains(f))
    }

    fn transform_grad(&self, _param_name: &str, grad: &[f32]) -> Vec<f32> {
        grad.to_vec()
    }
}

/// LoRA configuration for a single adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank matrices
    pub r: usize,
    /// Alpha scaling factor
    pub alpha: f64,
    /// Dropout rate
    pub dropout: f64,
    /// Target module names (e.g., "q_proj", "v_proj")
    pub target_modules: Vec<String>,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

impl From<&LoRAParams> for LoRAConfig {
    fn from(params: &LoRAParams) -> Self {
        Self {
            r: params.r,
            alpha: params.alpha,
            dropout: params.dropout,
            ..Default::default()
        }
    }
}

/// LoRA adapter weights for a single module
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Module name
    pub name: String,
    /// Low-rank matrix A (input_dim x r)
    pub lora_a: Vec<f32>,
    /// Low-rank matrix B (r x output_dim)
    pub lora_b: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Rank
    pub r: usize,
    /// Scaling factor: alpha / r
    pub scaling: f64,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter with Kaiming initialization for A, zero for B
    pub fn new(name: String, input_dim: usize, output_dim: usize, r: usize, alpha: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Kaiming uniform for A
        let bound = (1.0 / input_dim as f64).sqrt() as f32;
        let lora_a: Vec<f32> = (0..input_dim * r)
            .map(|_| rng.gen_range(-bound..bound))
            .collect();

        // Zero-initialize B so LoRA starts as identity
        let lora_b = vec![0.0f32; r * output_dim];

        Self {
            name,
            lora_a,
            lora_b,
            input_dim,
            output_dim,
            r,
            scaling: alpha / r as f64,
        }
    }

    /// Number of trainable parameters in this adapter
    pub fn num_params(&self) -> usize {
        self.input_dim * self.r + self.r * self.output_dim
    }
}

/// LoRA fine-tuning method
pub struct LoRAFineTuning {
    config: LoRAConfig,
    adapters: HashMap<String, LoRAAdapter>,
    total_params: usize,
}

impl LoRAFineTuning {
    /// Create a new LoRA fine-tuning method
    pub fn new(config: LoRAConfig, total_params: usize) -> Self {
        Self {
            config,
            adapters: HashMap::new(),
            total_params,
        }
    }

    /// Add an adapter for a module
    pub fn add_adapter(&mut self, module_name: &str, input_dim: usize, output_dim: usize) {
        let adapter = LoRAAdapter::new(
            module_name.to_string(),
            input_dim,
            output_dim,
            self.config.r,
            self.config.alpha,
        );
        self.adapters.insert(module_name.to_string(), adapter);
    }

    /// Get an adapter
    pub fn get_adapter(&self, module_name: &str) -> Option<&LoRAAdapter> {
        self.adapters.get(module_name)
    }

    /// Get a mutable adapter
    pub fn get_adapter_mut(&mut self, module_name: &str) -> Option<&mut LoRAAdapter> {
        self.adapters.get_mut(module_name)
    }

    /// Get all adapters
    pub fn adapters(&self) -> &HashMap<String, LoRAAdapter> {
        &self.adapters
    }

    /// Merge LoRA weights back into the base model weights
    pub fn merge_weights(&self, base_weights: &mut HashMap<String, Vec<f32>>) {
        for (name, adapter) in &self.adapters {
            if let Some(weights) = base_weights.get_mut(name) {
                // W' = W + scaling * (B @ A)
                // Simple matrix multiply for merging
                for i in 0..adapter.output_dim {
                    for j in 0..adapter.input_dim {
                        let mut sum = 0.0f32;
                        for k in 0..adapter.r {
                            sum += adapter.lora_b[k * adapter.output_dim + i]
                                * adapter.lora_a[j * adapter.r + k];
                        }
                        let idx = i * adapter.input_dim + j;
                        if idx < weights.len() {
                            weights[idx] += sum * adapter.scaling as f32;
                        }
                    }
                }
            }
        }
    }
}

impl FineTuningMethod for LoRAFineTuning {
    fn name(&self) -> &str { "lora" }

    fn num_trainable_params(&self) -> usize {
        self.adapters.values().map(|a| a.num_params()).sum()
    }

    fn num_total_params(&self) -> usize { self.total_params }

    fn is_trainable(&self, param_name: &str) -> bool {
        // Only LoRA adapter parameters are trainable
        self.config.target_modules.iter().any(|m| param_name.contains(m))
    }

    fn transform_grad(&self, _param_name: &str, grad: &[f32]) -> Vec<f32> {
        grad.to_vec()
    }
}

/// QLoRA (quantized LoRA) fine-tuning
pub struct QLoRAFineTuning {
    inner: LoRAFineTuning,
    /// Quantization bits for base model (4 or 8)
    quant_bits: u8,
}

impl QLoRAFineTuning {
    /// Create a new QLoRA fine-tuning method
    pub fn new(config: LoRAConfig, total_params: usize, quant_bits: u8) -> Self {
        Self {
            inner: LoRAFineTuning::new(config, total_params),
            quant_bits,
        }
    }

    /// Add an adapter
    pub fn add_adapter(&mut self, module_name: &str, input_dim: usize, output_dim: usize) {
        self.inner.add_adapter(module_name, input_dim, output_dim);
    }

    /// Get quantization bits
    pub fn quant_bits(&self) -> u8 {
        self.quant_bits
    }

    /// Estimated memory savings vs full fine-tuning
    pub fn memory_savings_ratio(&self) -> f64 {
        let full_bytes = self.inner.total_params as f64 * 2.0; // FP16
        let quant_bytes = self.inner.total_params as f64 * (self.quant_bits as f64 / 8.0);
        let lora_bytes = self.inner.num_trainable_params() as f64 * 2.0; // FP16 adapters
        (quant_bytes + lora_bytes) / full_bytes
    }
}

impl FineTuningMethod for QLoRAFineTuning {
    fn name(&self) -> &str { "qlora" }

    fn num_trainable_params(&self) -> usize {
        self.inner.num_trainable_params()
    }

    fn num_total_params(&self) -> usize {
        self.inner.num_total_params()
    }

    fn is_trainable(&self, param_name: &str) -> bool {
        self.inner.is_trainable(param_name)
    }

    fn transform_grad(&self, param_name: &str, grad: &[f32]) -> Vec<f32> {
        self.inner.transform_grad(param_name, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_adapter_creation() {
        let adapter = LoRAAdapter::new("test".to_string(), 4096, 4096, 16, 32.0);
        assert_eq!(adapter.num_params(), 4096 * 16 + 16 * 4096);
        assert_eq!(adapter.scaling, 2.0); // 32 / 16
    }

    #[test]
    fn test_lora_trainable_percentage() {
        let config = LoRAConfig::default();
        let mut lora = LoRAFineTuning::new(config, 7_000_000_000);
        lora.add_adapter("q_proj", 4096, 4096);
        lora.add_adapter("v_proj", 4096, 4096);

        // Should be a tiny fraction of total params
        assert!(lora.trainable_percentage() < 1.0);
    }

    #[test]
    fn test_full_fine_tuning() {
        let mut full = FullFineTuning::new(1000);
        assert!(full.is_trainable("any_param"));

        full.freeze(vec!["embed".to_string()]);
        assert!(!full.is_trainable("embed_tokens"));
        assert!(full.is_trainable("attention.q_proj"));
    }

    #[test]
    fn test_qlora_memory_savings() {
        let config = LoRAConfig::default();
        let qlora = QLoRAFineTuning::new(config, 7_000_000_000, 4);
        assert!(qlora.memory_savings_ratio() < 0.5); // Should save >50% memory
    }
}
