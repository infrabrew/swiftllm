// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      config.rs
// PATH:      /crates/swiftllm-core/src/config.rs
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

//! Configuration types for SwiftLLM engine
//!
//! This module provides configuration structs for all aspects of the inference engine,
//! including model settings, memory management, scheduling, and sampling parameters.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main engine configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Model configuration
    pub model: ModelConfig,

    /// Scheduler configuration
    pub scheduler: SchedulerConfig,

    /// Memory configuration
    pub memory: MemoryConfig,

    /// Device configuration
    pub device: DeviceConfig,

    /// Speculative decoding configuration (optional)
    pub speculative: Option<SpeculativeConfig>,
}

impl EngineConfig {
    /// Create a new engine configuration with a model path
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model: ModelConfig {
                path: model_path.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Set the tensor parallel size
    pub fn with_tensor_parallel(mut self, tp_size: usize) -> Self {
        self.device.tensor_parallel_size = tp_size;
        self
    }

    /// Set the maximum sequence length
    pub fn with_max_seq_len(mut self, max_len: usize) -> Self {
        self.model.max_seq_len = max_len;
        self
    }

    /// Enable speculative decoding with a draft model
    pub fn with_speculative_decoding(mut self, draft_model: impl Into<PathBuf>) -> Self {
        self.speculative = Some(SpeculativeConfig {
            draft_model_path: draft_model.into(),
            ..Default::default()
        });
        self
    }
}

/// Model-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model (local path or HuggingFace model ID)
    pub path: PathBuf,

    /// Model architecture (auto-detected if not specified)
    pub architecture: Option<ModelArchitecture>,

    /// Data type for model weights
    pub dtype: DataType,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Maximum number of tokens in a batch
    pub max_batch_tokens: usize,

    /// Trust remote code (for HuggingFace models)
    pub trust_remote_code: bool,

    /// Quantization configuration (optional)
    pub quantization: Option<QuantizationConfig>,

    /// Rope scaling configuration (optional)
    pub rope_scaling: Option<RopeScalingConfig>,

    /// EOS token ID (model-specific, defaults to 2 if unset)
    pub eos_token_id: Option<u32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            architecture: None,
            dtype: DataType::Float16,
            max_seq_len: 4096,
            max_batch_tokens: 8192,
            trust_remote_code: false,
            quantization: None,
            rope_scaling: None,
            eos_token_id: None,
        }
    }
}

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// LLaMA family (LLaMA, LLaMA 2, LLaMA 3)
    Llama,
    /// Mistral family (Mistral, Mixtral)
    Mistral,
    /// Mixtral MoE
    Mixtral,
    /// Qwen family
    Qwen,
    /// Qwen 2
    Qwen2,
    /// Phi family
    Phi,
    /// Phi-3
    Phi3,
    /// Falcon
    Falcon,
    /// GPT-NeoX
    GptNeox,
    /// GPT-J
    GptJ,
    /// MPT
    Mpt,
    /// Bloom
    Bloom,
    /// Gemma
    Gemma,
    /// DeepSeek
    DeepSeek,
    /// Jamba: Hybrid Mamba-SSM + Transformer + (optional) MoE
    /// AI21 Labs ICLR 2025 — 1:7 attention ratio, 256K context, single 80GB GPU
    Jamba,
    /// Pure Mamba-2/3 SSM architecture (no attention layers)
    Mamba,
    /// Zamba: weight-shared attention + Mamba hybrid (Zyphra 2024)
    Zamba,
    /// Nemotron-H: NVIDIA hybrid 56B — matches Llama-3.1-70B at 3x throughput
    NemotronH,
}

/// Data types for model weights and computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// Brain floating point (16-bit)
    BFloat16,
    /// 8-bit floating point (E4M3)
    Float8E4M3,
    /// 8-bit floating point (E5M2)
    Float8E5M2,
    /// 8-bit integer
    Int8,
    /// 4-bit integer
    Int4,
}

impl DataType {
    /// Get the size in bytes for one element of this data type.
    ///
    /// **Note:** For sub-byte types like `Int4`, this returns 1 because
    /// elements are packed into bytes (2 elements per byte). Use
    /// [`size_bytes_for_elements`] to compute the correct byte count
    /// for a given number of elements.
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Float8E4M3 | DataType::Float8E5M2 | DataType::Int8 => 1,
            DataType::Int4 => 1, // Packed: 2 elements per byte (see size_bytes_for_elements)
        }
    }

    /// Get the bit width of a single element.
    pub fn element_bit_width(&self) -> usize {
        match self {
            DataType::Float32 => 32,
            DataType::Float16 | DataType::BFloat16 => 16,
            DataType::Float8E4M3 | DataType::Float8E5M2 | DataType::Int8 => 8,
            DataType::Int4 => 4,
        }
    }

    /// Compute the number of bytes needed to store `num_elements` values.
    ///
    /// This correctly handles sub-byte packing (e.g., INT4 packs 2 elements
    /// per byte, rounding up for odd counts).
    pub fn size_bytes_for_elements(&self, num_elements: usize) -> usize {
        let bits = num_elements * self.element_bit_width();
        bits.div_ceil(8)
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method
    pub method: QuantizationMethod,

    /// Number of bits for weights
    pub bits: u8,

    /// Group size for quantization
    pub group_size: usize,

    /// Whether to use symmetric quantization
    pub symmetric: bool,
}

/// Supported quantization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMethod {
    /// GPTQ quantization
    Gptq,
    /// AWQ quantization
    Awq,
    /// GGML/GGUF quantization
    Ggml,
    /// SqueezeLLM
    SqueezeLlm,
    /// Marlin (optimized GPTQ)
    Marlin,
    /// FP8 quantization
    Fp8,
    /// TurboQuant — online vector quantization with near-optimal distortion
    /// (Zandieh et al., ICLR 2026). Random-rotation + Beta-distribution
    /// scalar quantizer for KV cache compression.
    TurboQuant,
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type
    pub scaling_type: RopeScalingType,

    /// Scaling factor
    pub factor: f32,

    /// Original maximum position embeddings
    pub original_max_position_embeddings: Option<usize>,
}

/// RoPE scaling types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingType {
    /// Linear scaling
    Linear,
    /// Dynamic NTK-aware scaling
    Dynamic,
    /// YaRN scaling
    Yarn,
    /// Longrope scaling
    Longrope,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of sequences in a batch
    pub max_num_seqs: usize,

    /// Maximum number of tokens per iteration
    pub max_num_batched_tokens: usize,

    /// Maximum padding percentage (0.0 - 1.0)
    pub max_padding_percentage: f32,

    /// Enable preemption
    pub enable_preemption: bool,

    /// Preemption mode
    pub preemption_mode: PreemptionMode,

    /// Delay factor for request scheduling (0.0 - 1.0)
    pub delay_factor: f32,

    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,

    /// Maximum number of tokens per prefill chunk
    pub max_prefill_tokens: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_padding_percentage: 0.2,
            enable_preemption: true,
            preemption_mode: PreemptionMode::Recompute,
            delay_factor: 0.0,
            enable_chunked_prefill: true,
            max_prefill_tokens: 2048,
            request_timeout_secs: 300,
        }
    }
}

/// Preemption modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreemptionMode {
    /// Recompute KV cache after preemption
    Recompute,
    /// Swap KV cache to CPU memory
    Swap,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Block size for PagedAttention (number of tokens per block)
    pub block_size: usize,

    /// GPU memory utilization (0.0 - 1.0)
    pub gpu_memory_utilization: f32,

    /// Swap space in GiB
    pub swap_space_gib: f32,

    /// CPU offload fraction (0.0 - 1.0)
    pub cpu_offload_fraction: f32,

    /// Enable prefix caching
    pub enable_prefix_caching: bool,

    /// Maximum number of blocks to cache for prefix
    pub max_prefix_cache_blocks: Option<usize>,

    /// Enable sliding window attention (if supported by model)
    pub sliding_window: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            gpu_memory_utilization: 0.90,
            swap_space_gib: 4.0,
            cpu_offload_fraction: 0.0,
            enable_prefix_caching: true,
            max_prefix_cache_blocks: None,
            sliding_window: None,
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Device type
    pub device: DeviceType,

    /// Tensor parallel size (number of GPUs for tensor parallelism)
    pub tensor_parallel_size: usize,

    /// Pipeline parallel size (number of stages)
    pub pipeline_parallel_size: usize,

    /// GPU IDs to use (if None, use all available)
    pub gpu_ids: Option<Vec<usize>>,

    /// Enable CUDA graphs
    pub enable_cuda_graphs: bool,

    /// Maximum number of captured CUDA graphs
    pub max_cuda_graphs: usize,

    /// Enforce eager mode (disable compilation optimizations)
    pub enforce_eager: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: DeviceType::Cuda,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            gpu_ids: None,
            enable_cuda_graphs: true,
            max_cuda_graphs: 10,
            enforce_eager: false,
        }
    }
}

/// Device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    /// CUDA GPU
    Cuda,
    /// ROCm GPU
    Rocm,
    /// CPU only
    Cpu,
    /// Apple Metal
    Metal,
}

/// Speculative decoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Path to the draft model
    pub draft_model_path: PathBuf,

    /// Number of speculative tokens to generate
    pub num_speculative_tokens: usize,

    /// Draft model tensor parallel size
    pub draft_tensor_parallel_size: usize,

    /// Enable ngram speculation
    pub enable_ngram_speculation: bool,

    /// Ngram prompt lookup window size
    pub ngram_prompt_lookup_max: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_model_path: PathBuf::new(),
            num_speculative_tokens: 5,
            draft_tensor_parallel_size: 1,
            enable_ngram_speculation: false,
            ngram_prompt_lookup_max: 4,
        }
    }
}

/// Sampling configuration (generation parameters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for sampling
    pub temperature: f32,

    /// Top-p (nucleus) sampling
    pub top_p: f32,

    /// Top-k sampling
    pub top_k: i32,

    /// Minimum probability for top-p
    pub min_p: f32,

    /// Repetition penalty
    pub repetition_penalty: f32,

    /// Frequency penalty (OpenAI style)
    pub frequency_penalty: f32,

    /// Presence penalty (OpenAI style)
    pub presence_penalty: f32,

    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Stop sequences
    pub stop: Vec<String>,

    /// Stop token IDs
    pub stop_token_ids: Vec<u32>,

    /// Whether to include stop sequence in output
    pub include_stop_str_in_output: bool,

    /// Skip special tokens in output
    pub skip_special_tokens: bool,

    /// Number of sequences to return
    pub n: usize,

    /// Best-of sampling (generate n sequences, return best)
    pub best_of: Option<usize>,

    /// Seed for reproducibility
    pub seed: Option<u64>,

    /// Return log probabilities
    pub logprobs: Option<usize>,

    /// Return prompt log probabilities
    pub prompt_logprobs: Option<usize>,

    /// Logit bias
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 256,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            include_stop_str_in_output: false,
            skip_special_tokens: true,
            n: 1,
            best_of: None,
            seed: None,
            logprobs: None,
            prompt_logprobs: None,
            logit_bias: None,
        }
    }
}

impl SamplingConfig {
    /// Create a greedy sampling configuration (temperature=0)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Create a sampling configuration for creative generation
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Create a sampling configuration for balanced generation
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            ..Default::default()
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Add stop sequences
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,

    /// Port to bind to
    pub port: u16,

    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,

    /// Enable CORS
    pub enable_cors: bool,

    /// Allowed origins for CORS
    pub cors_origins: Vec<String>,

    /// API key (optional)
    pub api_key: Option<String>,

    /// Enable request logging
    pub enable_logging: bool,

    /// Log level
    pub log_level: String,

    /// Enable metrics endpoint
    pub enable_metrics: bool,

    /// Response timeout in seconds
    pub response_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            max_concurrent_requests: 1000,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            api_key: None,
            enable_logging: true,
            log_level: "info".to_string(),
            enable_metrics: true,
            response_timeout_secs: 600,
        }
    }
}

/// TurboQuant KV cache compression configuration
///
/// TurboQuant (Zandieh et al., ICLR 2026) is an online vector quantization
/// algorithm that compresses KV cache vectors via random rotation followed by
/// per-coordinate scalar quantization against a precomputed Beta-distribution
/// codebook. It achieves near-optimal distortion rate with no training
/// required.
///
/// Two algorithm variants are supported:
///   - **MSE** (`TurboQuantMse`): minimises mean-squared reconstruction error.
///   - **InnerProduct** (`TurboQuantProd`): unbiased inner-product estimation
///     (uses b−1 bits for MSE quantization + 1 bit for JL residual sign).
///
/// Reference: "TurboQuant: Online Vector Quantization with Near-optimal
/// Distortion Rate" (arXiv 2504.19874)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantConfig {
    /// Bit-width per channel for key cache (default: 4).
    ///
    /// 3–4 bits achieves near-lossless quality; 2 bits incurs marginal
    /// degradation (≈0.1 ppl on LLaMA-2-7B).
    pub key_bits: u8,

    /// Bit-width per channel for value cache (default: 4).
    pub value_bits: u8,

    /// Residual length for the rotation matrix (number of random sign flips
    /// in the fast Walsh-Hadamard transform). `None` = use dimension of head
    /// (recommended).
    pub residual_length: Option<usize>,

    /// Whether to use the inner-product–optimised variant (`TurboQuantProd`).
    ///
    /// When `true`, the quantizer allocates `bits - 1` to MSE quantization
    /// and 1 bit to a Johnson–Lindenstrauss sign sketch of the residual,
    /// producing unbiased dot-product estimates. When `false`, the plain
    /// `TurboQuantMse` variant is used.
    pub use_inner_product_variant: bool,

    /// Group size for per-group quantization (default: 0 = per-head).
    ///
    /// Larger groups amortise rotation overhead but may reduce quality.
    /// 0 means treat each attention head independently (recommended).
    pub group_size: usize,

    /// Whether to quantize during prefill (default: true).
    ///
    /// Disabling this keeps full-precision KV during the prefill phase and
    /// only quantizes tokens generated during decode, which can improve
    /// long-context quality at the cost of higher prefill memory.
    pub quantize_prefill: bool,

    /// Random seed for the rotation matrix (deterministic reproducibility).
    pub seed: u64,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            key_bits: 4,
            value_bits: 4,
            residual_length: None,
            use_inner_product_variant: false,
            group_size: 0,
            quantize_prefill: true,
            seed: 42,
        }
    }
}

impl TurboQuantConfig {
    /// Create a config with the given bit-widths for keys and values.
    pub fn new(key_bits: u8, value_bits: u8) -> Self {
        Self {
            key_bits,
            value_bits,
            ..Default::default()
        }
    }

    /// Create the recommended "quality-neutral" preset (3.5-bit effective).
    ///
    /// Uses 4-bit keys and 3-bit values — matches full-precision quality
    /// on most benchmarks while cutting KV cache memory by ≈4×.
    pub fn quality_neutral() -> Self {
        Self {
            key_bits: 4,
            value_bits: 3,
            ..Default::default()
        }
    }

    /// Create the "aggressive" preset (2.5-bit effective).
    ///
    /// Uses 3-bit keys and 2-bit values — ≈5× memory reduction with
    /// marginal quality degradation.
    pub fn aggressive() -> Self {
        Self {
            key_bits: 3,
            value_bits: 2,
            ..Default::default()
        }
    }

    /// Enable the inner-product–optimised variant.
    pub fn with_inner_product(mut self) -> Self {
        self.use_inner_product_variant = true;
        self
    }

    /// Compute the compression ratio relative to FP16 storage.
    ///
    /// Returns the ratio of compressed size to original size (e.g., 0.25
    /// means 4× compression).
    pub fn compression_ratio(&self) -> f32 {
        let original_bits = 16.0_f32; // FP16 baseline
        let avg_bits = (self.key_bits as f32 + self.value_bits as f32) / 2.0;
        avg_bits / original_bits
    }

    /// Validate the configuration, returning an error on invalid settings.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.key_bits == 0 || self.key_bits > 16 {
            return Err(crate::error::Error::InvalidConfig(format!(
                "TurboQuant key_bits must be in [1, 16], got {}",
                self.key_bits
            )));
        }
        if self.value_bits == 0 || self.value_bits > 16 {
            return Err(crate::error::Error::InvalidConfig(format!(
                "TurboQuant value_bits must be in [1, 16], got {}",
                self.value_bits
            )));
        }
        if self.use_inner_product_variant && (self.key_bits < 2 || self.value_bits < 2) {
            return Err(crate::error::Error::InvalidConfig(
                "TurboQuantProd requires at least 2 bits (1 for MSE + 1 for JL sign)".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_builder() {
        let config = EngineConfig::new("meta-llama/Llama-2-7b-hf")
            .with_tensor_parallel(2)
            .with_max_seq_len(8192);

        assert_eq!(config.device.tensor_parallel_size, 2);
        assert_eq!(config.model.max_seq_len, 8192);
    }

    #[test]
    fn test_sampling_config_presets() {
        let greedy = SamplingConfig::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let creative = SamplingConfig::creative();
        assert_eq!(creative.temperature, 0.9);
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float16.size_bytes(), 2);
        assert_eq!(DataType::Int8.size_bytes(), 1);
    }

    #[test]
    fn test_turbo_quant_config_default() {
        let cfg = TurboQuantConfig::default();
        assert_eq!(cfg.key_bits, 4);
        assert_eq!(cfg.value_bits, 4);
        assert!(!cfg.use_inner_product_variant);
        assert_eq!(cfg.group_size, 0);
        assert!(cfg.quantize_prefill);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_turbo_quant_config_presets() {
        let qn = TurboQuantConfig::quality_neutral();
        assert_eq!(qn.key_bits, 4);
        assert_eq!(qn.value_bits, 3);
        assert!((qn.compression_ratio() - 3.5 / 16.0).abs() < 1e-6);
        assert!(qn.validate().is_ok());

        let ag = TurboQuantConfig::aggressive();
        assert_eq!(ag.key_bits, 3);
        assert_eq!(ag.value_bits, 2);
        assert!((ag.compression_ratio() - 2.5 / 16.0).abs() < 1e-6);
        assert!(ag.validate().is_ok());
    }

    #[test]
    fn test_turbo_quant_config_validation() {
        // Zero bits is invalid
        let cfg_zero = TurboQuantConfig { key_bits: 0, ..TurboQuantConfig::default() };
        assert!(cfg_zero.validate().is_err());

        // Over 16 bits is invalid
        let cfg_over = TurboQuantConfig { key_bits: 17, ..TurboQuantConfig::default() };
        assert!(cfg_over.validate().is_err());

        // Inner product with 1 bit is invalid (needs ≥2)
        let mut cfg2 = TurboQuantConfig::new(1, 2).with_inner_product();
        assert!(cfg2.validate().is_err());
        cfg2.key_bits = 2;
        assert!(cfg2.validate().is_ok());
    }

    #[test]
    fn test_turbo_quant_compression_ratio() {
        let cfg = TurboQuantConfig::new(4, 4);
        assert!((cfg.compression_ratio() - 0.25).abs() < 1e-6);

        let cfg2 = TurboQuantConfig::new(8, 8);
        assert!((cfg2.compression_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_turbo_quant_serialization() {
        let cfg = TurboQuantConfig::quality_neutral();
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: TurboQuantConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.key_bits, cfg.key_bits);
        assert_eq!(deserialized.value_bits, cfg.value_bits);
        assert_eq!(
            deserialized.use_inner_product_variant,
            cfg.use_inner_product_variant
        );
    }

    #[test]
    fn test_quantization_method_turboquant() {
        let method = QuantizationMethod::TurboQuant;
        let json = serde_json::to_string(&method).unwrap();
        assert_eq!(json, "\"turboquant\"");
        let deserialized: QuantizationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, QuantizationMethod::TurboQuant);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: config.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/config.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
