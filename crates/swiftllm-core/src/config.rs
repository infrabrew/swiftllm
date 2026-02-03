//! Configuration types for SwiftLLM engine
//!
//! This module provides configuration structs for all aspects of the inference engine,
//! including model settings, memory management, scheduling, and sampling parameters.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            scheduler: SchedulerConfig::default(),
            memory: MemoryConfig::default(),
            device: DeviceConfig::default(),
            speculative: None,
        }
    }
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
    /// Get the size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Float8E4M3 | DataType::Float8E5M2 | DataType::Int8 => 1,
            DataType::Int4 => 1, // Packed, actual size is 0.5 bytes per element
        }
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
}
