//! HuggingFace Model Loader
//!
//! Loads models from HuggingFace Hub or local HuggingFace-format directories.

use super::WeightLoader;
use crate::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use swiftllm_core::config::ModelArchitecture;
use swiftllm_core::error::{Error, Result};
use swiftllm_core::tensor::Tensor;

/// HuggingFace config.json structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfConfig {
    /// Model architecture (e.g., "LlamaForCausalLM")
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Model type (e.g., "llama")
    #[serde(default)]
    pub model_type: Option<String>,

    /// Hidden size
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate size (MLP)
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Number of attention heads
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of hidden layers
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,

    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Maximum position embeddings
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: Option<f32>,

    /// Layer norm epsilon (for non-RMS models)
    #[serde(default)]
    pub layer_norm_eps: Option<f32>,

    /// Rope theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: Option<f32>,

    /// Head dimension
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Sliding window attention size
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Whether to use bias in attention
    #[serde(default)]
    pub attention_bias: Option<bool>,

    /// Whether to use bias in MLP
    #[serde(default)]
    pub mlp_bias: Option<bool>,

    /// Tie word embeddings
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    /// BOS token ID
    #[serde(default = "default_bos")]
    pub bos_token_id: Option<u32>,

    /// EOS token ID
    #[serde(default = "default_eos")]
    pub eos_token_id: Option<u32>,

    /// Pad token ID
    #[serde(default)]
    pub pad_token_id: Option<u32>,

    /// Torch dtype
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Rope scaling configuration
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
}

fn default_hidden_size() -> usize { 4096 }
fn default_num_heads() -> usize { 32 }
fn default_num_layers() -> usize { 32 }
fn default_vocab_size() -> usize { 32000 }
fn default_max_position() -> usize { 4096 }
fn default_rms_eps() -> Option<f32> { Some(1e-5) }
fn default_rope_theta() -> Option<f32> { Some(10000.0) }
fn default_bos() -> Option<u32> { Some(1) }
fn default_eos() -> Option<u32> { Some(2) }

/// Rope scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(rename = "type")]
    pub scaling_type: Option<String>,
    pub factor: Option<f32>,
}

/// Load model configuration from a path
pub fn load_config(path: impl AsRef<Path>) -> Result<ModelConfig> {
    let path = path.as_ref();
    let config_path = if path.is_dir() {
        path.join("config.json")
    } else {
        path.to_path_buf()
    };

    let config_str = fs::read_to_string(&config_path).map_err(|e| {
        Error::ModelLoad(format!("Failed to read config.json: {}", e))
    })?;

    let hf_config: HfConfig = serde_json::from_str(&config_str).map_err(|e| {
        Error::InvalidConfig(format!("Failed to parse config.json: {}", e))
    })?;

    convert_hf_config(hf_config)
}

/// Detect model architecture from config
pub fn detect_architecture(path: impl AsRef<Path>) -> Result<ModelArchitecture> {
    let path = path.as_ref();
    let config_path = if path.is_dir() {
        path.join("config.json")
    } else {
        path.to_path_buf()
    };

    let config_str = fs::read_to_string(&config_path).map_err(|e| {
        Error::ModelLoad(format!("Failed to read config.json: {}", e))
    })?;

    let hf_config: HfConfig = serde_json::from_str(&config_str).map_err(|e| {
        Error::InvalidConfig(format!("Failed to parse config.json: {}", e))
    })?;

    // Try to detect from architectures field
    for arch in &hf_config.architectures {
        let arch_lower = arch.to_lowercase();
        if arch_lower.contains("llama") {
            return Ok(ModelArchitecture::Llama);
        } else if arch_lower.contains("mistral") {
            return Ok(ModelArchitecture::Mistral);
        } else if arch_lower.contains("mixtral") {
            return Ok(ModelArchitecture::Mixtral);
        } else if arch_lower.contains("qwen2") {
            return Ok(ModelArchitecture::Qwen2);
        } else if arch_lower.contains("qwen") {
            return Ok(ModelArchitecture::Qwen);
        } else if arch_lower.contains("phi3") || arch_lower.contains("phi-3") {
            return Ok(ModelArchitecture::Phi3);
        } else if arch_lower.contains("phi") {
            return Ok(ModelArchitecture::Phi);
        } else if arch_lower.contains("falcon") {
            return Ok(ModelArchitecture::Falcon);
        } else if arch_lower.contains("gemma") {
            return Ok(ModelArchitecture::Gemma);
        }
    }

    // Try model_type
    if let Some(model_type) = &hf_config.model_type {
        let model_type_lower = model_type.to_lowercase();
        if model_type_lower == "llama" {
            return Ok(ModelArchitecture::Llama);
        } else if model_type_lower == "mistral" {
            return Ok(ModelArchitecture::Mistral);
        } else if model_type_lower == "mixtral" {
            return Ok(ModelArchitecture::Mixtral);
        } else if model_type_lower == "qwen2" {
            return Ok(ModelArchitecture::Qwen2);
        } else if model_type_lower == "qwen" {
            return Ok(ModelArchitecture::Qwen);
        } else if model_type_lower == "phi3" {
            return Ok(ModelArchitecture::Phi3);
        } else if model_type_lower == "phi" {
            return Ok(ModelArchitecture::Phi);
        }
    }

    Err(Error::UnsupportedArchitecture(
        hf_config.architectures.first()
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    ))
}

/// Convert HuggingFace config to our ModelConfig
fn convert_hf_config(hf: HfConfig) -> Result<ModelConfig> {
    let architecture = if let Some(ref model_type) = hf.model_type {
        match model_type.to_lowercase().as_str() {
            "llama" => ModelArchitecture::Llama,
            "mistral" => ModelArchitecture::Mistral,
            "mixtral" => ModelArchitecture::Mixtral,
            "qwen" => ModelArchitecture::Qwen,
            "qwen2" => ModelArchitecture::Qwen2,
            "phi" => ModelArchitecture::Phi,
            "phi3" => ModelArchitecture::Phi3,
            _ => ModelArchitecture::Llama, // Default
        }
    } else {
        ModelArchitecture::Llama
    };

    let num_kv_heads = hf.num_key_value_heads.unwrap_or(hf.num_attention_heads);
    let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / hf.num_attention_heads);

    // Calculate intermediate size if not provided
    let intermediate_size = hf.intermediate_size.unwrap_or_else(|| {
        // Common formula: 8/3 * hidden_size, rounded to multiple of 256
        let size = (8 * hf.hidden_size) / 3;
        ((size + 255) / 256) * 256
    });

    Ok(ModelConfig {
        architecture,
        hidden_size: hf.hidden_size,
        intermediate_size,
        num_attention_heads: hf.num_attention_heads,
        num_key_value_heads: num_kv_heads,
        num_hidden_layers: hf.num_hidden_layers,
        vocab_size: hf.vocab_size,
        max_position_embeddings: hf.max_position_embeddings,
        rms_norm_eps: hf.rms_norm_eps.or(hf.layer_norm_eps).unwrap_or(1e-5),
        rope_theta: hf.rope_theta.unwrap_or(10000.0),
        head_dim,
        attention_bias: hf.attention_bias.unwrap_or(false),
        mlp_bias: hf.mlp_bias.unwrap_or(false),
        sliding_window: hf.sliding_window,
        tie_word_embeddings: hf.tie_word_embeddings.unwrap_or(false),
        bos_token_id: hf.bos_token_id.unwrap_or(1),
        eos_token_id: hf.eos_token_id.unwrap_or(2),
        pad_token_id: hf.pad_token_id,
    })
}

/// HuggingFace model loader
pub struct HuggingFaceLoader {
    /// Model path
    path: PathBuf,

    /// Model configuration
    config: ModelConfig,

    /// Weight files
    weight_files: Vec<PathBuf>,
}

impl HuggingFaceLoader {
    /// Create a new HuggingFace loader
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let config = load_config(&path)?;

        // Find weight files
        let mut weight_files = Vec::new();

        // Check for safetensors
        let safetensors_index = path.join("model.safetensors.index.json");
        if safetensors_index.exists() {
            // Load sharded safetensors
            let index_str = fs::read_to_string(&safetensors_index)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;

            if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();
                for filename in weight_map.values() {
                    if let Some(f) = filename.as_str() {
                        files.insert(f.to_string());
                    }
                }
                for file in files {
                    weight_files.push(path.join(file));
                }
            }
        } else {
            // Check for single safetensors file
            let single_safetensors = path.join("model.safetensors");
            if single_safetensors.exists() {
                weight_files.push(single_safetensors);
            } else {
                // Check for pytorch weights
                for entry in fs::read_dir(&path)? {
                    let entry = entry?;
                    let entry_path = entry.path();
                    if entry_path.extension().map_or(false, |e| e == "bin") {
                        weight_files.push(entry_path);
                    }
                }
            }
        }

        if weight_files.is_empty() {
            return Err(Error::ModelLoad("No weight files found".to_string()));
        }

        Ok(Self {
            path,
            config,
            weight_files,
        })
    }

    /// Get the model path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the weight files
    pub fn weight_files(&self) -> &[PathBuf] {
        &self.weight_files
    }
}

impl WeightLoader for HuggingFaceLoader {
    fn load_weights(&self) -> Result<HashMap<String, Tensor>> {
        // In a real implementation, this would load all weights from the files
        Err(Error::not_implemented("Full weight loading"))
    }

    fn load_weight(&self, _name: &str) -> Result<Tensor> {
        Err(Error::not_implemented("Single weight loading"))
    }

    fn weight_names(&self) -> Vec<String> {
        // Would parse the index file or safetensors metadata
        Vec::new()
    }

    fn has_weight(&self, _name: &str) -> bool {
        false
    }

    fn config(&self) -> Result<ModelConfig> {
        Ok(self.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hf_config() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        }"#;

        let hf_config: HfConfig = serde_json::from_str(json).unwrap();
        let config = convert_hf_config(hf_config).unwrap();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.num_attention_heads, 32);
    }

    #[test]
    fn test_detect_architecture() {
        // Test with architectures field
        let json = r#"{"architectures": ["LlamaForCausalLM"], "hidden_size": 4096}"#;
        let hf: HfConfig = serde_json::from_str(json).unwrap();

        // Check that architecture detection works
        let arch_str = hf.architectures.first().unwrap().to_lowercase();
        assert!(arch_str.contains("llama"));
    }
}
