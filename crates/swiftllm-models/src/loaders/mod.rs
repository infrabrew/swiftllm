//! Model Loaders
//!
//! This module provides loaders for various model formats.

pub mod gguf;
pub mod huggingface;
pub mod safetensors;

pub use huggingface::{HuggingFaceLoader, load_config, detect_architecture};

use crate::ModelConfig;
use swiftllm_core::error::Result;
use swiftllm_core::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Weight loader trait
pub trait WeightLoader: Send + Sync {
    /// Load all weights
    fn load_weights(&self) -> Result<HashMap<String, Tensor>>;

    /// Load a specific weight by name
    fn load_weight(&self, name: &str) -> Result<Tensor>;

    /// Get all weight names
    fn weight_names(&self) -> Vec<String>;

    /// Check if a weight exists
    fn has_weight(&self, name: &str) -> bool;

    /// Get the model configuration
    fn config(&self) -> Result<ModelConfig>;
}

/// Model format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// HuggingFace Transformers format
    HuggingFace,
    /// SafeTensors format
    SafeTensors,
    /// GGUF format (llama.cpp)
    Gguf,
    /// GGML format (legacy)
    Ggml,
    /// PyTorch checkpoint
    PyTorch,
}

/// Detect model format from path
pub fn detect_format(path: impl AsRef<Path>) -> Result<ModelFormat> {
    let path = path.as_ref();

    if path.is_dir() {
        // Check for HuggingFace format
        if path.join("config.json").exists() {
            if path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
            {
                return Ok(ModelFormat::SafeTensors);
            }
            return Ok(ModelFormat::HuggingFace);
        }
    } else if path.is_file() {
        let extension = path.extension().and_then(|e| e.to_str());
        match extension {
            Some("safetensors") => return Ok(ModelFormat::SafeTensors),
            Some("gguf") => return Ok(ModelFormat::Gguf),
            Some("ggml" | "bin") => {
                // Could be GGML or PyTorch, check magic bytes
                return Ok(ModelFormat::Ggml);
            }
            Some("pt" | "pth") => return Ok(ModelFormat::PyTorch),
            _ => {}
        }
    }

    Err(swiftllm_core::error::Error::ModelNotFound(
        path.display().to_string(),
    ))
}

/// Weight name mapping utilities
pub struct WeightMapper {
    /// Mapping from standard names to model-specific names
    mappings: HashMap<String, String>,
}

impl WeightMapper {
    /// Create a new weight mapper for a specific architecture
    pub fn new(architecture: swiftllm_core::config::ModelArchitecture) -> Self {
        let mappings = match architecture {
            swiftllm_core::config::ModelArchitecture::Llama => Self::llama_mappings(),
            swiftllm_core::config::ModelArchitecture::Mistral => Self::mistral_mappings(),
            swiftllm_core::config::ModelArchitecture::Qwen |
            swiftllm_core::config::ModelArchitecture::Qwen2 => Self::qwen_mappings(),
            swiftllm_core::config::ModelArchitecture::Phi |
            swiftllm_core::config::ModelArchitecture::Phi3 => Self::phi_mappings(),
            _ => HashMap::new(),
        };

        Self { mappings }
    }

    fn llama_mappings() -> HashMap<String, String> {
        // LLaMA weight names are already standard
        HashMap::new()
    }

    fn mistral_mappings() -> HashMap<String, String> {
        // Mistral uses same naming as LLaMA
        HashMap::new()
    }

    fn qwen_mappings() -> HashMap<String, String> {
        let mut m = HashMap::new();
        // Qwen has slightly different naming
        m.insert("transformer.wte".into(), "model.embed_tokens".into());
        m.insert("transformer.ln_f".into(), "model.norm".into());
        m
    }

    fn phi_mappings() -> HashMap<String, String> {
        let mut m = HashMap::new();
        // Phi uses different naming conventions
        m.insert("transformer.embd".into(), "model.embed_tokens".into());
        m.insert("transformer.h".into(), "model.layers".into());
        m.insert("lm_head".into(), "lm_head".into());
        m
    }

    /// Map a weight name
    pub fn map(&self, name: &str) -> String {
        self.mappings.get(name).cloned().unwrap_or_else(|| name.to_string())
    }
}
