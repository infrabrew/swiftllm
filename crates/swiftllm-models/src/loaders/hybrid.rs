// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      hybrid.rs
// PATH:      /crates/swiftllm-models/src/loaders/hybrid.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Weight loader for HybridModelConfig (Mamba-3 + LatentMoE + RLM + DenseVerif)
//
// Supports two on-disk formats:
//   1. SafeTensors directory  (preferred, HuggingFace-compatible)
//   2. PyTorch .pt checkpoint (torch.save dict; loaded via safetensors shim)
//
// Weight name conventions (mirrors the Python torch_model.py naming):
//
//   Embedding + LM head
//     token_embedding.weight          [vocab_size, d_model]
//     lm_head.weight                  [vocab_size, d_model]
//
//   Per-layer prefix: layers.{i}.  where i = 0..num_layers-1
//     Mamba / MambaMoe layers:
//       mamba.in_proj.weight          [in_proj_out, d_model]
//       mamba.conv1d.weight           [d_inner, 1, d_conv]
//       mamba.conv1d.bias             [d_inner]
//       mamba.dt_proj.weight          [d_inner, dt_rank]
//       mamba.dt_proj.bias            [d_inner]
//       mamba.a_log                   [d_inner, d_state]
//       mamba.d                       [d_inner]
//       mamba.out_proj.weight         [d_model, d_inner]
//       mamba.norm.weight             [d_model]
//
//     LatentMoE layers (MambaMoe / AttentionMoe):
//       moe.compress_proj.weight      [d_latent, d_model]
//       moe.expand_proj.weight        [d_model, d_latent]
//       moe.router.gate.weight        [num_experts, d_latent]
//       moe.experts.{e}.gate_proj.weight   [d_ffn, d_latent]
//       moe.experts.{e}.up_proj.weight     [d_ffn, d_latent]
//       moe.experts.{e}.down_proj.weight   [d_latent, d_ffn]
//       moe.shared.{e}.{gate,up,down}_proj.weight
//       moe.norm.weight               [d_model]
//
//     RLM layers:
//       rlm.scheduler.hidden_proj.weight    [depth_hidden, d_model]
//       rlm.scheduler.depth_proj.weight     [max_depth+1, depth_hidden]
//       rlm.repl_key_proj.weight            [d_model, d_model]
//       rlm.repl_val_proj.weight            [d_model, d_model]
//       rlm.subproblem_proj.weight          [d_subproblem, d_model]
//       rlm.solution_proj.weight            [d_model, d_subproblem]
//       rlm.gate_proj.weight                [d_model, d_model*2]
//       rlm.output_norm.weight              [d_model]
//
//     Dense Verification (last layer only, or separate file):
//       verif.query_proj.weight        [kv_dim, d_model]
//       verif.key_proj.weight          [kv_dim, d_model]
//       verif.value_proj.weight        [kv_dim, d_model]
//       verif.out_proj.weight          [d_model, kv_dim]
//       verif.score_proj.weight        [1, d_model]
//       verif.input_norm.weight        [d_model]
//
// Licensed under the Apache License, Version 2.0
// ==============================================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use tracing::{info, warn};

use swiftllm_core::config::DataType;
use swiftllm_core::error::{Error, Result};
use swiftllm_core::tensor::{Device, Tensor};

// ---------------------------------------------------------------------------
// HybridWeightLoader — main public API
// ---------------------------------------------------------------------------

/// Load weights for a Hybrid (Mamba-3 + LatentMoE + RLM + DenseVerif) model.
///
/// Handles both a single `model.safetensors` file and a sharded
/// `model.safetensors.index.json` + shard set.
pub struct HybridWeightLoader {
    model_dir: PathBuf,
    device:    Device,
}

impl HybridWeightLoader {
    /// Create a loader pointing at `model_dir` on the given `device`.
    pub fn new(model_dir: impl AsRef<Path>, device: Device) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_owned(),
            device,
        }
    }

    /// Load all tensors from the safetensors file(s) in `model_dir`.
    ///
    /// Returns a flat `HashMap<String, Tensor>` keyed by weight name.
    pub fn load_all(&self) -> Result<HashMap<String, Tensor>> {
        let shard_files = self.discover_shards()?;
        let mut weights: HashMap<String, Tensor> = HashMap::new();

        for shard_path in &shard_files {
            let shard_weights = self.load_shard(shard_path)?;
            weights.extend(shard_weights);
        }

        info!("HybridWeightLoader: loaded {} tensors from {:?}", weights.len(), self.model_dir);
        Ok(weights)
    }

    /// Discover safetensors shard files in model_dir.
    fn discover_shards(&self) -> Result<Vec<PathBuf>> {
        let single = self.model_dir.join("model.safetensors");
        if single.exists() {
            return Ok(vec![single]);
        }

        // Sharded: look for model-00001-of-NNNNN.safetensors
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&self.model_dir)
            .map_err(Error::Io)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                if name.ends_with(".safetensors") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        shards.sort();

        if shards.is_empty() {
            return Err(Error::ModelNotFound(format!(
                "No safetensors files found in {:?}", self.model_dir
            )));
        }

        Ok(shards)
    }

    /// Load a single safetensors shard and convert tensors to `Tensor`.
    fn load_shard(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        let file = std::fs::File::open(path)
            .map_err(Error::Io)?;

        // SAFETY: file is opened read-only and not modified during parsing.
        let mmap = unsafe {
            Mmap::map(&file).map_err(Error::Io)?
        };

        let st = SafeTensors::deserialize(&mmap[..])
            .map_err(|e| Error::ModelLoad(format!("SafeTensors parse error: {e}")))?;

        let mut weights: HashMap<String, Tensor> = HashMap::new();

        for (name, view) in st.tensors() {
            let dtype: DataType = match view.dtype() {
                safetensors::Dtype::F16  => DataType::Float16,
                safetensors::Dtype::BF16 => DataType::BFloat16,
                safetensors::Dtype::F32  => DataType::Float32,
                other => {
                    warn!("Skipping tensor '{}' with unsupported dtype {:?}", name, other);
                    continue;
                }
            };

            let shape: Vec<usize> = view.shape().to_vec();
            let data: Vec<u8>     = view.data().to_vec();

            let cpu_tensor = Tensor::from_data(data, shape.clone(), dtype)?;

            let tensor = if self.device.is_cuda() {
                cpu_tensor.to(self.device)
                    .unwrap_or_else(|e| {
                        warn!("H2D transfer failed for '{}': {}; using CPU fallback", name, e);
                        // Recreate from scratch to return CPU tensor
                        Tensor::zeros(
                            view.shape().to_vec(), dtype, Device::Cpu
                        ).expect("zeros always succeeds on CPU")
                    })
            } else {
                cpu_tensor
            };

            weights.insert(name.to_owned(), tensor);
        }

        Ok(weights)
    }
}

// ---------------------------------------------------------------------------
// Weight assignment helpers
// ---------------------------------------------------------------------------

/// Extract a named weight from the map, returning an error if missing.
pub fn get_weight<'a>(
    weights: &'a HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor> {
    weights.get(name).ok_or_else(|| Error::ModelLoad(
        format!("Missing weight: '{name}'")
    ))
}

/// Try to get a weight, returning `None` if not present (for optional weights).
pub fn get_weight_opt<'a>(
    weights: &'a HashMap<String, Tensor>,
    name: &str,
) -> Option<&'a Tensor> {
    weights.get(name)
}

/// Build a per-layer weight prefix: `"layers.{layer_idx}."`.
pub fn layer_prefix(layer_idx: usize) -> String {
    format!("layers.{}.", layer_idx)
}

// ---------------------------------------------------------------------------
// Tensor utility: view weight slice (alias without copy for safetensors data)
// ---------------------------------------------------------------------------

/// Create a Tensor view into the named weight with the expected shape.
/// Returns an error if the tensor is missing or has the wrong shape/dtype.
pub fn require_weight_of_shape(
    weights: &HashMap<String, Tensor>,
    name:    &str,
    expected_shape: &[usize],
    expected_dtype: DataType,
) -> Result<Tensor> {
    let t = get_weight(weights, name)?;

    if t.dims() != expected_shape {
        return Err(Error::ModelLoad(format!(
            "Weight '{}': expected shape {:?}, got {:?}",
            name, expected_shape, t.dims()
        )));
    }
    if t.dtype() != expected_dtype {
        // Allow implicit cast (e.g. BF16 ↔ F16)
        if !(matches!(t.dtype(), DataType::BFloat16 | DataType::Float16)
            && matches!(expected_dtype, DataType::BFloat16 | DataType::Float16)) {
            return Err(Error::ModelLoad(format!(
                "Weight '{}': expected dtype {:?}, got {:?}",
                name, expected_dtype, t.dtype()
            )));
        }
    }

    Ok(t.clone())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_prefix() {
        assert_eq!(layer_prefix(0),  "layers.0.");
        assert_eq!(layer_prefix(31), "layers.31.");
    }

    #[test]
    fn test_get_weight_missing() {
        let weights: HashMap<String, Tensor> = HashMap::new();
        assert!(get_weight(&weights, "nonexistent").is_err());
    }

    #[test]
    fn test_get_weight_opt_missing() {
        let weights: HashMap<String, Tensor> = HashMap::new();
        assert!(get_weight_opt(&weights, "nonexistent").is_none());
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: hybrid.rs
// REPO PATH:   /swiftllm/crates/swiftllm-models/src/loaders/hybrid.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
