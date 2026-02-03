//! SafeTensors Loader
//!
//! Loads weights from SafeTensors format files.

use super::WeightLoader;
use crate::ModelConfig;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use swiftllm_core::config::DataType;
use swiftllm_core::error::{Error, Result};
use swiftllm_core::tensor::{Device, Tensor};

/// SafeTensors file loader
pub struct SafeTensorsLoader {
    /// File paths
    files: Vec<PathBuf>,

    /// Weight metadata (name -> file index, offset, shape, dtype)
    metadata: HashMap<String, WeightMetadata>,

    /// Memory-mapped files
    mmaps: Vec<Mmap>,
}

/// Metadata for a single weight
#[derive(Debug, Clone)]
struct WeightMetadata {
    /// File index
    file_idx: usize,
    /// Byte offset in file
    offset: usize,
    /// Byte length
    length: usize,
    /// Shape
    shape: Vec<usize>,
    /// Data type
    dtype: DataType,
}

impl SafeTensorsLoader {
    /// Create a new SafeTensors loader
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut metadata = HashMap::new();

        for (file_idx, path) in paths.iter().enumerate() {
            let file = File::open(path).map_err(|e| {
                Error::ModelLoad(format!("Failed to open {}: {}", path.display(), e))
            })?;

            // Safety: We're memory-mapping for read-only access
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                Error::ModelLoad(format!("Failed to mmap {}: {}", path.display(), e))
            })?;

            // Parse SafeTensors header
            if mmap.len() < 8 {
                return Err(Error::ModelLoad("File too small".to_string()));
            }

            // First 8 bytes are header length (little-endian u64)
            let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;

            if mmap.len() < 8 + header_len {
                return Err(Error::ModelLoad("Invalid header length".to_string()));
            }

            // Parse JSON header
            let header_json = std::str::from_utf8(&mmap[8..8 + header_len]).map_err(|e| {
                Error::ModelLoad(format!("Invalid header encoding: {}", e))
            })?;

            let header: serde_json::Value = serde_json::from_str(header_json).map_err(|e| {
                Error::ModelLoad(format!("Invalid header JSON: {}", e))
            })?;

            // Parse tensor metadata
            if let Some(obj) = header.as_object() {
                for (name, info) in obj {
                    if name == "__metadata__" {
                        continue;
                    }

                    if let Some(tensor_info) = info.as_object() {
                        let dtype = tensor_info
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .map(|s| parse_dtype(s))
                            .unwrap_or(DataType::Float16);

                        let shape: Vec<usize> = tensor_info
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                                    .collect()
                            })
                            .unwrap_or_default();

                        let offsets = tensor_info
                            .get("data_offsets")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();

                        if offsets.len() == 2 {
                            let data_offset = 8 + header_len;
                            metadata.insert(
                                name.clone(),
                                WeightMetadata {
                                    file_idx,
                                    offset: data_offset + offsets[0],
                                    length: offsets[1] - offsets[0],
                                    shape,
                                    dtype,
                                },
                            );
                        }
                    }
                }
            }

            mmaps.push(mmap);
        }

        Ok(Self {
            files: paths,
            metadata,
            mmaps,
        })
    }

    /// Load a single safetensors file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(vec![path.as_ref().to_path_buf()])
    }

    /// Get tensor data bytes
    pub fn get_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let meta = self.metadata.get(name)?;
        let mmap = self.mmaps.get(meta.file_idx)?;
        Some(&mmap[meta.offset..meta.offset + meta.length])
    }

    /// Get tensor shape
    pub fn get_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.metadata.get(name).map(|m| m.shape.as_slice())
    }

    /// Get tensor dtype
    pub fn get_tensor_dtype(&self, name: &str) -> Option<DataType> {
        self.metadata.get(name).map(|m| m.dtype)
    }
}

impl WeightLoader for SafeTensorsLoader {
    fn load_weights(&self) -> Result<HashMap<String, Tensor>> {
        let mut weights = HashMap::new();

        for (name, meta) in &self.metadata {
            if let Some(bytes) = self.get_tensor_bytes(name) {
                let tensor = Tensor::from_data(
                    bytes.to_vec(),
                    meta.shape.clone(),
                    meta.dtype,
                )?;
                weights.insert(name.clone(), tensor);
            }
        }

        Ok(weights)
    }

    fn load_weight(&self, name: &str) -> Result<Tensor> {
        let meta = self.metadata.get(name).ok_or_else(|| {
            Error::ModelLoad(format!("Weight not found: {}", name))
        })?;

        let bytes = self.get_tensor_bytes(name).ok_or_else(|| {
            Error::ModelLoad(format!("Failed to read weight: {}", name))
        })?;

        Tensor::from_data(bytes.to_vec(), meta.shape.clone(), meta.dtype)
    }

    fn weight_names(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }

    fn has_weight(&self, name: &str) -> bool {
        self.metadata.contains_key(name)
    }

    fn config(&self) -> Result<ModelConfig> {
        Err(Error::not_implemented(
            "Config not available from SafeTensors"
        ))
    }
}

/// Parse dtype string to DataType
fn parse_dtype(s: &str) -> DataType {
    match s.to_uppercase().as_str() {
        "F32" | "FLOAT32" => DataType::Float32,
        "F16" | "FLOAT16" => DataType::Float16,
        "BF16" | "BFLOAT16" => DataType::BFloat16,
        "I8" | "INT8" => DataType::Int8,
        "I4" | "INT4" => DataType::Int4,
        "F8_E4M3" => DataType::Float8E4M3,
        "F8_E5M2" => DataType::Float8E5M2,
        _ => DataType::Float16, // Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert_eq!(parse_dtype("F32"), DataType::Float32);
        assert_eq!(parse_dtype("F16"), DataType::Float16);
        assert_eq!(parse_dtype("BF16"), DataType::BFloat16);
        assert_eq!(parse_dtype("I8"), DataType::Int8);
    }
}
