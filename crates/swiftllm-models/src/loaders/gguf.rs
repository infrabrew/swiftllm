//! GGUF Model Loader
//!
//! Loads models from GGUF format (used by llama.cpp).

use super::WeightLoader;
use crate::ModelConfig;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use swiftllm_core::config::{DataType, ModelArchitecture};
use swiftllm_core::error::{Error, Result};
use swiftllm_core::tensor::Tensor;

/// GGUF magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// GGUF version
const GGUF_VERSION: u32 = 3;

/// GGUF data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Bf16 = 29,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            24 => Some(Self::I8),
            28 => Some(Self::F64),
            29 => Some(Self::Bf16),
            _ => None,
        }
    }

    fn to_dtype(&self) -> DataType {
        match self {
            Self::F32 => DataType::Float32,
            Self::F16 => DataType::Float16,
            Self::Bf16 => DataType::BFloat16,
            Self::I8 => DataType::Int8,
            // Quantized types - map to closest
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::Iq4Nl | Self::Iq4Xs => DataType::Int4,
            _ => DataType::Float16, // Default for other quantized types
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 256,
            Self::Q5K => 256,
            Self::Q6K => 256,
            Self::Q8K => 256,
            _ => 32, // Default block size
        }
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Tensor info from GGUF file
#[derive(Debug, Clone)]
struct GgufTensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    dtype: GgufType,
    offset: u64,
}

/// GGUF file loader
pub struct GgufLoader {
    /// File path
    path: PathBuf,

    /// Metadata
    metadata: HashMap<String, GgufValue>,

    /// Tensor infos
    tensors: HashMap<String, GgufTensorInfo>,

    /// Data offset in file
    data_offset: u64,
}

impl GgufLoader {
    /// Create a new GGUF loader
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| {
            Error::ModelLoad(format!("Failed to open {}: {}", path.display(), e))
        })?;
        let mut reader = BufReader::new(file);

        // Read magic
        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(Error::ModelLoad(format!(
                "Invalid GGUF magic: {:08x}",
                magic
            )));
        }

        // Read version
        let version = read_u32(&mut reader)?;
        if version > GGUF_VERSION {
            return Err(Error::ModelLoad(format!(
                "Unsupported GGUF version: {}",
                version
            )));
        }

        // Read counts
        let n_tensors = read_u64(&mut reader)?;
        let n_kv = read_u64(&mut reader)?;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut reader)?;
            let value = read_value(&mut reader)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = read_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)?;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut reader)?);
            }

            let dtype_id = read_u32(&mut reader)?;
            let dtype = GgufType::from_u32(dtype_id).ok_or_else(|| {
                Error::ModelLoad(format!("Unknown GGUF type: {}", dtype_id))
            })?;

            let offset = read_u64(&mut reader)?;

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    n_dims,
                    dims,
                    dtype,
                    offset,
                },
            );
        }

        // Calculate data offset (aligned to 32 bytes)
        let current_pos = reader.stream_position()?;
        let data_offset = (current_pos + 31) & !31;

        Ok(Self {
            path,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get architecture from metadata
    pub fn architecture(&self) -> Option<ModelArchitecture> {
        let arch = self.get_metadata("general.architecture")?;
        match arch.as_str()? {
            "llama" => Some(ModelArchitecture::Llama),
            "mistral" => Some(ModelArchitecture::Mistral),
            "qwen" | "qwen2" => Some(ModelArchitecture::Qwen),
            "phi" | "phi3" => Some(ModelArchitecture::Phi),
            _ => None,
        }
    }
}

impl WeightLoader for GgufLoader {
    fn load_weights(&self) -> Result<HashMap<String, Tensor>> {
        let file = File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;

        let mut weights = HashMap::new();

        for (name, info) in &self.tensors {
            let offset = (self.data_offset + info.offset) as usize;
            let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
            let numel: usize = shape.iter().product();

            // Calculate byte size based on dtype
            let dtype = info.dtype.to_dtype();
            let bytes_per_elem = dtype.size_bytes();
            let byte_len = numel * bytes_per_elem;

            if offset + byte_len <= mmap.len() {
                let data = mmap[offset..offset + byte_len].to_vec();
                let tensor = Tensor::from_data(data, shape, dtype)?;
                weights.insert(name.clone(), tensor);
            }
        }

        Ok(weights)
    }

    fn load_weight(&self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::ModelLoad(format!("Weight not found: {}", name))
        })?;

        let file = File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;

        let offset = (self.data_offset + info.offset) as usize;
        let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let numel: usize = shape.iter().product();
        let dtype = info.dtype.to_dtype();
        let byte_len = numel * dtype.size_bytes();

        let data = mmap[offset..offset + byte_len].to_vec();
        Tensor::from_data(data, shape, dtype)
    }

    fn weight_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn has_weight(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn config(&self) -> Result<ModelConfig> {
        let arch = self.architecture().unwrap_or(ModelArchitecture::Llama);

        // Extract config from metadata
        let hidden_size = self
            .get_metadata("llama.embedding_length")
            .or_else(|| self.get_metadata("phi.embedding_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        let num_layers = self
            .get_metadata("llama.block_count")
            .or_else(|| self.get_metadata("phi.block_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let num_heads = self
            .get_metadata("llama.attention.head_count")
            .or_else(|| self.get_metadata("phi.attention.head_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let num_kv_heads = self
            .get_metadata("llama.attention.head_count_kv")
            .or_else(|| self.get_metadata("phi.attention.head_count_kv"))
            .and_then(|v| v.as_u32())
            .unwrap_or(num_heads as u32) as usize;

        let vocab_size = self
            .get_metadata("llama.vocab_size")
            .or_else(|| self.get_metadata("general.vocab_size"))
            .and_then(|v| v.as_u32())
            .unwrap_or(32000) as usize;

        let max_seq_len = self
            .get_metadata("llama.context_length")
            .or_else(|| self.get_metadata("phi.context_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        Ok(ModelConfig {
            architecture: arch,
            hidden_size,
            intermediate_size: hidden_size * 4, // Approximation
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            num_hidden_layers: num_layers,
            vocab_size,
            max_position_embeddings: max_seq_len,
            head_dim: hidden_size / num_heads,
            ..Default::default()
        })
    }
}

// Helper functions for reading GGUF data

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| Error::ModelLoad(format!("Invalid string: {}", e)))
}

fn read_value<R: Read>(reader: &mut R) -> Result<GgufValue> {
    let value_type = read_u32(reader)?;

    match value_type {
        0 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Uint8(buf[0]))
        }
        1 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Int8(buf[0] as i8))
        }
        2 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
        }
        3 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
        }
        4 => Ok(GgufValue::Uint32(read_u32(reader)?)),
        5 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
        }
        6 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
        }
        7 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Bool(buf[0] != 0))
        }
        8 => Ok(GgufValue::String(read_string(reader)?)),
        9 => {
            // Array
            let arr_type = read_u32(reader)?;
            let arr_len = read_u64(reader)? as usize;
            let mut arr = Vec::with_capacity(arr_len);
            for _ in 0..arr_len {
                // Re-read value with stored type
                // This is simplified - full implementation would need proper type handling
                arr.push(GgufValue::Uint32(read_u32(reader)?));
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(reader)?)),
        11 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
        }
        12 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
        }
        _ => Err(Error::ModelLoad(format!(
            "Unknown GGUF value type: {}",
            value_type
        ))),
    }
}
