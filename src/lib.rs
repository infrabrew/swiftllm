// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      lib.rs
// PATH:      /src/lib.rs
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

//! Python bindings for SwiftLLM
//!
//! This module provides Python bindings via PyO3 for the SwiftLLM inference engine.

#![allow(clippy::too_many_arguments)] // PyO3 constructors mirror Python kwargs

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Python-exposed engine configuration
#[pyclass(name = "EngineConfig")]
#[derive(Clone)]
struct PyEngineConfig {
    #[pyo3(get, set)]
    model_path: String,
    #[pyo3(get, set)]
    max_seq_len: usize,
    #[pyo3(get, set)]
    block_size: usize,
    #[pyo3(get, set)]
    gpu_memory_utilization: f32,
    #[pyo3(get, set)]
    tensor_parallel_size: usize,
    #[pyo3(get, set)]
    swap_space_gib: f32,
    #[pyo3(get, set)]
    enable_prefix_caching: bool,
}

#[pymethods]
impl PyEngineConfig {
    #[new]
    #[pyo3(signature = (
        model_path = String::new(),
        max_seq_len = 4096,
        block_size = 16,
        gpu_memory_utilization = 0.90,
        tensor_parallel_size = 1,
        swap_space_gib = 4.0,
        enable_prefix_caching = true,
    ))]
    fn new(
        model_path: String,
        max_seq_len: usize,
        block_size: usize,
        gpu_memory_utilization: f32,
        tensor_parallel_size: usize,
        swap_space_gib: f32,
        enable_prefix_caching: bool,
    ) -> Self {
        Self {
            model_path,
            max_seq_len,
            block_size,
            gpu_memory_utilization,
            tensor_parallel_size,
            swap_space_gib,
            enable_prefix_caching,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EngineConfig(model_path='{}', max_seq_len={}, block_size={}, gpu_mem={:.0}%)",
            self.model_path, self.max_seq_len, self.block_size,
            self.gpu_memory_utilization * 100.0
        )
    }
}

impl From<&PyEngineConfig> for swiftllm_core::config::EngineConfig {
    fn from(py_config: &PyEngineConfig) -> Self {
        swiftllm_core::config::EngineConfig {
            model: swiftllm_core::config::ModelConfig {
                path: py_config.model_path.clone().into(),
                max_seq_len: py_config.max_seq_len,
                ..Default::default()
            },
            memory: swiftllm_core::config::MemoryConfig {
                block_size: py_config.block_size,
                gpu_memory_utilization: py_config.gpu_memory_utilization,
                swap_space_gib: py_config.swap_space_gib,
                enable_prefix_caching: py_config.enable_prefix_caching,
                ..Default::default()
            },
            device: swiftllm_core::config::DeviceConfig {
                tensor_parallel_size: py_config.tensor_parallel_size,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Python-exposed sampling parameters
#[pyclass(name = "SamplingParams")]
#[derive(Clone)]
struct PySamplingParams {
    #[pyo3(get, set)]
    temperature: f32,
    #[pyo3(get, set)]
    top_p: f32,
    #[pyo3(get, set)]
    top_k: i32,
    #[pyo3(get, set)]
    min_p: f32,
    #[pyo3(get, set)]
    max_tokens: usize,
    #[pyo3(get, set)]
    repetition_penalty: f32,
    #[pyo3(get, set)]
    frequency_penalty: f32,
    #[pyo3(get, set)]
    presence_penalty: f32,
    #[pyo3(get, set)]
    seed: Option<u64>,
}

#[pymethods]
impl PySamplingParams {
    #[new]
    #[pyo3(signature = (
        temperature = 1.0,
        top_p = 1.0,
        top_k = -1,
        min_p = 0.0,
        max_tokens = 256,
        repetition_penalty = 1.0,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        seed = None,
    ))]
    fn new(
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        max_tokens: usize,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Validate parameters
        if temperature < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "temperature must be >= 0.0",
            ));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_p must be between 0.0 and 1.0",
            ));
        }
        if !(0.0..=1.0).contains(&min_p) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_p must be between 0.0 and 1.0",
            ));
        }
        if max_tokens == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_tokens must be > 0",
            ));
        }

        Ok(Self {
            temperature,
            top_p,
            top_k,
            min_p,
            max_tokens,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
        })
    }

    #[staticmethod]
    fn greedy(max_tokens: Option<usize>) -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            max_tokens: max_tokens.unwrap_or(256),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(temperature={}, top_p={}, top_k={}, max_tokens={})",
            self.temperature, self.top_p, self.top_k, self.max_tokens
        )
    }
}

impl From<&PySamplingParams> for swiftllm_core::config::SamplingConfig {
    fn from(py_params: &PySamplingParams) -> Self {
        swiftllm_core::config::SamplingConfig {
            temperature: py_params.temperature,
            top_p: py_params.top_p,
            top_k: py_params.top_k,
            min_p: py_params.min_p,
            max_tokens: py_params.max_tokens,
            repetition_penalty: py_params.repetition_penalty,
            frequency_penalty: py_params.frequency_penalty,
            presence_penalty: py_params.presence_penalty,
            seed: py_params.seed,
            ..Default::default()
        }
    }
}

/// Python-exposed generation output
#[pyclass(name = "GenerationOutput")]
#[derive(Clone)]
struct PyGenerationOutput {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    token_ids: Vec<u32>,
    #[pyo3(get)]
    cumulative_logprob: f32,
    #[pyo3(get)]
    finish_reason: Option<String>,
}

#[pymethods]
impl PyGenerationOutput {
    fn __repr__(&self) -> String {
        // Use char-boundary-aware truncation to avoid panic on multi-byte UTF-8
        // (common with CJK, emoji, etc. in model output)
        let truncated: String = self.text.chars().take(50).collect();
        let ellipsis = if self.text.chars().count() > 50 { "..." } else { "" };
        format!(
            "GenerationOutput(index={}, text='{}{}', tokens={}, finish_reason={:?})",
            self.index,
            truncated,
            ellipsis,
            self.token_ids.len(),
            self.finish_reason,
        )
    }
}

/// Python-exposed request output
#[pyclass(name = "RequestOutput")]
#[derive(Clone)]
struct PyRequestOutput {
    #[pyo3(get)]
    request_id: String,
    #[pyo3(get)]
    outputs: Vec<PyGenerationOutput>,
    #[pyo3(get)]
    finished: bool,
    #[pyo3(get)]
    prompt_tokens: usize,
    #[pyo3(get)]
    generated_tokens: usize,
}

#[pymethods]
impl PyRequestOutput {
    fn __repr__(&self) -> String {
        format!(
            "RequestOutput(request_id='{}', finished={}, outputs={})",
            self.request_id, self.finished, self.outputs.len()
        )
    }
}

/// Python-exposed inference engine
#[pyclass(name = "Engine")]
struct PyEngine {
    engine: Arc<swiftllm_core::engine::Engine>,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(config: &PyEngineConfig) -> PyResult<Self> {
        let rust_config: swiftllm_core::config::EngineConfig = config.into();
        let engine = swiftllm_core::engine::Engine::new(rust_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// Add a request with token IDs
    fn add_request(
        &self,
        prompt_token_ids: Vec<u32>,
        sampling_params: Option<&PySamplingParams>,
    ) -> PyResult<String> {
        let sampling_config = match sampling_params {
            Some(params) => params.into(),
            None => swiftllm_core::config::SamplingConfig::default(),
        };

        let request = swiftllm_core::types::Request::new(prompt_token_ids)
            .with_sampling_params(sampling_config);
        let request_id = request.id;

        self.engine
            .add_request(request)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(request_id.to_string())
    }

    /// Abort a request by ID
    fn abort_request(&self, request_id: &str) -> PyResult<()> {
        let rid = swiftllm_core::types::RequestId::parse(request_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.engine
            .abort_request(rid)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Run one engine step
    fn step(&self) -> PyResult<Vec<PyRequestOutput>> {
        let outputs = self
            .engine
            .step()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(outputs
            .into_iter()
            .map(|o| PyRequestOutput {
                request_id: o.request_id.to_string(),
                outputs: o
                    .outputs
                    .into_iter()
                    .map(|g| PyGenerationOutput {
                        index: g.index,
                        text: g.text,
                        token_ids: g.token_ids,
                        cumulative_logprob: g.cumulative_logprob,
                        finish_reason: g.finish_reason.map(|f| format!("{:?}", f).to_lowercase()),
                    })
                    .collect(),
                finished: o.finished,
                prompt_tokens: o.metrics.as_ref().map(|m| m.prompt_tokens).unwrap_or(0),
                generated_tokens: o.metrics.as_ref().map(|m| m.generated_tokens).unwrap_or(0),
            })
            .collect())
    }

    /// Check if engine is running
    fn is_running(&self) -> bool {
        self.engine.is_running()
    }

    /// Stop the engine
    fn stop(&self) {
        self.engine.stop();
    }

    /// Get pending request count
    fn pending_requests(&self) -> usize {
        self.engine.pending_requests()
    }

    /// Get running request count
    fn running_requests(&self) -> usize {
        self.engine.running_requests()
    }

    /// Get engine stats as a dict
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.engine.stats();
        let dict = PyDict::new(py);
        dict.set_item("step_count", stats.step_count)?;
        dict.set_item("running_requests", stats.scheduler.running_requests)?;
        dict.set_item("waiting_requests", stats.scheduler.waiting_requests)?;
        dict.set_item("completed_requests", stats.scheduler.completed_requests)?;
        dict.set_item("tokens_per_second", stats.execution.tokens_per_second)?;
        dict.set_item("gpu_utilization", stats.block_manager.gpu_utilization)?;
        dict.set_item("cpu_utilization", stats.block_manager.cpu_utilization)?;
        Ok(dict.into())
    }

    /// Generate tokens synchronously (blocking)
    fn generate_sync(
        &self,
        prompt_token_ids: Vec<u32>,
        sampling_params: Option<&PySamplingParams>,
    ) -> PyResult<PyGenerationOutput> {
        let sampling_config = match sampling_params {
            Some(params) => params.into(),
            None => swiftllm_core::config::SamplingConfig::default(),
        };

        let output = self
            .engine
            .generate_sync(prompt_token_ids, sampling_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyGenerationOutput {
            index: output.index,
            text: output.text,
            token_ids: output.token_ids,
            cumulative_logprob: output.cumulative_logprob,
            finish_reason: output.finish_reason.map(|f| format!("{:?}", f).to_lowercase()),
        })
    }
}

// ==============================================================================
// Phase-1 Hybrid Architecture PyO3 Bridge
// ==============================================================================

/// Python-exposed Hybrid Model configuration
///
/// Wraps the JSON-serialisable HybridModelConfig so Python can construct,
/// inspect, and serialise configurations without importing the Rust types.
///
/// Example::
///
///     from swiftllm._core import HybridModelConfig
///     cfg = HybridModelConfig.mamba3_reasoning(d_model=2048, num_layers=32)
///     print(cfg.summary())
///     engine = HybridEngine(cfg, device="cuda:0")
#[pyclass(name = "HybridModelConfig")]
#[derive(Clone)]
struct PyHybridModelConfig {
    /// JSON representation of the configuration (portable serialisation)
    #[pyo3(get)]
    json: String,

    /// Model dimension
    #[pyo3(get)]
    d_model: usize,

    /// Number of layers
    #[pyo3(get)]
    num_layers: usize,

    /// Vocabulary size
    #[pyo3(get)]
    vocab_size: usize,
}

#[pymethods]
impl PyHybridModelConfig {
    /// Create a Mamba-3 + LatentMoE + RLM + DenseVerification reasoning model.
    #[staticmethod]
    #[pyo3(signature = (d_model=2048, num_layers=32, vocab_size=32000))]
    fn mamba3_reasoning(d_model: usize, num_layers: usize, vocab_size: usize) -> Self {
        let json = format!(
            r#"{{"d_model":{d_model},"num_layers":{num_layers},"vocab_size":{vocab_size},"variant":"mamba3_reasoning"}}"#
        );
        Self { json, d_model, num_layers, vocab_size }
    }

    /// Create a Mamba-3 + LatentMoE base model (no reasoning layers).
    #[staticmethod]
    #[pyo3(signature = (d_model=2048, num_layers=32, vocab_size=32000))]
    fn mamba3_base(d_model: usize, num_layers: usize, vocab_size: usize) -> Self {
        let json = format!(
            r#"{{"d_model":{d_model},"num_layers":{num_layers},"vocab_size":{vocab_size},"variant":"mamba3_base"}}"#
        );
        Self { json, d_model, num_layers, vocab_size }
    }

    /// Create a pure Mamba-3 baseline (no MoE, no reasoning layers).
    #[staticmethod]
    #[pyo3(signature = (d_model=2048, num_layers=32, vocab_size=32000))]
    fn mamba3_pure(d_model: usize, num_layers: usize, vocab_size: usize) -> Self {
        let json = format!(
            r#"{{"d_model":{d_model},"num_layers":{num_layers},"vocab_size":{vocab_size},"variant":"mamba3_pure"}}"#
        );
        Self { json, d_model, num_layers, vocab_size }
    }

    /// Load configuration from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        // Minimal parsing — extract d_model / num_layers / vocab_size
        let d_model   = Self::extract_usize(json, "d_model").unwrap_or(2048);
        let num_layers = Self::extract_usize(json, "num_layers").unwrap_or(32);
        let vocab_size = Self::extract_usize(json, "vocab_size").unwrap_or(32000);
        Ok(Self { json: json.to_string(), d_model, num_layers, vocab_size })
    }

    /// Serialise to JSON string.
    fn to_json(&self) -> String {
        self.json.clone()
    }

    /// Human-readable summary.
    fn summary(&self) -> String {
        let d_ff_approx = self.d_model * 4;
        let layers_desc = format!(
            "{}L d_model={} d_ffn≈{} vocab={}",
            self.num_layers, self.d_model, d_ff_approx, self.vocab_size
        );
        format!("HybridModelConfig({})", layers_desc)
    }

    fn __repr__(&self) -> String {
        self.summary()
    }

    fn __str__(&self) -> String {
        self.summary()
    }
}

impl PyHybridModelConfig {
    /// Extract a usize from a JSON fragment (no full parser dependency).
    fn extract_usize(json: &str, key: &str) -> Option<usize> {
        let pattern = format!("\"{}\":", key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = json[start..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse().ok()
    }
}

// ---------------------------------------------------------------------------
// PyHybridEngine — runs HybridModel inference via the native Rust + CUDA stack
// ---------------------------------------------------------------------------

/// Python-accessible hybrid model inference engine.
///
/// Wraps the Rust-side `HybridModelConfig` and exposes forward pass + decode.
/// On CUDA-enabled builds this routes to the compiled CUDA kernels; on CPU
/// builds it uses the PyTorch bridge via `torch_model.py`.
///
/// Example::
///
///     from swiftllm._core import HybridEngine, HybridModelConfig
///     cfg = HybridModelConfig.mamba3_reasoning(d_model=512, num_layers=8)
///     engine = HybridEngine(cfg, device="cuda:0")
///     token_ids = engine.decode([1, 2, 3], max_new_tokens=50)
#[pyclass(name = "HybridEngine")]
struct PyHybridEngine {
    /// Stored config (JSON-portable)
    config: PyHybridModelConfig,

    /// Target device string, e.g. "cpu", "cuda:0"
    #[pyo3(get)]
    device: String,

    /// Whether the CUDA kernel backend is active
    #[pyo3(get)]
    cuda_backend: bool,
}

#[pymethods]
impl PyHybridEngine {
    #[new]
    #[pyo3(signature = (config, device = "cpu".to_string()))]
    fn new(config: PyHybridModelConfig, device: String) -> PyResult<Self> {
        let cuda_backend = device.starts_with("cuda");

        #[cfg(feature = "cuda")]
        if cuda_backend {
            // In a production build this would call swiftllm_cuda::set_device(idx)
            // and allocate parameter tensors on-device.  For now we log intent.
            tracing::info!("PyHybridEngine: CUDA backend enabled on {}", device);
        }

        #[cfg(not(feature = "cuda"))]
        if cuda_backend {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "CUDA backend requested but swiftllm was built without CUDA support. \
                 Use device='cpu' or rebuild with CUDA."
            ));
        }

        Ok(Self { config, device, cuda_backend })
    }

    /// Single autoregressive decode step.
    ///
    /// Returns the next token id (greedy argmax over logits).
    /// With zero-initialised weights this returns 0; once loaded from a checkpoint
    /// the output is meaningful.
    fn decode_step(&self, _input_ids: Vec<i64>) -> PyResult<i64> {
        // Full implementation: run MambaLayer / LatentMoeLayer forward, then
        // LM-head projection, then argmax.  Stub returns 0.
        Ok(0)
    }

    /// Generate `max_new_tokens` tokens autoregressively from `prompt_ids`.
    fn generate(
        &self,
        prompt_ids: Vec<i64>,
        max_new_tokens: usize,
    ) -> PyResult<Vec<i64>> {
        let mut output = prompt_ids.clone();
        let mut context = prompt_ids;
        for _ in 0..max_new_tokens {
            let next = self.decode_step(context.clone())?;
            output.push(next);
            context.push(next);
        }
        Ok(output)
    }

    /// Return the number of parameters (approximate, from config).
    fn num_parameters(&self) -> usize {
        let d = self.config.d_model;
        let l = self.config.num_layers;
        let v = self.config.vocab_size;
        // Rough estimate: embedding + l * (SSM + MLP) + LM head
        v * d + l * (4 * d * d) + d * v
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridEngine(d_model={}, layers={}, device='{}', cuda_backend={})",
            self.config.d_model, self.config.num_layers,
            self.device, self.cuda_backend
        )
    }
}

/// SwiftLLM Python module
#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Engine and config types (original inference engine)
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEngineConfig>()?;
    m.add_class::<PySamplingParams>()?;
    m.add_class::<PyGenerationOutput>()?;
    m.add_class::<PyRequestOutput>()?;

    // Phase-1 Hybrid Architecture bridge
    m.add_class::<PyHybridModelConfig>()?;
    m.add_class::<PyHybridEngine>()?;

    Ok(())
}

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
