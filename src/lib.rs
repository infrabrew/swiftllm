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

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
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
        if top_p < 0.0 || top_p > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_p must be between 0.0 and 1.0",
            ));
        }
        if min_p < 0.0 || min_p > 1.0 {
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
        format!(
            "GenerationOutput(index={}, text='{}...', tokens={}, finish_reason={:?})",
            self.index,
            if self.text.len() > 50 { &self.text[..50] } else { &self.text },
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
        let rid = swiftllm_core::types::RequestId::from_str(request_id)
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

/// SwiftLLM Python module
#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Engine and config types
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEngineConfig>()?;
    m.add_class::<PySamplingParams>()?;
    m.add_class::<PyGenerationOutput>()?;
    m.add_class::<PyRequestOutput>()?;

    Ok(())
}

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
