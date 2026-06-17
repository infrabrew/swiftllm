// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      tuning.rs
// PATH:      /crates/swiftllm-core/src/tuning.rs
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

//! Deployment tuning helpers.
//!
//! * [`PerformanceMode`] — a single `--performance-mode` flag that applies a
//!   coherent set of scheduler tunings for a deployment scenario (balanced,
//!   interactivity, or throughput).
//! * [`MaxModelLen`] / [`auto_context_length`] — the `--max-model-len auto`
//!   behaviour: fit the context length to available GPU memory so the engine
//!   does not OOM at startup.
//! * [`ModelInspection`] — the `SWIFTLLM_LOG_MODEL_INSPECTION=1` view of a
//!   model's internal structure, attention backend, and quantization.

use crate::config::SchedulerConfig;
use serde::{Deserialize, Serialize};
use std::fmt;

// ----------------------------------------------------------------------------
// Performance mode
// ----------------------------------------------------------------------------

/// Deployment scenario selected via `--performance-mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerformanceMode {
    /// Sensible middle ground (the default).
    Balanced,
    /// Minimise per-request latency for chat-like, low-concurrency workloads.
    Interactivity,
    /// Maximise aggregate tokens/sec for batch/offline workloads.
    Throughput,
}

impl Default for PerformanceMode {
    fn default() -> Self {
        PerformanceMode::Balanced
    }
}

impl fmt::Display for PerformanceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PerformanceMode::Balanced => "balanced",
            PerformanceMode::Interactivity => "interactivity",
            PerformanceMode::Throughput => "throughput",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for PerformanceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "balanced" | "balance" => Ok(PerformanceMode::Balanced),
            "interactivity" | "interactive" | "latency" => Ok(PerformanceMode::Interactivity),
            "throughput" | "batch" => Ok(PerformanceMode::Throughput),
            other => Err(format!("unknown performance mode: '{}'", other)),
        }
    }
}

impl PerformanceMode {
    /// Apply this mode's tunings to a scheduler configuration in place.
    pub fn apply(&self, scheduler: &mut SchedulerConfig) {
        match self {
            PerformanceMode::Balanced => {
                scheduler.max_num_seqs = 256;
                scheduler.max_num_batched_tokens = 8192;
                scheduler.max_prefill_tokens = 2048;
                scheduler.enable_chunked_prefill = true;
                scheduler.delay_factor = 0.0;
            }
            PerformanceMode::Interactivity => {
                // Small batches + small prefill chunks keep TTFT/ITL low.
                scheduler.max_num_seqs = 32;
                scheduler.max_num_batched_tokens = 2048;
                scheduler.max_prefill_tokens = 512;
                scheduler.enable_chunked_prefill = true;
                scheduler.delay_factor = 0.0;
            }
            PerformanceMode::Throughput => {
                // Large batches maximise GPU occupancy.
                scheduler.max_num_seqs = 512;
                scheduler.max_num_batched_tokens = 16384;
                scheduler.max_prefill_tokens = 8192;
                scheduler.enable_chunked_prefill = true;
                scheduler.delay_factor = 0.5;
            }
        }
    }

    /// Recommended asynchronous-scheduling pipeline depth for this mode.
    pub fn async_pipeline_depth(&self) -> usize {
        match self {
            PerformanceMode::Interactivity => 1, // strictly serial — lowest latency
            PerformanceMode::Balanced => 2,
            PerformanceMode::Throughput => 4,
        }
    }
}

// ----------------------------------------------------------------------------
// Automatic context length (--max-model-len auto)
// ----------------------------------------------------------------------------

/// The `--max-model-len` setting: either an explicit length or `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaxModelLen {
    /// Fit the context length to available memory automatically.
    Auto,
    /// Use a fixed context length.
    Fixed(usize),
}

impl Default for MaxModelLen {
    fn default() -> Self {
        MaxModelLen::Auto
    }
}

impl std::str::FromStr for MaxModelLen {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") {
            return Ok(MaxModelLen::Auto);
        }
        s.parse::<usize>()
            .map(MaxModelLen::Fixed)
            .map_err(|_| format!("max-model-len must be a positive integer or 'auto', got '{}'", s))
    }
}

impl MaxModelLen {
    /// Resolve to a concrete context length.
    ///
    /// For [`MaxModelLen::Auto`], fits to `available_kv_bytes` given the model's
    /// per-token KV footprint; for [`MaxModelLen::Fixed`], returns the requested
    /// length clamped to `model_max_supported`.
    pub fn resolve(
        &self,
        available_kv_bytes: usize,
        kv_bytes_per_token: usize,
        model_max_supported: usize,
    ) -> usize {
        match self {
            MaxModelLen::Auto => {
                auto_context_length(available_kv_bytes, kv_bytes_per_token, model_max_supported)
            }
            MaxModelLen::Fixed(n) => (*n).min(model_max_supported).max(1),
        }
    }
}

/// Compute the largest context length whose KV cache fits in `available_kv_bytes`,
/// capped at the model's architectural maximum. Returns at least 1.
pub fn auto_context_length(
    available_kv_bytes: usize,
    kv_bytes_per_token: usize,
    model_max_supported: usize,
) -> usize {
    if kv_bytes_per_token == 0 {
        return model_max_supported.max(1);
    }
    let fits = available_kv_bytes / kv_bytes_per_token;
    fits.min(model_max_supported).max(1)
}

/// Compute the KV bytes available for the cache from total device memory, after
/// reserving space for weights and activations.
pub fn available_kv_bytes(
    total_device_bytes: usize,
    weight_bytes: usize,
    utilization: f32,
    activation_reserve_bytes: usize,
) -> usize {
    let usable = (total_device_bytes as f64 * utilization.clamp(0.0, 1.0) as f64) as usize;
    usable
        .saturating_sub(weight_bytes)
        .saturating_sub(activation_reserve_bytes)
}

// ----------------------------------------------------------------------------
// Model inspection view (SWIFTLLM_LOG_MODEL_INSPECTION=1)
// ----------------------------------------------------------------------------

/// A human-readable inspection of a loaded model's internals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInspection {
    /// Architecture name (e.g. "Gemma", "Llama").
    pub architecture: String,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Hidden size.
    pub hidden_size: usize,
    /// Attention heads.
    pub num_attention_heads: usize,
    /// Key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Compute dtype (e.g. "float16").
    pub dtype: String,
    /// Attention backend in use (e.g. "paged", "flashinfer").
    pub attention_backend: String,
    /// Quantization scheme, if any (e.g. "fp8", "nvfp4").
    pub quantization: Option<String>,
    /// KV-cache dtype (e.g. "float16", "fp8").
    pub kv_cache_dtype: String,
}

impl ModelInspection {
    /// Whether inspection logging is enabled via `SWIFTLLM_LOG_MODEL_INSPECTION`.
    pub fn is_enabled() -> bool {
        std::env::var("SWIFTLLM_LOG_MODEL_INSPECTION")
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    }

    /// Render the inspection as a readable, multi-line report.
    pub fn render(&self) -> String {
        let gqa = if self.num_key_value_heads < self.num_attention_heads {
            format!(
                " (GQA {}:{})",
                self.num_attention_heads, self.num_key_value_heads
            )
        } else {
            String::new()
        };
        format!(
            "SwiftLLM model inspection\n\
             ├─ architecture      : {}\n\
             ├─ layers            : {}\n\
             ├─ hidden_size       : {}\n\
             ├─ attention_heads   : {}{}\n\
             ├─ dtype             : {}\n\
             ├─ attention_backend : {}\n\
             ├─ kv_cache_dtype    : {}\n\
             └─ quantization      : {}",
            self.architecture,
            self.num_layers,
            self.hidden_size,
            self.num_attention_heads,
            gqa,
            self.dtype,
            self.attention_backend,
            self.kv_cache_dtype,
            self.quantization.as_deref().unwrap_or("none"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn performance_mode_parses() {
        assert_eq!(PerformanceMode::from_str("throughput").unwrap(), PerformanceMode::Throughput);
        assert_eq!(PerformanceMode::from_str("INTERACTIVE").unwrap(), PerformanceMode::Interactivity);
        assert_eq!(PerformanceMode::from_str("balanced").unwrap(), PerformanceMode::Balanced);
        assert!(PerformanceMode::from_str("nonsense").is_err());
        assert_eq!(PerformanceMode::default().to_string(), "balanced");
    }

    #[test]
    fn performance_modes_order_batch_sizes() {
        let mut interactive = SchedulerConfig::default();
        let mut balanced = SchedulerConfig::default();
        let mut throughput = SchedulerConfig::default();
        PerformanceMode::Interactivity.apply(&mut interactive);
        PerformanceMode::Balanced.apply(&mut balanced);
        PerformanceMode::Throughput.apply(&mut throughput);

        assert!(interactive.max_num_seqs < balanced.max_num_seqs);
        assert!(balanced.max_num_seqs < throughput.max_num_seqs);
        assert!(interactive.max_num_batched_tokens < throughput.max_num_batched_tokens);
        assert!(interactive.max_prefill_tokens < throughput.max_prefill_tokens);
    }

    #[test]
    fn async_depth_scales_with_mode() {
        assert_eq!(PerformanceMode::Interactivity.async_pipeline_depth(), 1);
        assert!(
            PerformanceMode::Throughput.async_pipeline_depth()
                > PerformanceMode::Balanced.async_pipeline_depth()
        );
    }

    #[test]
    fn auto_context_fits_memory() {
        // 1 GiB available, 1 MiB per token -> 1024 tokens, under the 8k cap.
        let len = auto_context_length(1024 * 1024 * 1024, 1024 * 1024, 8192);
        assert_eq!(len, 1024);
    }

    #[test]
    fn auto_context_capped_at_model_max() {
        // Plenty of memory, but model only supports 4096.
        let len = auto_context_length(1usize << 40, 1024, 4096);
        assert_eq!(len, 4096);
    }

    #[test]
    fn auto_context_minimum_one() {
        // Tiny memory still yields at least 1.
        assert_eq!(auto_context_length(10, 1024, 8192), 1);
        // Zero per-token cost degenerates to the model max.
        assert_eq!(auto_context_length(0, 0, 4096), 4096);
    }

    #[test]
    fn max_model_len_parsing_and_resolution() {
        assert_eq!(MaxModelLen::from_str("auto").unwrap(), MaxModelLen::Auto);
        assert_eq!(MaxModelLen::from_str("2048").unwrap(), MaxModelLen::Fixed(2048));
        assert!(MaxModelLen::from_str("-5").is_err());

        // Fixed is clamped to the model maximum.
        assert_eq!(MaxModelLen::Fixed(100000).resolve(0, 1, 8192), 8192);
        // Auto fits to memory.
        assert_eq!(MaxModelLen::Auto.resolve(2048 * 64, 64, 8192), 2048);
    }

    #[test]
    fn available_kv_bytes_reserves_weights_and_activations() {
        // 16 GiB total, 90% util, 8 GiB weights, 1 GiB activations.
        let gib = 1024 * 1024 * 1024usize;
        let kv = available_kv_bytes(16 * gib, 8 * gib, 0.9, gib);
        // 16*0.9 = 14.4 GiB usable; minus 8 weights minus 1 activation ≈ 5.4 GiB.
        assert!(kv > 5 * gib && kv < 6 * gib);
    }

    #[test]
    fn model_inspection_renders_fields() {
        let inspection = ModelInspection {
            architecture: "Gemma".into(),
            num_layers: 28,
            hidden_size: 3072,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            dtype: "bfloat16".into(),
            attention_backend: "paged".into(),
            quantization: Some("fp8".into()),
            kv_cache_dtype: "fp8".into(),
        };
        let report = inspection.render();
        assert!(report.contains("architecture      : Gemma"));
        assert!(report.contains("GQA 16:8"));
        assert!(report.contains("quantization      : fp8"));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: tuning.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/tuning.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
