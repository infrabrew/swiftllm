// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      disaggregated.rs
// PATH:      /crates/swiftllm-core/src/serving/disaggregated.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// USES:
//   (no intra-crate imports — KvTransferMetadata references block_size from config)
// USED BY:
//   - swiftllm-core/src/serving/mod.rs      re-exports DisaggregatedScheduler, WorkerSpec, etc.
//   - swiftllm-core/src/lib.rs              indirectly via serving module
// SEE ALSO:
//   - swiftllm-core/src/engine.rs                      Engine is instantiated per role (prefill/decode)
//   - swiftllm-core/src/memory/block_manager.rs        KV block allocation; block_indices transferred here
//   - swiftllm-core/src/memory/paged_attention.rs      PagedAttention block layout matches KvTransferMetadata
//   - swiftllm-core/src/scheduler/continuous_batching.rs  decode-side ContinuousBatchingScheduler
//   - swiftllm-core/src/scheduler/mod.rs               Scheduler per worker role
//   - swiftllm-core/src/config.rs                      EngineConfig.block_size feeds DisaggregatedConfig
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

//! Disaggregated Prefill/Decode Serving.
//!
//! LLM inference has two qualitatively different phases:
//!
//! - **Prefill** — compute-bound: processes the entire prompt in one forward
//!   pass; benefits from large batches and high FLOP/s GPUs (A100, H100).
//! - **Decode** — memory-bandwidth-bound: generates one token at a time;
//!   bottlenecked by KV cache reads and benefits from high HBM bandwidth.
//!
//! By disaggregating the two phases onto dedicated worker pools, SwiftLLM can
//! independently scale each to match its bottleneck and avoid the
//! prefill-decode interference that degrades throughput in coupled systems.
//!
//! ## Architecture
//! ```text
//! Client
//!   │  Request
//!   ▼
//! DisaggregatedScheduler
//!   ├─ PrefillPool ──(KV transfer)──► DecodePool
//!   │    Worker₀                        Worker₀
//!   │    Worker₁                        Worker₁
//!   │    …                              …
//!   └─────────────────────────────────► Response
//! ```
//!
//! ## References
//! - "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"
//!   (Patel et al., 2023) — https://arxiv.org/abs/2311.18677
//! - "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM
//!   Serving" (Zhong et al., 2024)
//! - Mooncake (ByteDance, 2024): KV cache as a service

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The role a worker plays in the disaggregated pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerRole {
    /// Prefill worker — processes prompts, produces KV cache.
    Prefill,
    /// Decode worker — generates tokens autoregressively from a KV cache.
    Decode,
}

impl WorkerRole {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            WorkerRole::Prefill => "prefill",
            WorkerRole::Decode => "decode",
        }
    }
}

/// Specification for one physical or virtual worker instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSpec {
    /// Unique worker ID.
    pub id: usize,
    /// Role: prefill or decode.
    pub role: WorkerRole,
    /// Which device this worker runs on (GPU index or node address).
    pub device_id: usize,
    /// Maximum tokens this worker can prefill in one batch.
    pub max_prefill_tokens: usize,
    /// Maximum concurrent decoding sequences.
    pub max_decode_sequences: usize,
    /// Estimated peak throughput (tokens/s) for this worker's role.
    pub throughput_tokens_per_sec: f64,
}

impl WorkerSpec {
    /// Create a new prefill worker spec.
    pub fn prefill(id: usize, device_id: usize, max_prefill_tokens: usize) -> Self {
        Self {
            id,
            role: WorkerRole::Prefill,
            device_id,
            max_prefill_tokens,
            max_decode_sequences: 0,
            throughput_tokens_per_sec: 50_000.0,
        }
    }

    /// Create a new decode worker spec.
    pub fn decode(id: usize, device_id: usize, max_decode_sequences: usize) -> Self {
        Self {
            id,
            role: WorkerRole::Decode,
            device_id,
            max_prefill_tokens: 0,
            max_decode_sequences,
            throughput_tokens_per_sec: 2_000.0,
        }
    }
}

/// Metadata describing a KV cache transfer between prefill and decode workers.
///
/// After a prefill worker processes a prompt it ships the KV tensors to the
/// decode worker that will generate the continuation.
#[derive(Debug, Clone)]
pub struct KvTransferMetadata {
    /// The request whose KV cache is being transferred.
    pub request_id: u64,
    /// Layer indices whose KV blocks are included.
    pub layer_indices: Vec<usize>,
    /// Physical block indices being transferred.
    pub block_indices: Vec<usize>,
    /// Total byte size of the transfer.
    pub byte_size: usize,
    /// Estimated transfer latency in milliseconds.
    pub estimated_latency_ms: f64,
}

impl KvTransferMetadata {
    /// Estimate transfer latency assuming a given NIC bandwidth (GiB/s).
    pub fn estimated_latency_ms_for_bandwidth(&self, bandwidth_gibs: f64) -> f64 {
        let bytes_per_sec = bandwidth_gibs * 1024.0 * 1024.0 * 1024.0;
        (self.byte_size as f64 / bytes_per_sec) * 1000.0
    }
}

/// Scheduling policy for assigning requests to workers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// Round-robin across workers of each role.
    RoundRobin,
    /// Least-loaded worker (fewest tokens/sequences in flight).
    LeastLoaded,
    /// Locality-aware: prefer to keep prefill and decode on the same NVLink domain.
    LocalityAware,
}

/// Assignment of one request to a specific prefill and decode worker.
#[derive(Debug, Clone)]
pub struct WorkerAssignment {
    /// Request ID.
    pub request_id: u64,
    /// Worker that will run prefill.
    pub prefill_worker_id: usize,
    /// Worker that will run decode.
    pub decode_worker_id: usize,
    /// Estimated end-to-end latency (ms) for this assignment.
    pub estimated_latency_ms: f64,
}

/// Runtime load snapshot for a single worker.
#[derive(Debug, Clone, Default)]
pub struct WorkerLoad {
    /// Number of tokens currently being prefilled.
    pub prefill_tokens_in_flight: usize,
    /// Number of sequences currently being decoded.
    pub decode_sequences_in_flight: usize,
    /// Total tokens processed since startup.
    pub total_tokens_processed: u64,
}

impl WorkerLoad {
    /// Score for "least loaded" scheduling (lower = prefer this worker).
    fn load_score(&self, spec: &WorkerSpec) -> f64 {
        match spec.role {
            WorkerRole::Prefill => {
                self.prefill_tokens_in_flight as f64
                    / spec.max_prefill_tokens.max(1) as f64
            }
            WorkerRole::Decode => {
                self.decode_sequences_in_flight as f64
                    / spec.max_decode_sequences.max(1) as f64
            }
        }
    }
}

/// Configuration for the disaggregated scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggregatedConfig {
    /// Number of dedicated prefill workers.
    pub num_prefill_workers: usize,
    /// Number of dedicated decode workers.
    pub num_decode_workers: usize,
    /// Request assignment policy.
    pub policy: SchedulingPolicy,
    /// Target KV transfer bandwidth in GiB/s (NVLink: ~600, PCIe: ~32).
    pub kv_transfer_bandwidth_gibs: f64,
    /// Bytes per KV element (2 for bf16/fp16, 1 for int8).
    pub kv_bytes_per_element: usize,
    /// Number of attention heads × head dimension (for transfer size estimation).
    pub kv_heads_times_head_dim: usize,
    /// Number of model layers.
    pub num_layers: usize,
    /// KV cache block size in tokens.
    pub block_size: usize,
}

impl Default for DisaggregatedConfig {
    fn default() -> Self {
        Self {
            num_prefill_workers: 2,
            num_decode_workers: 4,
            policy: SchedulingPolicy::LeastLoaded,
            kv_transfer_bandwidth_gibs: 600.0, // NVLink
            kv_bytes_per_element: 2,           // bf16
            kv_heads_times_head_dim: 128 * 128, // 128 heads × 128 dim
            num_layers: 32,
            block_size: 16,
        }
    }
}

impl DisaggregatedConfig {
    /// Estimate the byte size of one KV cache block.
    ///
    /// Shape: `[2, num_layers, block_size, kv_heads_times_head_dim]` where
    /// the first dim is key vs value.
    pub fn kv_block_byte_size(&self) -> usize {
        2 * self.num_layers * self.block_size * self.kv_heads_times_head_dim
            * self.kv_bytes_per_element
    }
}

/// Disaggregated scheduler: routes requests to prefill and decode workers.
pub struct DisaggregatedScheduler {
    /// Configuration.
    pub config: DisaggregatedConfig,

    /// All worker specifications (prefill and decode).
    workers: Vec<WorkerSpec>,

    /// Per-worker load counters.
    load: HashMap<usize, WorkerLoad>,

    /// Round-robin cursor for prefill workers.
    rr_prefill: usize,
    /// Round-robin cursor for decode workers.
    rr_decode: usize,
}

impl DisaggregatedScheduler {
    /// Create a scheduler from a config.
    ///
    /// Workers are auto-provisioned: prefill workers get devices 0..N_prefill,
    /// decode workers get devices N_prefill..N_prefill+N_decode.
    pub fn new(config: DisaggregatedConfig) -> Self {
        let mut workers = Vec::new();
        let np = config.num_prefill_workers;
        let nd = config.num_decode_workers;

        let max_pt = 8192; // default max prefill tokens per worker
        let max_ds = 256;  // default max decode sequences per worker

        for i in 0..np {
            workers.push(WorkerSpec::prefill(i, i, max_pt));
        }
        for i in 0..nd {
            workers.push(WorkerSpec::decode(np + i, np + i, max_ds));
        }

        let load: HashMap<usize, WorkerLoad> = workers
            .iter()
            .map(|w| (w.id, WorkerLoad::default()))
            .collect();

        Self { config, workers, load, rr_prefill: 0, rr_decode: 0 }
    }

    /// Create from an explicit list of worker specs.
    pub fn from_workers(config: DisaggregatedConfig, workers: Vec<WorkerSpec>) -> Self {
        let load: HashMap<usize, WorkerLoad> = workers
            .iter()
            .map(|w| (w.id, WorkerLoad::default()))
            .collect();
        Self { config, workers, load, rr_prefill: 0, rr_decode: 0 }
    }

    /// Workers with a given role.
    fn workers_with_role(&self, role: WorkerRole) -> Vec<&WorkerSpec> {
        self.workers.iter().filter(|w| w.role == role).collect()
    }

    /// Schedule a request: pick prefill and decode workers.
    pub fn schedule(&mut self, request_id: u64, prompt_len: usize) -> Option<WorkerAssignment> {
        let prefill_id = self.pick_worker(WorkerRole::Prefill)?;
        let decode_id = self.pick_worker(WorkerRole::Decode)?;

        // Estimate transfer: how many blocks does this prompt use?
        let num_blocks = prompt_len.div_ceil(self.config.block_size);
        let byte_size = num_blocks * self.config.kv_block_byte_size();
        let bandwidth_bytes_per_ms =
            self.config.kv_transfer_bandwidth_gibs * 1024.0 * 1024.0 * 1024.0 / 1000.0;
        let transfer_ms = byte_size as f64 / bandwidth_bytes_per_ms;

        // Estimate decode latency: prompt_len / throughput + transfer
        let prefill_spec = self.workers.iter().find(|w| w.id == prefill_id)?;
        let prefill_ms = prompt_len as f64 / prefill_spec.throughput_tokens_per_sec * 1000.0;
        let estimated_latency_ms = prefill_ms + transfer_ms;

        // Update load tracking.
        if let Some(load) = self.load.get_mut(&prefill_id) {
            load.prefill_tokens_in_flight += prompt_len;
        }
        if let Some(load) = self.load.get_mut(&decode_id) {
            load.decode_sequences_in_flight += 1;
        }

        Some(WorkerAssignment {
            request_id,
            prefill_worker_id: prefill_id,
            decode_worker_id: decode_id,
            estimated_latency_ms,
        })
    }

    /// Mark a request as complete, freeing worker capacity.
    pub fn complete(&mut self, assignment: &WorkerAssignment, prompt_len: usize, gen_len: usize) {
        if let Some(load) = self.load.get_mut(&assignment.prefill_worker_id) {
            load.prefill_tokens_in_flight =
                load.prefill_tokens_in_flight.saturating_sub(prompt_len);
            load.total_tokens_processed += (prompt_len + gen_len) as u64;
        }
        if let Some(load) = self.load.get_mut(&assignment.decode_worker_id) {
            load.decode_sequences_in_flight =
                load.decode_sequences_in_flight.saturating_sub(1);
            load.total_tokens_processed += gen_len as u64;
        }
    }

    /// Build KV transfer metadata for a request.
    pub fn kv_transfer_metadata(
        &self,
        request_id: u64,
        prompt_len: usize,
        num_layers: Option<usize>,
    ) -> KvTransferMetadata {
        let layers = num_layers.unwrap_or(self.config.num_layers);
        let num_blocks = prompt_len.div_ceil(self.config.block_size);
        let byte_size = num_blocks * self.config.kv_block_byte_size();
        let bandwidth_bytes_per_ms =
            self.config.kv_transfer_bandwidth_gibs * 1024.0 * 1024.0 * 1024.0 / 1000.0;
        let estimated_latency_ms = byte_size as f64 / bandwidth_bytes_per_ms;

        KvTransferMetadata {
            request_id,
            layer_indices: (0..layers).collect(),
            block_indices: (0..num_blocks).collect(),
            byte_size,
            estimated_latency_ms,
        }
    }

    /// Total number of workers.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Current load for a given worker.
    pub fn worker_load(&self, worker_id: usize) -> Option<&WorkerLoad> {
        self.load.get(&worker_id)
    }

    /// Aggregate throughput estimate (tokens/s) for all decode workers.
    pub fn decode_throughput_estimate(&self) -> f64 {
        self.workers_with_role(WorkerRole::Decode)
            .iter()
            .map(|w| w.throughput_tokens_per_sec)
            .sum()
    }

    // --- private ---

    fn pick_worker(&mut self, role: WorkerRole) -> Option<usize> {
        // Collect worker IDs first to avoid holding a reference into `self.workers`
        // while we also need to mutate cursor fields.
        let worker_ids: Vec<usize> = self
            .workers
            .iter()
            .filter(|w| w.role == role)
            .map(|w| w.id)
            .collect();

        if worker_ids.is_empty() {
            return None;
        }

        let id = match self.config.policy {
            SchedulingPolicy::RoundRobin => {
                let cursor = match role {
                    WorkerRole::Prefill => &mut self.rr_prefill,
                    WorkerRole::Decode => &mut self.rr_decode,
                };
                let idx = *cursor % worker_ids.len();
                *cursor = cursor.wrapping_add(1);
                worker_ids[idx]
            }
            SchedulingPolicy::LeastLoaded | SchedulingPolicy::LocalityAware => {
                // Snapshot the load scores before the sort so we don't hold
                // a borrow on `self.load` and `self.workers` simultaneously.
                let scores: Vec<(usize, f64)> = worker_ids
                    .iter()
                    .map(|&wid| {
                        let spec = self.workers.iter().find(|w| w.id == wid).unwrap();
                        let score = self.load.get(&wid).map(|l| l.load_score(spec)).unwrap_or(0.0);
                        (wid, score)
                    })
                    .collect();

                scores
                    .into_iter()
                    .min_by(|(_, sa), (_, sb)| {
                        sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(id, _)| id)?
            }
        };
        Some(id)
    }
}

/// Compute the optimal prefill-to-decode worker ratio for a given model
/// and hardware configuration.
///
/// Returns `(num_prefill, num_decode)` such that the prefill and decode
/// throughputs are balanced: `N_p * T_p ≈ N_d * T_d`.
///
/// - `prefill_tokens_per_sec` — peak prefill throughput of one GPU (e.g. 50_000)
/// - `decode_tokens_per_sec`  — peak decode throughput of one GPU (e.g. 2_000)
/// - `total_workers`          — total GPU budget
pub fn optimal_worker_ratio(
    prefill_tokens_per_sec: f64,
    decode_tokens_per_sec: f64,
    total_workers: usize,
) -> (usize, usize) {
    if prefill_tokens_per_sec <= 0.0 || decode_tokens_per_sec <= 0.0 || total_workers == 0 {
        return (1, 1);
    }
    // Each prefill worker can "feed" (prefill_tps / decode_tps) decode workers.
    // So: n_decode / n_prefill = prefill_tps / decode_tps
    // And: n_prefill + n_decode = total
    // => n_prefill = total / (1 + prefill_tps/decode_tps)
    let ratio = prefill_tokens_per_sec / decode_tokens_per_sec;
    let n_prefill = ((total_workers as f64 / (1.0 + ratio)).round() as usize).max(1);
    let n_decode = (total_workers - n_prefill).max(1);
    (n_prefill, n_decode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scheduler() -> DisaggregatedScheduler {
        DisaggregatedScheduler::new(DisaggregatedConfig::default())
    }

    // ── WorkerSpec ────────────────────────────────────────────────────────────

    #[test]
    fn test_prefill_worker_role() {
        let w = WorkerSpec::prefill(0, 0, 8192);
        assert_eq!(w.role, WorkerRole::Prefill);
        assert_eq!(w.max_prefill_tokens, 8192);
    }

    #[test]
    fn test_decode_worker_role() {
        let w = WorkerSpec::decode(1, 1, 256);
        assert_eq!(w.role, WorkerRole::Decode);
        assert_eq!(w.max_decode_sequences, 256);
    }

    // ── DisaggregatedConfig ───────────────────────────────────────────────────

    #[test]
    fn test_kv_block_byte_size() {
        let cfg = DisaggregatedConfig {
            num_layers: 32,
            block_size: 16,
            kv_heads_times_head_dim: 128,
            kv_bytes_per_element: 2,
            ..Default::default()
        };
        // 2 × 32 × 16 × 128 × 2 = 262_144 bytes = 256 KiB
        let expected = 2 * 32 * 16 * 128 * 2;
        assert_eq!(cfg.kv_block_byte_size(), expected);
    }

    // ── DisaggregatedScheduler ────────────────────────────────────────────────

    #[test]
    fn test_scheduler_worker_count() {
        let sched = default_scheduler();
        let cfg = DisaggregatedConfig::default();
        assert_eq!(
            sched.num_workers(),
            cfg.num_prefill_workers + cfg.num_decode_workers
        );
    }

    #[test]
    fn test_schedule_returns_assignment() {
        let mut sched = default_scheduler();
        let assignment = sched.schedule(1, 512);
        assert!(assignment.is_some());
    }

    #[test]
    fn test_schedule_prefill_decode_different_workers() {
        let mut sched = default_scheduler();
        let a = sched.schedule(1, 512).unwrap();
        assert_ne!(a.prefill_worker_id, a.decode_worker_id);
    }

    #[test]
    fn test_schedule_updates_load() {
        let mut sched = default_scheduler();
        let a = sched.schedule(42, 1024).unwrap();
        let load = sched.worker_load(a.prefill_worker_id).unwrap();
        assert_eq!(load.prefill_tokens_in_flight, 1024);
    }

    #[test]
    fn test_complete_decrements_load() {
        let mut sched = default_scheduler();
        let a = sched.schedule(1, 512).unwrap();
        sched.complete(&a, 512, 64);
        let load = sched.worker_load(a.prefill_worker_id).unwrap();
        assert_eq!(load.prefill_tokens_in_flight, 0);
    }

    #[test]
    fn test_complete_tracks_total_tokens() {
        let mut sched = default_scheduler();
        let a = sched.schedule(1, 100).unwrap();
        sched.complete(&a, 100, 50);
        let load = sched.worker_load(a.prefill_worker_id).unwrap();
        assert_eq!(load.total_tokens_processed, 150);
    }

    #[test]
    fn test_round_robin_distributes() {
        let cfg = DisaggregatedConfig {
            policy: SchedulingPolicy::RoundRobin,
            num_prefill_workers: 2,
            num_decode_workers: 2,
            ..Default::default()
        };
        let mut sched = DisaggregatedScheduler::new(cfg);
        let a1 = sched.schedule(1, 100).unwrap();
        let a2 = sched.schedule(2, 100).unwrap();
        // Second request should go to a different prefill worker.
        assert_ne!(a1.prefill_worker_id, a2.prefill_worker_id);
    }

    #[test]
    fn test_kv_transfer_metadata() {
        let sched = default_scheduler();
        let meta = sched.kv_transfer_metadata(99, 512, None);
        assert_eq!(meta.request_id, 99);
        assert!(!meta.block_indices.is_empty());
        assert!(meta.byte_size > 0);
        assert!(meta.estimated_latency_ms >= 0.0);
    }

    #[test]
    fn test_estimated_latency_bandwidth() {
        let meta = KvTransferMetadata {
            request_id: 0,
            layer_indices: vec![],
            block_indices: vec![],
            byte_size: 1024 * 1024 * 1024, // 1 GiB
            estimated_latency_ms: 0.0,
        };
        let ms = meta.estimated_latency_ms_for_bandwidth(1.0); // 1 GiB/s
        assert!((ms - 1000.0).abs() < 1.0); // ≈ 1000 ms = 1 second
    }

    // ── optimal_worker_ratio ─────────────────────────────────────────────────

    #[test]
    fn test_optimal_ratio_balanced() {
        // Prefill 25× faster than decode → need 25× more decode workers
        let (np, nd) = optimal_worker_ratio(50_000.0, 2_000.0, 26);
        assert!(nd > np, "need more decode workers when decode is slower");
    }

    #[test]
    fn test_optimal_ratio_total_workers() {
        let (np, nd) = optimal_worker_ratio(10_000.0, 2_000.0, 12);
        assert_eq!(np + nd, 12);
    }

    #[test]
    fn test_optimal_ratio_zero_guard() {
        let (np, nd) = optimal_worker_ratio(0.0, 0.0, 0);
        assert_eq!(np, 1);
        assert_eq!(nd, 1);
    }

    #[test]
    fn test_decode_throughput_estimate() {
        let sched = default_scheduler();
        let cfg = DisaggregatedConfig::default();
        // N decode workers × default decode throughput per worker
        let expected =
            cfg.num_decode_workers as f64 * WorkerSpec::decode(0, 0, 256).throughput_tokens_per_sec;
        assert!((sched.decode_throughput_estimate() - expected).abs() < 1.0);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: disaggregated.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/serving/disaggregated.rs
// INTEGRATES:  engine.rs · memory/block_manager.rs · memory/paged_attention.rs · scheduler/continuous_batching.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
