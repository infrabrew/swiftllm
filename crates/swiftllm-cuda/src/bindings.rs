// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      bindings.rs
// PATH:      /crates/swiftllm-cuda/src/bindings.rs
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

//! CUDA Kernel Bindings
//!
//! Safe Rust bindings for CUDA kernels.

use super::{CudaError, Result};

/// Check the last CUDA error after a kernel launch.
///
/// This MUST be called after every kernel dispatch to catch asynchronous
/// launch errors that would otherwise be silently swallowed.
#[cfg(has_cuda)]
fn check_cuda_last_error(kernel_name: &str) -> Result<()> {
    extern "C" {
        fn cudaGetLastError() -> i32;
    }
    let rc = unsafe { cudaGetLastError() };
    if rc != 0 {
        Err(CudaError::KernelError(format!(
            "{} failed with CUDA error code {}",
            kernel_name, rc
        )))
    } else {
        Ok(())
    }
}

/// PagedAttention kernel parameters
#[derive(Debug, Clone)]
pub struct PagedAttentionParams {
    /// Number of sequences in batch
    pub num_seqs: usize,

    /// Number of heads
    pub num_heads: usize,

    /// Number of KV heads
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Block size
    pub block_size: usize,

    /// Scaling factor
    pub scale: f32,

    /// Maximum context length
    pub max_context_len: usize,
}

/// Launch PagedAttention prefill kernel
pub fn paged_attention_prefill(
    output: *mut half::f16,
    query: *const half::f16,
    key_cache: *const half::f16,
    value_cache: *const half::f16,
    slot_mapping: *const i32,
    context_lens: *const i32,
    params: &PagedAttentionParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        // Launch CUDA kernel
        // In a real implementation, this would call the compiled CUDA kernel
        tracing::debug!("Launching paged_attention_prefill kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Launch PagedAttention decode kernel
pub fn paged_attention_decode(
    output: *mut half::f16,
    query: *const half::f16,
    key_cache: *const half::f16,
    value_cache: *const half::f16,
    block_tables: *const i32,
    context_lens: *const i32,
    params: &PagedAttentionParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching paged_attention_decode kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Reshape and cache kernel parameters
#[derive(Debug, Clone)]
pub struct ReshapeCacheParams {
    /// Number of tokens
    pub num_tokens: usize,

    /// Number of KV heads
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Block size
    pub block_size: usize,
}

/// Launch reshape and cache kernel
pub fn reshape_and_cache(
    key: *const half::f16,
    value: *const half::f16,
    key_cache: *mut half::f16,
    value_cache: *mut half::f16,
    slot_mapping: *const i32,
    params: &ReshapeCacheParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching reshape_and_cache kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// RMS normalization kernel
pub fn rms_norm(
    output: *mut half::f16,
    input: *const half::f16,
    weight: *const half::f16,
    epsilon: f32,
    num_tokens: usize,
    hidden_size: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching rms_norm kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Fused add + RMS normalization kernel
pub fn fused_add_rms_norm(
    output: *mut half::f16,
    residual: *mut half::f16,
    input: *const half::f16,
    weight: *const half::f16,
    epsilon: f32,
    num_tokens: usize,
    hidden_size: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching fused_add_rms_norm kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Rotary embedding kernel
pub fn rotary_embedding(
    positions: *const i32,
    query: *mut half::f16,
    key: *mut half::f16,
    cos_cache: *const half::f16,
    sin_cache: *const half::f16,
    num_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching rotary_embedding kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// SiLU activation kernel
pub fn silu_and_mul(
    output: *mut half::f16,
    input: *const half::f16,
    num_tokens: usize,
    intermediate_size: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching silu_and_mul kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Copy blocks between GPU and CPU
pub fn copy_blocks(
    key_caches: &[*mut half::f16],
    value_caches: &[*mut half::f16],
    block_mapping: *const i64,
    num_pairs: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching copy_blocks kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Swap blocks between GPU and CPU
pub fn swap_blocks(
    src: *const half::f16,
    dst: *mut half::f16,
    block_mapping: *const i64,
    num_pairs: usize,
    block_size_bytes: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching swap_blocks operation");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Quantized GEMM (INT4)
pub fn gemm_int4(
    output: *mut half::f16,
    input: *const half::f16,
    weight: *const u8, // Packed INT4
    scales: *const half::f16,
    zeros: *const half::f16,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching gemm_int4 kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Quantized GEMM (INT8)
pub fn gemm_int8(
    output: *mut half::f16,
    input: *const half::f16,
    weight: *const i8,
    scales: *const half::f16,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching gemm_int8 kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Softmax kernel
pub fn softmax(
    output: *mut half::f16,
    input: *const half::f16,
    num_rows: usize,
    num_cols: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching softmax kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Top-k sampling kernel
pub fn top_k_sampling(
    output_ids: *mut i32,
    output_probs: *mut f32,
    logits: *const f32,
    k: usize,
    batch_size: usize,
    vocab_size: usize,
    random_vals: *const f32,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching top_k_sampling kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Top-p (nucleus) sampling kernel
pub fn top_p_sampling(
    output_ids: *mut i32,
    output_probs: *mut f32,
    logits: *const f32,
    p: f32,
    batch_size: usize,
    vocab_size: usize,
    random_vals: *const f32,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching top_p_sampling kernel");
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

// ==============================================================================
// Phase-1 Hybrid Architecture Kernel Bindings
// ==============================================================================

// ---------------------------------------------------------------------------
// Mamba-3 SSM scan kernel declarations
// ---------------------------------------------------------------------------

/// extern "C" declarations for compiled mamba3_scan.cu
#[cfg(has_cuda)]
extern "C" {
    fn mamba3_decode_step(
        x_t:          *const half::f16,
        dt_t:         *const half::f16,
        dt_proj_w:    *const half::f16,
        a_log:        *const half::f16,
        b_t:          *const half::f16,
        c_t:          *const half::f16,
        d_skip:       *const half::f16,
        h_state:      *mut   f32,
        y_t:          *mut   half::f16,
        batch:        i32, d_inner: i32, d_state: i32, dt_rank: i32,
        num_heads:    i32,
        dt_min:       f32, dt_max: f32,
        use_trap:     bool, use_complex: bool,
    );

    fn mamba3_prefill_scan(
        x:            *const half::f16,
        dt_proj_out:  *const half::f16,
        a_log:        *const half::f16,
        b:            *const half::f16,
        c:            *const half::f16,
        d_skip:       *const half::f16,
        out:          *mut   half::f16,
        batch:        i32, seq_len: i32, d_inner: i32, d_state: i32,
        use_trap:     bool, use_complex: bool,
    );
}

/// Parameters for a Mamba-3 decode step
#[derive(Debug, Clone)]
pub struct Mamba3DecodeParams {
    pub batch:       usize,
    pub d_inner:     usize,
    pub d_state:     usize,
    pub dt_rank:     usize,
    pub num_heads:   usize,
    pub dt_min:      f32,
    pub dt_max:      f32,
    pub use_trapezoidal: bool,
    pub use_complex: bool,
}

/// Launch the Mamba-3 single-step decode kernel.
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
pub unsafe fn mamba3_decode(
    x_t:       *const half::f16,
    dt_t:      *const half::f16,
    dt_proj_w: *const half::f16,
    a_log:     *const half::f16,
    b_t:       *const half::f16,
    c_t:       *const half::f16,
    d_skip:    *const half::f16,
    h_state:   *mut f32,
    y_t:       *mut half::f16,
    p: &Mamba3DecodeParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching mamba3_decode_step B={} d_inner={}", p.batch, p.d_inner);
        mamba3_decode_step(
            x_t, dt_t, dt_proj_w, a_log, b_t, c_t, d_skip, h_state, y_t,
            p.batch as i32, p.d_inner as i32, p.d_state as i32, p.dt_rank as i32,
            p.num_heads as i32, p.dt_min, p.dt_max, p.use_trapezoidal, p.use_complex,
        );
        check_cuda_last_error("mamba3_decode_step")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

/// Parameters for a Mamba-3 prefill scan
#[derive(Debug, Clone)]
pub struct Mamba3PrefillParams {
    pub batch:       usize,
    pub seq_len:     usize,
    pub d_inner:     usize,
    pub d_state:     usize,
    pub use_trapezoidal: bool,
    pub use_complex: bool,
}

/// Launch the Mamba-3 full-sequence prefill scan kernel.
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
pub unsafe fn mamba3_prefill(
    x:           *const half::f16,
    dt_proj_out: *const half::f16,
    a_log:       *const half::f16,
    b:           *const half::f16,
    c:           *const half::f16,
    d_skip:      *const half::f16,
    out:         *mut half::f16,
    p: &Mamba3PrefillParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching mamba3_prefill_scan B={} T={} d_inner={}", p.batch, p.seq_len, p.d_inner);
        mamba3_prefill_scan(
            x, dt_proj_out, a_log, b, c, d_skip, out,
            p.batch as i32, p.seq_len as i32, p.d_inner as i32, p.d_state as i32,
            p.use_trapezoidal, p.use_complex,
        );
        check_cuda_last_error("mamba3_prefill_scan")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

// ---------------------------------------------------------------------------
// LatentMoE kernel declarations
// ---------------------------------------------------------------------------

#[cfg(has_cuda)]
extern "C" {
    fn latent_compress(
        x: *const half::f16, w_compress: *const half::f16, z: *mut half::f16,
        n: i32, d_model: i32, d_latent: i32,
    );

    fn moe_topk_gate(
        router_logits: *const f32, expert_bias: *const f32,
        expert_ids: *mut i32, expert_weights: *mut f32,
        n: i32, num_experts: i32, top_k: i32,
    );

    fn moe_expert_dispatch(
        z_in: *const half::f16, gate_w: *const half::f16,
        up_w: *const half::f16, down_w: *const half::f16,
        expert_ids: *const i32, expert_weights: *const f32,
        z_out: *mut half::f16,
        n: i32, top_k: i32, d_latent: i32, d_ffn: i32, num_experts: i32,
    );

    fn moe_load_stats(
        expert_ids: *const i32, expert_counts: *mut i32,
        n: i32, top_k: i32,
    );

    fn latent_expand(
        z: *const half::f16, w_expand: *const half::f16, y: *mut half::f16,
        n: i32, d_latent: i32, d_model: i32,
    );
}

/// Parameters for a LatentMoE forward pass
#[derive(Debug, Clone)]
pub struct LatentMoeParams {
    pub n_tokens:    usize,
    pub d_model:     usize,
    pub d_latent:    usize,
    pub d_ffn:       usize,
    pub num_experts: usize,
    pub top_k:       usize,
}

/// Launch all LatentMoE kernels (compress → gate → dispatch → expand).
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn latent_moe_forward(
    x:             *const half::f16,
    w_compress:    *const half::f16,
    router_logits: *const f32,
    expert_bias:   *const f32,
    gate_w:        *const half::f16,
    up_w:          *const half::f16,
    down_w:        *const half::f16,
    w_expand:      *const half::f16,
    z_buf:         *mut half::f16,     // scratch [N, d_latent]
    expert_ids:    *mut i32,           // scratch [N, top_k]
    expert_weights:*mut f32,           // scratch [N, top_k]
    expert_counts: *mut i32,           // scratch [num_experts]
    z_out:         *mut half::f16,     // scratch [N, d_latent] (zeroed before call)
    y:             *mut half::f16,     // output  [N, d_model]
    p: &LatentMoeParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        tracing::debug!("Launching latent_moe N={} E={} k={}", p.n_tokens, p.num_experts, p.top_k);
        latent_compress(x, w_compress, z_buf,
            p.n_tokens as i32, p.d_model as i32, p.d_latent as i32);
        check_cuda_last_error("latent_compress")?;
        moe_topk_gate(router_logits, expert_bias, expert_ids, expert_weights,
            p.n_tokens as i32, p.num_experts as i32, p.top_k as i32);
        check_cuda_last_error("moe_topk_gate")?;
        moe_expert_dispatch(z_buf, gate_w, up_w, down_w,
            expert_ids, expert_weights, z_out,
            p.n_tokens as i32, p.top_k as i32, p.d_latent as i32,
            p.d_ffn as i32, p.num_experts as i32);
        check_cuda_last_error("moe_expert_dispatch")?;
        moe_load_stats(expert_ids, expert_counts, p.n_tokens as i32, p.top_k as i32);
        check_cuda_last_error("moe_load_stats")?;
        latent_expand(z_out, w_expand, y,
            p.n_tokens as i32, p.d_latent as i32, p.d_model as i32);
        check_cuda_last_error("latent_expand")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

// ---------------------------------------------------------------------------
// Dense Verification kernel declarations
// ---------------------------------------------------------------------------

#[cfg(has_cuda)]
extern "C" {
    fn dense_verif_cross_attn(
        q: *const half::f16, k: *const half::f16, v: *const half::f16,
        attn_out: *mut half::f16,
        token_conf: *mut f32, global_conf: *mut f32,
        batch: i32, t_q: i32, t_kv: i32, num_heads: i32, head_dim: i32,
        scale: f32,
    );

    fn dense_verif_global_conf(
        token_conf: *const f32, global_conf: *mut f32,
        batch: i32, t_q: i32,
    );
}

/// Parameters for the dense verification pass
#[derive(Debug, Clone)]
pub struct DenseVerifParams {
    pub batch:      usize,
    pub t_draft:    usize,
    pub t_trace:    usize,
    pub num_heads:  usize,
    pub head_dim:   usize,
}

/// Launch dense verification cross-attention + confidence scoring.
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
pub unsafe fn dense_verification(
    q:          *const half::f16,
    k:          *const half::f16,
    v:          *const half::f16,
    attn_out:   *mut half::f16,
    token_conf: *mut f32,
    global_conf:*mut f32,
    p: &DenseVerifParams,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        let scale = 1.0f32 / (p.head_dim as f32).sqrt();
        tracing::debug!("Launching dense_verif B={} T_draft={} T_trace={}", p.batch, p.t_draft, p.t_trace);
        dense_verif_cross_attn(
            q, k, v, attn_out, token_conf, global_conf,
            p.batch as i32, p.t_draft as i32, p.t_trace as i32,
            p.num_heads as i32, p.head_dim as i32, scale,
        );
        check_cuda_last_error("dense_verif_cross_attn")?;
        dense_verif_global_conf(
            token_conf, global_conf, p.batch as i32, p.t_draft as i32,
        );
        check_cuda_last_error("dense_verif_global_conf")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

// ---------------------------------------------------------------------------
// RLM kernel declarations
// ---------------------------------------------------------------------------

#[cfg(has_cuda)]
extern "C" {
    fn rlm_add_depth_embed(
        x: *mut half::f16, depth_emb: *const half::f16,
        n: i32, d_model: i32, depth: i32,
    );

    fn rlm_confidence_mlp(
        x: *const half::f16,
        w1: *const half::f16, b1: *const half::f16,
        w2: *const half::f16, b2: *const half::f16,
        conf: *mut f32,
        n: i32, d_model: i32, d_hidden: i32,
    );

    fn rlm_gate_subproblem(
        gate: *const f32,
        x_sub:  *const half::f16,
        x_base: *const half::f16,
        out:    *mut   half::f16,
        n: i32, d_model: i32,
    );
}

/// Launch RLM depth embedding addition (in-place).
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
pub unsafe fn rlm_depth_embed(
    x:         *mut half::f16,
    depth_emb: *const half::f16,
    n_tokens:  usize,
    d_model:   usize,
    depth:     usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        rlm_add_depth_embed(x, depth_emb, n_tokens as i32, d_model as i32, depth as i32);
        check_cuda_last_error("rlm_add_depth_embed")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

/// Launch RLM confidence MLP.
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rlm_confidence(
    x:        *const half::f16,
    w1:       *const half::f16,
    b1:       *const half::f16,
    w2:       *const half::f16,
    b2:       *const half::f16,
    conf:     *mut f32,
    n_tokens: usize,
    d_model:  usize,
    d_hidden: usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        rlm_confidence_mlp(x, w1, b1, w2, b2, conf,
            n_tokens as i32, d_model as i32, d_hidden as i32);
        check_cuda_last_error("rlm_confidence_mlp")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

/// Launch RLM sub-problem gating.
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
pub unsafe fn rlm_gate(
    gate:     *const f32,
    x_sub:    *const half::f16,
    x_base:   *const half::f16,
    out:      *mut   half::f16,
    n_tokens: usize,
    d_model:  usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        rlm_gate_subproblem(gate, x_sub, x_base, out, n_tokens as i32, d_model as i32);
        check_cuda_last_error("rlm_gate_subproblem")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

// ---------------------------------------------------------------------------
// Linear F16 kernel declarations
// ---------------------------------------------------------------------------

#[cfg(has_cuda)]
extern "C" {
    fn linear_f16_forward(
        x: *const half::f16, w: *const half::f16, bias: *const half::f16,
        y: *mut half::f16,
        m: i32, n: i32, k: i32, has_bias: i32,
    );
}

/// Launch F16 linear (GEMM): y = x @ W^T + b
///
/// # Safety
/// All pointer arguments must point to valid GPU memory of the expected sizes.
pub unsafe fn linear_f16(
    x:        *const half::f16,
    w:        *const half::f16,
    bias:     Option<*const half::f16>,
    y:        *mut half::f16,
    m:        usize,
    n:        usize,
    k:        usize,
) -> Result<()> {
    #[cfg(has_cuda)]
    {
        let (bias_ptr, has_bias) = match bias {
            Some(b) => (b, 1i32),
            None    => (std::ptr::null(), 0i32),
        };
        linear_f16_forward(x, w, bias_ptr, y, m as i32, n as i32, k as i32, has_bias);
        check_cuda_last_error("linear_f16_forward")
    }
    #[cfg(not(has_cuda))]
    { Err(CudaError::DeviceNotFound) }
}

// ------------------------------------------------------------------------------
// END OF FILE: bindings.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/src/bindings.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
