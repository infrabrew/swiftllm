// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      dense_verif_attn.cu
// PATH:      /crates/swiftllm-cuda/kernels/dense_verif_attn.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Dense Verification Cross-Attention Kernel
//
// Computes cross-attention where:
//   Q = draft token hidden states  [B, T_draft, num_heads, head_dim]
//   K = REPL trace hidden states   [B, T_trace, num_heads, head_dim]
//   V = REPL trace hidden states   [B, T_trace, num_heads, head_dim]
//
// Produces per-token confidence scores:
//   token_confidence : [B, T_draft]       — mean attention entropy per token
//   step_confidence  : [B, T_draft]       — geometric mean across heads
//   global_confidence: [B]                — scalar summary
//
// The cross-attention itself uses the standard scaled dot-product formula:
//   Attn(Q,K,V) = softmax(QK^T / sqrt(head_dim)) * V
//
// Confidence = 1 - H(attn_weights) / log(T_trace)
// where H = -sum(p * log(p + eps)) is the entropy of attention distribution.
// Low entropy → confident; high entropy → uncertain.
//
// Grid : [B, T_draft, num_heads]
// Block: [head_dim]   (≤256 for head_dim=64/128)
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <float.h>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// ---------------------------------------------------------------------------
// Dense verification cross-attention + confidence scoring
// ---------------------------------------------------------------------------

extern "C" __global__ void dense_verif_cross_attn(
    // Query: draft token hiddens
    const __half* __restrict__ Q,            // [B, T_q, H, D]
    // Key/Value: trace hiddens
    const __half* __restrict__ K,            // [B, T_kv, H, D]
    const __half* __restrict__ V,            // [B, T_kv, H, D]
    // Output
    __half* __restrict__ attn_out,           // [B, T_q, H, D]
    float*  __restrict__ token_conf,         // [B, T_q]    (atomicAdd from heads)
    float*  __restrict__ global_conf,        // [B]
    // Dims
    int B, int T_q, int T_kv, int num_heads, int head_dim,
    float scale                              // 1/sqrt(head_dim)
) {
    int b    = blockIdx.x;
    int t_q  = blockIdx.y;
    int h    = blockIdx.z;

    if (b >= B || t_q >= T_q || h >= num_heads) return;

    // Shared memory: attention logits [T_kv], then V-accumulator [head_dim]
    extern __shared__ float smem[];
    float* logits   = smem;                  // [T_kv]
    float* v_acc    = smem + T_kv;           // [head_dim]

    // --- Load Q for this (b, t_q, h) ---
    const __half* Qptr = Q + ((b * T_q + t_q) * num_heads + h) * head_dim;

    // --- Compute QK^T ---
    for (int t_kv = threadIdx.x; t_kv < T_kv; t_kv += blockDim.x) {
        const __half* Kptr = K + ((b * T_kv + t_kv) * num_heads + h) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(Qptr[d]) * __half2float(Kptr[d]);
        }
        logits[t_kv] = dot * scale;
    }
    __syncthreads();

    // --- Softmax (online, thread 0 does sequential pass for simplicity) ---
    if (threadIdx.x == 0) {
        float max_l = -FLT_MAX;
        for (int t = 0; t < T_kv; ++t) max_l = fmaxf(max_l, logits[t]);
        float sum_e = 0.f;
        for (int t = 0; t < T_kv; ++t) {
            logits[t] = expf(logits[t] - max_l);
            sum_e += logits[t];
        }
        float inv_sum = 1.f / fmaxf(sum_e, 1e-12f);
        for (int t = 0; t < T_kv; ++t) logits[t] *= inv_sum;
    }
    __syncthreads();

    // --- Compute attention entropy for confidence score ---
    // H = -sum(p * log(p + eps))
    if (threadIdx.x == 0) {
        float H = 0.f;
        for (int t = 0; t < T_kv; ++t) {
            float p = logits[t];
            H -= p * logf(p + 1e-10f);
        }
        float max_H = logf((float)T_kv + 1.f);
        float conf  = 1.f - H / max_H;     // 1 = perfectly confident
        atomicAdd(&token_conf[b * T_q + t_q], conf / (float)num_heads);
    }

    // --- Weighted sum over V ---
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) v_acc[d] = 0.f;
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int t_kv = 0; t_kv < T_kv; ++t_kv) {
            const __half* Vptr = V + ((b * T_kv + t_kv) * num_heads + h) * head_dim;
            acc += logits[t_kv] * __half2float(Vptr[d]);
        }
        __half* Optr = attn_out + ((b * T_q + t_q) * num_heads + h) * head_dim;
        Optr[d] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Reduce token_conf → global_conf  (mean over T_q per batch)
// Grid: [B]  Block: [min(T_q, 256)]
// ---------------------------------------------------------------------------

extern "C" __global__ void dense_verif_global_conf(
    const float* __restrict__ token_conf, // [B, T_q]
    float*       __restrict__ global_conf,// [B]
    int B, int T_q
) {
    int b = blockIdx.x;
    float sum = 0.f;
    for (int t = threadIdx.x; t < T_q; t += blockDim.x)
        sum += token_conf[b * T_q + t];
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0)
        global_conf[b] = sum / (float)T_q;
}

// ==============================================================================
// END OF FILE: dense_verif_attn.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
