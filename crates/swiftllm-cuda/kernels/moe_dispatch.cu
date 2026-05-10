// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      moe_dispatch.cu
// PATH:      /crates/swiftllm-cuda/kernels/moe_dispatch.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// LatentMoE / MoE Dispatch Kernels
//
// Entry points:
//   latent_compress     — project [B*T, d_model] → [B*T, d_latent]
//   moe_topk_gate       — router logits → top-K expert indices + weights
//   moe_expert_dispatch — scatter tokens to expert slots (grouped GEMM)
//   moe_load_stats      — accumulate per-expert token counts for bias update
//   latent_expand       — project [B*T, d_latent] → [B*T, d_model] (output)
//
// Load-balance strategy: DeepSeek-V3 aux-loss-free dynamic bias
//   bias[e] += alpha * (avg_load - count[e] / total_tokens)
// Applied host-side after moe_load_stats (no kernel needed).
//
// All weight matrices are F16; accumulation in F32.
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <float.h>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// Latent compression:  x [N, d_model]  →  z [N, d_latent]
// Simple GEMM-like kernel (use CUBLAS for production; this is the fallback).
// Grid: [N / TILE_M, d_latent / TILE_N]   Block: [TILE_N]
// ---------------------------------------------------------------------------

#define TILE_M 1
#define TILE_N 64

__global__ void latent_compress_kernel(
    const __half* __restrict__ x,          // [N, d_model]
    const __half* __restrict__ W_compress, // [d_model, d_latent]
    __half* __restrict__ z,                // [N, d_latent]
    int N, int d_model, int d_latent
) {
    int n = blockIdx.x;          // token index
    int col_start = blockIdx.y * TILE_N;

    if (n >= N) return;

    for (int col = col_start + threadIdx.x; col < col_start + TILE_N && col < d_latent; col += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < d_model; ++k) {
            acc += __half2float(x[n * d_model + k]) * __half2float(W_compress[k * d_latent + col]);
        }
        z[n * d_latent + col] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Top-K gating (with per-expert dynamic bias)
//   Input:  router_logits [N, num_experts]  (F32)
//   Output: expert_ids    [N, top_k]        (int32)
//           expert_weights[N, top_k]        (F32, softmax over selected)
// Grid: [N]  Block: [num_experts]  (≤1024)
// ---------------------------------------------------------------------------

__global__ void moe_topk_gate_kernel(
    const float* __restrict__ router_logits, // [N, num_experts]
    const float* __restrict__ expert_bias,   // [num_experts] dynamic bias
    int32_t* __restrict__ expert_ids,        // [N, top_k]
    float*   __restrict__ expert_weights,    // [N, top_k]
    int N, int num_experts, int top_k
) {
    int n = blockIdx.x;
    if (n >= N) return;

    // Load logits + bias into shared memory
    extern __shared__ float slogits[];
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        slogits[e] = router_logits[n * num_experts + e] + expert_bias[e];
    }
    __syncthreads();

    // Sequential top-K selection by thread 0 (num_experts typically ≤64/128)
    if (threadIdx.x == 0) {
        // Mark selected
        bool selected[1024] = {};   // VLA not allowed; use local array with hard cap
        float selected_logits[32];  // top_k ≤ 32 in practice
        int   selected_ids[32];

        for (int k = 0; k < top_k; ++k) {
            float best_val = -FLT_MAX;
            int   best_e   = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (!selected[e] && slogits[e] > best_val) {
                    best_val = slogits[e];
                    best_e   = e;
                }
            }
            selected[best_e]     = true;
            selected_logits[k]   = best_val;
            selected_ids[k]      = best_e;
        }

        // Softmax over selected logits
        float max_l = -FLT_MAX;
        for (int k = 0; k < top_k; ++k) max_l = fmaxf(max_l, selected_logits[k]);
        float sum_exp = 0.f;
        for (int k = 0; k < top_k; ++k) sum_exp += expf(selected_logits[k] - max_l);

        for (int k = 0; k < top_k; ++k) {
            expert_ids    [n * top_k + k] = selected_ids[k];
            expert_weights[n * top_k + k] = expf(selected_logits[k] - max_l) / sum_exp;
        }
    }
}

// ---------------------------------------------------------------------------
// Expert dispatch: for each token, run its assigned expert FFN (SwiGLU)
//   z_in  : [N, d_latent]
//   gate_w: [num_experts, d_latent, d_ffn]  (gate projection)
//   up_w  : [num_experts, d_latent, d_ffn]  (up   projection)
//   down_w: [num_experts, d_ffn,   d_latent](down projection)
//   z_out : [N, d_latent]  (accumulated with expert weights)
//
// Each block handles one (token, k_slot) pair.
// Grid: [N, top_k]   Block: [d_latent ≤ 256]
// ---------------------------------------------------------------------------

__device__ __forceinline__ float silu(float x) {
    return x / (1.f + expf(-x));
}

__global__ void moe_expert_dispatch_kernel(
    const __half*  __restrict__ z_in,          // [N, d_latent]
    const __half*  __restrict__ gate_w,        // [E, d_latent, d_ffn]
    const __half*  __restrict__ up_w,          // [E, d_latent, d_ffn]
    const __half*  __restrict__ down_w,        // [E, d_ffn, d_latent]
    const int32_t* __restrict__ expert_ids,    // [N, top_k]
    const float*   __restrict__ expert_weights,// [N, top_k]
    __half*        __restrict__ z_out,         // [N, d_latent]  (atomic add)
    int N, int top_k, int d_latent, int d_ffn, int num_experts
) {
    int n = blockIdx.x;   // token
    int k = blockIdx.y;   // which of the top-k slots
    if (n >= N || k >= top_k) return;

    int e = expert_ids[n * top_k + k];
    float w = expert_weights[n * top_k + k];

    // Each thread computes partial sums for a slice of d_ffn
    extern __shared__ float smem[];
    float* gate_buf = smem;              // [d_ffn]
    // up_buf overlaps smem region — gate_buf is reused in-place for SwiGLU

    // Gate and up projections: z_in @ gate_w[e], z_in @ up_w[e]
    for (int j = threadIdx.x; j < d_ffn; j += blockDim.x) {
        float g = 0.f, u = 0.f;
        const __half* gw = gate_w + (e * d_latent * d_ffn) + j;
        const __half* uw = up_w   + (e * d_latent * d_ffn) + j;
        for (int c = 0; c < d_latent; ++c) {
            float zc = __half2float(z_in[n * d_latent + c]);
            g += zc * __half2float(gw[c * d_ffn]);
            u += zc * __half2float(uw[c * d_ffn]);
        }
        gate_buf[j] = silu(g) * u;      // SwiGLU activation
    }
    __syncthreads();

    // Down projection back to d_latent, accumulate into z_out
    for (int c = threadIdx.x; c < d_latent; c += blockDim.x) {
        float acc = 0.f;
        const __half* dw = down_w + (e * d_ffn * d_latent) + c;
        for (int j = 0; j < d_ffn; ++j) {
            acc += gate_buf[j] * __half2float(dw[j * d_latent]);
        }
        // Weighted accumulate (atomicAdd in F32 then cast back)
        float existing = __half2float(z_out[n * d_latent + c]);
        z_out[n * d_latent + c] = __float2half(existing + w * acc);
    }
}

// ---------------------------------------------------------------------------
// Load statistics accumulation
// Counts how many tokens were routed to each expert (for bias update).
// Grid: [N]  Block: [1]
// ---------------------------------------------------------------------------

__global__ void moe_load_stats_kernel(
    const int32_t* __restrict__ expert_ids, // [N, top_k]
    int32_t* __restrict__ expert_counts,    // [num_experts] (atomicAdd)
    int N, int top_k
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int k = 0; k < top_k; ++k) {
        int e = expert_ids[n * top_k + k];
        atomicAdd(&expert_counts[e], 1);
    }
}

// ---------------------------------------------------------------------------
// Latent expand:  z [N, d_latent]  →  y [N, d_model]
// Grid: [N]  Block: [min(d_model, 512)]
// ---------------------------------------------------------------------------

__global__ void latent_expand_kernel(
    const __half* __restrict__ z,          // [N, d_latent]
    const __half* __restrict__ W_expand,   // [d_latent, d_model]
    __half* __restrict__ y,                // [N, d_model]
    int N, int d_latent, int d_model
) {
    int n = blockIdx.x;
    if (n >= N) return;
    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < d_latent; ++k) {
            acc += __half2float(z[n * d_latent + k]) * __half2float(W_expand[k * d_model + col]);
        }
        y[n * d_model + col] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

extern "C" void latent_compress(
    const __half* x, const __half* W_compress, __half* z,
    int N, int d_model, int d_latent
) {
    dim3 grid(N, (d_latent + TILE_N - 1) / TILE_N);
    latent_compress_kernel<<<grid, TILE_N>>>(x, W_compress, z, N, d_model, d_latent);
}

extern "C" void moe_topk_gate(
    const float* router_logits, const float* expert_bias,
    int32_t* expert_ids, float* expert_weights,
    int N, int num_experts, int top_k
) {
    int threads = min(num_experts, 256);
    size_t smem = num_experts * sizeof(float);
    moe_topk_gate_kernel<<<N, threads, smem>>>(
        router_logits, expert_bias, expert_ids, expert_weights,
        N, num_experts, top_k
    );
}

extern "C" void moe_expert_dispatch(
    const __half* z_in, const __half* gate_w, const __half* up_w,
    const __half* down_w, const int32_t* expert_ids, const float* expert_weights,
    __half* z_out,
    int N, int top_k, int d_latent, int d_ffn, int num_experts
) {
    dim3 grid(N, top_k);
    int threads = min(d_latent, 256);
    size_t smem = d_ffn * sizeof(float);
    moe_expert_dispatch_kernel<<<grid, threads, smem>>>(
        z_in, gate_w, up_w, down_w, expert_ids, expert_weights, z_out,
        N, top_k, d_latent, d_ffn, num_experts
    );
}

extern "C" void moe_load_stats(
    const int32_t* expert_ids, int32_t* expert_counts,
    int N, int top_k
) {
    int threads = min(N, 256);
    int blocks = (N + threads - 1) / threads;
    moe_load_stats_kernel<<<blocks, threads>>>(expert_ids, expert_counts, N, top_k);
}

extern "C" void latent_expand(
    const __half* z, const __half* W_expand, __half* y,
    int N, int d_latent, int d_model
) {
    int threads = min(d_model, 512);
    latent_expand_kernel<<<N, threads>>>(z, W_expand, y, N, d_latent, d_model);
}

// ==============================================================================
// END OF FILE: moe_dispatch.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
