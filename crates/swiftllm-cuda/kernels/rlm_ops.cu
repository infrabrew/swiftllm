// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      rlm_ops.cu
// PATH:      /crates/swiftllm-cuda/kernels/rlm_ops.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Recursive Language Model (RLM) GPU Kernels
//
// Entry points:
//   rlm_add_depth_embed  — add depth embedding to hidden states
//   rlm_confidence_mlp   — 2-layer MLP → scalar confidence per token
//   rlm_gate_subproblem  — sigmoid gate that blends base and sub-problem hiddens
//
// The RLM layer operates as:
//   1. x ← x + depth_embedding[depth]         (rlm_add_depth_embed)
//   2. conf ← sigmoid(W2 * relu(W1 * x + b1) + b2)  (rlm_confidence_mlp)
//   3. x_out ← gate * x_sub + (1-gate) * x_base      (rlm_gate_subproblem)
//
// All operations in F16; confidence output in F32.
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// 1. Add depth embedding
//    x : [B, T, d_model]   (in-place update)
//    depth_emb : [max_depth, d_model]
//    depth : scalar (current recursion depth)
// Grid: [B * T]  Block: [min(d_model, 512)]
// ---------------------------------------------------------------------------

__global__ void rlm_add_depth_embed_kernel(
    __half*       __restrict__ x,           // [B, T, d_model] — modified in place
    const __half* __restrict__ depth_emb,   // [max_depth, d_model]
    int N,                                   // B * T (total tokens)
    int d_model,
    int depth
) {
    int n = blockIdx.x;
    if (n >= N) return;

    const __half* emb = depth_emb + depth * d_model;
    __half* xn = x + n * d_model;

    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        xn[d] = __float2half(__half2float(xn[d]) + __half2float(emb[d]));
    }
}

// ---------------------------------------------------------------------------
// 2. Confidence MLP
//    W1: [d_model, d_hidden]   b1: [d_hidden]
//    W2: [d_hidden, 1]         b2: [1]
//    x:  [N, d_model]
//    conf: [N]   (F32 sigmoid output)
// Grid: [N]  Block: [d_hidden]
// ---------------------------------------------------------------------------

__global__ void rlm_confidence_mlp_kernel(
    const __half* __restrict__ x,    // [N, d_model]
    const __half* __restrict__ W1,   // [d_model, d_hidden]
    const __half* __restrict__ b1,   // [d_hidden]
    const __half* __restrict__ W2,   // [d_hidden, 1]
    const __half* __restrict__ b2,   // [1]
    float*        __restrict__ conf, // [N]
    int N, int d_model, int d_hidden
) {
    int n = blockIdx.x;
    if (n >= N) return;

    extern __shared__ float h1[];   // [d_hidden]

    // Layer 1: h1 = relu(x @ W1 + b1)
    for (int j = threadIdx.x; j < d_hidden; j += blockDim.x) {
        float acc = __half2float(b1[j]);
        for (int k = 0; k < d_model; ++k) {
            acc += __half2float(x[n * d_model + k]) * __half2float(W1[k * d_hidden + j]);
        }
        h1[j] = fmaxf(0.f, acc);    // ReLU
    }
    __syncthreads();

    // Layer 2: scalar = sum(h1 * W2) + b2  → sigmoid
    if (threadIdx.x == 0) {
        float logit = __half2float(b2[0]);
        for (int j = 0; j < d_hidden; ++j) {
            logit += h1[j] * __half2float(W2[j]);
        }
        conf[n] = 1.f / (1.f + expf(-logit));  // sigmoid
    }
}

// ---------------------------------------------------------------------------
// 3. Gate sub-problem hiddens
//    gate:    [N]     (scalar per token, from var-binding attention)
//    x_sub:  [N, d_model]
//    x_base: [N, d_model]
//    out:    [N, d_model] = gate * x_sub + (1-gate) * x_base
// Grid: [N]  Block: [min(d_model, 512)]
// ---------------------------------------------------------------------------

__global__ void rlm_gate_subproblem_kernel(
    const float*  __restrict__ gate,    // [N]
    const __half* __restrict__ x_sub,   // [N, d_model]
    const __half* __restrict__ x_base,  // [N, d_model]
    __half*       __restrict__ out,     // [N, d_model]
    int N, int d_model
) {
    int n = blockIdx.x;
    if (n >= N) return;

    float g = gate[n];
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float xs = __half2float(x_sub [n * d_model + d]);
        float xb = __half2float(x_base[n * d_model + d]);
        out[n * d_model + d] = __float2half(g * xs + (1.f - g) * xb);
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

extern "C" void rlm_add_depth_embed(
    __half* x, const __half* depth_emb,
    int N, int d_model, int depth
) {
    int threads = min(d_model, 512);
    rlm_add_depth_embed_kernel<<<N, threads>>>(x, depth_emb, N, d_model, depth);
}

extern "C" void rlm_confidence_mlp(
    const __half* x, const __half* W1, const __half* b1,
    const __half* W2, const __half* b2, float* conf,
    int N, int d_model, int d_hidden
) {
    int threads = min(d_hidden, 256);
    size_t smem = d_hidden * sizeof(float);
    rlm_confidence_mlp_kernel<<<N, threads, smem>>>(
        x, W1, b1, W2, b2, conf, N, d_model, d_hidden
    );
}

extern "C" void rlm_gate_subproblem(
    const float* gate, const __half* x_sub, const __half* x_base,
    __half* out, int N, int d_model
) {
    int threads = min(d_model, 512);
    rlm_gate_subproblem_kernel<<<N, threads>>>(gate, x_sub, x_base, out, N, d_model);
}

// ==============================================================================
// END OF FILE: rlm_ops.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
