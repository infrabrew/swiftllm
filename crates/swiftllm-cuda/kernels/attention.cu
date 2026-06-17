// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      attention.cu
// PATH:      /crates/swiftllm-cuda/kernels/attention.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Multi-head scaled-dot-product attention (F16 in/out, F32 compute).
//
//   out[i,h,:] = sum_j softmax_j( (Q[i,h,:]·K[j,h,:]) * scale ) * V[j,h,:]
//
// One thread per (query position, head). This is a correctness-first reference
// kernel (two-pass softmax with a local score buffer); production should use a
// fused / flash-attention kernel. Sequence length is capped at ATTN_MAX_SEQ.
//
// Layout: Q, K, V, out are [seq_len, num_heads * head_dim], row-major.
// `causal != 0` restricts each query i to keys j <= i.
//
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define ATTN_MAX_SEQ 256

__global__ void attention_kernel(
    const __half* __restrict__ q,   // [S, H*D]
    const __half* __restrict__ k,   // [S, H*D]
    const __half* __restrict__ v,   // [S, H*D]
    __half*       __restrict__ out, // [S, H*D]
    int S, int H, int D, float scale, int causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H;
    if (idx >= total) return;

    int i = idx / H;  // query position
    int h = idx % H;  // head
    int hd = H * D;
    const __half* qi = q + (size_t)i * hd + (size_t)h * D;

    int jmax = causal ? (i + 1) : S;
    if (jmax > ATTN_MAX_SEQ) jmax = ATTN_MAX_SEQ;

    float scores[ATTN_MAX_SEQ];

    // 1. Scores and running max.
    float maxv = -1e30f;
    for (int j = 0; j < jmax; ++j) {
        const __half* kj = k + (size_t)j * hd + (size_t)h * D;
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += __half2float(qi[d]) * __half2float(kj[d]);
        s *= scale;
        scores[j] = s;
        if (s > maxv) maxv = s;
    }

    // 2. Softmax (subtract max for stability).
    float denom = 0.f;
    for (int j = 0; j < jmax; ++j) {
        scores[j] = expf(scores[j] - maxv);
        denom += scores[j];
    }
    float inv_denom = denom > 0.f ? 1.f / denom : 0.f;

    // 3. Weighted sum of V.
    __half* oi = out + (size_t)i * hd + (size_t)h * D;
    for (int d = 0; d < D; ++d) {
        float acc = 0.f;
        for (int j = 0; j < jmax; ++j) {
            const __half* vj = v + (size_t)j * hd + (size_t)h * D;
            acc += scores[j] * __half2float(vj[d]);
        }
        oi[d] = __float2half(acc * inv_denom);
    }
}

extern "C" void attention_f16_launch(
    const __half* q, const __half* k, const __half* v, __half* out,
    int seq_len, int num_heads, int head_dim, float scale, int causal
) {
    int total = seq_len * num_heads;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;
    attention_kernel<<<blocks, threads>>>(q, k, v, out, seq_len, num_heads, head_dim, scale, causal);
}

// ==============================================================================
// END OF FILE: attention.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
