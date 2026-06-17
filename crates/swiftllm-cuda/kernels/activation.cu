// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      activation.cu
// PATH:      /crates/swiftllm-cuda/kernels/activation.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Element-wise activation kernels (F16 in/out, F32 compute).
//
// Entry points:
//   geglu_f16   — out = gelu_tanh(gate) * up   (Gemma-style gated FFN activation)
//   silu_mul_f16 — out = silu(gate) * up        (SwiGLU / LLaMA gated activation)
//
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Tanh-approximation GELU (matches swiftllm-models Gemma and forward.rs).
__device__ __forceinline__ float gelu_tanh(float x) {
    const float kSqrt2OverPi = 0.7978845608f; // sqrt(2/pi)
    float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(inner));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.f + expf(-x));
}

// ---------------------------------------------------------------------------
// GeGLU: out[i] = gelu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
__global__ void geglu_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    out[i] = __float2half(gelu_tanh(g) * u);
}

extern "C" void geglu_f16_launch(const __half* gate, const __half* up, __half* out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    geglu_kernel<<<blocks, threads>>>(gate, up, out, n);
}

// ---------------------------------------------------------------------------
// SwiGLU: out[i] = silu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
__global__ void silu_mul_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    out[i] = __float2half(silu(g) * u);
}

extern "C" void silu_mul_f16_launch(const __half* gate, const __half* up, __half* out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads>>>(gate, up, out, n);
}

// ==============================================================================
// END OF FILE: activation.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
