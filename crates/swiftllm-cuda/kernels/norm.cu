// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      norm.cu
// PATH:      /crates/swiftllm-cuda/kernels/norm.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// RMS normalization kernel (F16 in/out, F32 reduction).
//
//   out[row, i] = x[row, i] * rsqrt(mean(x[row, :]^2) + eps) * (weight[i] + offset)
//
// `offset` = 0 gives standard RMSNorm; `offset` = 1 gives Gemma's unit-offset
// RMSNorm (weight applied as 1 + w). One block per row; F32 shared-memory
// reduction. blockDim.x must be a power of two.
//
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void rmsnorm_kernel(
    const __half* __restrict__ x,       // [rows, dim]
    const __half* __restrict__ weight,  // [dim]
    __half*       __restrict__ out,     // [rows, dim]
    int rows, int dim, float eps, float weight_offset
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const __half* xr = x + (size_t)row * dim;
    __half* yr = out + (size_t)row * dim;

    // 1. Sum of squares (strided over dim, F32 accumulation).
    float local = 0.f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(xr[i]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    // 2. Tree reduction (blockDim.x is a power of two).
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)dim + eps);

    // 3. Normalize and scale.
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float w = __half2float(weight[i]) + weight_offset;
        yr[i] = __float2half(v * inv_rms * w);
    }
}

extern "C" void rmsnorm_f16_launch(
    const __half* x, const __half* weight, __half* out,
    int rows, int dim, float eps, float weight_offset
) {
    int threads = 256; // power of two; reduction handles dim != threads via striding
    size_t shmem = (size_t)threads * sizeof(float);
    rmsnorm_kernel<<<rows, threads, shmem>>>(x, weight, out, rows, dim, eps, weight_offset);
}

// ==============================================================================
// END OF FILE: norm.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
