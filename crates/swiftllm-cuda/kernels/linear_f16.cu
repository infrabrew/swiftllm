// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      linear_f16.cu
// PATH:      /crates/swiftllm-cuda/kernels/linear_f16.cu
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// F16 Linear (GEMM) Kernels
//
// Entry points:
//   linear_f16_forward   — y = x @ W^T + b   (row-major, F16 in/out, F32 accum)
//   linear_f16_bias_add  — y += b             (fused bias add, avoids extra kernel)
//
// For large M/N/K, production code should call cuBLAS cublasGemmEx with
// CUBLAS_COMPUTE_32F_FAST_16F.  These kernels are the fallback / reference
// implementation that matches the Python bridge numerics exactly.
//
// Tiling:  TILE_M × TILE_N output tile, each block owns one tile.
//          TILE_K = 32 for register accumulation.
//
// Shape conventions:
//   x : [M, K]   (input, row-major)
//   W : [N, K]   (weight, row-major — transposed in the inner loop)
//   b : [N]      (optional bias)
//   y : [M, N]   (output)
//
// Licensed under the Apache License, Version 2.0
// ==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// ---------------------------------------------------------------------------
// Tiled GEMM: y = x @ W^T  (F16 in, F32 accum, F16 out)
// Grid : [ceil(M/TILE_M), ceil(N/TILE_N)]
// Block: [TILE_N, TILE_M]
// ---------------------------------------------------------------------------

extern "C" __global__ void linear_f16_forward(
    const __half* __restrict__ x,    // [M, K]
    const __half* __restrict__ W,    // [N, K]
    const __half* __restrict__ bias, // [N]  (may be nullptr)
    __half*       __restrict__ y,    // [M, N]
    int M, int N, int K,
    int has_bias
) {
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int row = tile_m * TILE_M + threadIdx.y;
    int col = tile_n * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    float acc = 0.f;

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load A tile (x[row, k_start..k_start+TILE_K])
        int k_a = k_start + threadIdx.x;
        if (k_a < K)
            As[threadIdx.y][threadIdx.x] = __half2float(x[row * K + k_a]);
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        // Load B tile (W[col, k_start..k_start+TILE_K])
        int k_b = k_start + threadIdx.y;
        if (k_b < K)
            Bs[threadIdx.x][threadIdx.y] = __half2float(W[col * K + k_b]);
        else
            Bs[threadIdx.x][threadIdx.y] = 0.f;

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k)
            acc += As[threadIdx.y][k] * Bs[threadIdx.x][k];

        __syncthreads();
    }

    if (has_bias) acc += __half2float(bias[col]);
    y[row * N + col] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Fused bias add: y += b  (in-place, broadcast over M)
// Grid: [ceil(M*N / 256)]  Block: [256]
// ---------------------------------------------------------------------------

extern "C" __global__ void linear_f16_bias_add(
    __half*       __restrict__ y,    // [M, N]
    const __half* __restrict__ bias, // [N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int col = idx % N;
    y[idx] = __float2half(__half2float(y[idx]) + __half2float(bias[col]));
}

// ==============================================================================
// END OF FILE: linear_f16.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
