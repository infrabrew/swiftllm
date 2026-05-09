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

__global__ void linear_f16_forward_kernel(
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

    // TILE_K stride for shared memory: use TILE_N as the loading width
    // since block is [TILE_N, TILE_M]. Each thread loads 2 elements along K.
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    float acc = 0.f;

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load A tile: each thread loads TILE_K/TILE_N elements along K
        #pragma unroll
        for (int step = 0; step < TILE_K; step += TILE_N) {
            int k_a = k_start + step + threadIdx.x;
            if (k_a < K)
                As[threadIdx.y][step + threadIdx.x] = __half2float(x[row * K + k_a]);
            else
                As[threadIdx.y][step + threadIdx.x] = 0.f;
        }

        // Load B tile: each thread loads TILE_K/TILE_M elements along K
        #pragma unroll
        for (int step = 0; step < TILE_K; step += TILE_M) {
            int k_b = k_start + step + threadIdx.y;
            if (k_b < K)
                Bs[threadIdx.x][step + threadIdx.y] = __half2float(W[col * K + k_b]);
            else
                Bs[threadIdx.x][step + threadIdx.y] = 0.f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k)
            acc += As[threadIdx.y][k] * Bs[threadIdx.x][k];

        __syncthreads();
    }

    if (has_bias) acc += __half2float(bias[col]);
    y[row * N + col] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Host-side launcher for linear_f16_forward_kernel
// ---------------------------------------------------------------------------

extern "C" void linear_f16_forward(
    const __half* x, const __half* W, const __half* bias,
    __half* y, int M, int N, int K, int has_bias
) {
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(TILE_N, TILE_M);
    linear_f16_forward_kernel<<<grid, block>>>(x, W, bias, y, M, N, K, has_bias);
}

// ---------------------------------------------------------------------------
// Fused bias add: y += b  (in-place, broadcast over M)
// Grid: [ceil(M*N / 256)]  Block: [256]
// ---------------------------------------------------------------------------

__global__ void linear_f16_bias_add_kernel(
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

// ---------------------------------------------------------------------------
// Host-side launcher for linear_f16_bias_add_kernel
// ---------------------------------------------------------------------------

extern "C" void linear_f16_bias_add(
    __half* y, const __half* bias, int M, int N
) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    linear_f16_bias_add_kernel<<<blocks, threads>>>(y, bias, M, N);
}

// ==============================================================================
// END OF FILE: linear_f16.cu
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ==============================================================================
