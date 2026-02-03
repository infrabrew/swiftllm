// PagedAttention CUDA Kernel
//
// Implements efficient attention computation with paged KV cache,
// enabling non-contiguous memory allocation and memory sharing.
//
// Based on the PagedAttention algorithm from vLLM.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Utility functions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// PagedAttention V1: Standard paged attention for decode
// Each thread block handles one query head
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ output,           // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ query,      // [num_seqs, num_heads, head_dim]
    const scalar_t* __restrict__ key_cache,  // [num_blocks, num_kv_heads, head_dim, block_size]
    const scalar_t* __restrict__ value_cache,// [num_blocks, num_kv_heads, head_dim, block_size]
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,    // [num_seqs]
    const float scale,
    const int num_kv_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head_idx = head_idx / (gridDim.y / num_kv_heads);
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for query, logits, and output
    extern __shared__ char smem[];
    float* shared_query = reinterpret_cast<float*>(smem);
    float* shared_logits = shared_query + HEAD_DIM;
    float* shared_output = shared_logits + BLOCK_SIZE;

    // Load query into shared memory
    const scalar_t* query_ptr = query + seq_idx * q_stride + head_idx * HEAD_DIM;
    for (int i = thread_idx; i < HEAD_DIM; i += NUM_THREADS) {
        shared_query[i] = static_cast<float>(query_ptr[i]);
    }
    __syncthreads();

    // Initialize output accumulator
    float output_acc[HEAD_DIM / WARP_SIZE] = {0.0f};
    float max_logit = -INFINITY;
    float sum_exp = 0.0f;

    // Process each block
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_idx = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        const int block_offset = block_idx * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - block_offset);

        // Compute attention scores for this block
        const scalar_t* key_block_ptr = key_cache +
            physical_block_idx * kv_block_stride +
            kv_head_idx * kv_head_stride;

        // Each thread computes dot product for one token in the block
        float logit = 0.0f;
        if (thread_idx < tokens_in_block) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                float k = static_cast<float>(key_block_ptr[d * BLOCK_SIZE + thread_idx]);
                logit += shared_query[d] * k;
            }
            logit *= scale;
        } else {
            logit = -INFINITY;
        }

        // Store logits in shared memory
        shared_logits[thread_idx] = logit;
        __syncthreads();

        // Compute softmax and accumulate output
        // (Simplified - full implementation would use online softmax)
        float block_max = warp_reduce_max(logit);
        if (lane_idx == 0) {
            atomicMax(reinterpret_cast<int*>(&max_logit), __float_as_int(block_max));
        }
        __syncthreads();

        // Accumulate weighted values
        const scalar_t* value_block_ptr = value_cache +
            physical_block_idx * kv_block_stride +
            kv_head_idx * kv_head_stride;

        if (thread_idx < tokens_in_block) {
            float weight = expf(logit - max_logit);
            sum_exp += weight;

            for (int d = warp_idx; d < HEAD_DIM; d += NUM_THREADS / WARP_SIZE) {
                float v = static_cast<float>(value_block_ptr[d * BLOCK_SIZE + thread_idx]);
                output_acc[d / (NUM_THREADS / WARP_SIZE)] += weight * v;
            }
        }
        __syncthreads();
    }

    // Normalize output and write
    scalar_t* output_ptr = output + seq_idx * q_stride + head_idx * HEAD_DIM;
    for (int d = thread_idx; d < HEAD_DIM; d += NUM_THREADS) {
        // Reduce across warps and normalize
        float val = 0.0f;  // Would aggregate output_acc here
        output_ptr[d] = static_cast<scalar_t>(val / sum_exp);
    }
}

// PagedAttention V2: Optimized for longer sequences
// Uses flash-attention style tiling
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_v2_kernel(
    scalar_t* __restrict__ output,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale,
    const int num_kv_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride
) {
    // V2 uses a two-pass algorithm for numerical stability
    // Pass 1: Compute partial softmax statistics
    // Pass 2: Normalize and compute output

    // Implementation would go here...
}

// Reshape and cache kernel
// Stores key/value into the cache at specified slots
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const scalar_t* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    scalar_t* __restrict__ key_cache,        // [num_blocks, num_kv_heads, head_dim, block_size]
    scalar_t* __restrict__ value_cache,      // [num_blocks, num_kv_heads, head_dim, block_size]
    const int* __restrict__ slot_mapping,    // [num_tokens]
    const int num_kv_heads,
    const int key_stride,
    const int value_stride,
    const int kv_block_stride,
    const int kv_head_stride
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;

    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / BLOCK_SIZE;
    const int block_offset = slot_idx % BLOCK_SIZE;

    // Copy key
    if (dim_idx < HEAD_DIM) {
        const scalar_t k = key[token_idx * key_stride + head_idx * HEAD_DIM + dim_idx];
        key_cache[block_idx * kv_block_stride + head_idx * kv_head_stride + dim_idx * BLOCK_SIZE + block_offset] = k;
    }

    // Copy value
    if (dim_idx < HEAD_DIM) {
        const scalar_t v = value[token_idx * value_stride + head_idx * HEAD_DIM + dim_idx];
        value_cache[block_idx * kv_block_stride + head_idx * kv_head_stride + dim_idx * BLOCK_SIZE + block_offset] = v;
    }
}

// Copy blocks kernel (for CoW)
__global__ void copy_blocks_kernel(
    const int64_t* __restrict__ block_mapping,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int kv_block_stride,
    const int kv_head_stride
) {
    const int pair_idx = blockIdx.x;
    const int src_block = static_cast<int>(block_mapping[2 * pair_idx]);
    const int dst_block = static_cast<int>(block_mapping[2 * pair_idx + 1]);

    const int head_idx = blockIdx.y;
    const int idx = threadIdx.x;

    // Copy key
    const int src_offset = src_block * kv_block_stride + head_idx * kv_head_stride + idx;
    const int dst_offset = dst_block * kv_block_stride + head_idx * kv_head_stride + idx;

    if (idx < head_dim * block_size) {
        key_cache[dst_offset] = key_cache[src_offset];
        value_cache[dst_offset] = value_cache[src_offset];
    }
}

// C interface for launching kernels
extern "C" {

void launch_paged_attention_v1(
    void* output,
    void* query,
    void* key_cache,
    void* value_cache,
    int* block_tables,
    int* context_lens,
    float scale,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_num_blocks_per_seq,
    cudaStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(256);

    // Launch appropriate kernel based on head_dim
    if (head_dim == 128) {
        paged_attention_v1_kernel<half, 128, 16, 256><<<grid, block, 0, stream>>>(
            static_cast<half*>(output),
            static_cast<const half*>(query),
            static_cast<const half*>(key_cache),
            static_cast<const half*>(value_cache),
            block_tables,
            context_lens,
            scale,
            num_kv_heads,
            max_num_blocks_per_seq,
            num_heads * head_dim,
            num_kv_heads * head_dim * block_size,
            head_dim * block_size
        );
    }
    // Add other head_dim cases...
}

void launch_reshape_and_cache(
    void* key,
    void* value,
    void* key_cache,
    void* value_cache,
    int* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream
) {
    dim3 grid(num_tokens, num_kv_heads);
    dim3 block(head_dim);

    if (head_dim == 128 && block_size == 16) {
        reshape_and_cache_kernel<half, 128, 16><<<grid, block, 0, stream>>>(
            static_cast<const half*>(key),
            static_cast<const half*>(value),
            static_cast<half*>(key_cache),
            static_cast<half*>(value_cache),
            slot_mapping,
            num_kv_heads,
            num_kv_heads * head_dim,
            num_kv_heads * head_dim,
            num_kv_heads * head_dim * block_size,
            head_dim * block_size
        );
    }
}

void launch_copy_blocks(
    int64_t* block_mapping,
    void* key_cache,
    void* value_cache,
    int num_pairs,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream
) {
    dim3 grid(num_pairs, num_kv_heads);
    dim3 block(head_dim * block_size);

    copy_blocks_kernel<<<grid, block, 0, stream>>>(
        block_mapping,
        static_cast<half*>(key_cache),
        static_cast<half*>(value_cache),
        num_kv_heads,
        head_dim,
        block_size,
        num_kv_heads * head_dim * block_size,
        head_dim * block_size
    );
}

} // extern "C"
