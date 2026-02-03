#!/usr/bin/env python3
"""Multi-GPU inference example with SwiftLLM.

This example demonstrates how to use tensor parallelism for
running large models across multiple GPUs.
"""

import os
from swiftllm import LLM, SamplingParams


def check_gpu_availability():
    """Check available GPUs."""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"CUDA available: Yes")
            print(f"Number of GPUs: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
            return num_gpus
        else:
            print("CUDA available: No")
            return 0
    except ImportError:
        print("PyTorch not installed, cannot check GPU availability")
        return 0


def single_gpu_inference():
    """Run inference on a single GPU."""
    print("\n" + "=" * 60)
    print("Single GPU Inference")
    print("=" * 60)

    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        tensor_parallel_size=1,  # Single GPU
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )

    prompts = ["Explain tensor parallelism in one paragraph:"]
    outputs = llm.generate(prompts, sampling_params)

    print(f"Prompt: {prompts[0]}")
    print(f"Response: {outputs[0].outputs[0].text}")


def multi_gpu_tensor_parallel():
    """Run inference with tensor parallelism across multiple GPUs."""
    print("\n" + "=" * 60)
    print("Tensor Parallel Inference (Multi-GPU)")
    print("=" * 60)

    # Use 2 GPUs for tensor parallelism
    # This is useful for models that don't fit on a single GPU
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",  # Large model
        tensor_parallel_size=2,  # Split across 2 GPUs
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
    )

    prompts = [
        "What are the key differences between GPT and LLaMA architectures?",
        "Explain the attention mechanism in transformers:",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {output.outputs[0].text}")


def multi_gpu_with_memory_optimization():
    """Run inference with memory optimizations for large models."""
    print("\n" + "=" * 60)
    print("Multi-GPU with Memory Optimization")
    print("=" * 60)

    # For very large models, combine tensor parallelism with
    # aggressive memory optimization
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=4,  # 4 GPUs
        gpu_memory_utilization=0.95,  # Use more GPU memory
        swap_space=8.0,  # 8 GB swap space for overflow
        enable_prefix_caching=True,  # Cache common prefixes
    )

    # Use conservative sampling for long generations
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=500,
    )

    # Example: Long-form content generation
    prompt = """Write a detailed technical blog post about the following topic:

Topic: How PagedAttention Works in Modern LLM Inference Engines

Include sections on:
1. The memory problem in LLM inference
2. How PagedAttention solves it
3. Implementation details
4. Performance benefits

Blog post:"""

    outputs = llm.generate([prompt], sampling_params)

    print(f"Generated {len(outputs[0].outputs[0].token_ids)} tokens")
    print(f"\nGenerated content:\n{outputs[0].outputs[0].text}")


def benchmark_scaling():
    """Benchmark scaling across different numbers of GPUs."""
    print("\n" + "=" * 60)
    print("Multi-GPU Scaling Benchmark")
    print("=" * 60)

    import time

    model = "meta-llama/Llama-2-13b-hf"
    num_prompts = 10
    max_tokens = 100

    # Generate test prompts
    prompts = [f"Question {i}: Explain concept {i} in machine learning:" for i in range(num_prompts)]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)

    gpu_configs = [1, 2, 4]  # Test with 1, 2, and 4 GPUs

    results = []

    for num_gpus in gpu_configs:
        print(f"\nTesting with {num_gpus} GPU(s)...")

        try:
            llm = LLM(
                model=model,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.90,
            )

            # Warmup
            _ = llm.generate(prompts[:1], sampling_params, use_tqdm=False)

            # Benchmark
            start = time.time()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            elapsed = time.time() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed

            results.append({
                "gpus": num_gpus,
                "time": elapsed,
                "throughput": throughput,
            })

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.1f} tokens/s")

        except Exception as e:
            print(f"  Failed: {e}")

    # Print summary
    print("\n" + "-" * 40)
    print("Scaling Summary")
    print("-" * 40)
    if results:
        baseline = results[0]["throughput"]
        for r in results:
            speedup = r["throughput"] / baseline
            print(f"  {r['gpus']} GPU(s): {r['throughput']:.1f} tokens/s ({speedup:.2f}x)")


def environment_setup_guide():
    """Print guide for setting up multi-GPU environment."""
    print("\n" + "=" * 60)
    print("Multi-GPU Environment Setup Guide")
    print("=" * 60)

    print("""
1. Hardware Requirements:
   - Multiple NVIDIA GPUs (same model recommended)
   - NVLink for best performance (optional but recommended)
   - Sufficient host memory for model loading

2. Software Requirements:
   - CUDA 11.8 or higher
   - cuDNN 8.6 or higher
   - NCCL for multi-GPU communication

3. Environment Variables:
   # Specify which GPUs to use
   export CUDA_VISIBLE_DEVICES=0,1,2,3

   # For debugging
   export NCCL_DEBUG=INFO

   # For optimal performance
   export NCCL_P2P_DISABLE=0
   export NCCL_IB_DISABLE=0

4. Model Selection:
   - 7B models: 1 GPU (16GB)
   - 13B models: 1-2 GPUs
   - 30B models: 2-4 GPUs
   - 70B models: 4-8 GPUs

5. Memory Estimation:
   - FP16: ~2 bytes per parameter
   - INT8: ~1 byte per parameter
   - INT4: ~0.5 bytes per parameter

   Example for Llama-2-70B:
   - FP16: 70B * 2 = 140GB
   - With 4x A100 80GB: ~35GB per GPU (fits!)

6. Performance Tips:
   - Use power-of-2 tensor parallel sizes (1, 2, 4, 8)
   - Enable prefix caching for repeated prompts
   - Use quantization for memory-constrained setups
   - Balance batch size with available memory
""")


def main():
    """Run multi-GPU examples."""
    num_gpus = check_gpu_availability()

    if num_gpus == 0:
        print("\nNo GPUs available. Showing configuration examples only.")
        environment_setup_guide()
        return

    if num_gpus >= 1:
        single_gpu_inference()

    if num_gpus >= 2:
        # multi_gpu_tensor_parallel()
        print("\nMulti-GPU inference available with 2+ GPUs")

    if num_gpus >= 4:
        # multi_gpu_with_memory_optimization()
        # benchmark_scaling()
        print("\nAdvanced multi-GPU features available with 4+ GPUs")

    environment_setup_guide()


if __name__ == "__main__":
    main()
