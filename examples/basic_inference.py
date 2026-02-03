#!/usr/bin/env python3
"""Basic inference example with SwiftLLM.

This example demonstrates how to run simple inference with SwiftLLM.
"""

from swiftllm import LLM, SamplingParams


def main():
    # Initialize the LLM with a model
    # You can use a HuggingFace model ID or a local path
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        # Optional: Adjust for your GPU memory
        gpu_memory_utilization=0.90,
        # Optional: Use multiple GPUs
        tensor_parallel_size=1,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,      # Controls randomness (0 = deterministic)
        top_p=0.9,            # Nucleus sampling
        max_tokens=256,       # Maximum tokens to generate
        stop=["\n\n"],        # Stop sequences
    )

    # Single prompt generation
    prompts = ["What is the capital of France?"]
    outputs = llm.generate(prompts, sampling_params)

    print("=" * 60)
    print("Single Prompt Generation")
    print("=" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()

    # Batch generation - process multiple prompts efficiently
    batch_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a haiku about artificial intelligence:",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis:",
    ]

    print("=" * 60)
    print("Batch Generation")
    print("=" * 60)

    batch_outputs = llm.generate(batch_prompts, sampling_params)

    for i, output in enumerate(batch_outputs):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

    # Generation with different parameters per prompt
    print("\n" + "=" * 60)
    print("Different Parameters Per Prompt")
    print("=" * 60)

    varied_prompts = [
        "Tell me a joke:",
        "Explain gravity:",
    ]

    varied_params = [
        SamplingParams(temperature=0.9, max_tokens=100),  # More creative
        SamplingParams(temperature=0.3, max_tokens=200),  # More focused
    ]

    varied_outputs = llm.generate(varied_prompts, varied_params)

    for output, params in zip(varied_outputs, varied_params):
        print(f"\nPrompt: {output.prompt}")
        print(f"Temperature: {params.temperature}")
        print(f"Generated: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
