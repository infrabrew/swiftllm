#!/usr/bin/env python3
"""Batch processing example with SwiftLLM.

This example demonstrates efficient batch processing for high-throughput
inference scenarios like processing large datasets.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from swiftllm import LLM, SamplingParams


@dataclass
class ProcessingResult:
    """Result of processing a single item."""
    input_text: str
    output_text: str
    tokens_generated: int
    processing_time: float


def process_dataset(
    llm: LLM,
    data: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    prompt_template: str = "{text}",
    batch_size: int = 32,
) -> List[ProcessingResult]:
    """Process a dataset with batched inference.

    Args:
        llm: The LLM instance to use.
        data: List of items to process.
        sampling_params: Sampling parameters for generation.
        prompt_template: Template for formatting prompts.
        batch_size: Number of items per batch.

    Returns:
        List of processing results.
    """
    results = []
    total_items = len(data)

    print(f"Processing {total_items} items in batches of {batch_size}")

    for i in range(0, total_items, batch_size):
        batch = data[i:i + batch_size]
        batch_start = time.time()

        # Format prompts
        prompts = [prompt_template.format(**item) for item in batch]

        # Generate
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        batch_time = time.time() - batch_start
        time_per_item = batch_time / len(batch)

        # Collect results
        for j, (item, output) in enumerate(zip(batch, outputs)):
            results.append(ProcessingResult(
                input_text=item.get("text", str(item)),
                output_text=output.outputs[0].text,
                tokens_generated=len(output.outputs[0].token_ids),
                processing_time=time_per_item,
            ))

        # Progress report
        processed = min(i + batch_size, total_items)
        print(f"Processed {processed}/{total_items} items "
              f"({batch_time:.2f}s for batch, "
              f"{time_per_item*1000:.1f}ms/item)")

    return results


def summarization_pipeline():
    """Example: Text summarization pipeline."""
    print("=" * 60)
    print("Text Summarization Pipeline")
    print("=" * 60)

    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.3,  # Lower temperature for factual summarization
        max_tokens=100,
        top_p=0.9,
    )

    # Sample documents to summarize
    documents = [
        {
            "id": 1,
            "text": "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet. Pangrams are often used to display fonts and test keyboards."
        },
        {
            "id": 2,
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It focuses on developing algorithms that can access data and use it to learn for themselves."
        },
        {
            "id": 3,
            "text": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels."
        },
    ]

    template = "Summarize the following text in one sentence:\n\n{text}\n\nSummary:"

    results = process_dataset(
        llm,
        documents,
        sampling_params,
        prompt_template=template,
        batch_size=2,
    )

    print("\n--- Results ---")
    for doc, result in zip(documents, results):
        print(f"\nDocument {doc['id']}:")
        print(f"  Original: {doc['text'][:100]}...")
        print(f"  Summary: {result.output_text.strip()}")


def classification_pipeline():
    """Example: Text classification pipeline."""
    print("\n" + "=" * 60)
    print("Text Classification Pipeline")
    print("=" * 60)

    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for classification
        max_tokens=10,
    )

    # Sample texts to classify
    texts = [
        {"text": "I absolutely loved this movie! Best film of the year."},
        {"text": "The product broke after one day. Terrible quality."},
        {"text": "It was okay, nothing special but not bad either."},
        {"text": "Amazing service! Will definitely come back."},
        {"text": "Worst experience ever. Would not recommend."},
    ]

    template = """Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: {text}

Sentiment:"""

    results = process_dataset(
        llm,
        texts,
        sampling_params,
        prompt_template=template,
        batch_size=5,
    )

    print("\n--- Classification Results ---")
    for text, result in zip(texts, results):
        sentiment = result.output_text.strip().split()[0]  # Get first word
        print(f"  Text: \"{text['text'][:50]}...\"")
        print(f"  Sentiment: {sentiment}")
        print()


def question_answering_pipeline():
    """Example: Question answering over documents."""
    print("\n" + "=" * 60)
    print("Question Answering Pipeline")
    print("=" * 60)

    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50,
    )

    # Context and questions
    context = """
    SwiftLLM is a high-performance LLM inference engine built with Rust.
    It features PagedAttention for efficient memory management, continuous
    batching for high throughput, and supports tensor parallelism for
    multi-GPU inference. The engine is compatible with models from
    HuggingFace, GGUF, and SafeTensors formats.
    """

    questions = [
        {"question": "What language is SwiftLLM built with?"},
        {"question": "What memory management technique does SwiftLLM use?"},
        {"question": "What model formats does SwiftLLM support?"},
        {"question": "How does SwiftLLM achieve high throughput?"},
    ]

    template = f"""Context: {context}

Question: {{question}}

Answer:"""

    results = process_dataset(
        llm,
        questions,
        sampling_params,
        prompt_template=template,
        batch_size=4,
    )

    print("\n--- Q&A Results ---")
    for q, result in zip(questions, results):
        print(f"  Q: {q['question']}")
        print(f"  A: {result.output_text.strip()}")
        print()


def throughput_benchmark():
    """Benchmark throughput with varying batch sizes."""
    print("\n" + "=" * 60)
    print("Throughput Benchmark")
    print("=" * 60)

    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
    )

    # Generate dummy prompts
    num_prompts = 100
    prompts = [{"text": f"Question {i}: What is {i} + {i}?"} for i in range(num_prompts)]

    template = "{text}\nAnswer:"

    batch_sizes = [1, 4, 8, 16, 32]

    print(f"\nBenchmarking with {num_prompts} prompts...")

    for batch_size in batch_sizes:
        start = time.time()
        results = process_dataset(
            llm,
            prompts,
            sampling_params,
            prompt_template=template,
            batch_size=batch_size,
        )
        elapsed = time.time() - start

        total_tokens = sum(r.tokens_generated for r in results)
        print(f"\nBatch size {batch_size}:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {total_tokens/elapsed:.1f} tokens/s")
        print(f"  Latency: {elapsed/num_prompts*1000:.1f} ms/request")


def save_results_to_file():
    """Example: Save batch processing results to file."""
    print("\n" + "=" * 60)
    print("Save Results to File")
    print("=" * 60)

    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )

    prompts = [
        {"text": "Write a title for an article about AI:"},
        {"text": "Write a title for an article about climate:"},
        {"text": "Write a title for an article about space:"},
    ]

    results = process_dataset(llm, prompts, sampling_params, batch_size=3)

    # Save to JSON
    output_data = [
        {
            "input": r.input_text,
            "output": r.output_text.strip(),
            "tokens": r.tokens_generated,
            "time_ms": r.processing_time * 1000,
        }
        for r in results
    ]

    output_path = Path("batch_results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    """Run all batch processing examples."""
    summarization_pipeline()
    classification_pipeline()
    question_answering_pipeline()
    throughput_benchmark()
    save_results_to_file()


if __name__ == "__main__":
    main()
