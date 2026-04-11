#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      training.py
# PATH:      /examples/training.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Full training example using SwiftLLM.

This example demonstrates full-parameter fine-tuning and
how to use the training API programmatically with callbacks
and config management.

Usage:
    python examples/training.py
"""

from swiftllm.training import (
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    FineTuningMethod,
    LrScheduler,
    MixedPrecision,
)


def example_full_finetuning():
    """Full-parameter fine-tuning (no LoRA)."""
    print("=== Full Fine-Tuning ===\n")

    config = TrainingConfig(
        model="meta-llama/Llama-2-7b-hf",
        train_data="./data/train.jsonl",
        eval_data="./data/eval.jsonl",
        output_dir="./output/full-ft",
        fine_tuning_method=FineTuningMethod.FULL,
        num_epochs=2,
        per_device_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler=LrScheduler.COSINE,
        mixed_precision=MixedPrecision.BF16,
        max_seq_len=2048,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
    )

    trainer = Trainer(config)
    trainer.train()

    # Access final metrics
    m = trainer.metrics
    print(f"\nFinal: loss={m.train_loss:.4f}, ppl={m.perplexity:.2f}")


def example_config_management():
    """Demonstrate saving and loading training configs."""
    print("=== Config Management ===\n")

    config = TrainingConfig(
        model="mistralai/Mistral-7B-v0.1",
        train_data="./data/train.jsonl",
        output_dir="./output/config-demo",
        learning_rate=1e-4,
        num_epochs=5,
    )

    # Save
    config.save("./output/config-demo/config.json")
    print("Config saved to ./output/config-demo/config.json")
    print(f"Config dict: {config.to_dict()}\n")

    # Load
    loaded = TrainingConfig.load("./output/config-demo/config.json")
    print(f"Loaded model: {loaded.model}")
    print(f"Loaded lr: {loaded.learning_rate}")
    print(f"Loaded epochs: {loaded.num_epochs}")


def example_with_callbacks():
    """Training with custom callbacks for monitoring."""
    print("=== Training with Callbacks ===\n")

    config = TrainingConfig(
        model="meta-llama/Llama-2-7b-hf",
        train_data="./data/train.jsonl",
        output_dir="./output/callbacks-demo",
        num_epochs=1,
        logging_steps=5,
    )

    trainer = Trainer(config)

    # Track best loss
    best_loss = float("inf")

    def monitor(metrics: TrainingMetrics):
        nonlocal best_loss
        if metrics.train_loss < best_loss:
            best_loss = metrics.train_loss
            print(f"  * New best loss at step {metrics.step}: {best_loss:.4f}")

    trainer.add_callback(monitor)
    trainer.train()

    print(f"\nBest training loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwiftLLM Training Examples")
    parser.add_argument(
        "--example",
        choices=["full", "config", "callbacks", "all"],
        default="callbacks",
        help="Which example to run (default: callbacks)",
    )
    args = parser.parse_args()

    if args.example in ("full", "all"):
        example_full_finetuning()
    if args.example in ("config", "all"):
        example_config_management()
    if args.example in ("callbacks", "all"):
        example_with_callbacks()

# ------------------------------------------------------------------------------
# END OF FILE: training.py
# REPO PATH:   /swiftllm/examples/training.py
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
