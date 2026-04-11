#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      fine_tuning.py
# PATH:      /examples/fine_tuning.py
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

"""Fine-tuning example using SwiftLLM with LoRA.

This example demonstrates how to fine-tune a model using LoRA adapters,
which is memory-efficient and fast.

Usage:
    python examples/fine_tuning.py

    # Or with custom settings:
    python examples/fine_tuning.py --model meta-llama/Llama-2-7b-hf --data data.jsonl
"""

import argparse
from swiftllm.training import (
    Trainer,
    TrainingConfig,
    LoRAConfig,
    FineTuningMethod,
    LrScheduler,
    fine_tune,
)


def example_lora_basic():
    """Basic LoRA fine-tuning with the convenience function."""
    print("=== Basic LoRA Fine-Tuning ===\n")

    trainer = fine_tune(
        model="meta-llama/Llama-2-7b-hf",
        train_data="./data/train.jsonl",
        output_dir="./output/lora-basic",
        lora_r=16,
        lora_alpha=32.0,
        learning_rate=2e-4,
        num_epochs=1,
    )

    print(f"\nFinal metrics: loss={trainer.metrics.train_loss:.4f}")


def example_lora_advanced():
    """Advanced LoRA fine-tuning with full configuration."""
    print("=== Advanced LoRA Fine-Tuning ===\n")

    lora_config = LoRAConfig(
        r=32,
        alpha=64.0,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,
    )

    config = TrainingConfig(
        model="meta-llama/Llama-2-7b-hf",
        train_data="./data/train.jsonl",
        eval_data="./data/eval.jsonl",
        output_dir="./output/lora-advanced",
        fine_tuning_method=FineTuningMethod.LORA,
        lora=lora_config,
        num_epochs=3,
        per_device_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler=LrScheduler.COSINE,
        max_seq_len=4096,
        logging_steps=5,
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        seed=42,
    )

    # Save config for reproducibility
    config.save("./output/lora-advanced/config.json")

    trainer = Trainer(config)

    # Add a custom callback
    losses = []
    def log_callback(metrics):
        losses.append(metrics.train_loss)
        if len(losses) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"  [callback] Rolling avg loss (last 50): {avg:.4f}")

    trainer.add_callback(log_callback)
    trainer.train()


def example_qlora():
    """QLoRA fine-tuning (4-bit quantized base model + LoRA)."""
    print("=== QLoRA Fine-Tuning ===\n")

    config = TrainingConfig(
        model="meta-llama/Llama-2-13b-hf",
        train_data="./data/train.jsonl",
        output_dir="./output/qlora",
        fine_tuning_method=FineTuningMethod.QLORA,
        lora=LoRAConfig(r=16, alpha=32.0),
        num_epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_len=2048,
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwiftLLM Fine-Tuning Examples")
    parser.add_argument(
        "--example",
        choices=["basic", "advanced", "qlora", "all"],
        default="basic",
        help="Which example to run (default: basic)",
    )
    args = parser.parse_args()

    if args.example in ("basic", "all"):
        example_lora_basic()
    if args.example in ("advanced", "all"):
        example_lora_advanced()
    if args.example in ("qlora", "all"):
        example_qlora()

# ------------------------------------------------------------------------------
# END OF FILE: fine_tuning.py
# REPO PATH:   /swiftllm/examples/fine_tuning.py
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
