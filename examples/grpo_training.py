#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      grpo_training.py
# PATH:      /examples/grpo_training.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/training.py  GrpoTrainer, TrainingConfig, grpo_train()
#   - python/swiftllm/config.py    GrpoConfig, CgarConfig, PrmConfig, LongRewardConfig
# SEE ALSO:
#   - crates/swiftllm-training/src/grpo.rs          Rust GRPO optimizer
#   - crates/swiftllm-training/src/curriculum.rs    Rust CGAR scheduler
#   - crates/swiftllm-training/src/process_reward.rs Rust PRM step rewards
#   - crates/swiftllm-training/src/long_reward.rs   Rust LongR dense rewards
#   - examples/self_consistency.py                  Self-consistency inference demo
#   - examples/fine_tuning.py                       Standard LoRA fine-tuning
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

"""GRPO Training Example (Phase 2 Research Integration)

Demonstrates how to train a model with:

  1. GRPO (Group Relative Policy Optimization) — RL fine-tuning without a critic
  2. CGAR (Curriculum-Guided Adaptive Recursion) — progressive depth curriculum
  3. Process Reward Models (PRM) — step-level reasoning feedback
  4. LongR Dense Rewards — token-level NLL information gain

References:
  - GRPO: DeepSeekMath (Shao et al., 2024)
  - CGAR: Curriculum-Guided Adaptive Recursion (2024)
  - PRM: Let's Verify Step by Step (Lightman et al., 2023)
  - LongR: LongReward (2024)

Usage:
    python examples/grpo_training.py --model <model_path> --train-data <data.jsonl>

    # GRPO only (no curriculum or PRM):
    python examples/grpo_training.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --train-data data/math_prompts.jsonl \\
        --disable-cgar --no-prm

    # Full research stack:
    python examples/grpo_training.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --train-data data/math_prompts.jsonl \\
        --group-size 8 \\
        --enable-prm \\
        --long-reward-weight 0.1

Training data format (JSONL, one JSON object per line):
    {"prompt": "What is 12 × 15? Think step by step.", "answer": "180"}
    {"prompt": "Solve: 3x + 7 = 22", "answer": "5"}
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwiftLLM GRPO training demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Model path or HuggingFace ID")
    parser.add_argument("--train-data", default=None,
                        help="Path to training data JSONL (auto-generated if not provided)")
    parser.add_argument("-o", "--output-dir", default="./grpo_output",
                        help="Output directory for checkpoints")
    # GRPO hyperparameters
    parser.add_argument("--group-size", type=int, default=8,
                        help="Rollout samples per prompt (G)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO clipping ε")
    parser.add_argument("--kl-coeff", type=float, default=0.04,
                        help="KL divergence penalty β")
    parser.add_argument("--correctness-weight", type=float, default=1.0,
                        help="Weight for correctness reward")
    parser.add_argument("--format-weight", type=float, default=0.2,
                        help="Weight for format reward")
    # CGAR curriculum
    parser.add_argument("--disable-cgar", dest="enable_cgar", action="store_false",
                        default=True, help="Disable CGAR depth curriculum")
    parser.add_argument("--shallow-end", type=float, default=0.30,
                        help="CGAR: training fraction at end of shallow phase")
    parser.add_argument("--medium-end", type=float, default=0.60,
                        help="CGAR: training fraction at end of medium phase")
    # PRM
    parser.add_argument("--enable-prm", action="store_true", default=False,
                        help="Enable rule-based Process Reward Model")
    parser.add_argument("--prm-aggregation", default="last_step",
                        choices=["min", "mean", "product", "last_step", "weighted_mean"],
                        help="PRM step aggregation strategy")
    # LongR
    parser.add_argument("--long-reward-weight", type=float, default=0.0,
                        help="LongR dense reward weight (0.0 = disabled)")
    # Training
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Total model layers (for CGAR)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-config-only", action="store_true",
                        help="Print the resolved TrainingConfig and exit without training")
    return parser.parse_args()


def generate_synthetic_data(path: str, n: int = 50):
    """Generate synthetic math prompts for demonstration."""
    import random
    random.seed(42)

    samples = []
    for _ in range(n):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        op = random.choice(["+", "-", "*"])
        answer = {"+" : a + b, "-": a - b, "*": a * b}[op]
        samples.append({
            "prompt": (
                f"Solve the following arithmetic problem step by step: {a} {op} {b} = ?\n"
                "Show your work, then state 'The answer is X.' at the end."
            ),
            "answer": str(answer),
        })

    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {n} synthetic training examples → {path}")


def main():
    args = parse_args()

    from swiftllm.training import GrpoTrainer, TrainingConfig, FineTuningMethod
    from swiftllm.config import (
        GrpoConfig,
        CgarConfig,
        PrmConfig,
        LongRewardConfig,
        PrmAggregation,
    )

    # Auto-generate training data if not supplied
    train_data = args.train_data
    _tmpdir = None
    if train_data is None:
        import tempfile
        _tmpdir = tempfile.mkdtemp(prefix="swiftllm_grpo_")
        train_data = os.path.join(_tmpdir, "synthetic_math.jsonl")
        generate_synthetic_data(train_data)

    # Validate
    if not Path(train_data).exists():
        print(f"Error: training data not found: {train_data}", file=sys.stderr)
        sys.exit(1)

    # Build configs
    grpo_cfg = GrpoConfig(
        group_size=args.group_size,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        correctness_weight=args.correctness_weight,
        format_weight=args.format_weight,
    )

    cgar_cfg = None
    if args.enable_cgar:
        cgar_cfg = CgarConfig(
            shallow_end=args.shallow_end,
            medium_end=args.medium_end,
        )

    prm_cfg = None
    if args.enable_prm:
        prm_cfg = PrmConfig(
            aggregation=PrmAggregation(args.prm_aggregation),
            outcome_weight=0.5,
            prm_weight=0.5,
        )

    long_reward_cfg = None
    if args.long_reward_weight > 0:
        long_reward_cfg = LongRewardConfig(weight=args.long_reward_weight)

    config = TrainingConfig(
        model=args.model,
        train_data=train_data,
        output_dir=args.output_dir,
        fine_tuning_method=FineTuningMethod.FULL,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        num_layers=args.num_layers,
        grpo=grpo_cfg,
        cgar=cgar_cfg,
        prm=prm_cfg,
        long_reward=long_reward_cfg,
    )

    print("\nResolved TrainingConfig:")
    print("=" * 60)
    import json as _json
    print(_json.dumps(config.to_dict(), indent=2, default=str))
    print("=" * 60)

    if args.save_config_only:
        config_path = os.path.join(args.output_dir, "training_config.json")
        os.makedirs(args.output_dir, exist_ok=True)
        config.save(config_path)
        print(f"\nConfig saved to {config_path}. Exiting (--save-config-only).")
        return

    trainer = GrpoTrainer(config)

    # Attach a simple metric logger callback
    def log_callback(metrics):
        extra = ""
        if config.grpo:
            extra = f" [G={config.grpo.group_size}]"
        print(
            f"  [callback] step={metrics.step} | "
            f"loss={metrics.train_loss:.4f} | "
            f"lr={metrics.learning_rate:.2e}{extra}"
        )

    trainer.add_callback(log_callback)
    trainer.train()

    # Print summary
    final_metrics = trainer.metrics
    print("\nTraining summary:")
    print(f"  Final loss     : {final_metrics.train_loss:.4f}")
    print(f"  Total tokens   : {final_metrics.total_tokens:,}")
    print(f"  Elapsed        : {final_metrics.elapsed_secs:.1f}s")
    print(f"  Throughput     : {final_metrics.throughput:.0f} tok/s")
    print(f"  Output         : {args.output_dir}/")

    # Cleanup temp dir if we created one
    if _tmpdir:
        import shutil
        shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
# END OF FILE: grpo_training.py
# REPO PATH:   /swiftllm/examples/grpo_training.py
# INTEGRATES:  training.py · config.py
#              Rust: grpo.rs · curriculum.rs · process_reward.rs · long_reward.rs
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
