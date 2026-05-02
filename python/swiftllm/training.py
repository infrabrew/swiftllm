# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      training.py
# PATH:      /python/swiftllm/training.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/config.py   GrpoConfig, CgarConfig, PrmConfig, LongRewardConfig
# USED BY:
#   - python/swiftllm/__init__.py   lazy re-exports of Trainer, GrpoTrainer, fine_tune, grpo_train
#   - python/swiftllm/cli.py        cmd_train, cmd_grpo, cmd_finetune
# SEE ALSO:
#   - crates/swiftllm-training/src/grpo.rs          Rust GRPO optimizer (GrpoConfig mirror)
#   - crates/swiftllm-training/src/curriculum.rs    Rust CGAR scheduler (CgarConfig mirror)
#   - crates/swiftllm-training/src/process_reward.rs Rust PRM (PrmConfig mirror)
#   - crates/swiftllm-training/src/long_reward.rs   Rust LongR dense rewards (LongRewardConfig mirror)
#   - crates/swiftllm-training/src/trainer.rs       Rust trainer integrates all curriculum steps
#   - crates/swiftllm-training/src/config.rs        Rust TrainingConfig (superset of Python's)
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

"""SwiftLLM Training — Python API for training and fine-tuning LLMs

Includes Phase 2 research integrations:

  GRPO (Group Relative Policy Optimization)
    RL fine-tuning without a critic model using group-relative advantage
    estimation and PPO-style clipped policy gradients.

  CGAR (Curriculum-Guided Adaptive Recursion)
    Progressive depth curriculum: shallow (0–30%) → medium (30–60%) →
    full (60–100%), yielding up to 1.71× training speedup.

  Process Reward Models (PRM)
    Step-level feedback on reasoning chains via rule-based heuristics or a
    learned neural verifier; five aggregation strategies supported.

  LongR Dense Rewards
    Per-token relative NLL information gain vs. a reference model for
    long-context tasks; 9% LongBench v2 gain in original paper.

Example usage:

    Fine-tune with LoRA:

        from swiftllm.training import Trainer, TrainingConfig, LoRAConfig

        config = TrainingConfig(
            model="meta-llama/Llama-2-7b-hf",
            output_dir="./output",
            train_data="./data/train.jsonl",
            num_epochs=3,
            learning_rate=2e-4,
            lora=LoRAConfig(r=16, alpha=32),
        )
        trainer = Trainer(config)
        trainer.train()

    GRPO training with curriculum:

        from swiftllm.training import GrpoTrainer, TrainingConfig
        from swiftllm.config import GrpoConfig, CgarConfig, PrmConfig

        config = TrainingConfig(
            model="meta-llama/Llama-2-7b-hf",
            output_dir="./output",
            train_data="./data/rl_prompts.jsonl",
            fine_tuning_method="full",
            grpo=GrpoConfig(group_size=8, kl_coeff=0.04),
            cgar=CgarConfig(shallow_end=0.30, medium_end=0.60),
            prm=PrmConfig(aggregation="last_step"),
            long_reward_weight=0.1,
        )
        trainer = GrpoTrainer(config)
        trainer.train()
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .config import (
    GrpoConfig,
    CgarConfig,
    PrmConfig,
    LongRewardConfig,
    PrmAggregation,
    DenseAggregation,
)


class FineTuningMethod(Enum):
    """Fine-tuning method."""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class LrScheduler(Enum):
    """Learning rate scheduler type."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class MixedPrecision(Enum):
    """Mixed precision training mode."""
    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    Attributes:
        r: Rank of the low-rank matrices.
        alpha: Scaling factor (effective scale = alpha/r).
        dropout: Dropout probability for LoRA layers.
        target_modules: Which modules to apply LoRA to.
        use_rslora: Use Rank-Stabilized LoRA scaling.
    """
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    use_rslora: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
        }


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        model: Path to model or HuggingFace model ID.
        output_dir: Directory for checkpoints and logs.
        train_data: Path to training data (JSONL, CSV, or text).
        eval_data: Path to evaluation data (optional).
        num_epochs: Number of training epochs.
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate before optimizer step.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay (L2 regularization).
        warmup_steps: Warmup steps (int) or warmup ratio (float < 1.0).
        max_grad_norm: Maximum gradient norm for clipping.
        lr_scheduler: Learning rate scheduler type.
        mixed_precision: Mixed precision training mode.
        fine_tuning_method: Fine-tuning method (full, lora, qlora).
        lora: LoRA configuration (only used when method is lora/qlora).
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps (0 = per epoch).
        save_total_limit: Max checkpoints to keep.
        eval_steps: Evaluate every N steps (0 = per epoch).
        max_seq_len: Maximum sequence length.
        seed: Random seed.
        resume_from_checkpoint: Path to resume from.

        -- Phase 2: Research integrations --
        num_layers: Total number of model layers (required for CGAR/phased spec).
        grpo: GRPO optimizer config; set to enable RL fine-tuning.
        cgar: CGAR curriculum config; set to enable depth curriculum.
        prm: Process Reward Model config; set to enable step-level rewards.
        long_reward_weight: Weight for LongR dense rewards (0.0 = disabled).
        long_reward: Full LongR config; if None and long_reward_weight > 0,
            uses defaults with the specified weight.
    """
    model: str = ""
    output_dir: str = "./output"
    train_data: str = ""
    eval_data: Optional[str] = None
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: Union[int, float] = 100
    max_grad_norm: float = 1.0
    lr_scheduler: LrScheduler = LrScheduler.COSINE
    mixed_precision: MixedPrecision = MixedPrecision.FP16
    fine_tuning_method: FineTuningMethod = FineTuningMethod.LORA
    lora: Optional[LoRAConfig] = None
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: Optional[int] = 3
    eval_steps: int = 500
    max_seq_len: int = 2048
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    # Phase 2 — Research integrations
    num_layers: int = 32
    grpo: Optional[GrpoConfig] = None
    cgar: Optional[CgarConfig] = None
    prm: Optional[PrmConfig] = None
    long_reward_weight: float = 0.0
    long_reward: Optional[LongRewardConfig] = None

    def __post_init__(self):
        if self.fine_tuning_method in (FineTuningMethod.LORA, FineTuningMethod.QLORA):
            if self.lora is None:
                self.lora = LoRAConfig()
        # Promote long_reward_weight → full LongRewardConfig if needed
        if self.long_reward_weight > 0 and self.long_reward is None:
            self.long_reward = LongRewardConfig(weight=self.long_reward_weight)
        elif self.long_reward is not None and self.long_reward_weight == 0.0:
            self.long_reward_weight = self.long_reward.weight

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model": self.model,
            "output_dir": self.output_dir,
            "train_data": self.train_data,
            "eval_data": self.eval_data,
            "num_epochs": self.num_epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler": self.lr_scheduler.value,
            "mixed_precision": self.mixed_precision.value,
            "fine_tuning_method": self.fine_tuning_method.value,
            "lora": self.lora.to_dict() if self.lora else None,
            "max_seq_len": self.max_seq_len,
            "seed": self.seed,
            # Phase 2
            "num_layers": self.num_layers,
            "grpo": self.grpo.to_dict() if self.grpo else None,
            "cgar": self.cgar.to_dict() if self.cgar else None,
            "prm": self.prm.to_dict() if self.prm else None,
            "long_reward_weight": self.long_reward_weight,
            "long_reward": self.long_reward.to_dict() if self.long_reward else None,
        }
        return d

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)
        lora = LoRAConfig(**d.pop("lora")) if d.get("lora") else None
        d["lr_scheduler"] = LrScheduler(d.get("lr_scheduler", "cosine"))
        d["mixed_precision"] = MixedPrecision(d.get("mixed_precision", "fp16"))
        d["fine_tuning_method"] = FineTuningMethod(d.get("fine_tuning_method", "lora"))
        # Deserialise Phase 2 nested configs
        if d.get("grpo"):
            d["grpo"] = GrpoConfig(**d["grpo"])
        if d.get("cgar"):
            cgar_d = dict(d["cgar"])
            d["cgar"] = CgarConfig(**cgar_d)
        if d.get("prm"):
            prm_d = dict(d["prm"])
            if "aggregation" in prm_d:
                prm_d["aggregation"] = PrmAggregation(prm_d["aggregation"])
            d["prm"] = PrmConfig(**prm_d)
        if d.get("long_reward"):
            lr_d = dict(d["long_reward"])
            if "aggregation" in lr_d:
                lr_d["aggregation"] = DenseAggregation(lr_d["aggregation"])
            d["long_reward"] = LongRewardConfig(**lr_d)
        # Filter to known fields to tolerate future/extra keys in saved configs
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"lora"}
        d = {k: v for k, v in d.items() if k in known}
        return cls(lora=lora, **d)


@dataclass
class TrainingMetrics:
    """Snapshot of training metrics."""
    step: int = 0
    epoch: int = 0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    perplexity: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0
    total_tokens: int = 0
    elapsed_secs: float = 0.0


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration.

    Attributes:
        patience: Number of eval steps without improvement before stopping.
        min_delta: Minimum improvement to count as progress.
        metric: Metric to monitor ('eval_loss' or 'train_loss').
    """
    patience: int = 3
    min_delta: float = 0.0
    metric: str = "eval_loss"


class Trainer:
    """Main trainer class for fine-tuning and training LLMs.

    Example:
        >>> config = TrainingConfig(
        ...     model="meta-llama/Llama-2-7b-hf",
        ...     train_data="train.jsonl",
        ...     lora=LoRAConfig(r=16),
        ... )
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        early_stopping: Optional[EarlyStoppingConfig] = None,
    ):
        self.config = config
        self.early_stopping = early_stopping
        self._callbacks: List[Callable] = []
        self._metrics = TrainingMetrics()
        self._best_metric: float = float("inf")
        self._patience_counter: int = 0
        self._checkpoints: List[str] = []
        self._stopped_early: bool = False

    def add_callback(self, callback: Callable):
        """Add a training callback.

        The callback is called after each logging step with the current metrics.
        """
        self._callbacks.append(callback)

    def train(self):
        """Run the training loop.

        This method:
        1. Loads the model and tokenizer
        2. Sets up the optimizer and scheduler
        3. Loads and preprocesses the training data
        4. Runs the training loop with evaluation and early stopping
        5. Saves checkpoints and the final model
        """
        import math

        print("=" * 70)
        print("  [SIMULATED] SwiftLLM Training — stub backend")
        print("  The CUDA training backend is not yet wired up. This run executes")
        print("  a fixed 100-step simulated loop with synthetic loss curves to")
        print("  exercise the config, logging, and checkpoint plumbing. No model")
        print("  weights are loaded and no gradient updates are performed.")
        print("=" * 70)

        print(f"SwiftLLM Training")
        print(f"  Model: {self.config.model}")
        print(f"  Method: {self.config.fine_tuning_method.value}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.effective_batch_size}")
        print(f"  Learning rate: {self.config.learning_rate:.2e}")

        if self.config.lora:
            print(f"  LoRA rank: {self.config.lora.r}")
            print(f"  LoRA alpha: {self.config.lora.alpha}")
            print(f"  LoRA targets: {self.config.lora.target_modules}")

        if self.early_stopping:
            print(f"  Early stopping: patience={self.early_stopping.patience}, "
                  f"metric={self.early_stopping.metric}")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save config
        self.config.save(os.path.join(self.config.output_dir, "training_config.json"))

        # Note: full implementation requires CUDA backend.
        # Using simulated loop to demonstrate API, logging, and checkpoint management.

        total_steps = 100
        start_time = time.time()

        for step in range(1, total_steps + 1):
            # Simulate decreasing loss
            loss = 3.0 * math.exp(-step * 0.03) + 0.5 + math.sin(step * 0.1) * 0.05
            lr = self.config.learning_rate * min(step / 10, 1.0)

            self._metrics = TrainingMetrics(
                step=step,
                epoch=(step * self.config.num_epochs) // total_steps,
                train_loss=loss,
                perplexity=math.exp(loss),
                learning_rate=lr,
                throughput=1000.0,
                total_tokens=step * self.config.per_device_batch_size * self.config.max_seq_len,
                elapsed_secs=time.time() - start_time,
            )

            if step % self.config.logging_steps == 0:
                elapsed = time.time() - start_time
                tok_per_sec = self._metrics.total_tokens / max(elapsed, 1e-6)
                print(
                    f"  step {step}/{total_steps} | "
                    f"loss: {loss:.4f} | "
                    f"ppl: {math.exp(loss):.2f} | "
                    f"lr: {lr:.2e} | "
                    f"tok/s: {tok_per_sec:.0f}"
                )
                for cb in self._callbacks:
                    cb(self._metrics)

            # Checkpoint saving
            if self.config.save_steps > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

            # Evaluation + early stopping
            if self.config.eval_steps > 0 and step % self.config.eval_steps == 0:
                eval_loss = loss * 1.05  # Simulated eval loss
                self._metrics.eval_loss = eval_loss
                print(f"  eval | loss: {eval_loss:.4f} | ppl: {math.exp(eval_loss):.2f}")

                if self.early_stopping and self._check_early_stopping(eval_loss):
                    print(f"\n  Early stopping triggered at step {step} "
                          f"(no improvement for {self.early_stopping.patience} evals)")
                    self._stopped_early = True
                    break

        # Final checkpoint
        self._save_checkpoint(self._metrics.step, is_final=True)

        status = "Early stopped" if self._stopped_early else "Training complete"
        print(f"\n{status}! Output saved to {self.config.output_dir}")

    def _check_early_stopping(self, current_metric: float) -> bool:
        """Check if training should stop early. Returns True if should stop."""
        if not self.early_stopping:
            return False

        if current_metric < self._best_metric - self.early_stopping.min_delta:
            self._best_metric = current_metric
            self._patience_counter = 0
            return False

        self._patience_counter += 1
        return self._patience_counter >= self.early_stopping.patience

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save a training checkpoint."""
        name = "final" if is_final else f"checkpoint-{step}"
        ckpt_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            "step": step,
            "epoch": self._metrics.epoch,
            "train_loss": self._metrics.train_loss,
            "eval_loss": self._metrics.eval_loss,
            "learning_rate": self._metrics.learning_rate,
            "config": self.config.to_dict(),
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        if not is_final:
            self._checkpoints.append(ckpt_dir)

        # Enforce save_total_limit
        if self.config.save_total_limit is not None:
            while len(self._checkpoints) > self.config.save_total_limit:
                old = self._checkpoints.pop(0)
                import shutil
                if os.path.isdir(old):
                    shutil.rmtree(old, ignore_errors=True)

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: str) -> "Trainer":
        """Resume training from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            Trainer configured to resume from the checkpoint.
        """
        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        # Validate checkpoint path doesn't escape via ".." traversal
        real_state = os.path.realpath(state_path)
        real_ckpt = os.path.realpath(checkpoint_path)
        if not real_state.startswith(real_ckpt + os.sep):
            raise ValueError(f"Invalid checkpoint path: state file resolves outside checkpoint dir")
        with open(state_path) as f:
            state = json.load(f)
        config_path = os.path.realpath(
            os.path.join(checkpoint_path, "..", "training_config.json")
        )
        config = TrainingConfig.load(config_path)
        config.resume_from_checkpoint = checkpoint_path
        trainer = cls(config)
        trainer._metrics = TrainingMetrics(
            step=state["step"],
            epoch=state["epoch"],
            train_loss=state["train_loss"],
            eval_loss=state.get("eval_loss"),
        )
        return trainer

    def evaluate(self) -> TrainingMetrics:
        """Run evaluation on the eval dataset."""
        print("Running evaluation...")
        return self._metrics

    @property
    def metrics(self) -> TrainingMetrics:
        """Get the latest training metrics."""
        return self._metrics

    @property
    def stopped_early(self) -> bool:
        """Whether training was stopped early."""
        return self._stopped_early


class GrpoTrainer(Trainer):
    """Trainer subclass that applies GRPO with optional CGAR curriculum and PRM.

    ``GrpoTrainer`` wraps the standard ``Trainer`` loop with:

    - **GRPO reward shaping**: per-step group-relative advantage computation,
      PPO-style clipped policy gradient, and KL divergence penalty.
    - **CGAR depth curriculum**: progressively increasing active layer depth
      from shallow to full across training (1.71× speedup reported).
    - **PRM step rewards**: step-level feedback on reasoning chains blended
      with the outcome reward.
    - **LongR dense rewards**: token-level NLL relative information gain
      for long-context tasks.

    The ``TrainingConfig.grpo`` field must be set (not None) to activate GRPO.
    All other research additions are optional.

    Example::

        from swiftllm.training import GrpoTrainer, TrainingConfig
        from swiftllm.config import GrpoConfig, CgarConfig, PrmConfig

        config = TrainingConfig(
            model="meta-llama/Llama-2-7b-hf",
            train_data="rl_prompts.jsonl",
            output_dir="./grpo_output",
            fine_tuning_method="full",
            num_layers=32,
            grpo=GrpoConfig(group_size=8, kl_coeff=0.04),
            cgar=CgarConfig(shallow_end=0.30, medium_end=0.60),
            prm=PrmConfig(aggregation="last_step"),
            long_reward_weight=0.10,
        )
        trainer = GrpoTrainer(config)
        trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        early_stopping: Optional[EarlyStoppingConfig] = None,
    ):
        if config.grpo is None:
            raise ValueError(
                "GrpoTrainer requires TrainingConfig.grpo to be set. "
                "Pass grpo=GrpoConfig(...) to TrainingConfig."
            )
        super().__init__(config, early_stopping)
        self._grpo_config = config.grpo
        self._cgar_config = config.cgar
        self._prm_config = config.prm
        self._long_reward_config = config.long_reward

    def train(self):
        """Run the GRPO training loop with curriculum and reward shaping.

        This method extends the standard training loop with:
          1. GRPO group sampling (``group_size`` rollouts per prompt).
          2. Reward computation (correctness + format + length + PRM + LongR).
          3. Group-relative advantage normalisation.
          4. PPO-style clipped policy-gradient loss + KL penalty.
          5. CGAR depth scheduling via ``_apply_cgar_tick()``.
        """
        import math

        print("=" * 70)
        print("  [SIMULATED] SwiftLLM GrpoTrainer — stub backend")
        print("  Full CUDA GRPO + curriculum backend not yet wired up.")
        print("  Running a simulated loop to exercise config, logging, and")
        print("  checkpoint plumbing. No gradient updates are performed.")
        print("=" * 70)

        print(f"SwiftLLM GRPO Training")
        print(f"  Model:         {self.config.model}")
        print(f"  Group size:    {self._grpo_config.group_size}")
        print(f"  Clip eps:      {self._grpo_config.clip_eps}")
        print(f"  KL coeff:      {self._grpo_config.kl_coeff}")
        if self._cgar_config:
            print(f"  CGAR:          enabled (shallow_end={self._cgar_config.shallow_end}, "
                  f"medium_end={self._cgar_config.medium_end})")
        if self._prm_config:
            print(f"  PRM:           enabled (aggregation={self._prm_config.aggregation.value})")
        if self._long_reward_config:
            print(f"  LongR weight:  {self._long_reward_config.weight}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.config.save(os.path.join(self.config.output_dir, "training_config.json"))

        total_steps = 100
        start_time = time.time()

        for step in range(1, total_steps + 1):
            fraction = step / total_steps

            # Simulated active-layer depth from CGAR schedule
            if self._cgar_config:
                active_layers = self._compute_cgar_layers(fraction)
            else:
                active_layers = self.config.num_layers

            # Simulated reward signal (group-relative advantage)
            base_loss = 3.0 * math.exp(-step * 0.03) + 0.5
            reward = 1.0 - base_loss / 3.5  # crude reward proxy
            kl_penalty = self._grpo_config.kl_coeff * 0.05 * math.exp(-step * 0.02)
            policy_loss = max(0.0, 1.0 - reward) + kl_penalty

            lr = self.config.learning_rate * min(step / 10, 1.0)

            self._metrics = TrainingMetrics(
                step=step,
                epoch=(step * self.config.num_epochs) // total_steps,
                train_loss=policy_loss,
                perplexity=math.exp(policy_loss),
                learning_rate=lr,
                throughput=1000.0 * self._grpo_config.group_size,
                total_tokens=(
                    step
                    * self.config.per_device_batch_size
                    * self._grpo_config.group_size
                    * self.config.max_seq_len
                ),
                elapsed_secs=time.time() - start_time,
            )

            if step % self.config.logging_steps == 0:
                elapsed = time.time() - start_time
                print(
                    f"  step {step}/{total_steps} | "
                    f"policy_loss: {policy_loss:.4f} | "
                    f"reward: {reward:.4f} | "
                    f"kl: {kl_penalty:.4f} | "
                    f"lr: {lr:.2e} | "
                    f"layers: {active_layers}/{self.config.num_layers}"
                )
                for cb in self._callbacks:
                    cb(self._metrics)

            if self.config.save_steps > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(self._metrics.step, is_final=True)
        print(f"\nGRPO training complete! Output saved to {self.config.output_dir}")

    def _compute_cgar_layers(self, fraction: float) -> int:
        """Compute active layer count from CGAR curriculum schedule.

        Uses a smooth Hermite interpolation within each phase boundary
        to avoid discontinuous jumps in layer depth.
        """
        cfg = self._cgar_config
        n = self.config.num_layers
        min_l = cfg.min_layers if cfg.min_layers is not None else max(1, n // 3)
        max_l = cfg.max_layers if cfg.max_layers is not None else n

        def hermite(t: float) -> float:
            """Smooth step: 3t² - 2t³."""
            t = max(0.0, min(1.0, t))
            return t * t * (3.0 - 2.0 * t)

        if fraction < cfg.shallow_end:
            t = hermite(fraction / cfg.shallow_end)
            return round(min_l + t * (n // 2 - min_l))
        elif fraction < cfg.medium_end:
            t = hermite((fraction - cfg.shallow_end) / (cfg.medium_end - cfg.shallow_end))
            return round(n // 2 + t * (max_l - n // 2))
        else:
            return max_l


def fine_tune(
    model: str,
    train_data: str,
    output_dir: str = "./output",
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    **kwargs,
) -> Trainer:
    """Convenience function for fine-tuning with LoRA.

    Args:
        model: Model path or HuggingFace ID.
        train_data: Path to training data (JSONL format).
        output_dir: Output directory.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        learning_rate: Learning rate.
        num_epochs: Number of epochs.
        **kwargs: Additional TrainingConfig parameters.

    Returns:
        Trainer instance (already trained).

    Example:
        >>> trainer = fine_tune(
        ...     model="meta-llama/Llama-2-7b-hf",
        ...     train_data="data.jsonl",
        ...     lora_r=16,
        ... )
    """
    config = TrainingConfig(
        model=model,
        train_data=train_data,
        output_dir=output_dir,
        fine_tuning_method=FineTuningMethod.LORA,
        lora=LoRAConfig(r=lora_r, alpha=lora_alpha),
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        **kwargs,
    )
    trainer = Trainer(config)
    trainer.train()
    return trainer

def grpo_train(
    model: str,
    train_data: str,
    output_dir: str = "./grpo_output",
    group_size: int = 8,
    kl_coeff: float = 0.04,
    learning_rate: float = 1e-5,
    num_epochs: int = 1,
    enable_cgar: bool = True,
    enable_prm: bool = False,
    long_reward_weight: float = 0.0,
    **kwargs,
) -> GrpoTrainer:
    """Convenience function for GRPO training with sensible defaults.

    Args:
        model: Model path or HuggingFace ID.
        train_data: Path to prompts/data (JSONL format).
        output_dir: Output directory for checkpoints.
        group_size: Number of rollout samples per prompt (G).
        kl_coeff: KL divergence penalty coefficient (β).
        learning_rate: Peak learning rate.
        num_epochs: Number of training epochs.
        enable_cgar: Whether to enable CGAR depth curriculum.
        enable_prm: Whether to enable rule-based process reward model.
        long_reward_weight: Weight for LongR dense rewards (0.0 = off).
        **kwargs: Additional TrainingConfig parameters.

    Returns:
        GrpoTrainer instance (already trained).

    Example::

        trainer = grpo_train(
            model="meta-llama/Llama-2-7b-hf",
            train_data="prompts.jsonl",
            group_size=8,
            enable_cgar=True,
        )
    """
    config = TrainingConfig(
        model=model,
        train_data=train_data,
        output_dir=output_dir,
        fine_tuning_method=FineTuningMethod.FULL,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        grpo=GrpoConfig(group_size=group_size, kl_coeff=kl_coeff),
        cgar=CgarConfig() if enable_cgar else None,
        prm=PrmConfig() if enable_prm else None,
        long_reward_weight=long_reward_weight,
        **kwargs,
    )
    trainer = GrpoTrainer(config)
    trainer.train()
    return trainer

# ------------------------------------------------------------------------------
# END OF FILE: training.py
# REPO PATH:   /swiftllm/python/swiftllm/training.py
# INTEGRATES:  config.py · __init__.py · cli.py
#              Rust: grpo.rs · curriculum.rs · process_reward.rs · long_reward.rs
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
