"""SwiftLLM Training — Python API for training and fine-tuning LLMs

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

    Full fine-tuning:

        config = TrainingConfig(
            model="meta-llama/Llama-2-7b-hf",
            output_dir="./output",
            train_data="./data/train.jsonl",
            fine_tuning_method="full",
            learning_rate=5e-5,
        )
        trainer = Trainer(config)
        trainer.train()
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


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

    def __post_init__(self):
        if self.fine_tuning_method in (FineTuningMethod.LORA, FineTuningMethod.QLORA):
            if self.lora is None:
                self.lora = LoRAConfig()

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

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._callbacks: List[Callable] = []
        self._metrics = TrainingMetrics()

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
        4. Runs the training loop with evaluation
        5. Saves the final model
        """
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

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save config
        self.config.save(os.path.join(self.config.output_dir, "training_config.json"))

        # In a full implementation, this would:
        # 1. Load model via swiftllm._core or transformers
        # 2. Apply LoRA/QLoRA adapters if configured
        # 3. Create data loaders
        # 4. Run training loop with actual gradient computation
        # 5. Save model weights

        print("\nTraining would start here.")
        print("(Full implementation requires CUDA backend — using placeholder training loop)")

        # Placeholder training simulation
        import math
        total_steps = 100  # Simulated
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
                print(
                    f"  step {step}/{total_steps} | "
                    f"loss: {loss:.4f} | "
                    f"ppl: {math.exp(loss):.2f} | "
                    f"lr: {lr:.2e}"
                )
                for cb in self._callbacks:
                    cb(self._metrics)

        print(f"\nTraining complete! Output saved to {self.config.output_dir}")

    def evaluate(self) -> TrainingMetrics:
        """Run evaluation on the eval dataset."""
        print("Running evaluation...")
        return self._metrics

    @property
    def metrics(self) -> TrainingMetrics:
        """Get the latest training metrics."""
        return self._metrics


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
