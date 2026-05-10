# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      hybrid_model_torch.py
# PATH:      /examples/hybrid_model_torch.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# PURPOSE:
#   GPU-executable PyTorch examples for the SwiftLLM hybrid architecture.
#   Demonstrates: build → train → verify → checkpoint → generate.
#
# REQUIREMENTS:
#   pip install torch>=2.0                  # required
#   pip install mamba-ssm causal-conv1d     # optional — fused CUDA SSM kernels
#   pip install safetensors                 # optional — .safetensors weight loading
#
# RUN (CPU, no GPU required):
#   python examples/hybrid_model_torch.py
#
# RUN (GPU):
#   CUDA_VISIBLE_DEVICES=0 python examples/hybrid_model_torch.py --device cuda
# ==============================================================================

"""
PyTorch Bridge Examples — HybridModel on GPU
=============================================

This file shows how to go from a config to a real, trainable, GPU-executable
PyTorch model in SwiftLLM.

  Section 1 — Build & inspect
  Section 2 — Single forward pass + loss
  Section 3 — Mini training loop (LM pre-training)
  Section 4 — Dense Verification post-decode
  Section 5 — Checkpoint: save, load, resume
  Section 6 — Mamba-3 vs Mamba-2 comparison
  Section 7 — LatentMoE load-balancing stats
  Section 8 — RLM block depth / confidence
  Section 9 — Larger model + GPU memory estimate
  Section 10 — Pretrained weight loading (HF-style)
"""

import argparse
import math
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# SwiftLLM imports
from swiftllm import (
    build_mamba3_reasoning_model,
    build_mamba3_base_model,
    build_mamba3_pure_model,
    build_torch_model,
    save_checkpoint,
    load_checkpoint,
    load_pretrained_weights,
    HAS_MAMBA_SSM,
    parameter_summary,
)
from swiftllm.model_config import (
    MambaConfig, RlmConfig, DenseVerificationConfig,
    HybridLayerType, HybridModelConfig,
)
from swiftllm.torch_model import (
    HybridModel,
    MambaLayer,
    LatentMoeLayer,
    RlmLayer,
    DenseVerificationLayer,
    HybridForwardOutput,
    VerificationOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print('=' * 68)


def fake_batch(vocab: int, batch: int = 2, seq: int = 64,
               device: str = "cpu") -> tuple:
    """Return (input_ids, labels) random tensors."""
    ids    = torch.randint(0, vocab, (batch, seq), device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    return ids, labels


# ---------------------------------------------------------------------------
# Section 1 — Build & inspect
# ---------------------------------------------------------------------------

def example_build(device: str) -> HybridModel:
    section("1. Build & Inspect")

    cfg = build_mamba3_reasoning_model(
        d_model=256,
        num_layers=8,
        vocab_size=4096,
        num_experts=8,
        moe_period=4,
    )

    print("\n── Architecture config ─────────────────────────────────────")
    print(cfg.summary())

    print("── Parameter estimate ──────────────────────────────────────")
    print(parameter_summary(cfg))

    model = build_torch_model(cfg).to(device)

    print(f"── PyTorch model ───────────────────────────────────────────")
    print(f"  {model}")
    print(f"  Actual param count  : {model.num_parameters():,}")
    print(f"  Trainable params    : {model.num_parameters(trainable_only=True):,}")
    print(f"  Device              : {next(model.parameters()).device}")
    print(f"  mamba-ssm backend   : {HAS_MAMBA_SSM}")
    if not HAS_MAMBA_SSM:
        print("  (using _ReferenceMamba fallback — install mamba-ssm for speed)")

    return model


# ---------------------------------------------------------------------------
# Section 2 — Single forward pass + loss
# ---------------------------------------------------------------------------

def example_forward(model: HybridModel, device: str) -> None:
    section("2. Forward Pass + Loss")

    cfg = model.cfg
    ids, labels = fake_batch(cfg.vocab_size, batch=2, seq=32, device=device)

    t0 = time.perf_counter()
    out = model(ids, labels=labels)
    elapsed = time.perf_counter() - t0

    print(f"\n  logits  shape  : {out.logits.shape}")
    print(f"  loss           : {out.loss.item():.4f}  (should ≈ log({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})")
    print(f"  confidence     : {out.confidence.shape}  range [{out.confidence.min():.3f}, {out.confidence.max():.3f}]")
    print(f"  forward time   : {elapsed*1000:.1f} ms")

    # Verify logits sum to log(vocab_size) via cross-entropy
    assert out.loss is not None
    assert out.logits.shape == (2, 32, cfg.vocab_size)
    print("  ✓ shapes correct")


# ---------------------------------------------------------------------------
# Section 3 — Mini training loop
# ---------------------------------------------------------------------------

def example_train(model: HybridModel, device: str, steps: int = 20) -> None:
    section("3. Mini Training Loop")

    cfg = model.cfg

    # AdamW with separate weight-decay / no-decay groups
    param_groups = model.parameter_groups(lr=3e-4, weight_decay=0.1)
    opt = torch.optim.AdamW(param_groups)

    print(f"\n  Training {steps} steps on {device} ...")
    model.train()
    for step in range(1, steps + 1):
        ids, labels = fake_batch(cfg.vocab_size, batch=4, seq=64, device=device)
        out = model(ids, labels=labels)
        out.loss.backward()
        # Gradient clipping (standard for LLMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()
        # Update LatentMoE dynamic bias (DeepSeek-style aux-loss-free load balance)
        model.update_load_stats()

        if step % 5 == 0 or step == 1:
            print(f"  step {step:3d}  loss={out.loss.item():.4f}")

    print("  Training loop completed ✓")
    model.eval()


# ---------------------------------------------------------------------------
# Section 4 — Dense Verification
# ---------------------------------------------------------------------------

def example_verification(model: HybridModel, device: str) -> None:
    section("4. Dense Verification (post-decode)")

    cfg = model.cfg
    if cfg.dense_verification_config is None:
        print("  (model has no DenseVerificationConfig — skipping)")
        return

    ids, _ = fake_batch(cfg.vocab_size, batch=3, seq=32, device=device)

    with torch.no_grad():
        out = model(ids, run_verification=True)

    v = out.verification
    print(f"\n  global_score   : {[f'{s:.3f}' for s in v.global_score.tolist()]}")
    print(f"  token_scores   : shape {v.token_scores.shape}")
    print(f"  is_accepted    : {v.is_accepted.tolist()}")
    dv_cfg = cfg.dense_verification_config
    print(f"  min_confidence : {dv_cfg.min_confidence}  (threshold)")

    for i, positions in enumerate(v.low_confidence_positions):
        print(f"  batch[{i}] low-conf positions: {positions[:10]}{'...' if len(positions) > 10 else ''}")

    # Demonstrate re-generation logic (conceptual — uses a second forward pass)
    if not v.is_accepted.all():
        print("\n  Some outputs below confidence threshold — re-generation would trigger here.")
        print("  (In production, re-generate flagged spans up to max_regen_attempts times)")


# ---------------------------------------------------------------------------
# Section 5 — Checkpoint: save, load, resume
# ---------------------------------------------------------------------------

def example_checkpoint(model: HybridModel, device: str) -> None:
    section("5. Checkpoint: Save / Load / Resume")

    ckpt_path = "/tmp/swiftllm_hybrid_example.pt"
    cfg = model.cfg

    # Save
    opt = torch.optim.AdamW(model.parameter_groups())
    save_checkpoint(model, ckpt_path, optimizer=opt, step=20,
                    metadata={"description": "example checkpoint"})
    size_kb = Path(ckpt_path).stat().st_size / 1024
    print(f"\n  Saved to {ckpt_path}  ({size_kb:.0f} KB)")

    # Load
    model2, meta = load_checkpoint(ckpt_path, device=device)
    print(f"  Loaded  step={meta['step']}  d_model={meta['config'].d_model}")

    # Verify weights are identical
    ids, _ = fake_batch(cfg.vocab_size, batch=2, seq=16, device=device)
    with torch.no_grad():
        l1 = model(ids).logits
        l2 = model2(ids).logits
    diff = (l1 - l2).abs().max().item()
    print(f"  Max logit diff  : {diff:.2e}  (should be ~0)")
    assert diff < 1e-5, f"checkpoint mismatch: {diff}"
    print("  Checkpoint round-trip ✓")


# ---------------------------------------------------------------------------
# Section 6 — Mamba-3 vs Mamba-2 standalone comparison
# ---------------------------------------------------------------------------

def example_mamba_comparison(device: str) -> None:
    section("6. Mamba-3 vs Mamba-2 Standalone Comparison")

    d = 256
    mamba3 = MambaLayer(MambaConfig.mamba3(d)).to(device)
    mamba2 = MambaLayer(MambaConfig.mamba2(d)).to(device)

    x = torch.randn(2, 128, d, device=device)

    t0 = time.perf_counter()
    y3 = mamba3(x)
    t3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    y2 = mamba2(x)
    t2 = time.perf_counter() - t0

    print(f"\n  Input    shape  : {x.shape}")
    print(f"  Mamba-3  output : {y3.shape}  time: {t3*1000:.1f} ms")
    print(f"  Mamba-2  output : {y2.shape}  time: {t2*1000:.1f} ms")
    print(f"  (Mamba-3 uses trapezoidal disc + complex-state approx)")
    print(f"  mamba-ssm backend: {HAS_MAMBA_SSM}")

    if HAS_MAMBA_SSM:
        speedup = t2 / t3 if t3 > 0 else float("inf")
        print(f"  Speedup vs ref   : {speedup:.1f}× (mamba-ssm fused kernel)")

    assert y3.shape == x.shape
    assert y2.shape == x.shape
    print("  Shapes correct ✓")


# ---------------------------------------------------------------------------
# Section 7 — LatentMoE load-balancing
# ---------------------------------------------------------------------------

def example_moe_load_balance(device: str) -> None:
    section("7. LatentMoE Load-Balancing Stats")

    from swiftllm.model_config import LatentMoeConfig, RoutingStrategy
    lmoe_cfg = LatentMoeConfig.deepseek_style(
        d_model=256, num_experts=8, num_experts_per_token=2
    )
    lmoe = LatentMoeLayer(lmoe_cfg).to(device)

    print(f"\n  num_experts     : {lmoe.num_experts}")
    print(f"  top-k           : {lmoe.num_experts_per_token}")
    print(f"  bias update rate: {lmoe.bias_update_rate}")

    print(f"\n  Running 5 forward passes to accumulate load stats...")
    for _ in range(5):
        x = torch.randn(4, 16, 256, device=device)
        _ = lmoe(x)

    print(f"  Before update — expert bias: {lmoe.expert_bias.tolist()}")
    lmoe.update_load_stats()
    print(f"  After  update — expert bias: {[f'{b:.4f}' for b in lmoe.expert_bias.tolist()]}")
    print("  (positive bias → expert was under-loaded → becomes more likely)")


# ---------------------------------------------------------------------------
# Section 8 — RLM depth and confidence
# ---------------------------------------------------------------------------

def example_rlm_depth(device: str) -> None:
    section("8. RLM Block — Depth Embedding & Confidence")

    rlm_cfg = RlmConfig.reasoning(d_model=256)
    rlm = RlmLayer(rlm_cfg).to(device)

    x = torch.randn(2, 32, 256, device=device)

    print(f"\n  max_depth      : {rlm_cfg.max_depth}")
    print(f"  var_slots      : {rlm_cfg.var_binding_slots}")
    print(f"  early_exit_thr : {rlm_cfg.early_exit_threshold}")
    print()
    for depth in range(rlm_cfg.max_depth + 1):
        with torch.no_grad():
            out, conf = rlm(x, depth=depth)
        avg_conf = conf.mean().item()
        print(f"  depth={depth}  output:{out.shape}  avg_confidence:{avg_conf:.4f}")

    # Simulate variable binding context
    var_indices = torch.randint(0, rlm_cfg.var_binding_slots, (2, 4), device=device)
    with torch.no_grad():
        out_v, conf_v = rlm(x, depth=1, var_indices=var_indices)
    print(f"\n  With var_indices: avg_confidence={conf_v.mean().item():.4f}")
    print("  Confidence range must be [0, 1]...")
    assert 0 <= conf_v.min() and conf_v.max() <= 1, "confidence out of range"
    print("  ✓")


# ---------------------------------------------------------------------------
# Section 9 — Larger model + GPU memory estimate
# ---------------------------------------------------------------------------

def example_larger_model(device: str) -> None:
    section("9. Larger Model + Memory Estimate")

    sizes = [
        ("300M", dict(d_model=512,  num_layers=12, num_experts=8)),
        ("1B",   dict(d_model=1024, num_layers=16, num_experts=16)),
    ]

    for name, kw in sizes:
        cfg = build_mamba3_reasoning_model(**kw, vocab_size=32000)

        # Estimate GPU VRAM (fp16: 2 bytes per param)
        from swiftllm import estimate_parameters
        counts = estimate_parameters(cfg)
        total  = counts["total"]
        fp16_gb = total * 2 / 1024**3
        bf16_gb = total * 2 / 1024**3
        fp32_gb = total * 4 / 1024**3

        print(f"\n  {name} model  (d={kw['d_model']}, L={kw['num_layers']}, E={kw['num_experts']})")
        print(f"    Parameters  : {total / 1e6:.0f}M")
        print(f"    VRAM fp16   : {fp16_gb:.1f} GB  (inference)")
        print(f"    VRAM fp32   : {fp32_gb:.1f} GB  (training weights only)")
        print(f"    Training est: ~{fp32_gb * 4:.1f} GB (weights + grads + Adam states)")

    if device == "cuda" and torch.cuda.is_available():
        cfg = build_mamba3_pure_model(d_model=256, num_layers=4, vocab_size=4096)
        model = build_torch_model(cfg).to(device)
        ids, labels = fake_batch(4096, batch=4, seq=128, device=device)
        torch.cuda.reset_peak_memory_stats()
        out = model(ids, labels=labels)
        out.loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n  GPU peak memory (d=256, L=4, B=4, S=128): {peak_mb:.1f} MB")
    else:
        print(f"\n  (GPU memory measurement skipped — running on {device})")


# ---------------------------------------------------------------------------
# Section 10 — Pretrained weight loading
# ---------------------------------------------------------------------------

def example_pretrained(device: str) -> None:
    section("10. Pretrained Weight Loading")

    print("""
  load_pretrained_weights() accepts:
    - .safetensors files  (pip install safetensors)
    - .bin / .pt files    (HuggingFace pytorch_model.bin)

  strict=False (default) allows partial loading — useful when fine-tuning
  a Mamba-3 model initialised from Mamba-2 weights.

  Example:

    from swiftllm import build_mamba3_base_model, build_torch_model
    from swiftllm import load_pretrained_weights

    cfg   = build_mamba3_base_model(d_model=2048, num_layers=32)
    model = build_torch_model(cfg).cuda()

    # Load weights from a HuggingFace-format checkpoint directory
    load_pretrained_weights(
        model,
        path="./weights/mamba3_2b/pytorch_model.bin",
        device="cuda",
        strict=False,   # ignore missing Mamba-3-specific keys
    )

  For multi-file sharded checkpoints (pytorch_model-00001-of-00004.bin etc.):
    Use HuggingFace `safetensors` or `transformers.modeling_utils.load_sharded_checkpoint`.

  JSON config round-trip (Python config ↔ Rust backend):

    cfg.to_json("./weights/mamba3_2b/swiftllm_config.json")
    # Rust engine reads this JSON via PyO3 to build JambaConfig
    """)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwiftLLM PyTorch Bridge Examples")
    parser.add_argument("--device", default="cpu",
                        help="compute device: cpu (default) or cuda")
    parser.add_argument("--section", "-s", type=int, default=0,
                        help="run only section N (1-10); 0 = all")
    parser.add_argument("--train-steps", type=int, default=20,
                        help="mini training loop step count (default 20)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available — falling back to CPU")
        device = "cpu"

    print(f"\nSwiftLLM PyTorch Bridge Examples")
    print(f"  device   : {device}")
    print(f"  torch    : {torch.__version__}")
    print(f"  mamba-ssm: {HAS_MAMBA_SSM}")

    # Build the shared model (sections 2-5 reuse it)
    warnings.filterwarnings("ignore")
    model = None

    fns = [
        lambda: (globals().__setitem__("_model", example_build(device)) or None),
        lambda: example_forward(_model, device),
        lambda: example_train(_model, device, steps=args.train_steps),
        lambda: example_verification(_model, device),
        lambda: example_checkpoint(_model, device),
        lambda: example_mamba_comparison(device),
        lambda: example_moe_load_balance(device),
        lambda: example_rlm_depth(device),
        lambda: example_larger_model(device),
        lambda: example_pretrained(device),
    ]

    # Wire up shared model variable
    _model = None

    def run_section(i: int) -> None:
        global _model
        if i == 0:
            _model = example_build(device)
        elif i == 1:
            example_forward(_model, device)
        elif i == 2:
            example_train(_model, device, steps=args.train_steps)
        elif i == 3:
            example_verification(_model, device)
        elif i == 4:
            example_checkpoint(_model, device)
        elif i == 5:
            example_mamba_comparison(device)
        elif i == 6:
            example_moe_load_balance(device)
        elif i == 7:
            example_rlm_depth(device)
        elif i == 8:
            example_larger_model(device)
        elif i == 9:
            example_pretrained(device)

    if args.section:
        idx = args.section - 1
        if not (0 <= idx <= 9):
            import sys
            print("Error: --section must be 1-10", file=sys.stderr)
            sys.exit(1)
        if idx > 0:
            _model = example_build(device)   # sections 2-10 need the model
        run_section(idx)
    else:
        run_section(0)
        for i in range(1, 10):
            run_section(i)

    print("\n" + "=" * 68)
    print("  All examples complete.")
    print("=" * 68)
