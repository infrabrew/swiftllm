# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      hybrid_model.py
# PATH:      /examples/hybrid_model.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# PURPOSE:
#   End-to-end usage examples for the SwiftLLM hybrid model architecture:
#     Mamba-3 SSM + LatentMoE + RLM (Recursive Language Model) + Dense Verification
#
# WHAT IT DEMONSTRATES:
#   1. Quickstart — one-line flagship preset
#   2. Fluent builder — step-by-step fine-grained control
#   3. Component-level configs — direct dataclass construction
#   4. Architecture variants — base / pure / hybrid-attention models
#   5. Save / load configs — JSON round-trip
#   6. Size estimation — parameter count breakdown
#   7. Engine integration — pass config into LLM / EngineConfig
#   8. Custom layer schedules — per-layer type override
#   9. Training workflow — GRPO RL fine-tuning on a reasoning model
#  10. Serialisation-only mode — no GPU required (inspect configs offline)
#
# RUN:
#   python examples/hybrid_model.py
#
# REQUIREMENTS (GPU runs only):
#   pip install swiftllm          # for LLM / EngineConfig
#   pip install mamba-ssm         # optional PyTorch SSM backend
# ------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

"""
SwiftLLM Hybrid Architecture Examples
======================================

Architecture overview::

    Input tokens
         │
         ▼
    Embedding (vocab_size × d_model)
         │
    ┌────┴────────────────────────────────────────────────────────┐
    │  Backbone — 32 layers, schedule: MAMBA / MAMBA / MAMBA / MAMBA_MOE / …
    │
    │  [Mamba-3 SSM Block]  ← most layers
    │    • selective scan          O(L) compute, O(1) decode memory
    │    • complex-valued states   richer frequency representation
    │    • MIMO multi-head         d_model//64 heads
    │    • exp-trapezoidal disc.   improved long-range propagation
    │
    │  [LatentMoE Block]  ← every 4th layer
    │    • compress  d_model → d_model//8  (87.5 % less inter-GPU traffic)
    │    • route     64 experts, top-6 active, 2 shared
    │    • experts   FFN in latent space
    │    • expand    d_model//8 → d_model
    │    • load bal  DeepSeek-V3 dynamic bias (aux-loss-free)
    └────────────────────────────────────────────────────────────┘
         │
    [RLM Block]
         • REPL state machine  (Assign / Compute / Verify / Recurse)
         • variable binding table  (32 slots)
         • depth scheduler  (up to depth 3)
         • early exit  when step confidence ≥ 0.92
         │
    [Dense Verification]  (post-decode)
         • cross-attention: draft tokens ↔ REPL execution trace
         • token scores  [seq_len]
         • step scores   [num_repl_steps]
         • global score  scalar
         • re-generate any span where confidence < 0.80
         │
         ▼
    LM Head → logits → sample
"""

import json
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from swiftllm.model_config import (
    MambaConfig,
    MoeConfig,
    LatentMoeConfig,
    RlmConfig,
    DenseVerificationConfig,
    HybridLayerType,
    HybridModelConfig,
    RoutingStrategy,
    ModelBaseConfig,
)
from swiftllm.hybrid_model import (
    HybridModelBuilder,
    build_mamba3_reasoning_model,
    build_mamba3_base_model,
    build_mamba3_pure_model,
    build_mamba3_hybrid_attention_model,
    estimate_parameters,
    parameter_summary,
)


# ===========================================================================
# Helper
# ===========================================================================

def section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 68 - len(title)))


# ===========================================================================
# 1. Quickstart — one-line flagship preset
# ===========================================================================

def example_quickstart():
    section("1. Quickstart — flagship reasoning model (one line)")

    cfg = build_mamba3_reasoning_model(
        d_model=2048,
        num_layers=32,
        vocab_size=32000,
    )

    print(cfg.summary())
    print(parameter_summary(cfg))


# ===========================================================================
# 2. Fluent builder — step-by-step
# ===========================================================================

def example_fluent_builder():
    section("2. Fluent Builder — step-by-step fine-grained control")

    cfg = (
        HybridModelBuilder(
            d_model=2048,
            num_layers=32,
            vocab_size=32000,
            max_seq_len=131072,
        )
        # ── Mamba-3 SSM backbone ──────────────────────────────────────
        .with_mamba3(
            d_state=128,      # 128 complex states per head (Mamba-3)
            expand=2,         # inner_dim = d_model × 2
            ns_steps=5,       # Newton refinement steps for trapezoidal disc
        )
        # ── LatentMoE — appears every 4 layers ───────────────────────
        .with_latent_moe(
            num_experts=64,
            num_experts_per_token=6,    # top-6 routing
            num_shared_experts=2,       # always-active shared experts
            latent_compression_ratio=8, # compress d_model → d_model//8
            moe_period=4,               # every 4th layer is MoE
            routing=RoutingStrategy.TOP_K,
            aux_loss_coeff=0.0,         # aux-loss-free (dynamic bias handles balance)
        )
        # ── RLM reasoning block ───────────────────────────────────────
        .with_rlm(
            max_depth=3,                # up to 3 recursive sub-problems
            enable_repl=True,           # REPL state machine active
            var_binding_slots=32,       # variable binding table size
            early_exit_threshold=0.92,  # exit early on high confidence
        )
        # ── Dense Verification post-decode pass ───────────────────────
        .with_dense_verification(
            num_verification_heads=8,
            min_confidence=0.80,        # re-generate if global score < 0.80
            max_regen_attempts=3,
            score_repl_steps=True,      # per-REPL-step confidence scores
        )
        .build()
    )

    print(cfg.summary())

    subsection("Layer schedule (first 16 layers)")
    for i, lt in enumerate(cfg.layer_schedule[:16], 1):
        marker = " ← LatentMoE" if lt == HybridLayerType.MAMBA_MOE else ""
        print(f"  Layer {i:2d}: {lt.value}{marker}")
    print(f"  ... ({cfg.num_layers} total)")

    subsection("Architecture properties")
    print(f"  is_pure_mamba      : {cfg.is_pure_mamba}")
    print(f"  num_mamba_layers   : {cfg.num_mamba_layers}")
    print(f"  num_moe_layers     : {cfg.num_moe_layers}")
    print(f"  num_attention_layers: {cfg.num_attention_layers}")


# ===========================================================================
# 3. Component-level configs — direct dataclass construction
# ===========================================================================

def example_component_configs():
    section("3. Component Configs — direct dataclass construction")

    subsection("MambaConfig — Mamba-3 preset")
    mamba = MambaConfig.mamba3(d_model=2048)
    print(f"  d_model         : {mamba.d_model}")
    print(f"  d_state         : {mamba.d_state}")
    print(f"  num_heads       : {mamba.num_heads}  (d_model // 64)")
    print(f"  dt_rank         : {mamba.dt_rank}   (ceil(d_model / 16))")
    print(f"  complex_states  : {mamba.use_complex_states}")
    print(f"  mimo            : {mamba.use_mimo}")
    print(f"  trapezoidal_disc: {mamba.use_trapezoidal_disc}")
    print(f"  ns_steps        : {mamba.ns_steps}")

    subsection("MambaConfig — Mamba-2 (ablation baseline)")
    mamba2 = MambaConfig.mamba2(d_model=2048)
    print(f"  d_state         : {mamba2.d_state}   (16 for Mamba-2)")
    print(f"  complex_states  : {mamba2.use_complex_states}")
    print(f"  trapezoidal_disc: {mamba2.use_trapezoidal_disc}")

    subsection("LatentMoeConfig — DeepSeek-V3 style")
    lmoe = LatentMoeConfig.deepseek_style(
        d_model=2048, num_experts=64, num_experts_per_token=6
    )
    print(f"  d_model             : {lmoe.d_model}")
    print(f"  d_latent            : {lmoe.d_latent}  (d_model // {lmoe.latent_compression_ratio})")
    print(f"  num_experts         : {lmoe.moe.num_experts}")
    print(f"  num_experts_per_tok : {lmoe.moe.num_experts_per_token}")
    print(f"  shared_experts      : {lmoe.moe.num_shared_experts}")
    print(f"  routing             : {lmoe.moe.routing.value}")
    print(f"  bias_update_rate    : {lmoe.moe.bias_update_rate}  (DeepSeek dynamic bias)")
    print(f"  aux_loss_coeff      : {lmoe.moe.aux_loss_coeff}    (0 = aux-loss-free)")

    subsection("RlmConfig — reasoning preset")
    rlm = RlmConfig.reasoning(d_model=2048)
    print(f"  max_depth           : {rlm.max_depth}")
    print(f"  enable_repl         : {rlm.enable_repl}")
    print(f"  var_binding_slots   : {rlm.var_binding_slots}")
    print(f"  depth_hidden_size   : {rlm.depth_hidden_size}   (d_model // 4)")
    print(f"  d_subproblem        : {rlm.d_subproblem}   (d_model // 2)")
    print(f"  early_exit_threshold: {rlm.early_exit_threshold}")

    subsection("DenseVerificationConfig — standard")
    dv = DenseVerificationConfig.standard(d_model=2048)
    print(f"  num_verif_heads     : {dv.num_verification_heads}")
    print(f"  d_v_head            : {dv.d_v_head}   (d_model // 8)")
    print(f"  min_confidence      : {dv.min_confidence}")
    print(f"  max_regen_attempts  : {dv.max_regen_attempts}")
    print(f"  score_repl_steps    : {dv.score_repl_steps}")


# ===========================================================================
# 4. Architecture variants
# ===========================================================================

def example_variants():
    section("4. Architecture Variants")

    configs = {
        "Mamba-3 + LatentMoE + RLM + DenseVerif (flagship)":
            build_mamba3_reasoning_model(d_model=2048, num_layers=32),

        "Mamba-3 + LatentMoE base (no reasoning layers)":
            build_mamba3_base_model(d_model=2048, num_layers=32),

        "Pure Mamba-3 (ablation baseline, no MoE/reasoning)":
            build_mamba3_pure_model(d_model=2048, num_layers=32),

        "Mamba-3 + Attention (Jamba-style, attn every 8th layer)":
            build_mamba3_hybrid_attention_model(
                d_model=2048, num_layers=32, attn_period=8, moe_period=4
            ),
    }

    print(f"\n  {'Variant':<52} {'Mamba':>5} {'MoE':>5} {'Attn':>5} {'~Params':>10}")
    print("  " + "─" * 82)
    for name, cfg in configs.items():
        p = estimate_parameters(cfg)
        total = p["total"]
        label = f"{total / 1e9:.1f}B" if total >= 1e9 else f"{total / 1e6:.0f}M"
        print(
            f"  {name:<52} "
            f"{cfg.num_mamba_layers:>5} "
            f"{cfg.num_moe_layers:>5} "
            f"{cfg.num_attention_layers:>5} "
            f"{label:>10}"
        )

    # Show HybridModelConfig.make_schedule() directly
    subsection("Custom schedule: all-Mamba, no MoE, no attention (8 layers)")
    sched = HybridModelConfig.make_schedule(num_layers=8, moe_period=0, attn_period=0)
    print("  " + " · ".join(t.value for t in sched))

    subsection("Custom schedule: Jamba-style 16 layers (attn=8, moe=4)")
    sched2 = HybridModelConfig.make_schedule(num_layers=16, moe_period=4, attn_period=8)
    print("  " + " · ".join(t.value for t in sched2))


# ===========================================================================
# 5. Size scaling table
# ===========================================================================

def example_scaling():
    section("5. Size Scaling — parameter estimates across model sizes")

    configs_and_labels = [
        ("300M (d=1024, L=24)",  1024, 24,  8000, 16),
        ("1B   (d=2048, L=24)",  2048, 24, 32000, 32),
        ("3B   (d=2048, L=32)",  2048, 32, 32000, 64),
        ("7B   (d=4096, L=32)",  4096, 32, 32000, 64),
        ("13B  (d=5120, L=40)",  5120, 40, 32000, 64),
        ("32B  (d=7168, L=48)",  7168, 48, 32000, 128),
    ]

    print(f"\n  {'Size':22} {'d_model':>7} {'Layers':>6} {'Experts':>7} {'Est. Params':>12}")
    print("  " + "─" * 60)

    for label, d, L, vocab, experts in configs_and_labels:
        cfg = build_mamba3_reasoning_model(
            d_model=d, num_layers=L, vocab_size=vocab, num_experts=experts
        )
        p = estimate_parameters(cfg)
        total = p["total"]
        fmt = f"{total / 1e9:.2f}B" if total >= 1e9 else f"{total / 1e6:.0f}M"
        print(f"  {label:<22} {d:>7} {L:>6} {experts:>7} {fmt:>12}")


# ===========================================================================
# 6. JSON save / load round-trip
# ===========================================================================

def example_save_load(tmp_dir: str = "/tmp/swiftllm_examples"):
    section("6. Save / Load — JSON round-trip")

    cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)

    path = Path(tmp_dir) / "mamba3_reasoning_2048.json"
    cfg.to_json(path)
    print(f"  Saved config to: {path}")

    # Show a snippet of the JSON
    with path.open() as fh:
        raw = json.load(fh)
    print(f"  JSON keys       : {list(raw.keys())}")
    print(f"  mamba d_state   : {raw['mamba_config']['d_state']}")
    print(f"  rlm max_depth   : {raw['rlm_config']['max_depth']}")
    print(f"  dv min_conf     : {raw['dense_verification_config']['min_confidence']}")

    # Round-trip: load back
    cfg2 = HybridModelConfig.from_json(path)
    assert cfg2.d_model == cfg.d_model
    assert cfg2.rlm_config.max_depth == cfg.rlm_config.max_depth
    assert cfg2.dense_verification_config.min_confidence == cfg.dense_verification_config.min_confidence
    print("  Round-trip OK ✓")

    # Also demonstrate to_dict / from_dict
    d = cfg.to_dict()
    cfg3 = HybridModelConfig.from_dict(d)
    assert cfg3.num_layers == cfg.num_layers
    print("  to_dict/from_dict OK ✓")


# ===========================================================================
# 7. Custom layer schedule
# ===========================================================================

def example_custom_schedule():
    section("7. Custom Layer Schedule — per-layer type override")

    # Hand-craft a 12-layer schedule:
    #   layers 1-4  : Mamba-3
    #   layers 5-8  : Mamba-3 (with MoE on layer 8)
    #   layers 9-10 : Mamba-3
    #   layer  11   : attention + MoE
    #   layer  12   : Mamba-3

    schedule = [
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA_MOE,
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA_MOE,
        HybridLayerType.MAMBA,
        HybridLayerType.MAMBA,
        HybridLayerType.ATTENTION_MOE,
        HybridLayerType.MAMBA,
    ]

    cfg = (
        HybridModelBuilder(d_model=1024, num_layers=12, vocab_size=32000)
        .with_mamba3()
        .with_latent_moe(num_experts=16, moe_period=4)
        .with_rlm()
        .with_dense_verification()
        .with_layer_schedule(schedule)
        .build()
    )

    print("  Custom 12-layer schedule:")
    for i, lt in enumerate(cfg.layer_schedule, 1):
        print(f"    Layer {i:2d}: {lt.value}")

    print(f"\n  Mamba layers     : {cfg.num_mamba_layers}")
    print(f"  MoE layers       : {cfg.num_moe_layers}")
    print(f"  Attention layers : {cfg.num_attention_layers}")


# ===========================================================================
# 8. Engine integration (config-only, no GPU needed)
# ===========================================================================

def example_engine_integration():
    section("8. Engine Integration — EngineConfig + LLM (config-only demo)")

    print(textwrap.dedent("""
    To run a hybrid model via the SwiftLLM engine, pass the HybridModelConfig
    to EngineConfig.  The Rust backend deserialises the config and builds the
    full model graph from it.

    ── Code (requires GPU + compiled swiftllm._core) ────────────────────────

        from swiftllm import LLM, EngineConfig, SamplingParams
        from swiftllm import build_mamba3_reasoning_model

        # 1. Build the architecture config
        model_cfg = build_mamba3_reasoning_model(
            d_model=2048,
            num_layers=32,
            vocab_size=32000,
        )

        # 2. Pass it to the engine (alongside the weight checkpoint path)
        engine_cfg = EngineConfig(
            model="./weights/mamba3_reasoning_2b/",   # HF-style checkpoint dir
            model_config=model_cfg,                    # Python config → Rust bridge
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            max_num_seqs=128,
        )

        # 3. Create the engine and generate
        llm = LLM(engine_config=engine_cfg)

        outputs = llm.generate(
            ["Explain why the Riemann Hypothesis matters."],
            SamplingParams(temperature=0.7, max_tokens=512),
        )
        print(outputs[0].outputs[0].text)

    ── Async usage ───────────────────────────────────────────────────────────

        from swiftllm import AsyncLLM
        import asyncio

        async def main():
            llm = AsyncLLM(engine_config=engine_cfg)
            async for output in llm.generate_stream("Prove that √2 is irrational."):
                print(output.outputs[0].text, end="", flush=True)

        asyncio.run(main())

    ── RLM-specific sampling params ─────────────────────────────────────────

        from swiftllm import SamplingParams, RlmConfig
        from swiftllm.config import RlmConfig as InferenceRlmConfig

        # Override RLM depth per-request (model must have RLM block)
        outputs = llm.generate(
            ["What is 17 × 23 + 144 / 12?"],
            SamplingParams(
                temperature=0.0,     # greedy for math
                max_tokens=256,
                rlm_max_depth=2,     # override model default at inference time
            ),
        )
    """).rstrip())


# ===========================================================================
# 9. Training workflow (config-only)
# ===========================================================================

def example_training_workflow():
    section("9. Training Workflow — GRPO fine-tuning on reasoning model")

    print(textwrap.dedent("""
    Recommended two-phase training pipeline:

    Phase A — Supervised pre-training on the base model (no RLM/DV):

        from swiftllm import build_mamba3_base_model
        from swiftllm import Trainer, TrainingConfig, LoRAConfig

        base_cfg = build_mamba3_base_model(d_model=2048, num_layers=32)

        trainer = Trainer(
            model="./weights/mamba3_base_2b_init/",
            model_config=base_cfg,
            config=TrainingConfig(
                train_data="./data/openwebtext.jsonl",
                output_dir="./checkpoints/mamba3_base_2b/",
                num_epochs=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=3e-4,
            ),
        )
        trainer.train()

    Phase B — GRPO RL fine-tuning with RLM + Dense Verification enabled:

        from swiftllm import build_mamba3_reasoning_model
        from swiftllm import GrpoTrainer, GrpoConfig

        reasoning_cfg = build_mamba3_reasoning_model(d_model=2048, num_layers=32)

        grpo = GrpoTrainer(
            model="./checkpoints/mamba3_base_2b/",
            model_config=reasoning_cfg,   # adds RLM + DV on top of loaded weights
            config=GrpoConfig(
                train_data="./data/math_problems.jsonl",
                output_dir="./checkpoints/mamba3_reasoning_2b/",
                num_rollouts=8,
                kl_coeff=0.04,
                reward_model="./weights/math_reward_model/",
            ),
        )
        grpo.train()

    ── One-liner convenience function ───────────────────────────────────────

        from swiftllm import grpo_train, build_mamba3_reasoning_model

        grpo_train(
            model="./checkpoints/mamba3_base_2b/",
            model_config=build_mamba3_reasoning_model(d_model=2048, num_layers=32),
            train_data="hf:openai/gsm8k:train",
            output_dir="./checkpoints/mamba3_reasoning_gsm8k/",
            num_rollouts=8,
        )
    """).rstrip())


# ===========================================================================
# 10. Serialisation-only — inspect configs without GPU
# ===========================================================================

def example_serialisation_only():
    section("10. Serialisation-Only — inspect configs offline (no GPU)")

    cfg = build_mamba3_reasoning_model(d_model=4096, num_layers=48, num_experts=64)

    subsection("Full architecture summary (7B scale)")
    print(cfg.summary())

    subsection("Detailed parameter breakdown")
    print(parameter_summary(cfg))

    subsection("JSON snippet — mamba_config")
    d = cfg.to_dict()
    print(json.dumps(d["mamba_config"], indent=4))

    subsection("JSON snippet — rlm_config")
    print(json.dumps(d["rlm_config"], indent=4))

    subsection("JSON snippet — dense_verification_config")
    print(json.dumps(d["dense_verification_config"], indent=4))

    subsection("JSON snippet — first 8 layer types")
    print(json.dumps(d["layer_schedule"][:8], indent=4))


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SwiftLLM Hybrid Architecture Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--section", "-s",
        type=int,
        default=0,
        help="Run only this section number (1-10). Default: run all.",
    )
    args = parser.parse_args()

    examples = [
        example_quickstart,
        example_fluent_builder,
        example_component_configs,
        example_variants,
        example_scaling,
        example_save_load,
        example_custom_schedule,
        example_engine_integration,
        example_training_workflow,
        example_serialisation_only,
    ]

    if args.section:
        idx = args.section - 1
        if not (0 <= idx < len(examples)):
            print(f"Error: --section must be 1-{len(examples)}", file=sys.stderr)
            sys.exit(1)
        examples[idx]()
    else:
        for fn in examples:
            fn()

    print()
    print("=" * 72)
    print("  All examples complete.")
    print("=" * 72)
