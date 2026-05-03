#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      rlm_inference.py
# PATH:      /examples/rlm_inference.py
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

"""Recursive Language Model (RLM) Inference Example

Demonstrates the three operating modes of the RLM layer:

  1. SHALLOW  — single pass with a shallow recursion budget (depth=1).
                Good for simple question answering that might benefit from
                one decomposition step.

  2. REASONING — default reasoning mode (depth=3, REPL enabled).
                 The model can decompose hard problems into sub-problems,
                 bind intermediate results to named variables, and build
                 solutions bottom-up.  Best for math, coding, and multi-step
                 reasoning tasks.

  3. AGENTIC  — deep recursion with full REPL (depth=5).
                Suited to agentic tasks that require multi-level planning,
                tool simulation, and symbolic computation.

Architecture note
-----------------
The Python API wraps the Rust ``RlmLayer::forward_with_repl()`` path in
``crates/swiftllm-models/src/layers/rlm.rs``.  The Python implementation
parses ``REPL:`` annotations from the generated text as a lightweight bridge
until the Rust backend is fully wired to the Python runtime.

Usage
-----
    python examples/rlm_inference.py --model <path> [--mode reasoning] [--prompt "..."]
    python examples/rlm_inference.py --model <path> --demo-all
"""

import argparse
import json
import sys
import time

DEMO_PROMPTS = {
    "math": (
        "Prove by induction that the sum of the first n natural numbers equals "
        "n(n+1)/2.  Show each inductive step clearly."
    ),
    "coding": (
        "Write a Python function that returns the nth Fibonacci number using "
        "memoisation.  Also provide a brief correctness proof and complexity analysis."
    ),
    "planning": (
        "Design a step-by-step plan to migrate a 50 TB relational database from "
        "on-premises PostgreSQL to a cloud-managed Aurora PostgreSQL with zero "
        "downtime.  List risks and mitigations for each phase."
    ),
    "reasoning": (
        "A farmer has 100 metres of fencing and wants to enclose a rectangular "
        "field adjacent to a barn wall (so only three sides need fencing). "
        "What dimensions maximise the enclosed area?"
    ),
}


def _print_rlm_result(result, verbose: bool = False):
    """Pretty-print an RlmOutput."""
    print(f"\n{'=' * 70}")
    print(f"PROMPT:  {result.prompt[:80]}{'...' if len(result.prompt) > 80 else ''}")
    print(f"{'=' * 70}")
    print(result.text)
    print(f"\n--- RLM metadata ---")
    print(f"  Max recursion depth used : {result.recursion_depth_used}")
    print(f"  REPL steps logged        : {len(result.repl_trace)}")
    print(f"  Variable bindings        : {len(result.repl_variables)}")
    print(f"  Early exits (scheduler)  : {result.early_exits}")

    if result.repl_variables and verbose:
        print(f"\n  Variable bindings:")
        for name, val in result.repl_variables.items():
            print(f"    {name} = {str(val)[:80]}")

    if result.repl_trace and verbose:
        print(f"\n  REPL execution trace ({len(result.repl_trace)} steps):")
        for i, step in enumerate(result.repl_trace):
            step_type = step.get("type", "?").upper()
            if step_type == "ASSIGN":
                print(f"    [{i}] ASSIGN  {step.get('name')} = {str(step.get('output', ''))[:60]}")
            elif step_type == "COMPUTE":
                print(f"    [{i}] COMPUTE {str(step.get('expression', ''))[:70]}")
            elif step_type == "VERIFY":
                conf = step.get("confidence", 1.0)
                print(f"    [{i}] VERIFY  conf={conf:.2f}  {str(step.get('claim', ''))[:50]}")
            elif step_type == "RECURSE":
                print(f"    [{i}] RECURSE depth={step.get('depth')} "
                      f"{str(step.get('subproblem', ''))[:50]}")


def demo_shallow(llm, prompt: str, verbose: bool):
    """Mode 1 — SHALLOW (depth=1, REPL enabled)."""
    from swiftllm.config import RlmConfig, RlmMode, SamplingParams

    print("\n" + "#" * 70)
    print("#  MODE: SHALLOW  (max_depth=1, enable_repl=True)")
    print("#" * 70)

    cfg = RlmConfig(
        mode=RlmMode.SHALLOW,
        max_depth=1,
        enable_repl=True,
        early_exit_threshold=0.95,
    )
    params = SamplingParams(temperature=0.6, max_tokens=512)

    t0 = time.time()
    results = llm.generate_with_rlm([prompt], config=cfg, base_params=params)
    elapsed = time.time() - t0

    _print_rlm_result(results[0], verbose=verbose)
    print(f"  Generation time          : {elapsed:.2f}s")
    return results[0]


def demo_reasoning(llm, prompt: str, verbose: bool):
    """Mode 2 — REASONING (depth=3, REPL enabled) — default."""
    from swiftllm.config import RlmConfig, RlmMode, SamplingParams

    print("\n" + "#" * 70)
    print("#  MODE: REASONING  (max_depth=3, enable_repl=True)")
    print("#" * 70)

    cfg = RlmConfig(
        mode=RlmMode.REASONING,
        max_depth=3,
        enable_repl=True,
        var_binding_slots=32,
        early_exit_threshold=0.92,
    )
    params = SamplingParams(temperature=0.7, max_tokens=768)

    t0 = time.time()
    results = llm.generate_with_rlm([prompt], config=cfg, base_params=params)
    elapsed = time.time() - t0

    _print_rlm_result(results[0], verbose=verbose)
    print(f"  Generation time          : {elapsed:.2f}s")
    return results[0]


def demo_agentic(llm, prompt: str, verbose: bool):
    """Mode 3 — AGENTIC (depth=5, REPL enabled, larger token budget)."""
    from swiftllm.config import RlmConfig, RlmMode, SamplingParams

    print("\n" + "#" * 70)
    print("#  MODE: AGENTIC  (max_depth=5, enable_repl=True)")
    print("#" * 70)

    cfg = RlmConfig(
        mode=RlmMode.AGENTIC,
        max_depth=5,
        enable_repl=True,
        var_binding_slots=64,
        early_exit_threshold=0.90,
    )
    params = SamplingParams(temperature=0.8, max_tokens=512)

    t0 = time.time()
    results = llm.generate_with_rlm([prompt], config=cfg, base_params=params)
    elapsed = time.time() - t0

    _print_rlm_result(results[0], verbose=verbose)
    print(f"  Generation time          : {elapsed:.2f}s")
    return results[0]


def demo_no_repl(llm, prompt: str, verbose: bool):
    """Variant — REASONING without REPL (pure recursive decomposition)."""
    from swiftllm.config import RlmConfig, RlmMode, SamplingParams

    print("\n" + "#" * 70)
    print("#  MODE: REASONING  (max_depth=3, enable_repl=False)  — no REPL sandbox")
    print("#" * 70)

    cfg = RlmConfig(
        mode=RlmMode.REASONING,
        max_depth=3,
        enable_repl=False,
    )
    params = SamplingParams(temperature=0.7, max_tokens=512)

    t0 = time.time()
    results = llm.generate_with_rlm([prompt], config=cfg, base_params=params)
    elapsed = time.time() - t0

    _print_rlm_result(results[0], verbose=verbose)
    print(f"  Generation time          : {elapsed:.2f}s")
    return results[0]


def demo_variable_binding(llm, verbose: bool):
    """Show variable binding: a multi-step algebraic derivation."""
    from swiftllm.config import RlmConfig, RlmMode, SamplingParams

    prompt = (
        "Solve the following system step by step, storing each intermediate "
        "result as a named variable:\n\n"
        "  2x + 3y = 12\n"
        "  4x -  y =  2\n\n"
        "Use REPL:ASSIGN annotations to bind x and y once found."
    )

    print("\n" + "#" * 70)
    print("#  DEMO: Variable Binding  (REPL:ASSIGN annotations)")
    print("#" * 70)
    print(f"  Prompt: {prompt[:120]}...")

    cfg = RlmConfig(
        mode=RlmMode.REASONING,
        max_depth=2,
        enable_repl=True,
        var_binding_slots=16,
    )
    params = SamplingParams(temperature=0.3, max_tokens=512)

    t0 = time.time()
    results = llm.generate_with_rlm([prompt], config=cfg, base_params=params)
    elapsed = time.time() - t0

    _print_rlm_result(results[0], verbose=True)
    print(f"  Generation time          : {elapsed:.2f}s")

    # Show the final variable bindings prominently
    if results[0].repl_variables:
        print("\n  Final variable store:")
        for k, v in results[0].repl_variables.items():
            print(f"    {k:20s} = {v}")

    return results[0]


def main():
    parser = argparse.ArgumentParser(
        description="SwiftLLM RLM inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Custom prompt (uses a built-in reasoning prompt if omitted)")
    parser.add_argument("--prompt-type",
                        choices=list(DEMO_PROMPTS.keys()), default="math",
                        help="Built-in prompt type (default: math)")
    parser.add_argument("--mode",
                        choices=["shallow", "reasoning", "agentic", "no-repl"],
                        default="reasoning",
                        help="RLM operating mode (default: reasoning)")
    parser.add_argument("--demo-all", action="store_true",
                        help="Run all three modes + variable-binding demo in sequence")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full REPL trace and variable bindings")
    parser.add_argument("--json", action="store_true",
                        help="Emit results as JSON")
    parser.add_argument("--download-dir", default=None,
                        help="Directory for downloading models")
    args = parser.parse_args()

    # Build prompt
    prompt = args.prompt or DEMO_PROMPTS[args.prompt_type]

    # Initialise LLM
    print(f"Loading model: {args.model}")
    from swiftllm import LLM
    llm = LLM(
        model=args.model,
        download_dir=args.download_dir,
    )

    json_results = []

    if args.demo_all:
        r1 = demo_shallow(llm, prompt, verbose=args.verbose)
        r2 = demo_reasoning(llm, prompt, verbose=args.verbose)
        r3 = demo_agentic(llm, prompt, verbose=args.verbose)
        r4 = demo_no_repl(llm, prompt, verbose=args.verbose)
        r5 = demo_variable_binding(llm, verbose=args.verbose)
        json_results = [r1, r2, r3, r4, r5]
    else:
        dispatch = {
            "shallow": demo_shallow,
            "reasoning": demo_reasoning,
            "agentic": demo_agentic,
            "no-repl": demo_no_repl,
        }
        result = dispatch[args.mode](llm, prompt, verbose=args.verbose)
        json_results = [result]

    if args.json:
        out = []
        for r in json_results:
            out.append({
                "prompt": r.prompt,
                "text": r.text,
                "recursion_depth_used": r.recursion_depth_used,
                "repl_trace": r.repl_trace,
                "repl_variables": r.repl_variables,
                "early_exits": r.early_exits,
            })
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
# END OF FILE: rlm_inference.py
# REPO PATH:   /swiftllm/examples/rlm_inference.py
# SEE ALSO:
#   python/swiftllm/engine.py             LLM.generate_with_rlm()
#   python/swiftllm/config.py             RlmConfig, RlmMode
#   crates/swiftllm-models/src/layers/rlm.rs  RlmLayer::forward_with_repl()
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
