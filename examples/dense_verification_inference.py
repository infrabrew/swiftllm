#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      dense_verification_inference.py
# PATH:      /examples/dense_verification_inference.py
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

"""Dense Verification Layer Inference Example

Demonstrates all four verification strategies offered by the
``DenseVerificationLayer``:

  DISABLED      — passthrough; no scoring.  Useful as a baseline.
  SCORE_ONLY    — compute token/step scores but always accept the first draft.
                  Good for observability dashboards and research logging.
  GATE          — reject drafts whose global score falls below
                  ``min_confidence``; accept the best available draft without
                  regenerating (single-attempt).
  GATE_AND_REGEN — full pipeline: reject and regenerate up to
                  ``max_regen_attempts`` times until a draft is accepted or
                  the budget is exhausted.

Architecture note
-----------------
The Python API wraps the Rust ``DenseVerificationLayer::verify_and_correct()``
path in ``crates/swiftllm-models/src/layers/dense_verification.rs``.
The Python implementation uses a heuristic per-token coherence score as a
proxy for the cross-attention verification signal while the Rust backend is
being wired to the Python runtime.

Usage
-----
    python examples/dense_verification_inference.py --model <path> [--strategy gate_and_regen]
    python examples/dense_verification_inference.py --model <path> --compare-all
    python examples/dense_verification_inference.py --model <path> --score-batch prompts.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

DEMO_PROMPTS = [
    "Explain why the sky is blue in 3 concise sentences.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the steps to perform a binary search on a sorted array.",
    "Why does adding salt lower the boiling point of water? Be precise.",
]


def _confidence_bar(score: float, width: int = 30) -> str:
    """ASCII confidence bar for terminal output."""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1%}"


def _print_dv_result(result, show_token_scores: bool = False):
    """Pretty-print a DenseVerificationOutput."""
    print(f"\n{'=' * 70}")
    print(f"PROMPT   : {result.prompt[:80]}{'...' if len(result.prompt) > 80 else ''}")
    print(f"{'=' * 70}")
    print(result.text)
    print(f"\n--- Dense Verification metadata ---")
    print(f"  Global confidence  : {_confidence_bar(result.global_score)}")
    print(f"  Accepted on attempt: {result.accepted_on_attempt}")
    print(f"  Low-confidence pos : {len(result.low_confidence_positions)} tokens")
    if result.step_scores:
        mean_step = sum(result.step_scores) / len(result.step_scores)
        print(f"  REPL step scores   : {len(result.step_scores)} steps, "
              f"mean={mean_step:.2%}")

    if show_token_scores and result.token_scores:
        n_show = min(20, len(result.token_scores))
        scores_str = "  ".join(f"{s:.2f}" for s in result.token_scores[:n_show])
        print(f"\n  First {n_show} token scores: {scores_str}")
        if len(result.token_scores) > n_show:
            print(f"  ... ({len(result.token_scores) - n_show} more)")


def demo_strategy(llm, prompt: str, strategy_name: str, verbose: bool,
                  min_confidence: float = 0.80, max_regen: int = 3):
    """Run one strategy and print results."""
    from swiftllm.config import DenseVerificationConfig, VerificationStrategy, SamplingParams

    strategy_map = {
        "disabled":       VerificationStrategy.DISABLED,
        "score_only":     VerificationStrategy.SCORE_ONLY,
        "gate":           VerificationStrategy.GATE,
        "gate_and_regen": VerificationStrategy.GATE_AND_REGEN,
    }
    strategy = strategy_map[strategy_name]

    print(f"\n{'#' * 70}")
    print(f"#  STRATEGY: {strategy_name.upper()}")
    if strategy != VerificationStrategy.DISABLED:
        print(f"#  min_confidence={min_confidence:.0%}  max_regen={max_regen}")
    print(f"{'#' * 70}")

    cfg = DenseVerificationConfig(
        strategy=strategy,
        min_confidence=min_confidence,
        max_regen_attempts=max_regen,
        score_repl_steps=True,
    )
    params = SamplingParams(temperature=0.7, max_tokens=256)

    t0 = time.time()
    results = llm.generate_with_dense_verification([prompt], config=cfg, base_params=params)
    elapsed = time.time() - t0

    _print_dv_result(results[0], show_token_scores=verbose)
    print(f"  Generation time    : {elapsed:.2f}s")
    return results[0]


def compare_all_strategies(llm, prompt: str, verbose: bool):
    """Run all four strategies on the same prompt and print a comparison table."""
    strategies = ["disabled", "score_only", "gate", "gate_and_regen"]
    results = {}
    for s in strategies:
        r = demo_strategy(llm, prompt, s, verbose=verbose)
        results[s] = r

    print(f"\n{'=' * 70}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<20}  {'Global Score':>14}  {'Attempt':>8}  {'Low-conf':>9}")
    print(f"{'-'*20}  {'-'*14}  {'-'*8}  {'-'*9}")
    for s, r in results.items():
        print(f"{s:<20}  {r.global_score:>13.1%}  {r.accepted_on_attempt:>8}  "
              f"{len(r.low_confidence_positions):>9}")
    return results


def score_batch_from_file(llm, file_path: str, min_confidence: float,
                           output_path: str | None, verbose: bool):
    """Score every prompt in a text file and write a JSON report."""
    from swiftllm.config import DenseVerificationConfig, VerificationStrategy, SamplingParams

    prompts = Path(file_path).read_text().splitlines()
    prompts = [p.strip() for p in prompts if p.strip()]

    if not prompts:
        print(f"No prompts found in {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(prompts)} prompts from {file_path}…")

    cfg = DenseVerificationConfig(
        strategy=VerificationStrategy.SCORE_ONLY,
        min_confidence=min_confidence,
        score_repl_steps=True,
    )
    params = SamplingParams(temperature=0.7, max_tokens=256)

    t0 = time.time()
    results = llm.generate_with_dense_verification(prompts, config=cfg, base_params=params)
    elapsed = time.time() - t0

    report = []
    for r in results:
        report.append({
            "prompt": r.prompt,
            "text": r.text,
            "global_score": r.global_score,
            "accepted": r.global_score >= min_confidence,
            "token_scores": r.token_scores,
            "step_scores": r.step_scores,
            "low_confidence_positions": r.low_confidence_positions,
        })

    print(f"\n--- Batch scoring complete ({elapsed:.2f}s) ---")
    accepted = sum(1 for r in report if r["accepted"])
    print(f"  Prompts       : {len(report)}")
    print(f"  Accepted      : {accepted} ({accepted/len(report):.0%})")
    mean_score = sum(r["global_score"] for r in report) / len(report)
    print(f"  Mean score    : {mean_score:.2%}")

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"  Report saved  : {output_path}")
    else:
        print(json.dumps(report, indent=2))


def demo_confidence_calibration(llm, verbose: bool):
    """Show how confidence thresholds affect acceptance rate across a prompt set."""
    from swiftllm.config import DenseVerificationConfig, VerificationStrategy, SamplingParams

    print(f"\n{'#' * 70}")
    print("#  DEMO: Confidence Calibration")
    print(f"{'#' * 70}")
    print("Running SCORE_ONLY on a set of prompts, then showing acceptance")
    print("rate at different confidence thresholds.\n")

    cfg = DenseVerificationConfig(
        strategy=VerificationStrategy.SCORE_ONLY,
        min_confidence=0.0,   # accept everything; we'll threshold manually
        score_repl_steps=False,
    )
    params = SamplingParams(temperature=0.7, max_tokens=200)
    results = llm.generate_with_dense_verification(DEMO_PROMPTS, config=cfg, base_params=params)
    scores = [r.global_score for r in results]

    print(f"  Scored {len(scores)} prompts:")
    for prompt, score in zip(DEMO_PROMPTS, scores):
        short = prompt[:55] + ("..." if len(prompt) > 55 else "")
        print(f"    {score:.2%}  {short}")

    print(f"\n  Acceptance rate by threshold:")
    print(f"  {'Threshold':>12}  {'Accepted':>10}  {'Rate':>8}")
    for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        n_accepted = sum(1 for s in scores if s >= thresh)
        print(f"  {thresh:>12.0%}  {n_accepted:>10}/{len(scores)}  "
              f"{n_accepted/len(scores):>7.0%}")


def main():
    parser = argparse.ArgumentParser(
        description="SwiftLLM Dense Verification inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Custom prompt")
    parser.add_argument("--strategy",
                        choices=["disabled", "score_only", "gate", "gate_and_regen"],
                        default="gate_and_regen",
                        help="Verification strategy (default: gate_and_regen)")
    parser.add_argument("--min-confidence",
                        type=float, default=0.80,
                        help="Confidence threshold for GATE / GATE_AND_REGEN (default: 0.80)")
    parser.add_argument("--max-regen",
                        type=int, default=3,
                        help="Max regeneration attempts for GATE_AND_REGEN (default: 3)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all four strategies on the same prompt and compare")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run confidence calibration demo across built-in prompts")
    parser.add_argument("--score-batch",
                        metavar="FILE",
                        help="Score all prompts in FILE (one per line) with SCORE_ONLY")
    parser.add_argument("-o", "--output",
                        help="Output JSON file for --score-batch report")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-token confidence scores")
    parser.add_argument("--download-dir", default=None,
                        help="Directory for downloading models")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    from swiftllm import LLM
    llm = LLM(model=args.model, download_dir=args.download_dir)

    if args.score_batch:
        score_batch_from_file(llm, args.score_batch, args.min_confidence,
                              args.output, args.verbose)
        return

    if args.calibrate:
        demo_confidence_calibration(llm, verbose=args.verbose)
        return

    prompt = args.prompt or DEMO_PROMPTS[0]

    if args.compare_all:
        compare_all_strategies(llm, prompt, verbose=args.verbose)
    else:
        demo_strategy(
            llm, prompt, args.strategy,
            verbose=args.verbose,
            min_confidence=args.min_confidence,
            max_regen=args.max_regen,
        )


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
# END OF FILE: dense_verification_inference.py
# REPO PATH:   /swiftllm/examples/dense_verification_inference.py
# SEE ALSO:
#   python/swiftllm/engine.py                         LLM.generate_with_dense_verification()
#   python/swiftllm/config.py                         DenseVerificationConfig, VerificationStrategy
#   crates/swiftllm-models/src/layers/dense_verification.rs  DenseVerificationLayer::verify_and_correct()
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
