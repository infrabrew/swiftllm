#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      self_consistency.py
# PATH:      /examples/self_consistency.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/engine.py    LLM.generate_with_self_consistency()
#   - python/swiftllm/config.py    SelfConsistencyConfig, AnswerExtractor
#   - python/swiftllm/sampling.py  SelfConsistencySampler (offline vote helper)
# SEE ALSO:
#   - crates/swiftllm-core/src/sampling/self_consistency.rs  Rust implementation
#   - crates/swiftllm-core/src/inference/verification.rs     downstream verifier
#   - examples/grpo_training.py                              GRPO training example
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

"""Self-Consistency Inference Example (Wang et al., 2022)

Demonstrates three usage patterns for SwiftLLM's self-consistency module:

1. Basic self-consistency via LLM.generate_with_self_consistency()
2. Custom answer extractor (sentinel-based)
3. Offline vote over pre-generated texts using SelfConsistencySampler directly

Usage:
    python examples/self_consistency.py --model <model_path> [options]

    # Quick test with a small model:
    python examples/self_consistency.py \\
        --model TheBloke/Llama-2-7B-GGUF \\
        --model-file llama-2-7b.Q4_K_M.gguf \\
        --num-samples 4
"""

import argparse
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwiftLLM self-consistency inference demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of independent reasoning chains per question")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (should be > 0 for diversity)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per sample")
    parser.add_argument("--extractor", default="heuristic",
                        choices=["heuristic", "sentinel", "last_line", "xml_tag"],
                        help="Answer extraction strategy")
    parser.add_argument("--sentinel", default="The answer is",
                        help="Sentinel string (used when --extractor=sentinel)")
    parser.add_argument("--show-all-answers", action="store_true",
                        help="Print all extracted answers, not just the majority")
    return parser.parse_args()


MATH_QUESTIONS = [
    "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. "
    "What is the total distance travelled? Think step by step.",

    "A store sells apples for $0.75 each. A customer buys a dozen apples and "
    "pays with a $20 bill. How much change should they receive? Think step by step.",

    "If 3 workers can build a wall in 12 days, how many days will it take "
    "6 workers to build the same wall? Think step by step.",
]

CHAIN_OF_THOUGHT_PREFIX = (
    "Solve the following problem step by step. "
    "Show your reasoning clearly, then state 'The answer is X.' at the end.\n\n"
)


def demo_basic_self_consistency(llm, args):
    """Demo 1: Basic usage via LLM.generate_with_self_consistency()."""
    from swiftllm.config import SelfConsistencyConfig, AnswerExtractor

    extractor_map = {
        "heuristic": AnswerExtractor.HEURISTIC,
        "sentinel": AnswerExtractor.AFTER_SENTINEL,
        "last_line": AnswerExtractor.LAST_LINE,
        "xml_tag": AnswerExtractor.XML_TAG,
    }

    cfg = SelfConsistencyConfig(
        num_samples=args.num_samples,
        temperature=args.temperature,
        extractor=extractor_map[args.extractor],
        answer_sentinel=args.sentinel,
    )

    prompts = [CHAIN_OF_THOUGHT_PREFIX + q for q in MATH_QUESTIONS]

    print("\n" + "=" * 70)
    print("DEMO 1: Self-Consistency via LLM.generate_with_self_consistency()")
    print("=" * 70)

    results = llm.generate_with_self_consistency(
        prompts,
        config=cfg,
    )

    for question, result in zip(MATH_QUESTIONS, results):
        print(f"\nQuestion: {question[:80]}...")
        print(f"Majority answer : {result.answer!r}")
        print(f"Agreement       : {result.vote_fraction:.0%} of {args.num_samples} samples")

        if args.show_all_answers:
            print("All answers:")
            for i, ans in enumerate(result.all_answers):
                print(f"  Sample {i+1}: {ans!r}")


def demo_sentinel_extractor(llm, args):
    """Demo 2: Custom sentinel-based answer extraction."""
    from swiftllm.config import SelfConsistencyConfig, AnswerExtractor

    sentinel = "FINAL ANSWER:"
    sentinel_prompt = (
        "Solve step by step. End your response with exactly:\n"
        f"{sentinel} <your answer here>\n\n"
    )

    cfg = SelfConsistencyConfig(
        num_samples=args.num_samples,
        temperature=args.temperature,
        extractor=AnswerExtractor.AFTER_SENTINEL,
        answer_sentinel=sentinel,
    )

    question = MATH_QUESTIONS[0]
    prompt = sentinel_prompt + question

    print("\n" + "=" * 70)
    print("DEMO 2: Custom Sentinel-Based Extraction")
    print("=" * 70)
    print(f"Sentinel: {sentinel!r}")
    print(f"Question: {question[:80]}...")

    results = llm.generate_with_self_consistency([prompt], config=cfg)
    r = results[0]
    print(f"Majority answer : {r.answer!r}")
    print(f"Agreement       : {r.vote_fraction:.0%}")


def demo_offline_vote(args):
    """Demo 3: Offline vote over pre-existing generated texts (no LLM required)."""
    from swiftllm.sampling import SelfConsistencySampler
    from swiftllm.config import SelfConsistencyConfig, AnswerExtractor

    # Simulate pre-generated reasoning chains (as if from a prior batch run)
    pre_generated = [
        "The train goes 60 mph × 2.5 h = 150 miles, then 80 mph × 1.5 h = 120 miles. "
        "Total distance = 150 + 120 = 270 miles. The answer is 270.",

        "First segment: 60 * 2.5 = 150 miles. Second: 80 * 1.5 = 120 miles. "
        "Total: 270 miles. The answer is 270 miles.",

        "Distance = speed × time. Part 1: 60 × 2.5 = 150. Part 2: 80 × 1.5 = 120. "
        "Sum = 270. The answer is 270.",

        "The answer is 150 + 120 = 280 miles.",  # intentional wrong answer

        "I calculate 60*2.5=150 and 80*1.5=120, giving 270 total. The answer is 270.",

        "Part one: 150 miles. Part two: 120 miles. Grand total: 270 miles. "
        "The answer is 270.",

        "Using d=rt: 60(2.5)+80(1.5)=150+120=270. The answer is 270.",

        "I believe the total is 240 miles.",  # another wrong answer
    ]

    cfg = SelfConsistencyConfig(
        num_samples=len(pre_generated),
        extractor=AnswerExtractor.HEURISTIC,
        temperature=0.8,  # not used in offline mode but required by config
    )
    sampler = SelfConsistencySampler(cfg)
    result = sampler.vote(pre_generated)

    print("\n" + "=" * 70)
    print("DEMO 3: Offline Vote Over Pre-Generated Texts")
    print("=" * 70)
    print(f"Majority answer : {result.answer!r}")
    print(f"Agreement       : {result.vote_fraction:.0%} ({sum(1 for a in result.all_answers if a == result.answer)}"
          f"/{len(result.all_answers)} samples)")

    if args.show_all_answers:
        print("Per-sample extractions:")
        for i, ans in enumerate(result.all_answers):
            marker = "✓" if ans == result.answer else "✗"
            print(f"  [{marker}] Sample {i+1}: {ans!r}")


def main():
    args = parse_args()

    print("SwiftLLM Self-Consistency Demo")
    print(f"  Model      : {args.model}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Extractor  : {args.extractor}")

    # Demo 3 doesn't need the LLM — run it first
    demo_offline_vote(args)

    # Demos 1 and 2 require an LLM
    try:
        from swiftllm import LLM, SamplingParams
        llm = LLM(model=args.model)
        demo_basic_self_consistency(llm, args)
        demo_sentinel_extractor(llm, args)
    except Exception as exc:
        print(f"\n[Note] LLM demos skipped: {exc}", file=sys.stderr)
        print("Install a model and pass --model to run the full demo.", file=sys.stderr)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
# END OF FILE: self_consistency.py
# REPO PATH:   /swiftllm/examples/self_consistency.py
# INTEGRATES:  engine.py · sampling.py · config.py
#              Rust: self_consistency.rs · verification.rs
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
