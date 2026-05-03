#!/usr/bin/env python3
# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      dataset_ingestion.py
# PATH:      /examples/dataset_ingestion.py
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

"""SwiftLLM Dataset Ingestion Example

Shows how to convert directories and individual files of any supported format
into JSONL training data, then feed that data directly to fine_tune().

Supported input formats
-----------------------
  Plain text : .txt  .md  .rst  .log
  Code       : .py  .js  .ts  .rs  .go  .java  .c  .cpp  .cs  .rb  .sql  .sh
               .swift  .kt  .scala  .yaml  .toml  and ~40 more extensions
  Documents  : .pdf   (pip install pdfplumber)
               .docx  (pip install python-docx)
  Web        : .html  .htm  .xml  (pip install beautifulsoup4)
  Structured : .csv   .json  .jsonl

Output formats
--------------
  pretraining    {"text": "..."}
  sft_messages   {"messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}
  sft_completion {"prompt": "...", "completion": "..."}
  code           {"prompt": "# python\\n# File: foo.py\\n", "completion": "<code>"}

Usage
-----
    # Ingest a single directory
    python examples/dataset_ingestion.py --input ./docs/ --output train.jsonl

    # Ingest multiple sources (dir + PDF + CSV)
    python examples/dataset_ingestion.py \\
        --input ./docs/ paper.pdf qa_pairs.csv \\
        --output train.jsonl --format sft_completion

    # Code fine-tuning from a source tree
    python examples/dataset_ingestion.py \\
        --input ./src/ --output code_train.jsonl \\
        --format code --extensions .py,.rs,.go

    # Run all demo modes (creates synthetic sample files)
    python examples/dataset_ingestion.py --demo-all --output-dir ./demo_output/
"""

import argparse
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Synthetic sample file content (used when --demo-all is set)
# ---------------------------------------------------------------------------

_SAMPLE_TXT = textwrap.dedent("""\
    Introduction to Large Language Models

    Large language models (LLMs) are neural networks trained on vast corpora of
    text to predict the next token in a sequence. They are based on the
    Transformer architecture, which uses self-attention to model dependencies
    between all positions in the input simultaneously.

    Training involves minimising the cross-entropy loss over billions of
    parameters, typically using Adam or its variants with a cosine learning-rate
    schedule and mixed-precision arithmetic.

    Applications range from question answering and summarisation to code
    generation and mathematical reasoning.
""")

_SAMPLE_PY = textwrap.dedent("""\
    \"\"\"Fibonacci with memoisation.\"\"\"

    from functools import lru_cache


    @lru_cache(maxsize=None)
    def fib(n: int) -> int:
        \"\"\"Return the nth Fibonacci number (0-indexed).\"\"\"
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)


    def fib_sequence(length: int) -> list[int]:
        \"\"\"Return the first *length* Fibonacci numbers.\"\"\"
        return [fib(i) for i in range(length)]


    if __name__ == "__main__":
        print(fib_sequence(20))
""")

_SAMPLE_RS = textwrap.dedent("""\
    //! Fast Fibonacci in Rust.

    /// Return the nth Fibonacci number.
    pub fn fib(n: u64) -> u64 {
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 0..n {
            (a, b) = (b, a.wrapping_add(b));
        }
        a
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_fib() {
            assert_eq!(fib(0), 0);
            assert_eq!(fib(1), 1);
            assert_eq!(fib(10), 55);
        }
    }
""")

_SAMPLE_MD = textwrap.dedent("""\
    # SwiftLLM Quick-Start

    ## Installation

    ```bash
    pip install swiftllm
    ```

    ## Basic Inference

    ```python
    from swiftllm import LLM, SamplingParams

    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    outputs = llm.generate(
        ["What is the capital of France?"],
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    print(outputs[0].outputs[0].text)
    ```

    ## Fine-Tuning

    ```python
    from swiftllm.training import fine_tune

    trainer = fine_tune(
        model="meta-llama/Llama-2-7b-hf",
        train_data="./data/train.jsonl",
        lora_r=16,
        num_epochs=3,
    )
    ```
""")

_SAMPLE_CSV = textwrap.dedent("""\
    prompt,completion
    "What is a transformer model?","A transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel rather than sequentially."
    "Explain gradient descent.","Gradient descent is an optimisation algorithm that iteratively adjusts model parameters in the direction that most reduces the loss function."
    "What is LoRA?","Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that trains small adapter matrices rather than updating all model weights."
    "What does GRPO stand for?","Group Relative Policy Optimisation — a reinforcement learning fine-tuning method that uses group-relative advantage estimation without a critic model."
    "What is PagedAttention?","PagedAttention is a memory management technique for LLM serving that allocates KV-cache memory in non-contiguous pages, reducing fragmentation and enabling higher throughput."
""")

_SAMPLE_JSONL = "\n".join([
    json.dumps({"text": "The attention mechanism allows the model to focus on relevant parts of the input when generating each output token."}),
    json.dumps({"text": "PagedAttention divides the KV cache into fixed-size pages managed like virtual memory, enabling near-zero memory waste."}),
    json.dumps({"prompt": "What is beam search?", "completion": "Beam search maintains the top-k most probable sequences at each step, trading compute for higher-quality outputs."}),
    json.dumps({"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is temperature sampling?"},
        {"role": "assistant", "content": "Temperature scaling divides logits before softmax, increasing (T>1) or decreasing (T<1) output randomness."},
    ]}),
])


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _create_sample_files(base_dir: Path) -> None:
    """Write synthetic sample files for demo purposes."""
    (base_dir / "text").mkdir(parents=True, exist_ok=True)
    (base_dir / "code" / "python").mkdir(parents=True, exist_ok=True)
    (base_dir / "code" / "rust").mkdir(parents=True, exist_ok=True)
    (base_dir / "structured").mkdir(parents=True, exist_ok=True)

    (base_dir / "text" / "intro.txt").write_text(_SAMPLE_TXT)
    (base_dir / "text" / "quickstart.md").write_text(_SAMPLE_MD)
    (base_dir / "code" / "python" / "fib.py").write_text(_SAMPLE_PY)
    (base_dir / "code" / "rust" / "fib.rs").write_text(_SAMPLE_RS)
    (base_dir / "structured" / "qa_pairs.csv").write_text(_SAMPLE_CSV)
    (base_dir / "structured" / "extra.jsonl").write_text(_SAMPLE_JSONL)

    print(f"  Created sample files in {base_dir}/")
    for f in sorted(base_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(base_dir)}")


def _print_sample_records(jsonl_path: str, n: int = 3) -> None:
    """Print the first n records from a JSONL file."""
    with open(jsonl_path) as fh:
        for i, line in enumerate(fh):
            if i >= n:
                print(f"  … (see {jsonl_path} for all records)")
                break
            rec = json.loads(line)
            # Truncate long values for display
            display = {}
            for k, v in rec.items():
                if isinstance(v, str) and len(v) > 120:
                    display[k] = v[:117] + "..."
                elif isinstance(v, list) and v:
                    display[k] = f"[{len(v)} items]"
                else:
                    display[k] = v
            print(f"  [{i}] {json.dumps(display, ensure_ascii=False)}")


def demo_pretraining(sample_dir: Path, output_dir: Path) -> None:
    """Mode 1 — Pretraining format: {\"text\": \"...\"}"""
    from swiftllm.dataset import DatasetIngester, DatasetFormat, IngestionConfig

    print("\n" + "#" * 70)
    print("#  MODE: PRETRAINING  ({\"text\": \"...\"})")
    print("#  Reads all supported file types from a directory tree")
    print("#" * 70)

    out = str(output_dir / "pretraining.jsonl")
    cfg = IngestionConfig(
        input_paths=[str(sample_dir)],
        output_path=out,
        format=DatasetFormat.PRETRAINING,
        chunk_size=512,
        chunk_overlap=64,
        min_length=30,
        deduplicate=True,
        verbose=True,
    )
    result = DatasetIngester(cfg).ingest()
    print()
    print(result.summary())
    print("\n  Sample records:")
    _print_sample_records(out)


def demo_sft_messages(sample_dir: Path, output_dir: Path) -> None:
    """Mode 2 — SFT messages format: {\"messages\": [...]}"""
    from swiftllm.dataset import DatasetIngester, DatasetFormat, IngestionConfig

    print("\n" + "#" * 70)
    print("#  MODE: SFT_MESSAGES  ({\"messages\": [{\"role\": ...}, ...]})")
    print("#  Existing prompt/completion pairs are converted to messages format")
    print("#  Raw text chunks are wrapped in a user/assistant turn pair")
    print("#" * 70)

    out = str(output_dir / "sft_messages.jsonl")
    cfg = IngestionConfig(
        input_paths=[str(sample_dir)],
        output_path=out,
        format=DatasetFormat.SFT_MESSAGES,
        chunk_size=512,
        chunk_overlap=64,
        min_length=30,
        system_prompt="You are a knowledgeable AI assistant specialising in machine learning.",
        sft_user_template="Continue the following passage:\n\n{text}",
        deduplicate=True,
    )
    result = DatasetIngester(cfg).ingest()
    print()
    print(result.summary())
    print("\n  Sample records:")
    _print_sample_records(out)


def demo_sft_completion(sample_dir: Path, output_dir: Path) -> None:
    """Mode 3 — SFT completion format: {\"prompt\": \"...\", \"completion\": \"...\"}"""
    from swiftllm.dataset import DatasetIngester, DatasetFormat, IngestionConfig

    print("\n" + "#" * 70)
    print("#  MODE: SFT_COMPLETION  ({\"prompt\": \"...\", \"completion\": \"...\"})")
    print("#  CSV/JSONL with prompt+completion columns are passed through directly")
    print("#  Raw text is split 75/25 into a prompt and completion")
    print("#" * 70)

    out = str(output_dir / "sft_completion.jsonl")
    cfg = IngestionConfig(
        input_paths=[str(sample_dir)],
        output_path=out,
        format=DatasetFormat.SFT_COMPLETION,
        chunk_size=512,
        chunk_overlap=0,
        min_length=30,
        sft_user_template="Continue the following passage:\n\n{text}",
        deduplicate=True,
    )
    result = DatasetIngester(cfg).ingest()
    print()
    print(result.summary())
    print("\n  Sample records:")
    _print_sample_records(out)


def demo_code(sample_dir: Path, output_dir: Path) -> None:
    """Mode 4 — Code format: {\"prompt\": \"# lang\\n# File: ...\", \"completion\": code}"""
    from swiftllm.dataset import DatasetIngester, DatasetFormat, IngestionConfig

    print("\n" + "#" * 70)
    print("#  MODE: CODE  ({\"prompt\": \"# lang\\n# File: name\", \"completion\": code})")
    print("#  Only code files use the CODE schema; other files fall back to pretraining")
    print("#  Restrict to specific extensions with file_extensions=['.py', '.rs']")
    print("#" * 70)

    out = str(output_dir / "code.jsonl")
    cfg = IngestionConfig(
        input_paths=[str(sample_dir / "code")],
        output_path=out,
        format=DatasetFormat.CODE,
        chunk_size=1024,
        chunk_overlap=64,
        min_length=30,
        file_extensions=[".py", ".rs"],
        deduplicate=True,
        include_metadata=True,  # adds _source, _ext fields
    )
    result = DatasetIngester(cfg).ingest()
    print()
    print(result.summary())
    print("\n  Sample records:")
    _print_sample_records(out)


def demo_mixed_sources(output_dir: Path) -> None:
    """Mode 5 — Multiple heterogeneous input paths."""
    from swiftllm.dataset import ingest_dataset

    print("\n" + "#" * 70)
    print("#  MODE: MIXED SOURCES (convenience function)")
    print("#  ingest_dataset() accepts a list of files and directories")
    print("#" * 70)

    # Use files from demo_all sample directory
    sample_dir = output_dir / "samples"
    inputs = [
        str(sample_dir / "text" / "intro.txt"),
        str(sample_dir / "structured" / "qa_pairs.csv"),
        str(sample_dir / "structured" / "extra.jsonl"),
    ]
    out = str(output_dir / "mixed.jsonl")

    print(f"\n  Inputs:")
    for p in inputs:
        print(f"    {p}")

    result = ingest_dataset(
        input_paths=inputs,
        output_path=out,
        format="sft_completion",
        chunk_size=512,
        verbose=False,
    )
    print()
    print(result.summary())
    print("\n  Sample records:")
    _print_sample_records(out)


def demo_train_from_directory(sample_dir: Path, output_dir: Path) -> None:
    """Mode 6 — Pass a directory directly to fine_tune(); auto-ingest fires."""
    from swiftllm.training import fine_tune

    print("\n" + "#" * 70)
    print("#  MODE: AUTO-INGEST IN fine_tune()")
    print("#  Pass a directory as train_data — ingestion happens automatically")
    print("#" * 70)

    print(f"\n  Calling fine_tune(train_data='{sample_dir}', …)")
    print("  (fine_tune detects a directory and auto-ingests to output_dir/auto_train.jsonl)")

    trainer = fine_tune(
        model="meta-llama/Llama-2-7b-hf",   # placeholder — no model loaded
        train_data=str(sample_dir),
        output_dir=str(output_dir / "finetune_output"),
        lora_r=16,
        num_epochs=1,
    )
    print(f"\n  Training complete. Metrics: loss={trainer.metrics.train_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SwiftLLM Dataset Ingestion demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        metavar="PATH",
        default=None,
        help="Input files or directories to ingest.",
    )
    parser.add_argument(
        "-o", "--output",
        default="./ingested.jsonl",
        metavar="FILE",
        help="Output JSONL file (default: ./ingested.jsonl).",
    )
    parser.add_argument(
        "--format",
        choices=["pretraining", "sft_messages", "sft_completion", "code"],
        default="pretraining",
        help="Output JSONL format (default: pretraining).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int, default=2048,
        help="Max chars per chunk (default: 2048).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int, default=128,
        help="Overlap chars between chunks (default: 128).",
    )
    parser.add_argument(
        "--extensions",
        default=None,
        metavar="EXT[,EXT...]",
        help="Comma-separated extension whitelist (e.g. .py,.md,.txt).",
    )
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run all demo modes with synthetic sample files.",
    )
    parser.add_argument(
        "--demo-mode",
        choices=["pretraining", "sft_messages", "sft_completion", "code",
                 "mixed", "auto_ingest"],
        default=None,
        help="Run a single demo mode with synthetic sample files.",
    )
    parser.add_argument(
        "--output-dir",
        default="./demo_output",
        metavar="DIR",
        help="Directory for demo output files (default: ./demo_output).",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Attach _source/_ext fields to every output record.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file progress.",
    )
    args = parser.parse_args()

    # --- Demo modes ---
    if args.demo_all or args.demo_mode:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_dir = output_dir / "samples"

        print("Creating synthetic sample files…")
        _create_sample_files(sample_dir)

        if args.demo_all or args.demo_mode == "pretraining":
            demo_pretraining(sample_dir, output_dir)

        if args.demo_all or args.demo_mode == "sft_messages":
            demo_sft_messages(sample_dir, output_dir)

        if args.demo_all or args.demo_mode == "sft_completion":
            demo_sft_completion(sample_dir, output_dir)

        if args.demo_all or args.demo_mode == "code":
            demo_code(sample_dir, output_dir)

        if args.demo_all or args.demo_mode == "mixed":
            demo_mixed_sources(output_dir)

        if args.demo_all or args.demo_mode == "auto_ingest":
            demo_train_from_directory(sample_dir, output_dir)

        if args.demo_all:
            print("\n" + "=" * 70)
            print("All demo modes complete.")
            print(f"Output files written to: {output_dir}")
            for f in sorted(output_dir.glob("*.jsonl")):
                lines = sum(1 for _ in open(f))
                print(f"  {f.name:<30} {lines:>5} records")
        return

    # --- Direct ingestion mode ---
    if not args.input:
        parser.error("Provide --input PATH [PATH …] or use --demo-all / --demo-mode")

    from swiftllm.dataset import ingest_dataset

    ext_list = None
    if args.extensions:
        ext_list = [
            e.strip() if e.strip().startswith(".") else f".{e.strip()}"
            for e in args.extensions.split(",")
            if e.strip()
        ]

    print(f"Ingesting {len(args.input)} path(s) → {args.output}")
    result = ingest_dataset(
        input_paths=args.input,
        output_path=args.output,
        format=args.format,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        file_extensions=ext_list,
        include_metadata=args.include_metadata,
        verbose=args.verbose,
    )

    print()
    print(result.summary())

    print("\nFirst 3 records:")
    _print_sample_records(args.output, n=3)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------
# END OF FILE: dataset_ingestion.py
# REPO PATH:   /swiftllm/examples/dataset_ingestion.py
# SEE ALSO:
#   python/swiftllm/dataset.py     DatasetIngester, IngestionConfig, DatasetFormat
#   python/swiftllm/training.py    prepare_dataset(), fine_tune(), Trainer auto-ingest
#   python/swiftllm/cli.py         `swiftllm dataset` subcommand
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
