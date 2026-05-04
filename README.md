<!--
    ==============================================================================
    PROJECT:   SWIFTLLM
    FILE:      README.md
    PATH:      /README.md
    AUTHOR:    Peter A. Aldrich Jr.
    DATE:      2026
    ------------------------------------------------------------------------------
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ==============================================================================
-->

# <a href="https://infrabrew.github.io/swiftllm/">SwiftLLM</a>
[![Logo](https://github.com/infrabrew/infrabrew.github.io/blob/master/swiftllm/assets/logo-mark-128.png?raw=true)](https://infrabrew.github.io/swiftllm/)

<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.0--beta-yellow.svg" alt="v2.2.0-beta">
  <img src="https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
</p>

**SwiftLLM** is a high-performance LLM inference, serving, and training engine built with Rust for maximum speed and efficiency. It features state-of-the-art memory management, continuous batching, multi-GPU support, built-in LoRA/QLoRA fine-tuning, multi-format dataset ingestion, and a full suite of research-derived enhancements across three integrated phases:

- **Phase 1** — Hybrid model architectures: Mamba SSM layers (with MIMO multi-head scan), LatentMoE with dynamic-bias load balancing, and Jamba-style hybrid attention+SSM blocks
- **Phase 2** — Advanced training: GRPO reinforcement learning, CGAR depth curriculum, Process Reward Models, and LongR dense rewards
- **Phase 3** — Test-time inference: self-consistency majority voting, multi-round self-refinement, Best-of-N dense verification, disaggregated prefill/decode serving, **Recursive Language Model (RLM)** with REPL sandbox, and **Dense Verification Layer** with cross-attention token scoring
- **Dataset Ingestion** — Convert directories of text, code, PDF, DOCX, CSV, HTML, and JSON files into JSONL training data in one command; auto-ingest fires inside `Trainer` and `fine_tune()` transparently

---

## Table of Contents

- [Key Features](#key-features)
- [Research Integrations](#research-integrations)
  - [Phase 1: Hybrid Architectures](#phase-1-hybrid-architectures)
  - [Phase 2: Training Enhancements](#phase-2-training-enhancements)
  - [Phase 3: Inference Enhancements](#phase-3-inference-enhancements)
  - [Phase 3: Model-Level Reasoning — RLM & Dense Verification](#phase-3-model-level-reasoning--rlm--dense-verification)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Beginner's Guide — Start Here](#beginners-guide--start-here)
  - [What is SwiftLLM?](#what-is-swiftllm-plain-english)
  - [What Can I Do With It?](#what-can-i-do-with-it)
  - [What You Need](#what-you-need)
  - [Step 1 — Install](#step-1--install-swiftllm)
  - [Step 2 — Pick a Model](#step-2--pick-a-model)
  - [Step 3 — Download Your Model](#step-3--download-your-model)
  - [Chat With the AI](#chat-with-the-ai)
  - [Ask a Single Question](#ask-a-single-question)
  - [Run a Local AI Server](#run-a-local-ai-server)
  - [Teach It Your Own Documents](#teach-it-your-own-documents)
  - [Plain-English Glossary](#plain-english-glossary)
  - [Common Questions](#common-questions)
  - [Troubleshooting](#troubleshooting)
- [Quick Start](#quick-start)
- [Dataset Ingestion](#dataset-ingestion)
  - [Supported Input Sources](#supported-input-sources)
  - [HuggingFace Datasets](#huggingface-datasets)
  - [Output Formats](#output-formats-jsonl)
  - [Three Usage Modes](#three-usage-modes)
  - [CLI — swiftllm dataset](#cli--swiftllm-dataset)
  - [Python API — HuggingFaceSource](#python-api--huggingfacesource)
  - [Python API — ingest_dataset()](#python-api-dataset)
  - [Python API — DatasetIngester](#python-api--datasetingester)
  - [Auto-Ingest in Trainer](#auto-ingest-in-trainer)
  - [Optional Dependencies](#optional-dependencies-for-dataset-ingestion)
- [Training & Fine-Tuning](#training--fine-tuning)
  - [Quick Start — CLI](#quick-fine-tune-with-lora-cli)
  - [Python Training API](#python-training-api)
  - [Training Data Formats](#training-data-formats)
  - [LoRAConfig Reference](#loraconfig-reference)
  - [QLoRA Fine-Tuning](#qlora-fine-tuning)
  - [Full TrainingConfig Reference](#full-trainingconfig-reference)
  - [Gradient Accumulation & Effective Batch Size](#gradient-accumulation--effective-batch-size)
  - [Mixed Precision](#mixed-precision)
  - [Learning Rate Schedulers](#learning-rate-schedulers)
  - [Checkpoints & Resumption](#checkpoints--resumption)
  - [Early Stopping](#early-stopping)
  - [Callbacks & Metrics](#callbacks--metrics)
  - [Config File Workflow](#config-file-workflow)
  - [Multi-GPU Training](#multi-gpu-training)
  - [Memory Requirements](#memory-requirements)
  - [After Training: Loading Your Model](#after-training-loading-your-model)
  - [Task-Specific Tips](#task-specific-tips)
- [GRPO Reinforcement Learning Training](#grpo-reinforcement-learning-training)
- [Test-Time Inference Enhancements](#test-time-inference-enhancements)
- [Disaggregated Serving](#disaggregated-serving)
- [Recursive Language Model (RLM)](#recursive-language-model-rlm)
- [Dense Verification Layer](#dense-verification-layer)
- [Configuration Reference](#configuration-reference)
- [Environment Variables](#environment-variables)
- [CLI Commands](#cli-commands)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Security](#security)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Key Features

**Core Engine**
- **High Throughput** — Continuous batching and efficient scheduling for maximum tokens/second
- **Memory Efficient** — PagedAttention for optimal KV cache management
- **Low Latency** — Optimized CUDA kernels and speculative decoding
- **Tensor Parallelism** — Scale to multiple GPUs seamlessly
- **OpenAI Compatible** — Drop-in replacement for the OpenAI API with security hardening
- **Multiple Formats** — HuggingFace repos, GGUF quantized models, SafeTensors
- **GGUF Inference** — Run quantized GGUF models on GPU via llama-cpp-python
- **Air-Gapped Install** — Bundle and deploy on networks with no internet access

**Training & Data**
- **Dataset Ingestion** — One command converts any directory of `.txt`, `.md`, `.py`, `.rs`, `.pdf`, `.docx`, `.csv`, `.html`, `.jsonl` (and 40+ more) into JSONL; 4 output schemas; SHA-256 dedup; auto-fires inside `Trainer` and `fine_tune()`
- **HuggingFace Dataset Support** — Pull any dataset from the HuggingFace Hub with `--hf-dataset` (CLI) or `HuggingFaceSource` (Python); combine with local files in one command; auto-detects Alpaca, ShareGPT, OpenAI-messages, prompt/completion, Q&A, and plain-text schemas; streaming mode for large corpora
- **LoRA / QLoRA / Full Fine-Tuning** — Memory-efficient adapter training or full parameter updates
- **Muon, AdamW, SGD Optimizers** — Newton-Schulz orthogonalization, decoupled weight decay, Nesterov momentum
- **GRPO (RL Fine-Tuning)** — Group Relative Policy Optimization without a critic model (Phase 2)
- **CGAR Curriculum** — Curriculum-Guided Adaptive Recursion for 1.71× training speedup (Phase 2)
- **Process Reward Models** — Step-level reasoning feedback with 5 aggregation strategies (Phase 2)
- **LongR Dense Rewards** — Per-token NLL information gain for long-context tasks; +9% LongBench v2 (Phase 2)

**Inference**
- **Self-Consistency Voting** — Majority vote across N independent chains (Wang et al., 2022) (Phase 3)
- **Multi-Round Self-Refinement** — Iterative critique→revision cycles (Self-Refine, Madaan 2023) (Phase 3)
- **Best-of-N Verification** — Dense scoring and reranking of N candidates (Phase 3)
- **Disaggregated Serving** — Separate prefill/decode worker pools (Splitwise/DistServe) (Phase 3)
- **Recursive Language Model (RLM)** — Bounded recursive self-calling with REPL sandbox, variable binding, and recursion scheduler (Phase 3)
- **Dense Verification Layer** — Cross-attention draft↔REPL-trace scoring; per-token & per-step confidence; GATE_AND_REGEN pipeline (Phase 3)

**Model Architectures (Phase 1)**
- **Mamba-3 SSM Layers** — Selective SSM with MIMO multi-head scan, complex-valued states, and exponential-trapezoidal discretisation
- **LatentMoE FFN** — compress→MoE dispatch→expand with aux-loss-free dynamic-bias load balancing (DeepSeek-V3 style)
- **Jamba Hybrid** — Interleaved Attention + Mamba blocks with configurable ratio

---

## Research Integrations

### Phase 1: Hybrid Architectures

Implemented in `crates/swiftllm-models/src/`:

| Module | What It Adds |
|--------|-------------|
| `mamba.rs` | `MambaConfig`, `MambaLayer` — selective SSM with discretization (∆, A, B, C, D), hardware-aware parallel scan, `MambaBlock` with pre-norm |
| `moe.rs` | `MoeConfig`, `MoeLayer` — top-k sparse routing, load-balancing auxiliary loss, expert capacity enforcement |
| `jamba.rs` | `JambaConfig`, `JambaLayer` — interleaved attention and Mamba layers with configurable `attn_layer_offset`; enables Phased Specialisation training |

**Jamba Configuration Example (Rust)**

```rust
use swiftllm_models::JambaConfig;

let config = JambaConfig {
    hidden_size: 4096,
    num_attention_heads: 32,
    num_mamba_layers: 24,
    num_attention_layers: 8,
    state_size: 16,
    conv_size: 4,
    expand_factor: 2,
    num_experts: 16,
    top_k_experts: 2,
    attn_layer_offset: 4,  // attention every 4th layer
    ..Default::default()
};
```

---

### Phase 2: Training Enhancements

Implemented in `crates/swiftllm-training/src/` and exposed via `python/swiftllm/`:

#### GRPO — Group Relative Policy Optimization

Fine-tunes models using RL without a value/critic model. For each prompt, generates a group of `G` rollouts, computes group-relative advantages, then applies a PPO-style clipped policy gradient with a KL divergence penalty.

- **Paper**: DeepSeekMath (Shao et al., 2024)
- **Rewards**: Correctness (substring match), format (structural quality), length deviation
- **No critic model required** — advantages are normalized within the group

#### CGAR — Curriculum-Guided Adaptive Recursion

Progressive depth curriculum: trains with a subset of active layers that grows from shallow to full depth over training. Uses smooth Hermite interpolation at phase boundaries.

- **Paper**: Curriculum-Guided Adaptive Recursion (2024)
- **Speedup**: Up to 1.71× vs. full-depth training from the start
- **Phases**: Shallow (0–30%) → Medium (30–60%) → Full (60–100%)

#### Process Reward Models (PRM)

Provides step-level feedback on reasoning chains by scoring each reasoning step independently, then aggregating scores into a single reward signal blended with the outcome reward.

- **Paper**: Let's Verify Step by Step (Lightman et al., 2023)
- **Modes**: `RulePrm` (heuristic, no extra model) or `NeuralPrm` (learned verifier)
- **Aggregation**: Min, Mean, Product, LastStep, WeightedMean

#### LongR Dense Rewards

Computes per-token relative NLL information gain vs. a frozen reference model: `r_t = NLL_ref − NLL_model`. Provides a dense token-level reward signal that is more informative than sparse outcome rewards for long-context tasks.

- **Paper**: LongReward (2024)
- **Gain**: 9% improvement on LongBench v2 in the original paper

---

### Phase 3: Inference Enhancements

Implemented in `crates/swiftllm-core/src/inference/` and `crates/swiftllm-core/src/sampling/`:

#### Self-Consistency (Wang et al., 2022)

Generates `N` independent reasoning chains at temperature > 0, extracts a final answer from each using a configurable extractor, and returns the plurality-majority answer. Ties broken by mean sequence log-probability.

#### Multi-Round Self-Refinement (Madaan et al., 2023)

Iterative critique → revision loop. In each round, the model critiques its own previous output then produces a revised version. Stops when improvement (measured by normalised Levenshtein edit distance) falls below a threshold or the round limit is reached.

#### Best-of-N Dense Verification

Generates `N` candidate responses, scores each using a configurable strategy (rule-based heuristic, neural PRM, ensemble, or sequence log-prob), then returns the highest-scoring candidate. Mirrors `verify_and_rank()` in the Rust core.

#### Disaggregated Prefill/Decode Serving (Splitwise / DistServe)

Routes compute-bound prefill requests and bandwidth-bound decode requests to dedicated, independently-scaled worker pools. Scheduling policies: round-robin, least-loaded, locality-aware. Includes `optimal_worker_ratio()` for auto-sizing worker pools.

---

### Phase 3: Model-Level Reasoning — RLM & Dense Verification

Implemented in `crates/swiftllm-models/src/layers/rlm.rs` and `dense_verification.rs`:

#### Recursive Language Model (RLM)

The RLM extends autoregressive generation with bounded recursive self-calling.  When the model encounters a complex sub-problem it can decompose it, solve each sub-problem at shallower depth, then integrate the sub-solutions back into the main hidden state via a learned gating mechanism.

- **Recursion Scheduler** — a complexity-classifier MLP predicts the required depth for each token position; early exit when the predicted depth is 0 with confidence ≥ threshold
- **REPL Sandbox** — a symbolic execution environment with four step types: `Assign`, `Compute`, `Verify`, `Recurse`; creates an execution trace consumed by the Dense Verification Layer
- **Variable Binding Table** — soft-attention key-value store for intermediate results; lookup = softmax attention, write = gated update `g*new + (1−g)*old`
- **Operating Modes** — `DISABLED` (plain pass-through), `SHALLOW` (depth=1), `REASONING` (depth=3, default), `AGENTIC` (depth=5)

**Paper**: "Architecting the Next-Generation Agentic Paradigm: A Hybrid Synthesis of Mamba-3, Mixture of Experts, Recursive Language Models, and Dense Verification" (2024)

#### Dense Verification Layer

After the RLM generation completes, the Dense Verification Layer performs one additional pass on the full output, cross-attending draft hidden states (Q) against embedded REPL execution trace (K/V) to compute per-token and per-step confidence scores.

- **Cross-Attention Scoring** — multi-head cross-attention: draft tokens as queries, REPL trace steps as keys/values → per-token attention weights → sigmoid score projection
- **Verification Strategies** — `DISABLED`, `SCORE_ONLY` (always accept), `GATE` (reject below threshold, single attempt), `GATE_AND_REGEN` (reject and regenerate up to `max_regen_attempts`)
- **Step Scoring** — separately scores each `Verify` step in the REPL trace for fine-grained quality assessment
- **Integration with RLM** — `verify_and_correct()` accepts the model's hidden states and `ReplState`, returning a `VerificationResult` with `global_score`, `token_scores`, `step_scores`, `low_confidence_positions`

**Paper**: "Let's Verify Step by Step" (Lightman et al., 2023); §3.5 of the Hybrid Mamba-3 architecture paper

---

## Supported Models

| Architecture | Models | Notes |
|-------------|--------|-------|
| **LLaMA** | LLaMA, LLaMA 2, LLaMA 3, Code Llama | |
| **Mistral** | Mistral 7B, Mixtral 8x7B | Mixtral uses MoE FFN |
| **Qwen** | Qwen, Qwen 2, Qwen 3 | |
| **Phi** | Phi-2, Phi-3 | |
| **Falcon** | Falcon | |
| **Gemma** | Gemma | |
| **Mamba** | Mamba-130M … Mamba-3B | Phase 1 — pure SSM |
| **Jamba** | Jamba-v0.1, custom | Phase 1 — hybrid Attention + Mamba + MoE |

---

## Installation

### Quick Install (Recommended)

```bash
git clone https://github.com/swiftllm/swiftllm.git
cd swiftllm
./install.sh
```

The installer automatically:
- Detects your GPU and CUDA toolkit
- Creates a Python virtual environment
- Installs Rust if needed
- Builds SwiftLLM from source (CPU or CUDA, depending on what's detected)
- Installs llama-cpp-python with GPU support (if available)

#### Supported Platforms

| OS | x86_64 | aarch64 / arm64 |
|----|--------|-----------------|
| Linux | `manylinux2014_x86_64` | `manylinux2014_aarch64` |
| macOS | `macosx_10_15_x86_64` (≥10.15) | `macosx_11_0_arm64` (Apple Silicon) |

The wheel is Python-abi3 (`cp38-abi3`), meaning a single wheel works across Python 3.8–3.12. CUDA is opt-in via the `cuda` cargo feature; the default build (`./install.sh --cpu` or `./install.sh` on a host with no CUDA) produces a portable CPU wheel with no CUDA toolkit dependency.

#### Installer Options

```bash
./install.sh --cpu          # CPU-only (skip GPU detection)
./install.sh --gpu          # Force GPU/CUDA build
./install.sh --venv ~/sllm  # Custom venv location
./install.sh --no-venv      # Install into current Python environment
./install.sh --model-dir /data/models  # Set model storage directory
./install.sh --airgap       # Offline install from air-gap bundle (no network)
```

### Air-Gapped / Offline Install

For hosts with no internet access, create a bundle on a connected machine first:

```bash
# On a CONNECTED machine
git clone https://github.com/swiftllm/swiftllm.git && cd swiftllm

# Basic bundle (source + all Python wheels + Rust installer)
./airgap-bundle.sh

# Include a model in the bundle
./airgap-bundle.sh --model "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"

# CPU-only wheels + custom output path
./airgap-bundle.sh --cpu -o /mnt/usb/swiftllm-bundle.tar.gz

# Cross-architecture bundle (x86_64 host → ARM64 target)
./airgap-bundle.sh --arch aarch64 -o swiftllm-bundle-arm64.tar.gz

# macOS Apple Silicon
./airgap-bundle.sh --arch arm64 --platform macosx_11_0_arm64
```

Transfer the archive to the air-gapped host, then:

```bash
# On the AIR-GAPPED host
tar xzf swiftllm-airgap-bundle.tar.gz
cd swiftllm-airgap-bundle/swiftllm
./install.sh --airgap
```

To run in offline mode at runtime:

```bash
export SWIFTLLM_OFFLINE=1
swiftllm generate -m /path/to/local/model.gguf -p "Hello"
```

### Manual Install

```bash
git clone https://github.com/swiftllm/swiftllm.git
cd swiftllm

pip install maturin
maturin build --release
pip install target/wheels/swiftllm-*.whl

# GGUF support (CPU)
pip install llama-cpp-python

# GGUF support (CUDA GPU)
CMAKE_ARGS='-DGGML_CUDA=on' CUDACXX=/usr/local/cuda/bin/nvcc pip install llama-cpp-python
```

### Requirements

- Python 3.8+
- Rust 1.70+ (auto-installed by `install.sh` if missing)
- CUDA 11.8+ (optional, for GPU acceleration)

---

## Beginner's Guide — Start Here

> **This section uses plain English throughout.** No prior programming experience is required.
> If you are already comfortable with the command line and Python, feel free to jump straight to [Quick Start](#quick-start).

---

### What is SwiftLLM? (Plain English)

SwiftLLM is a tool that lets you run AI language models — the same kind of technology behind ChatGPT — **on your own computer**, for free, without sending anything to the internet.

Once a model is downloaded, everything runs locally. Your questions, documents, and responses never leave your machine. You can use it from a chat window in your terminal, call it from a Python script, or run it as a server that other apps can talk to.

You can also **teach it your own content**: point it at a folder of PDFs, Word documents, notes, or code files and SwiftLLM will train a version of the model that knows your specific material.

---

### What Can I Do With It?

| I want to… | How |
|------------|-----|
| Chat with an AI privately on my computer | `swiftllm chat` |
| Ask a question from the command line | `swiftllm generate` |
| Run a local AI server for other apps | `swiftllm serve` |
| Train the AI on my own PDFs / docs / code | `swiftllm dataset` then `swiftllm finetune` |
| Use it in a Python script | `from swiftllm import LLM` |

---

### What You Need

| Requirement | Details |
|-------------|---------|
| Operating system | Linux or macOS. Windows users: install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) first (free, built into Windows 10/11) |
| RAM | 8 GB minimum · 16 GB recommended for best experience |
| Disk space | 500 MB for SwiftLLM itself · 400 MB – 40 GB per model (you choose the size) |
| Internet | Only needed once to download SwiftLLM and your model |
| GPU | Optional — SwiftLLM works without one, but a GPU makes it 10–100× faster |

> **No coding knowledge is required** for the chat, generate, and serve features. A little familiarity with a terminal (typing commands) is all you need.

---

### Step 1 — Install SwiftLLM

Open a terminal and run these two commands one at a time:

```bash
git clone https://github.com/swiftllm/swiftllm.git
cd swiftllm
./install.sh
```

The installer automatically detects your GPU, sets up a Python environment, and builds everything. When it finishes you will see a success message. Close and reopen your terminal so the `swiftllm` command becomes available.

> **Don't have `git` installed?**
> - **macOS**: run `xcode-select --install` in your terminal, then try again
> - **Ubuntu / Debian Linux**: run `sudo apt install git` then try again

---

### Step 2 — Pick a Model

A **model** is the AI's "brain" — a large file it uses to generate text. Bigger models produce smarter, more detailed responses but need more memory and disk space. Start small and work up.

| Model name | Download size | RAM needed | Best for |
|------------|--------------|-----------|----------|
| **Qwen 2.5 0.5B** ← *great starting point* | ~400 MB | 2 GB | Testing, learning, fast replies |
| **Qwen 2.5 7B** | ~5 GB | 8 GB | General chat, Q&A, writing, summaries |
| **LLaMA 3 8B** | ~6 GB | 10 GB | Writing assistance, coding help, analysis |
| **LLaMA 2 13B** | ~10 GB | 16 GB | Complex reasoning, detailed long answers |
| **LLaMA 2 70B** | ~40 GB | 48 GB+ | Highest quality — needs a powerful machine |

> 💡 **Not sure which to pick?** Start with **Qwen 2.5 0.5B** — it downloads in under a minute and runs on almost any computer. You can always download a larger model later.

---

### Step 3 — Download Your Model

Copy and paste this command to download the recommended starter model:

```bash
swiftllm download -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

SwiftLLM downloads the model and stores it in a cache folder (`~/.cache/swiftllm/models`). The next time you use it the model loads instantly — no internet connection needed.

> **Want a different model?** Replace the model name with any of the sizes from the table above, or any model ID from [HuggingFace](https://huggingface.co/models).

---

### Chat With the AI

This opens an interactive chat session — just like a messaging app, but running entirely on your computer:

```bash
swiftllm chat -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

Type your message and press **Enter**. The AI replies. Keep the conversation going as long as you like. Press **Ctrl + C** or type `exit` to quit.

**Example conversation:**
```
You: Can you explain what machine learning is in simple terms?

AI: Machine learning is a way of teaching computers to learn from examples
    rather than following a fixed set of rules. Instead of programming every
    possible situation, you show the computer thousands of examples and it
    figures out the patterns on its own...

You: Can you give me a real-world example?

AI: Sure! Think of a spam filter for email. Instead of manually writing rules
    like "if the email contains the word 'lottery' then it's spam", you show
    the filter thousands of emails labelled "spam" and "not spam"...
```

---

### Ask a Single Question

Use this when you want a quick one-off answer without starting a full chat session:

```bash
swiftllm generate \
  -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  -p "Summarise the water cycle in three bullet points." \
  --max-tokens 200
```

| Option | What it does |
|--------|-------------|
| `-m` | Which model to use |
| `-p` | Your question or instruction (the "prompt") |
| `--max-tokens 200` | Maximum length of the reply (200 tokens ≈ 150 words) |
| `--temperature 0.7` | How creative the response is (0.0 = focused, 1.0 = more varied) |

---

### Run a Local AI Server

This starts SwiftLLM as a background server. Any app that supports the OpenAI API — Open WebUI, LangChain, your own scripts — can then send requests to it:

```bash
swiftllm serve \
  -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --port 8000
```

Leave that terminal running. Open a second terminal and test it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello! Who are you?"}]
  }'
```

You can also open a browser-based chat UI by pointing [Open WebUI](https://github.com/open-webui/open-webui) at `http://localhost:8000`.

> **Want to add a password?** Add `--api-key my-secret-key` to the serve command, then include `-H "Authorization: Bearer my-secret-key"` in your requests.

---

### Teach It Your Own Documents

This is where SwiftLLM becomes really powerful. You can take any existing AI model and train it on **your own files**, on **ready-made public datasets from the internet**, or on **both at the same time** — so it becomes an expert in exactly the content you care about.

You have two ways to bring in training data:

| Source | What it is | Example |
|--------|-----------|---------|
| **Your own files** | PDFs, Word docs, text files, code, spreadsheets on your computer | Company handbook, research papers, your codebase |
| **HuggingFace datasets** | Free, ready-made datasets from [huggingface.co/datasets](https://huggingface.co/datasets) | `tatsu-lab/alpaca` (52 k instructions), `HuggingFaceFW/fineweb` (billions of web pages) |

You can use either one alone, or mix them together.

---

**Option A — Train on your own files**

#### Step 1 — Put your files in a folder

Organise your files however you like. SwiftLLM reads sub-folders automatically:

```
my_documents/
├── handbook.pdf
├── product_specs.docx
├── meeting_notes.txt
├── research/
│   ├── paper1.pdf
│   └── paper2.pdf
└── codebase/
    ├── main.py
    └── utils.py
```

Supported file types: `.txt` `.md` `.pdf` `.docx` `.csv` `.py` `.js` `.ts` `.rs` `.go` `.java` `.html` and many more.

> **PDF / Word support needs one extra install** — run once:
> ```bash
> pip install pdfplumber   # for PDF files
> pip install python-docx  # for Word (.docx) files
> ```

#### Step 2 — Convert your files to training data

```bash
swiftllm dataset \
  --input ./my_documents/ \
  --output ./training_data.jsonl \
  --format pretraining
```

You'll see a summary when it finishes:

```
Dataset ingestion complete
  Total chunks    : 847
  ── Local files ──────────────────
  Files processed : 12
  Chunks written  : 847
  Total chars     : 1,204,392
  By extension    :
    .pdf               410 chunks
    .txt               280 chunks
    .docx              110 chunks
    .py                 47 chunks
```

---

**Option B — Train on a public HuggingFace dataset**

No files to organise — just pick a dataset name from [huggingface.co/datasets](https://huggingface.co/datasets) and SwiftLLM downloads and converts it automatically.

#### Step 1 — Install the HuggingFace datasets library (once)

```bash
pip install datasets
```

#### Step 2 — Pull the dataset and convert it

```bash
swiftllm dataset \
  --hf-dataset tatsu-lab/alpaca \
  --format sft_completion \
  --output ./training_data.jsonl
```

SwiftLLM downloads the dataset, auto-detects its format (question/answer, instruction/output, plain text, etc.), and converts it to training-ready JSONL.

---

**Option C — Mix your files with a public dataset (most powerful)**

Combine your own documents with a public dataset in one command — SwiftLLM merges them and removes any duplicates automatically.

```bash
swiftllm dataset \
  --input ./my_documents/ \
  --hf-dataset tatsu-lab/alpaca \
  --hf-max-samples 10000 \
  --format sft_completion \
  --output ./training_data.jsonl
```

You'll see both sources in the summary:

```
Dataset ingestion complete
  Total chunks    : 10,847
  ── Local files ──────────────────
  Files processed : 12
  Chunks written  : 847
  ── HuggingFace datasets ─────────
  Rows consumed   : 10,000
  Chunks written  : 10,000
    tatsu-lab/alpaca               10000 chunks
```

---

#### Step 3 — Fine-tune the model on your data

```bash
swiftllm finetune \
  -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --train-data ./training_data.jsonl \
  --output-dir ./my-custom-model/
```

This trains the model on your data. When it finishes, your custom model is saved in `./my-custom-model/final/`.

#### Step 4 — Use your custom model

```bash
swiftllm chat -m ./my-custom-model/final/
```

The model now has knowledge of your specific content on top of its general training.

> **How long does this take?**
> - With a GPU: roughly 20–60 minutes for a few hundred pages of documents
> - Without a GPU: a few hours — run it overnight
>
> **What does "fine-tuning" actually mean?** It's like giving a smart new employee your company handbook to study. They already know how to speak, reason, and answer questions — fine-tuning adds your specific knowledge on top of that without changing everything they already know.

---

### Plain-English Glossary

These words appear throughout the documentation. Here's what they mean in plain language:

| Term | Plain-English meaning |
|------|-----------------------|
| **Model** | The AI's "brain" — a large file containing billions of learned connections that generate text |
| **GGUF** | A compressed model file format. The `.gguf` extension means the model is stored in this efficient format — smaller file size, faster to load |
| **Parameter / B** | "B" stands for billion. A 7B model has 7 billion internal connections. More parameters = smarter but needs more memory |
| **Fine-tuning** | Taking an existing AI model and giving it extra training on your own documents so it learns your specific content |
| **LoRA** | A clever fine-tuning shortcut. Instead of retraining the entire model (which needs enormous compute power), LoRA trains only small "adapter" pieces. Same quality improvement, much less time and memory |
| **JSONL** | A simple data file format — one JSON record per line. SwiftLLM reads and writes these for training data. You don't need to create them manually — `swiftllm dataset` does it for you |
| **GPU** | A graphics card (e.g. NVIDIA RTX). Originally designed for video games, GPUs are also excellent at AI calculations — 10–100× faster than a regular CPU for model inference and training |
| **CPU** | Your computer's main processor. SwiftLLM works on CPU — it's just slower than GPU |
| **Inference** | Asking the model a question and getting a response (as opposed to training it) |
| **Token** | A unit of text roughly equal to 4 characters or ¾ of a word. `--max-tokens 200` means the response can be at most 200 tokens, about 150 words |
| **Temperature** | How random or creative the AI's responses are. `0.0` = very consistent and focused. `1.0` = more varied and creative. Start at `0.7` for general use |
| **Prompt** | The text you send to the AI — your question, instruction, or starting sentence |
| **Chunk** | When SwiftLLM reads a long document, it breaks it into smaller pieces called chunks. Each chunk becomes one training record |
| **Pretraining** | Teaching a model from scratch on massive amounts of text — already done for you by the model's original creators |
| **HuggingFace** | A popular website (`huggingface.co`) where thousands of free AI models are shared and downloaded from |
| **CUDA** | NVIDIA's software layer that lets programs use GPU acceleration. SwiftLLM uses it automatically if you have an NVIDIA card |

---

### Common Questions

**Do I need a GPU?**
No. SwiftLLM works entirely on CPU. A GPU (NVIDIA card) makes inference 10–100× faster and fine-tuning much quicker, but it is completely optional. Smaller models (0.5B–7B) are very usable on CPU for non-time-sensitive work.

**Is my data private?**
Yes, completely. Everything runs on your own machine. Your prompts, documents, training data, and model outputs never leave your computer. There is no cloud component, no telemetry, and no data collection.

**Can I use any model from HuggingFace?**
Yes. Pass any HuggingFace model ID directly with `-m`, for example: `-m meta-llama/Llama-2-7b-hf`. SwiftLLM downloads and caches it automatically.

**What file types can I train on?**
Plain text, Markdown, all major code languages, PDF (needs `pdfplumber`), Word documents (needs `python-docx`), HTML, CSV, JSON, and JSONL. See [Supported Input Formats](#supported-input-formats) for the full list.

**How long does fine-tuning take?**
For a 7B model with LoRA on a few hundred pages of documents: roughly 20–60 minutes with a mid-range GPU, or 4–8 hours on CPU. You can run it overnight and the result will be waiting for you in the morning.

**What if my computer runs out of memory?**
Switch to a smaller model, or add `--max-seq-len 512` to reduce how much text is processed at once. The [Memory Requirements](#memory-requirements) table in the Training section shows exactly how much RAM each model size needs.

**Can I use SwiftLLM with a chat UI in my browser?**
Yes. Run `swiftllm serve` then connect [Open WebUI](https://github.com/open-webui/open-webui) to `http://localhost:8000`. Open WebUI is a free, open-source chat interface that looks and works like ChatGPT.

**Does fine-tuning change the original model?**
No. The original model file is never modified. Fine-tuning creates a separate small adapter file that sits alongside the original model. You can always go back to using the base model.

**My fine-tuned model doesn't seem to know my documents. What happened?**
A few things to check: (1) Make sure your documents were actually processed — run `swiftllm dataset` with `--verbose` to see each file being read. (2) Try more training epochs: add `--num-epochs 3`. (3) More data helps — the more document content you provide, the better the results.

---

### Troubleshooting

| Symptom | What to try |
|---------|------------|
| `command not found: swiftllm` | Close and reopen your terminal after install. If still missing, run `source ~/.bashrc` (Linux) or `source ~/.zshrc` (macOS) |
| Model download is very slow | Downloads range from 400 MB to 40 GB depending on the model — this is normal. It only downloads once and is cached locally afterward |
| `CUDA out of memory` error | Switch to a smaller model, or add `--max-seq-len 512` to limit memory usage |
| `out of memory` on CPU | Add `--max-seq-len 512`, or try the 0.5B model instead of a larger one |
| PDF files are not being read | Run `pip install pdfplumber` then try again |
| Word (.docx) files are not being read | Run `pip install python-docx` then try again |
| Responses are too short | Increase `--max-tokens`, e.g. `--max-tokens 512` |
| Responses feel repetitive or boring | Add `--temperature 0.7` to introduce more variation |
| Responses are too random or off-topic | Lower temperature: `--temperature 0.3` |
| Fine-tuning seems to be doing nothing | Add `--num-epochs 3` for more training passes, and make sure your input files actually contain text |
| `FileNotFoundError` on `--train-data` | Check the path — use `ls ./training_data.jsonl` to confirm the file exists |
| Chat session is very slow | You are likely running on CPU — this is normal. Consider a smaller model or adding a GPU |
| I don't know which model to use | Start with `Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf` — it works on almost any computer |

---

## Quick Start

### Download a Model

```bash
# Download a full HuggingFace repo
swiftllm download -m meta-llama/Llama-2-7b-hf

# Download a single GGUF file by URL
swiftllm download -m "https://huggingface.co/TeichAI/Qwen3-32B-Kimi-K2-Thinking-Distill-GGUF/blob/main/Qwen3-32B-Kimi-K2-Thinking-Distill.q4_k_m.gguf"

# Download a single GGUF file with shorthand
swiftllm download -m "TeichAI/Qwen3-32B-Kimi-K2-Thinking-Distill-GGUF:Qwen3-32B-Kimi-K2-Thinking-Distill.q4_k_m.gguf"

# Specify where to store models
swiftllm download -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" --download-dir /data/models
```

### Generate Text

```bash
# Standard generation
swiftllm generate \
  -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  -p "What is the capital of France?" \
  --max-tokens 128

# Self-consistency: majority vote across 8 independent chains
swiftllm generate \
  -m /path/to/model.gguf \
  -p "Solve step by step: 3x + 7 = 22. The answer is?" \
  --self-consistency 8 --temperature 0.8

# Multi-round refinement: up to 3 critique→revision cycles
swiftllm generate \
  -m /path/to/model.gguf \
  -p "Write a concise summary of the water cycle." \
  --refinement-rounds 3

# Best-of-N: generate 8 candidates and return the highest-scoring one
swiftllm generate \
  -m /path/to/model.gguf \
  -p "Explain photosynthesis in one paragraph." \
  --best-of-n 8

# Interactive chat
swiftllm chat -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

### Python API — Basic Inference

```python
from swiftllm import LLM, SamplingParams

# Load a GGUF model (downloads automatically if not cached)
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf")

# Or from a local path
llm = LLM(model="/path/to/model.gguf")

# Standard generation
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0].outputs[0].text)
```

---

## Dataset Ingestion

SwiftLLM can build a JSONL training dataset from **three sources** — your own local files, public datasets from the HuggingFace Hub, or both combined in a single command.  No custom data-prep scripts needed; just point at what you have and pick an output format.

---

### Supported Input Sources

#### Local Files

| Category  | Extensions |
|-----------|-----------|
| Plain text | `.txt`  `.md`  `.rst`  `.log`  `.tex`  `.asciidoc` |
| Code | `.py`  `.js`  `.ts`  `.rs`  `.go`  `.java`  `.c`  `.cpp`  `.cs`  `.rb`  `.php`  `.swift`  `.kt`  `.scala`  `.sh`  `.sql`  `.toml`  `.yaml`  and ~30 more |
| Documents | `.pdf` *(pdfplumber / pypdf)*   `.docx` *(python-docx)* |
| Web | `.html`  `.htm`  `.xml` *(beautifulsoup4 recommended)* |
| Structured | `.csv`   `.json`   `.jsonl` |

CSV and JSONL files are auto-detected: if they already contain
`prompt`/`completion`, `messages`, or `text` columns/keys they are passed
through directly; otherwise the values are concatenated as plain text.

---

### HuggingFace Datasets

Pull any public (or private, with a token) dataset directly from the
[HuggingFace Hub](https://huggingface.co/datasets) using the `--hf-dataset`
CLI flag or the `HuggingFaceSource` Python class.

**Auto-detected schemas** — no field mapping needed for common formats:

| Schema | Detected columns | Example datasets |
|--------|-----------------|-----------------|
| Alpaca instruction | `instruction` + `output` (+ optional `input`) | `tatsu-lab/alpaca`, `yahma/alpaca-cleaned` |
| ShareGPT conversations | `conversations` with `from`/`value` keys | `WizardLM/WizardLM_evol_instruct_70k` |
| OpenAI messages | `messages` with `role`/`content` keys | `HuggingFaceH4/ultrachat_200k` |
| Prompt + completion | `prompt`/`completion` or `question`/`answer` | `openai/gsm8k`, `truthful_qa` |
| Plain text | `text` or `content` | `HuggingFaceFW/fineweb`, `EleutherAI/pile` |

All sources share the same SHA-256 deduplication pool — a chunk that appears in both a local file and an HF dataset is written only once.

---

### Output Formats (JSONL)

| `--format` | Record schema | Best for |
|---|---|---|
| `pretraining` | `{"text": "..."}` | Next-token LM training on raw corpora |
| `sft_messages` | `{"messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}` | Chat / instruction fine-tuning |
| `sft_completion` | `{"prompt": "...", "completion": "..."}` | Classic SFT (non-chat) |
| `code` | `{"prompt": "# python\n# File: foo.py\n\n", "completion": "<code>"}` | Code generation fine-tuning |

For raw text files in `sft_messages` / `sft_completion` mode, each chunk is
split ~75 / 25 into a user prompt and assistant completion so the model learns
both document style and continuation.

---

### Three Usage Modes

```
Mode 1: Local files only       → --input ./my_docs/
Mode 2: HuggingFace only       → --hf-dataset tatsu-lab/alpaca
Mode 3: Both combined          → --input ./my_docs/ --hf-dataset tatsu-lab/alpaca
```

All three modes write a single merged, deduplicated `.jsonl` file.

---

### CLI — `swiftllm dataset`

**Mode 1 — Local files:**

```bash
# Pretraining from a documentation tree
swiftllm dataset \
  --input ./docs/ \
  --output ./data/train.jsonl

# Code fine-tuning (Python + Rust only)
swiftllm dataset \
  --input ./src/ ./tests/ \
  --output ./data/code_train.jsonl \
  --format code \
  --extensions .py,.rs

# SFT from mixed local sources: PDF + CSV + notes directory
swiftllm dataset \
  --input paper.pdf qa_pairs.csv ./notes/ \
  --output ./data/sft.jsonl \
  --format sft_completion \
  --chunk-size 1024
```

**Mode 2 — HuggingFace datasets:**

```bash
# Pull the Alpaca instruction dataset from HuggingFace
swiftllm dataset \
  --hf-dataset tatsu-lab/alpaca \
  --format sft_completion \
  --output ./data/alpaca_train.jsonl

# Multiple HF datasets in one run
swiftllm dataset \
  --hf-dataset tatsu-lab/alpaca HuggingFaceH4/ultrachat_200k \
  --format sft_messages \
  --output ./data/combined_hf.jsonl

# Large corpus with streaming (avoids full download)
swiftllm dataset \
  --hf-dataset HuggingFaceFW/fineweb \
  --hf-subset sample-10BT \
  --hf-streaming \
  --hf-max-samples 100000 \
  --format pretraining \
  --output ./data/fineweb_100k.jsonl
```

**Mode 3 — Local files + HuggingFace combined:**

```bash
# Your documents + a public HF dataset, merged into one file
swiftllm dataset \
  --input ./my_docs/ ./research_papers/ \
  --hf-dataset tatsu-lab/alpaca \
  --hf-max-samples 10000 \
  --format sft_completion \
  --output ./data/combined_train.jsonl

# Custom field mapping for non-standard schemas
swiftllm dataset \
  --hf-dataset my-org/my-dataset \
  --hf-prompt-field query \
  --hf-completion-field answer \
  --format sft_completion \
  --output ./data/custom.jsonl
```

**Statistics only (dry-run):**

```bash
swiftllm dataset \
  --input ./my_corpus/ \
  --hf-dataset tatsu-lab/alpaca \
  --output /dev/null \
  --stats-only
```

**All flags:**

*Local file flags:*

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH …` | *(optional)* | Files or directories; omit when using `--hf-dataset` only |
| `--output FILE` | *(required)* | Destination `.jsonl` |
| `--format` | `pretraining` | Output schema: `pretraining` `sft_messages` `sft_completion` `code` |
| `--chunk-size N` | `2048` | Max characters per record |
| `--chunk-overlap N` | `128` | Overlap between consecutive chunks |
| `--min-length N` | `50` | Discard chunks shorter than N chars |
| `--max-file-size-mb MB` | `50` | Skip local files larger than this |
| `--extensions .ext[,…]` | all supported | Whitelist specific extensions |
| `--no-recursive` | off | Don't walk directories recursively |
| `--no-dedup` | off | Allow duplicate chunks |
| `--include-metadata` | off | Add `_source` / `_ext` keys to records |
| `--system-prompt TEXT` | `"You are a helpful assistant."` | System turn for `sft_messages` |
| `--stats-only` | off | Print statistics; skip writing output |
| `--verbose` | off | Print per-file / per-dataset progress |

*HuggingFace flags:*

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-dataset NAME …` | *(optional)* | HuggingFace dataset name(s); use multiple for several datasets |
| `--hf-split SPLIT` | `train` | Split to load; slice syntax supported: `train[:5000]` |
| `--hf-subset NAME` | — | Dataset config/subset, e.g. `sample-10BT` for FineWeb |
| `--hf-max-samples N` | all | Maximum rows to consume per dataset |
| `--hf-streaming` | off | Stream rows without full download (saves disk space) |
| `--hf-shuffle` | off | Shuffle before slicing (uses `--hf-seed`) |
| `--hf-seed N` | `42` | Random seed for shuffle |
| `--hf-trust-remote-code` | off | Required by some community datasets |
| `--hf-text-field COL` | auto | Override: column containing plain text |
| `--hf-prompt-field COL` | auto | Override: column containing prompt / question |
| `--hf-completion-field COL` | auto | Override: column containing completion / answer |
| `--hf-messages-field COL` | auto | Override: column containing a messages list |
| `--hf-instruction-field COL` | auto | Override: Alpaca-style instruction column |
| `--hf-output-field COL` | auto | Override: Alpaca-style output column |

---

### Python API — HuggingFaceSource

`HuggingFaceSource` describes one HuggingFace dataset to pull in.  Combine multiple in a list for multi-dataset runs.

```python
from swiftllm import HuggingFaceSource, ingest_dataset

# Minimal — auto-detects all fields
src = HuggingFaceSource("tatsu-lab/alpaca")

# Large corpus with streaming (avoids downloading to disk)
src_large = HuggingFaceSource(
    dataset_name="HuggingFaceFW/fineweb",
    subset="sample-10BT",
    split="train",
    streaming=True,
    max_samples=100_000,
)

# Explicit field mapping for a non-standard schema
src_custom = HuggingFaceSource(
    dataset_name="my-org/my-private-dataset",
    prompt_field="query",
    completion_field="response",
    split="train[:50%]",
    trust_remote_code=True,
)
```

**`HuggingFaceSource` parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | *(required)* | HuggingFace dataset ID, e.g. `"tatsu-lab/alpaca"` |
| `split` | `"train"` | Split or slice, e.g. `"train[:5000]"` |
| `subset` | `None` | Config/subset name (second arg to `load_dataset`) |
| `text_field` | auto | Column containing plain text |
| `prompt_field` | auto | Column containing prompt / question |
| `completion_field` | auto | Column containing completion / answer |
| `messages_field` | auto | Column containing a messages list |
| `instruction_field` | auto | Alpaca-style instruction column |
| `input_field` | auto | Alpaca-style optional context column |
| `output_field` | auto | Alpaca-style output column |
| `max_samples` | all | Cap rows per dataset |
| `shuffle` | `False` | Shuffle before slicing |
| `seed` | `42` | Seed for shuffle |
| `streaming` | `False` | HuggingFace streaming mode |
| `trust_remote_code` | `False` | Required by some community datasets |
| `cache_dir` | system default | Override local cache directory |

---

### Python API (Dataset)

**`ingest_dataset()` — one-liner for all three modes:**

```python
from swiftllm import ingest_dataset, HuggingFaceSource

# Mode 1 — local files only
result = ingest_dataset(
    input_paths="./docs/",
    output_path="./data/train.jsonl",
)
print(result.summary())

# Mode 2 — HuggingFace only
result = ingest_dataset(
    hf_sources=[HuggingFaceSource("tatsu-lab/alpaca")],
    output_path="./data/alpaca.jsonl",
    format="sft_completion",
)

# Mode 3 — local files + HuggingFace combined
result = ingest_dataset(
    input_paths=["./my_docs/", "domain_notes.pdf"],
    hf_sources=[
        HuggingFaceSource("tatsu-lab/alpaca"),
        HuggingFaceSource(
            "HuggingFaceFW/fineweb",
            subset="sample-10BT",
            streaming=True,
            max_samples=20_000,
        ),
    ],
    output_path="./data/combined.jsonl",
    format="sft_completion",
)
print(result.summary())
# Dataset ingestion complete
#   Total chunks    : 32,847
#   ── Local files ──────────────────
#   Files processed : 14
#   Chunks written  : 12,847
#   ── HuggingFace datasets ─────────
#   Rows consumed   : 52,002
#   Chunks written  : 20,000
#     tatsu-lab/alpaca               52002 chunks
#     HuggingFaceFW/fineweb          20000 chunks

# Code fine-tuning from a source tree
result = ingest_dataset(
    input_paths=["./src/", "./tests/"],
    output_path="./data/code_train.jsonl",
    format="code",
    file_extensions=[".py", ".rs", ".go"],
    chunk_size=1500,
)
```

**`prepare_dataset()` — two-step ingest-then-train:**

```python
from swiftllm.training import prepare_dataset, Trainer, TrainingConfig

result = prepare_dataset(
    input_paths=["./docs/", "paper.pdf"],
    output_path="./data/train.jsonl",
    format="sft_completion",
    chunk_size=1024,
)
print(result.summary())

config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    output_dir="./output",
    num_epochs=3,
)
Trainer(config).train()
```

---

### Python API — DatasetIngester

Full control via `DatasetIngester` + `IngestionConfig`:

```python
from swiftllm import (
    DatasetIngester, DatasetFormat, IngestionConfig, HuggingFaceSource
)

cfg = IngestionConfig(
    output_path="./data/train.jsonl",
    # Local sources (optional if hf_sources provided)
    input_paths=["./src/", "paper.pdf", "qa_pairs.csv"],
    # HuggingFace sources (optional if input_paths provided)
    hf_sources=[
        HuggingFaceSource("tatsu-lab/alpaca"),
        HuggingFaceSource(
            "HuggingFaceFW/fineweb",
            subset="sample-10BT",
            streaming=True,
            max_samples=50_000,
        ),
    ],
    format=DatasetFormat.SFT_MESSAGES,
    chunk_size=2048,
    chunk_overlap=128,
    min_length=50,
    max_file_size_mb=100.0,
    file_extensions=[".py", ".md", ".pdf", ".csv"],
    recursive=True,
    system_prompt="You are a helpful coding assistant.",
    deduplicate=True,       # shared across local + HF sources
    include_metadata=True,  # adds _source, _ext
    verbose=True,
)

result = DatasetIngester(cfg).ingest()
print(result.summary())
```

**`IngestionResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_chunks` | `int` | Total JSONL records written (local + HF) |
| `total_files_scanned` | `int` | Local files visited (including skipped) |
| `total_files_processed` | `int` | Local files that produced ≥1 chunk |
| `total_chars` | `int` | Raw characters extracted from local files |
| `format_counts` | `dict[str, int]` | Local records per extension, e.g. `{".py": 80}` |
| `total_hf_rows` | `int` | Rows consumed from all HuggingFace sources |
| `total_hf_chunks` | `int` | Records written from HuggingFace sources |
| `hf_dataset_counts` | `dict[str, int]` | Records per HF dataset, e.g. `{"tatsu-lab/alpaca": 52000}` |
| `skipped_files` | `list[(str, str)]` | `(path, reason)` for each skipped local file |
| `output_path` | `str` | Absolute path of the written `.jsonl` |

---

### Auto-Ingest in Trainer

`Trainer`, `fine_tune()`, and `GrpoTrainer` automatically ingest any non-JSONL `train_data` before training begins.  The produced JSONL is written to `<output_dir>/auto_train.jsonl` and persists alongside checkpoints.

```python
from swiftllm.training import fine_tune

# Mode 1 — directory of local files (existing behaviour)
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./my_codebase/",
    output_dir="./output",
    lora_r=16,
    num_epochs=3,
)

# Mode 2 — HuggingFace only (new)
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    hf_dataset="tatsu-lab/alpaca",
    dataset_format="sft_completion",
    output_dir="./output",
    lora_r=32,
    num_epochs=3,
)

# Mode 3 — local files + HuggingFace combined (new)
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./my_docs/",           # local files
    hf_dataset="tatsu-lab/alpaca",     # HF dataset merged in
    hf_max_samples=10_000,
    dataset_format="sft_completion",
    output_dir="./output",
    lora_r=16,
)

# Pass a mixed list of local paths
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data=["paper.pdf", "qa_pairs.csv", "./notes/"],
    dataset_format="sft_completion",
    output_dir="./output",
)

# "hf:" prefix shorthand inside TrainingConfig
from swiftllm.training import Trainer, TrainingConfig
config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="hf:tatsu-lab/alpaca",   # ← auto-ingested from HF Hub
    output_dir="./output",
)
Trainer(config).train()
```

**`fine_tune()` HuggingFace parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hf_dataset` | `None` | HuggingFace dataset name to pull in |
| `hf_split` | `"train"` | Dataset split |
| `hf_subset` | `None` | Config/subset name |
| `hf_max_samples` | all | Maximum rows to consume |
| `hf_streaming` | `False` | Streaming mode (no full download) |

---

### Optional Dependencies for Dataset Ingestion

| Format / Feature | Package | Install |
|------------------|---------|---------|
| HuggingFace datasets | `datasets` | `pip install datasets` |
| PDF | `pdfplumber` *(recommended)* | `pip install pdfplumber` |
| PDF *(fallback)* | `pypdf` | `pip install pypdf` |
| DOCX | `python-docx` | `pip install python-docx` |
| HTML/XML *(improved)* | `beautifulsoup4` | `pip install beautifulsoup4` |

Install all at once:

```bash
pip install datasets pdfplumber python-docx beautifulsoup4
```

HTML and XML work without `beautifulsoup4` via a regex fallback, but the output
quality is better with the full parser installed.  PDF and DOCX require their
respective libraries; a clear `ImportError` with install instructions is raised
if neither is available when a matching file is encountered.

---

## Training & Fine-Tuning

### Quick Fine-Tune with LoRA (CLI)

```bash
# Fine-tune from an existing JSONL file
swiftllm finetune \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/train.jsonl \
  --lora-r 16 --lora-alpha 32 \
  --learning-rate 2e-4

# Fine-tune from a directory of local files — ingestion is automatic
swiftllm dataset -i ./my_corpus/ -o ./data/train.jsonl --format sft_completion
swiftllm finetune -m meta-llama/Llama-2-7b-hf --train-data ./data/train.jsonl --lora-r 16

# Fine-tune from a HuggingFace dataset — ingest then train
swiftllm dataset --hf-dataset tatsu-lab/alpaca --format sft_completion -o ./data/alpaca.jsonl
swiftllm finetune -m meta-llama/Llama-2-7b-hf --train-data ./data/alpaca.jsonl --lora-r 16

# Fine-tune from HuggingFace + your own files (combined in one step)
swiftllm dataset \
  --input ./my_docs/ \
  --hf-dataset tatsu-lab/alpaca \
  --hf-max-samples 10000 \
  --format sft_completion \
  -o ./data/combined.jsonl
swiftllm finetune -m meta-llama/Llama-2-7b-hf --train-data ./data/combined.jsonl --lora-r 16

# Full training command with all options
swiftllm train \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/train.jsonl \
  --eval-data ./data/eval.jsonl \
  --method lora \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lr-scheduler cosine \
  -o ./output/my-model
```

### Python Training API

```python
from swiftllm import Trainer, TrainingConfig, LoRAConfig, fine_tune

# ── From an existing JSONL file ──────────────────────────────────────────────
config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    output_dir="./output",
    num_epochs=3,
    learning_rate=2e-4,
    lora=LoRAConfig(r=16, alpha=32),
)
Trainer(config).train()

# ── From a local directory (auto-ingested) ───────────────────────────────────
config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./my_corpus/",       # any directory; writes auto_train.jsonl first
    output_dir="./output",
)
Trainer(config).train()

# ── From a HuggingFace dataset only (new) ────────────────────────────────────
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    hf_dataset="tatsu-lab/alpaca",
    dataset_format="sft_completion",
    output_dir="./output",
    lora_r=32,
    num_epochs=3,
)

# ── From HuggingFace + local files combined (new) ────────────────────────────
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./my_docs/",           # local files
    hf_dataset="tatsu-lab/alpaca",     # HuggingFace dataset merged in
    hf_max_samples=10_000,
    dataset_format="sft_completion",
    output_dir="./output",
    lora_r=16,
)

# ── "hf:" shorthand in TrainingConfig ────────────────────────────────────────
config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="hf:tatsu-lab/alpaca",  # pulled from Hub automatically
    output_dir="./output",
)
Trainer(config).train()

# ── Mixed list of local paths ─────────────────────────────────────────────────
trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data=["paper.pdf", "qa_pairs.csv", "./notes/"],
    dataset_format="sft_completion",
    lora_r=16,
    num_epochs=3,
)

# ── Explicit two-step: ingest first, then train ───────────────────────────────
from swiftllm.training import prepare_dataset

result = prepare_dataset(
    input_paths=["./docs/", "paper.pdf"],
    output_path="./data/train.jsonl",
    format="sft_messages",
)
print(result.summary())

trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    lora_r=16,
)
```

### Supported Fine-Tuning Methods

| Method | Description | Memory |
|--------|-------------|--------|
| **LoRA** | Low-Rank Adaptation — trains small adapter matrices | Low |
| **QLoRA** | 4-bit quantized base model + LoRA adapters | Very Low |
| **Full** | Full parameter fine-tuning | High |

### Optimizers

| Optimizer | Best For | Description |
|-----------|----------|-------------|
| **Muon** | Matrix-shaped params (linear layers) | Newton-Schulz orthogonalization on Nesterov momentum; faster convergence than Adam for ≥2D weights. Auto-falls back to AdamW for 1D params (biases, norms). [arXiv:2409.20325](https://arxiv.org/abs/2409.20325) |
| **AdamW** | General purpose | Decoupled weight decay Adam; default for most fine-tuning |
| **SGD** | Large-batch training | SGD with optional Nesterov momentum |

### Muon Optimizer (Rust API)

```rust
use swiftllm_training::{Muon, MuonConfig};
use swiftllm_training::Optimizer;

let mut opt = Muon::new(MuonConfig {
    lr: 0.02,
    momentum: 0.95,
    ns_steps: 5,
    weight_decay: 0.0,
    adamw_lr: 3e-4,
    ..Default::default()
});

opt.set_shape("layer0.weight", 4096, 4096);
let mut param = vec![0.01f32; 4096 * 4096];
let grad = compute_gradient(&model, &batch);
opt.step(&mut param, &grad, "layer0.weight");
```

---

### Training Data Formats

> **Have raw files or want to use a public dataset?**  See [Dataset Ingestion](#dataset-ingestion) — one command converts directories of `.txt`, `.md`, `.py`, `.pdf`, `.docx`, `.csv`, `.html`, and more into any of the formats below, or pulls directly from any HuggingFace Hub dataset with `--hf-dataset`.  You can mix both sources in a single run.

SwiftLLM accepts three input formats for supervised fine-tuning.

#### JSONL — Instruction/Chat (Recommended)

Each line is one JSON object. The `messages` field follows the OpenAI chat format and is the recommended structure for instruction tuning and chat models:

```jsonl
{"messages": [{"role": "user", "content": "What is 12 × 15?"}, {"role": "assistant", "content": "12 × 15 = 180."}]}
{"messages": [{"role": "system", "content": "You are a helpful math tutor."}, {"role": "user", "content": "Solve 3x + 7 = 22"}, {"role": "assistant", "content": "Subtract 7: 3x = 15. Divide by 3: x = 5."}]}
```

For simpler prompt-completion pairs, use `prompt` + `completion`:

```jsonl
{"prompt": "Translate to French: Hello, how are you?", "completion": "Bonjour, comment allez-vous?"}
{"prompt": "Summarize in one sentence: [long article]", "completion": "The article discusses..."}
```

For GRPO/RL training, use `prompt` + optional `answer`:

```jsonl
{"prompt": "Solve step by step: 3x + 7 = 22. What is x?", "answer": "5"}
{"prompt": "A train travels 60 mph for 2 h then 80 mph for 1 h. Total distance?", "answer": "200"}
```

#### JSONL — Long-Form / Document

For long-context tasks (LongR rewards), include a `text` field:

```jsonl
{"text": "Chapter 1: Introduction\n\nLarge language models have..."}
{"text": "Abstract: We propose a new method for..."}
```

#### CSV

The trainer auto-detects CSV files (`.csv` extension). Required columns: `prompt` and `completion` (or `text` for language modelling).

```csv
prompt,completion
"Translate to Spanish: Good morning","Buenos días"
"What is the capital of Japan?","Tokyo"
```

#### Plain Text

Files with no recognized extension, or `.txt` files, are treated as raw text for language modelling pretraining. Each line becomes one training example.

```
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
```

#### Recommended Data Sizes

| Task | Min examples | Recommended | Notes |
|------|-------------|-------------|-------|
| LoRA instruction tuning | 500 | 5 000–50 000 | Quality > quantity |
| QLoRA instruction tuning | 500 | 5 000–20 000 | Same data, less VRAM |
| Full fine-tuning | 10 000 | 100 000+ | Needs larger dataset to avoid overfitting |
| GRPO RL | 200 | 1 000–10 000 | Prompts only; no answers required |
| Domain adaptation | 1 000 | 10 000–1 M | Plain text for continued pretraining |

---

### LoRAConfig Reference

```python
from swiftllm.training import LoRAConfig

lora = LoRAConfig(
    r=16,                                           # Rank of the adapter matrices. Higher rank = more capacity.
                                                    # Common values: 4 (tiny), 8, 16 (default), 32, 64.
    alpha=32.0,                                     # Scaling factor. Effective scale = alpha / r.
                                                    # Rule of thumb: alpha = 2×r for stable training.
    dropout=0.05,                                   # Dropout applied inside LoRA layers (0.0 = no dropout).
                                                    # Use 0.1 for small datasets to prevent overfitting.
    target_modules=["q_proj", "k_proj",             # Which linear layers to apply LoRA to.
                    "v_proj", "o_proj"],             # See architecture-specific recommendations below.
    use_rslora=False,                               # Rank-Stabilized LoRA: scales by 1/√r instead of 1/r.
                                                    # More stable for high ranks (r ≥ 32).
)
```

**Rank (`r`) guidelines**

| Rank | Parameters added | Use case |
|------|-----------------|----------|
| 4 | ~1–2 M | Fast domain adaptation, very limited VRAM |
| 8 | ~2–4 M | Lightweight instruction tuning |
| **16** | **~4–8 M** | **Default — good balance for most tasks** |
| 32 | ~8–16 M | Math, code, or complex reasoning tasks |
| 64 | ~16–32 M | Approaches QLoRA quality for complex tasks |

**Target modules by architecture**

| Architecture | Recommended `target_modules` |
|-------------|------------------------------|
| LLaMA / Mistral / Qwen | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| LLaMA + MLP | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| Falcon | `["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]` |
| Phi-2 / Phi-3 | `["q_proj", "k_proj", "v_proj", "dense"]` |
| Jamba (attention layers only) | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| All linear layers | `"all-linear"` *(string shorthand)* |

---

### QLoRA Fine-Tuning

QLoRA trains LoRA adapters on top of a 4-bit quantized base model — requiring ~65–70% less VRAM than standard LoRA while preserving most of the quality.

```bash
# CLI — QLoRA
swiftllm train \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/train.jsonl \
  --method qlora \
  --lora-r 16 \
  --lora-alpha 32 \
  --learning-rate 2e-4 \
  --mixed-precision bf16 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  -o ./qlora_output
```

```python
from swiftllm.training import Trainer, TrainingConfig, LoRAConfig, FineTuningMethod, MixedPrecision

config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    output_dir="./qlora_output",
    fine_tuning_method=FineTuningMethod.QLORA,  # 4-bit base + LoRA adapters
    lora=LoRAConfig(
        r=16,
        alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    learning_rate=2e-4,
    per_device_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch = 2 × 8 = 16
    mixed_precision=MixedPrecision.BF16, # bf16 preferred for QLoRA
    max_seq_len=2048,
    num_epochs=3,
)

trainer = Trainer(config)
trainer.train()
```

**QLoRA VRAM usage** (approximate, fp16 activations):

| Model | Full FT | LoRA (r=16) | QLoRA (r=16, 4-bit) |
|-------|---------|-------------|---------------------|
| 7B | ~56 GB | ~28 GB | ~10 GB |
| 13B | ~104 GB | ~52 GB | ~20 GB |
| 34B | >200 GB | ~120 GB | ~48 GB |
| 70B | >400 GB | ~240 GB | ~96 GB |

---

### Full TrainingConfig Reference

```python
from swiftllm.training import TrainingConfig, LoRAConfig, FineTuningMethod, LrScheduler, MixedPrecision

config = TrainingConfig(
    # ── Model & data ──────────────────────────────────────────────────────
    model="meta-llama/Llama-2-7b-hf",  # HF model ID or local path
    train_data="./data/train.jsonl",   # training data (JSONL, CSV, or text)
    eval_data="./data/eval.jsonl",     # optional evaluation data
    output_dir="./output",             # checkpoints + logs are saved here

    # ── Training loop ─────────────────────────────────────────────────────
    num_epochs=3,                      # full passes over the training set
    per_device_batch_size=4,           # batch size per GPU
    gradient_accumulation_steps=1,     # accumulate N micro-batches before stepping
                                       # effective_batch = per_device × accum × num_gpus
    max_seq_len=2048,                  # truncate / pad sequences to this length

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    learning_rate=5e-5,                # peak LR (after warmup)
    weight_decay=0.01,                 # L2 regularisation coefficient
    warmup_steps=100,                  # int = absolute steps; float < 1.0 = ratio of total steps
    max_grad_norm=1.0,                 # gradient clipping threshold
    lr_scheduler=LrScheduler.COSINE,   # LINEAR | COSINE | COSINE_WITH_RESTARTS | CONSTANT | CONSTANT_WITH_WARMUP

    # ── Precision ─────────────────────────────────────────────────────────
    mixed_precision=MixedPrecision.FP16,  # NO | FP16 | BF16

    # ── Fine-tuning method ────────────────────────────────────────────────
    fine_tuning_method=FineTuningMethod.LORA,  # FULL | LORA | QLORA
    lora=LoRAConfig(r=16, alpha=32),            # only used when method is LORA or QLORA

    # ── Logging & evaluation ──────────────────────────────────────────────
    logging_steps=10,                  # print metrics every N optimizer steps
    eval_steps=500,                    # run evaluation every N steps (0 = per epoch only)

    # ── Checkpointing ─────────────────────────────────────────────────────
    save_steps=500,                    # save a checkpoint every N steps (0 = per epoch)
    save_total_limit=3,                # keep at most N checkpoints (oldest are deleted)
    resume_from_checkpoint=None,       # path to a checkpoint dir to resume from

    # ── Reproducibility ───────────────────────────────────────────────────
    seed=42,

    # ── Phase 2 research integrations (all optional) ──────────────────────
    num_layers=32,                     # total model layers (required for CGAR)
    grpo=None,                         # GrpoConfig — enables RL fine-tuning (GrpoTrainer only)
    cgar=None,                         # CgarConfig — enables depth curriculum
    prm=None,                          # PrmConfig  — enables step-level rewards
    long_reward_weight=0.0,            # float > 0  — enables LongR dense rewards
)
```

**`effective_batch_size` property**

```python
# effective_batch_size = per_device_batch_size × gradient_accumulation_steps
print(config.effective_batch_size)  # 4 × 1 = 4
```

---

### Gradient Accumulation & Effective Batch Size

Large-batch training generally improves stability and final quality, but increasing `per_device_batch_size` directly requires proportionally more VRAM. Use `gradient_accumulation_steps` to simulate a larger batch without extra memory:

```python
# 16-sample effective batch on a single 24 GB GPU
config = TrainingConfig(
    per_device_batch_size=2,           # 2 examples fit in GPU memory at once
    gradient_accumulation_steps=8,     # accumulate 8 micro-batches → effective = 16
)

# Same effective batch across 4 GPUs
config = TrainingConfig(
    per_device_batch_size=4,
    gradient_accumulation_steps=1,
    # tensor_parallel_size=4 set via EngineConfig or environment variable
)
```

**Effective batch size formula**:
```
effective_batch = per_device_batch_size × gradient_accumulation_steps × num_gpus
```

Typical starting points:
- Instruction tuning: effective batch 32–128
- RL / GRPO: effective batch = `per_device_batch_size × group_size`
- Pretraining: effective batch 256–2048

---

### Mixed Precision

| Mode | Description | When to use |
|------|-------------|------------|
| `NO` | Full fp32 training | Debugging; very small models |
| `FP16` | Loss scaled fp16 (AMP) | NVIDIA Volta/Turing/Ampere (A100, V100, RTX 30xx) |
| `BF16` | Brain float 16 — wider exponent range, no loss scaling | Ampere+ (A100, H100) and Apple Silicon; preferred for QLoRA and GRPO |

```python
from swiftllm.training import MixedPrecision

config = TrainingConfig(
    mixed_precision=MixedPrecision.BF16,  # recommended for A100 / H100
)
```

> **Tip**: If you see NaN losses with `FP16`, switch to `BF16`. BF16 has the same range as FP32 and doesn't require loss scaling.

---

### Learning Rate Schedulers

| Scheduler | Behaviour | Good for |
|-----------|-----------|---------|
| `LINEAR` | Linear decay from peak LR to 0 | Short runs, quick experiments |
| `COSINE` | Cosine annealing from peak to ~0 | **Default — best for most tasks** |
| `COSINE_WITH_RESTARTS` | Cosine with periodic warm restarts | Long runs; helps escape local minima |
| `CONSTANT` | Constant LR after warmup | Hyperparameter search |
| `CONSTANT_WITH_WARMUP` | Constant LR (with warmup) | Debugging, ablations |

```python
from swiftllm.training import LrScheduler

config = TrainingConfig(
    learning_rate=2e-4,
    warmup_steps=100,          # int: first 100 steps ramp from 0 → peak LR
    # warmup_steps=0.03,       # float < 1.0: first 3% of steps are warmup
    lr_scheduler=LrScheduler.COSINE,
)
```

**Recommended learning rates by method**:

| Method | Typical LR range | Notes |
|--------|-----------------|-------|
| Full fine-tuning | `1e-5` – `5e-5` | Lower LR to avoid catastrophic forgetting |
| LoRA (r=16) | `1e-4` – `3e-4` | Adapters train faster; can use higher LR |
| QLoRA (r=16) | `1e-4` – `2e-4` | Similar to LoRA |
| GRPO | `1e-6` – `1e-5` | RL is sensitive; keep LR small |

---

### Checkpoints & Resumption

SwiftLLM saves two files per checkpoint:
- `output_dir/checkpoint-{step}/trainer_state.json` — step, epoch, loss, LR
- `output_dir/training_config.json` — full `TrainingConfig` (written once at start)
- `output_dir/final/trainer_state.json` — final checkpoint after training ends

**Configure checkpointing:**

```python
config = TrainingConfig(
    output_dir="./output",
    save_steps=500,         # save every 500 optimizer steps
    save_total_limit=3,     # keep the 3 most recent checkpoints; older are auto-deleted
                            # set None to keep all checkpoints (may use a lot of disk)
)
```

**Resume from a checkpoint:**

```python
from swiftllm.training import Trainer

# Class method — loads config + trainer state automatically
trainer = Trainer.resume_from_checkpoint("./output/checkpoint-1000")
trainer.train()
```

Or via CLI:

```bash
swiftllm train \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/train.jsonl \
  --resume-from-checkpoint ./output/checkpoint-1000 \
  -o ./output
```

**Checkpoint directory layout:**

```
output/
├── training_config.json         ← saved once at start
├── checkpoint-500/
│   └── trainer_state.json
├── checkpoint-1000/
│   └── trainer_state.json
└── final/
    └── trainer_state.json
```

---

### Early Stopping

Stop training automatically when the monitored metric stops improving.

```python
from swiftllm.training import Trainer, TrainingConfig, EarlyStoppingConfig

config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    eval_data="./data/eval.jsonl",
    eval_steps=200,          # evaluate every 200 steps
    num_epochs=10,           # upper bound — early stopping may trigger sooner
)

early_stop = EarlyStoppingConfig(
    patience=3,              # stop after 3 consecutive evals with no improvement
    min_delta=0.001,         # improvement must be > 0.001 to count as progress
    metric="eval_loss",      # monitor eval_loss (or "train_loss")
)

trainer = Trainer(config, early_stopping=early_stop)
trainer.train()

if trainer.stopped_early:
    print(f"Stopped early at step {trainer.metrics.step}")
```

---

### Callbacks & Metrics

Attach callbacks to react to training events or build custom loggers, experiment trackers, or dashboards.

```python
from swiftllm.training import Trainer, TrainingConfig, TrainingMetrics

config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="data.jsonl",
    logging_steps=10,    # callbacks fire every logging_steps steps
)

trainer = Trainer(config)

# ── Simple loss logger ─────────────────────────────────────────────────────
trainer.add_callback(lambda m: print(
    f"step={m.step}  loss={m.train_loss:.4f}  ppl={m.perplexity:.2f}  "
    f"lr={m.learning_rate:.2e}  tok/s={m.throughput:.0f}"
))

# ── Weights & Biases integration ───────────────────────────────────────────
import wandb
wandb.init(project="swiftllm", config=config.to_dict())

def wandb_callback(m: TrainingMetrics):
    wandb.log({
        "train/loss": m.train_loss,
        "train/perplexity": m.perplexity,
        "train/learning_rate": m.learning_rate,
        "train/throughput_tok_s": m.throughput,
        "eval/loss": m.eval_loss,
    }, step=m.step)

trainer.add_callback(wandb_callback)

# ── Custom early-exit based on loss threshold ─────────────────────────────
def loss_guard(m: TrainingMetrics):
    if m.train_loss > 10.0:
        raise RuntimeError(f"Loss exploded to {m.train_loss:.2f} at step {m.step}")

trainer.add_callback(loss_guard)

trainer.train()

# ── Read metrics after training ────────────────────────────────────────────
metrics = trainer.metrics   # TrainingMetrics dataclass
print(f"Final step  : {metrics.step}")
print(f"Final loss  : {metrics.train_loss:.4f}")
print(f"Final ppl   : {metrics.perplexity:.2f}")
print(f"Eval loss   : {metrics.eval_loss}")
print(f"Total tokens: {metrics.total_tokens:,}")
print(f"Elapsed     : {metrics.elapsed_secs:.0f}s")
```

**`TrainingMetrics` fields**:

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current optimizer step |
| `epoch` | int | Current epoch (0-indexed) |
| `train_loss` | float | Training loss at this step |
| `eval_loss` | float \| None | Evaluation loss (None if not yet evaluated) |
| `perplexity` | float | `exp(train_loss)` |
| `learning_rate` | float | Current LR value |
| `throughput` | float | Tokens per second |
| `total_tokens` | int | Cumulative tokens processed |
| `elapsed_secs` | float | Wall-clock seconds since training start |

---

### Config File Workflow

Save a config to JSON and reuse it for reproducible runs:

```python
from swiftllm.training import TrainingConfig, LoRAConfig

config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    num_epochs=3,
    learning_rate=2e-4,
    lora=LoRAConfig(r=16, alpha=32),
)

# Save to file
config.save("./my_run.json")

# Load later (tolerates extra/missing keys gracefully)
config2 = TrainingConfig.load("./my_run.json")
```

CLI with a config file:

```bash
# Run from a saved config (all other flags are ignored)
swiftllm train --config ./my_run.json

# GRPO with a config file
swiftllm grpo --config ./grpo_run.json
```

The saved JSON file contains all `TrainingConfig` fields, including nested `grpo`, `cgar`, `prm`, and `lora` objects. It can be version-controlled and shared with collaborators.

---

### Multi-GPU Training

**Tensor parallelism** splits each model layer across multiple GPUs — best for inference and for models that exceed a single GPU's VRAM:

```bash
# Fine-tune across 4 GPUs with tensor parallelism
swiftllm train \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/train.jsonl \
  --method lora \
  --tensor-parallel-size 4 \
  -o ./output
```

```python
# Via environment variable (read automatically by EngineConfig)
import os
os.environ["SWIFTLLM_TENSOR_PARALLEL_SIZE"] = "4"

from swiftllm.training import Trainer, TrainingConfig
config = TrainingConfig(model="meta-llama/Llama-2-7b-hf", ...)
trainer = Trainer(config)
trainer.train()
```

**NCCL tuning for multi-GPU**:

```bash
# Disable P2P on systems where GPUs are not directly connected
export NCCL_P2P_DISABLE=1

# Enable verbose NCCL logging for debugging hangs
export NCCL_DEBUG=INFO

# Restrict to specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**Distributed data parallelism** (DDP) — scale to multiple nodes:

```bash
# Node 0 (master)
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  WORLD_SIZE=2 RANK=0 \
  swiftllm train -m meta-llama/Llama-2-7b-hf --train-data data.jsonl

# Node 1 (worker)
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  WORLD_SIZE=2 RANK=1 \
  swiftllm train -m meta-llama/Llama-2-7b-hf --train-data data.jsonl
```

---

### Memory Requirements

Approximate VRAM requirements for training (fp16 activations, batch size = 1 per device):

| Model size | Full fine-tuning | LoRA r=16 | QLoRA r=16 (4-bit) |
|-----------|-----------------|-----------|-------------------|
| **1–3B** | 12–24 GB | 6–12 GB | 4–6 GB |
| **7B** | ~56 GB | ~28 GB | ~10 GB |
| **13B** | ~104 GB | ~52 GB | ~20 GB |
| **34B** | >200 GB | ~120 GB | ~48 GB |
| **70B** | >400 GB | ~240 GB | ~96 GB |

**Tips to reduce memory usage:**

```python
config = TrainingConfig(
    per_device_batch_size=1,           # smallest batch
    gradient_accumulation_steps=16,    # maintain effective batch = 16
    mixed_precision=MixedPrecision.BF16,
    max_seq_len=1024,                  # shorter sequences = less KV cache memory
    fine_tuning_method=FineTuningMethod.QLORA,
    lora=LoRAConfig(r=8),              # smaller rank = fewer parameters
)
```

```bash
# Also reduce GPU overhead reservation
export SWIFTLLM_GPU_OVERHEAD_MB=256
export SWIFTLLM_GPU_MEMORY_UTILIZATION=0.92
```

---

### After Training: Loading Your Model

After training completes, your output directory contains:

```
output/
├── training_config.json     ← full config (for reproducibility)
├── final/
│   └── trainer_state.json   ← step, epoch, loss at completion
└── checkpoint-*/
    └── trainer_state.json
```

> **Note**: The CUDA backend is not yet fully wired — weight files are not written in the current release. When the backend is complete, LoRA adapters will be saved as `adapter_model.safetensors` + `adapter_config.json`, and full fine-tuned weights as `model.safetensors` shards.

**Planned loading API** (once backend is wired):

```python
# Load the fine-tuned model for inference
from swiftllm import LLM

# LoRA: pass the adapter directory; base model is loaded separately
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",   # base model
    enable_lora=True,
)
# Load LoRA adapter at request time (hot-swap supported)
from swiftllm.config import LoRARequest
outputs = llm.generate(
    ["Hello, world!"],
    lora_request=LoRARequest(lora_name="my-adapter", lora_path="./output/final"),
)

# Full fine-tune: point directly at the output directory
llm = LLM(model="./output/final")
outputs = llm.generate(["Hello!"])
```

---

### Task-Specific Tips

#### Instruction Tuning

Use the `messages` JSONL format. Add a system prompt that describes the model's role:

```jsonl
{"messages": [{"role": "system", "content": "You are a concise technical assistant."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Recommended settings: LoRA r=16, LR=2e-4, 3 epochs, max_seq_len=2048.

#### Code Generation

Use longer sequences and include complete, runnable code examples in both prompt and completion. Avoid truncating mid-function:

```python
config = TrainingConfig(
    max_seq_len=4096,            # code examples can be long
    lora=LoRAConfig(
        r=32,                    # higher rank for code
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # include MLP
    ),
    learning_rate=1e-4,
)
```

#### Mathematical Reasoning

Use chain-of-thought data with explicit `<think>` / step delimiters. Pair with a Process Reward Model for step-level feedback:

```python
from swiftllm.config import PrmConfig

config = TrainingConfig(
    max_seq_len=2048,
    learning_rate=5e-5,
    prm=PrmConfig(
        aggregation="last_step",
        step_separator="\n\n",   # blank line between reasoning steps
        outcome_weight=0.5,
        prm_weight=0.5,
    ),
)
```

#### Domain Adaptation (Continued Pretraining)

Use plain-text data with full fine-tuning or high-rank LoRA. Keep learning rate low to avoid forgetting:

```python
config = TrainingConfig(
    fine_tuning_method=FineTuningMethod.LORA,
    lora=LoRAConfig(r=64, use_rslora=True),  # high rank, RSLoRA for stability
    learning_rate=5e-5,                       # lower LR than instruction tuning
    num_epochs=1,                             # one pass avoids overfitting raw text
    max_seq_len=4096,
)
```

#### Dialogue / Chat Fine-Tuning

Mask the loss on system prompt and user turns — only compute loss on assistant turns. Use `messages` format:

```jsonl
{"messages": [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "What is photosynthesis?"},
  {"role": "assistant", "content": "Photosynthesis is the process by which plants..."}
]}
```

Recommended: LoRA r=16–32, LR=2e-4, 2–3 epochs, max_seq_len=2048, eval on held-out dialogues.

#### LoRA vs QLoRA Decision Guide

```
Available VRAM for 7B model?
├── ≥ 28 GB → LoRA (full quality, fastest training)
├── 10–27 GB → QLoRA (near-LoRA quality, ~40% slower)
└── < 10 GB → QLoRA + r=8 + max_seq_len=1024 + gradient_accum=16
```

---

## GRPO Reinforcement Learning Training

GRPO (Group Relative Policy Optimization) fine-tunes models using RL without requiring a critic/value model. It samples a group of `G` rollouts per prompt, computes group-relative advantages, then applies a PPO-style clipped policy gradient update plus a KL divergence penalty against a frozen reference model.

### CLI

```bash
# GRPO with CGAR curriculum (default: enabled)
swiftllm grpo \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/rl_prompts.jsonl \
  --group-size 8 \
  --kl-coeff 0.04 \
  --num-epochs 1 \
  -o ./grpo_output

# Full research stack: GRPO + CGAR + PRM + LongR
swiftllm grpo \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/math_prompts.jsonl \
  --group-size 8 \
  --enable-prm \
  --long-reward-weight 0.1 \
  --num-layers 32 \
  -o ./grpo_full_output

# GRPO without curriculum
swiftllm grpo \
  -m meta-llama/Llama-2-7b-hf \
  --train-data data.jsonl \
  --disable-cgar \
  --group-size 4
```

### Python API

```python
from swiftllm import GrpoTrainer, TrainingConfig
from swiftllm.config import GrpoConfig, CgarConfig, PrmConfig, LongRewardConfig

# Configure the full research stack
config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/math_prompts.jsonl",
    output_dir="./grpo_output",
    fine_tuning_method="full",
    learning_rate=1e-5,
    num_epochs=1,
    num_layers=32,

    # GRPO: group-relative policy optimization
    grpo=GrpoConfig(
        group_size=8,           # G rollouts per prompt
        clip_eps=0.2,           # PPO clipping threshold ε
        kl_coeff=0.04,          # KL penalty coefficient β
        correctness_weight=1.0,
        format_weight=0.2,
    ),

    # CGAR: progressive depth curriculum (1.71× speedup)
    cgar=CgarConfig(
        shallow_end=0.30,       # shallow phase ends at 30% of training
        medium_end=0.60,        # medium phase ends at 60% of training
    ),

    # PRM: step-level process reward model
    prm=PrmConfig(
        aggregation="last_step",
        outcome_weight=0.5,
        prm_weight=0.5,
    ),

    # LongR: dense token-level rewards (+9% LongBench v2)
    long_reward=LongRewardConfig(weight=0.10),
)

trainer = GrpoTrainer(config)
trainer.add_callback(lambda m: print(f"step {m.step} | loss {m.train_loss:.4f}"))
trainer.train()
```

### Convenience Function

```python
from swiftllm import grpo_train

trainer = grpo_train(
    model="meta-llama/Llama-2-7b-hf",
    train_data="math_prompts.jsonl",
    group_size=8,
    enable_cgar=True,
    enable_prm=False,
    long_reward_weight=0.1,
)
```

### Training Data Format

JSONL with one JSON object per line:

```jsonl
{"prompt": "What is 12 × 15? Think step by step.", "answer": "180"}
{"prompt": "Solve: 3x + 7 = 22", "answer": "5"}
{"prompt": "A train travels at 60 mph for 2 hours, then 80 mph for 1 hour. Total distance?", "answer": "200"}
```

---

## Test-Time Inference Enhancements

SwiftLLM provides three test-time compute enhancements that require no training — they improve output quality by spending more tokens at inference time.

### Self-Consistency Majority Voting

Generate `N` independent reasoning chains and return the plurality-majority answer. More chains → higher accuracy on reasoning tasks, with diminishing returns beyond ~16 samples.

```python
from swiftllm import LLM
from swiftllm.config import SelfConsistencyConfig, AnswerExtractor

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Heuristic extractor: finds last number or \boxed{} expression
results = llm.generate_with_self_consistency(
    "What is 17 × 23? Think step by step.",
    config=SelfConsistencyConfig(
        num_samples=8,
        extractor=AnswerExtractor.HEURISTIC,
        temperature=0.8,
    ),
)
result = results[0]
print(f"Answer: {result.answer}")
print(f"Agreement: {result.vote_fraction:.0%} of {len(result.raw_outputs)} samples")

# Sentinel extractor: text after "The answer is"
results = llm.generate_with_self_consistency(
    "Solve: 3x + 7 = 22. Show your work, then state 'The answer is X.'",
    config=SelfConsistencyConfig(
        num_samples=8,
        extractor=AnswerExtractor.AFTER_SENTINEL,
        answer_sentinel="The answer is",
        temperature=0.8,
    ),
)

# XML tag extractor: content of <answer>…</answer>
results = llm.generate_with_self_consistency(
    "What is the capital of France? Respond with <answer>…</answer>.",
    config=SelfConsistencyConfig(
        num_samples=4,
        extractor=AnswerExtractor.XML_TAG,
        answer_tag="answer",
        temperature=0.6,
    ),
)
```

**Standalone majority voting** (over pre-generated texts):

```python
from swiftllm.sampling import SelfConsistencySampler
from swiftllm.config import SelfConsistencyConfig

sampler = SelfConsistencySampler(SelfConsistencyConfig(num_samples=4))
result = sampler.vote([
    "The answer is 270 miles.",
    "Total distance: 270 miles.",
    "I get 240 miles.",
    "The answer is 270.",
])
print(result.answer, result.vote_fraction)  # "270"  0.75
```

### Multi-Round Self-Refinement

Iteratively critique and revise an initial response until improvement stalls or the round limit is reached.

```python
from swiftllm.config import RefinementConfig, StoppingCriterion, ImprovementMetric

results = llm.generate_with_refinement(
    "Write a concise summary of the water cycle.",
    config=RefinementConfig(
        max_rounds=3,
        min_improvement=0.05,                       # stop when < 5% edit-distance improvement
        stopping_criterion=StoppingCriterion.EITHER, # stop at max_rounds OR min_improvement
        improvement_metric=ImprovementMetric.EDIT_DISTANCE,
    ),
)
r = results[0]
print(f"Rounds used: {r.num_rounds_used}")
print(f"Initial: {r.initial_output[:100]}...")
print(f"Refined: {r.final_output[:100]}...")

# Inspect per-round details
for round_info in r.rounds:
    print(f"  Round {round_info['round']}: improvement={round_info['improvement_score']:.3f}")
```

**Custom critique function:**

```python
def my_critic(current_output: str) -> str:
    return (
        f"The following answer may have errors. Identify any factual mistakes, "
        f"unclear reasoning, or missing steps:\n\n{current_output}\n\nCritique:"
    )

results = llm.generate_with_refinement(
    "Explain Newton's second law of motion.",
    config=RefinementConfig(max_rounds=2),
    critique_fn=my_critic,
)
```

### Best-of-N Dense Verification

Generate N candidate responses, score each one, and return the highest-scoring candidate.

```python
from swiftllm.config import VerificationConfig, ScoringStrategy

# Rule-based scoring (no extra model required)
results = llm.generate_best_of_n(
    "Explain the difference between supervised and unsupervised learning.",
    config=VerificationConfig(
        num_candidates=8,
        scoring_strategy=ScoringStrategy.RULE_BASED,
    ),
)
r = results[0]
print(f"Best score: {r.best_score:.3f}")
print(r.best_text)

# Ensemble scoring: combine rule-based, neural PRM, and sequence log-prob
results = llm.generate_best_of_n(
    "Write a haiku about machine learning.",
    config=VerificationConfig(
        num_candidates=8,
        scoring_strategy=ScoringStrategy.ENSEMBLE,
        rule_weight=0.5,
        neural_weight=0.3,
        logprob_weight=0.2,
    ),
)
```

### Configuring Inference Features on EngineConfig

You can set default inference configs on `EngineConfig` so they apply automatically:

```python
from swiftllm import LLM, EngineConfig
from swiftllm.config import SelfConsistencyConfig, RefinementConfig

llm = LLM.__new__(LLM)
llm.config = EngineConfig(
    model="meta-llama/Llama-2-7b-hf",
    self_consistency=SelfConsistencyConfig(num_samples=8),
    refinement=RefinementConfig(max_rounds=3),
)
# Now generate_with_self_consistency() and generate_with_refinement()
# work without passing config= each time.
```

---

## Disaggregated Serving

Disaggregated prefill/decode serving routes compute-bound **prefill** (processing the prompt) and bandwidth-bound **decode** (generating tokens) to dedicated, independently-scaled worker pools. This matches the Splitwise and DistServe architectures.

### Python Configuration

```python
from swiftllm.config import DisaggregatedServingConfig, DisaggregatedPolicy

ds_config = DisaggregatedServingConfig(
    num_prefill_workers=2,
    num_decode_workers=6,
    scheduling_policy=DisaggregatedPolicy.LEAST_LOADED,
    kv_transfer_timeout_ms=100,
    enable_auto_ratio=True,  # auto-compute optimal worker ratio at startup
)
```

### Optimal Worker Ratio (Rust API)

```rust
use swiftllm_core::serving::optimal_worker_ratio;

// prefill_tps: tokens per second a prefill worker processes
// decode_tps:  tokens per second a decode worker produces
let (n_prefill, n_decode) = optimal_worker_ratio(500.0, 50.0, 8);
// → (1, 7) — one prefill worker can saturate 7 decode workers
println!("Prefill workers: {n_prefill}, Decode workers: {n_decode}");
```

### Scheduling Policies

| Policy | Behavior |
|--------|---------|
| `ROUND_ROBIN` | Cycle through workers in order |
| `LEAST_LOADED` | Route to the worker with the fewest active requests |
| `LOCALITY_AWARE` | Prefer the worker that already holds the KV cache for this request |

---

## Recursive Language Model (RLM)

```python
from swiftllm import LLM, RlmConfig, RlmMode, SamplingParams

llm = LLM(model="path/to/model")

# REASONING mode: depth=3, REPL enabled (default)
results = llm.generate_with_rlm(
    "Prove by induction that 1+2+…+n = n(n+1)/2.",
    config=RlmConfig(
        mode=RlmMode.REASONING,
        max_depth=3,
        enable_repl=True,         # symbolic REPL sandbox
        var_binding_slots=32,     # soft key-value memory slots
        early_exit_threshold=0.92,
    ),
    base_params=SamplingParams(temperature=0.7, max_tokens=768),
)

result = results[0]
print(result.text)
print(f"Depth used : {result.recursion_depth_used}")
print(f"REPL steps : {len(result.repl_trace)}")
print(f"Variables  : {result.repl_variables}")

# AGENTIC mode: depth=5, larger token budget
results = llm.generate_with_rlm(
    "Plan a zero-downtime migration of a 10TB PostgreSQL database to Aurora.",
    config=RlmConfig(mode=RlmMode.AGENTIC, max_depth=5),
)

# SHALLOW mode: single decomposition step
results = llm.generate_with_rlm(
    "What is the derivative of sin(x)·cos(x)?",
    config=RlmConfig(mode=RlmMode.SHALLOW, max_depth=1),
)

# Disable REPL (pure recursive generation, no variable binding)
results = llm.generate_with_rlm(
    "Explain the halting problem.",
    config=RlmConfig(mode=RlmMode.REASONING, max_depth=3, enable_repl=False),
)
```

**CLI**

```bash
# RLM with max recursion depth 3
swiftllm generate -m /path/to/model -p "Prove that sqrt(2) is irrational." --rlm 3

# RLM without REPL sandbox
swiftllm generate -m /path/to/model -p "..." --rlm 3 --rlm-no-repl

# Agentic depth
swiftllm generate -m /path/to/model -p "Plan a database migration." --rlm 5
```

The `RlmOutput` dataclass contains:

| Field | Description |
|-------|-------------|
| `text` | Final generated text |
| `recursion_depth_used` | Actual max depth reached (0 = direct solve) |
| `repl_variables` | Final variable bindings `{name: value}` |
| `repl_trace` | Ordered list of REPL steps (type, args, output) |
| `early_exits` | Scheduler shortcuts (confidence ≥ threshold) |

---

## Dense Verification Layer

```python
from swiftllm import LLM, DenseVerificationConfig, VerificationStrategy, SamplingParams

llm = LLM(model="path/to/model")

# GATE_AND_REGEN: reject and regenerate until confidence ≥ 80% (up to 3 attempts)
results = llm.generate_with_dense_verification(
    "Explain Gödel's incompleteness theorems.",
    config=DenseVerificationConfig(
        strategy=VerificationStrategy.GATE_AND_REGEN,
        min_confidence=0.80,
        max_regen_attempts=3,
        score_repl_steps=True,    # score REPL:VERIFY annotations
    ),
    base_params=SamplingParams(temperature=0.7, max_tokens=512),
)

result = results[0]
print(result.text)
print(f"Global confidence  : {result.global_score:.1%}")
print(f"Accepted on attempt: {result.accepted_on_attempt}")
print(f"Low-conf positions : {result.low_confidence_positions}")

# SCORE_ONLY: always accept first draft but collect scores
results = llm.generate_with_dense_verification(
    "What causes northern lights?",
    config=DenseVerificationConfig(strategy=VerificationStrategy.SCORE_ONLY),
)
print(results[0].global_score)

# GATE: reject but do NOT regenerate (accept best-effort)
results = llm.generate_with_dense_verification(
    "Summarise the water cycle.",
    config=DenseVerificationConfig(
        strategy=VerificationStrategy.GATE,
        min_confidence=0.75,
    ),
)
```

**CLI**

```bash
# Dense verification with GATE_AND_REGEN (default)
swiftllm generate -m /path/to/model -p "..." --dense-verification

# Custom confidence threshold and regen limit
swiftllm generate -m /path/to/model -p "..." \
  --dense-verification --dv-min-confidence 0.75 --dv-max-regen 2

# Score only (no gating)
swiftllm generate -m /path/to/model -p "..." --dense-verification --dv-score-only
```

**Verification Strategies**

| Strategy | Behaviour |
|----------|-----------|
| `DISABLED` | Passthrough; no scoring. |
| `SCORE_ONLY` | Score all tokens/steps; always accept the first draft. |
| `GATE` | Reject drafts below `min_confidence`; accept best-effort without regenerating. |
| `GATE_AND_REGEN` | Reject and regenerate up to `max_regen_attempts` times; return the highest-scoring attempt. |

The `DenseVerificationOutput` dataclass contains:

| Field | Description |
|-------|-------------|
| `text` | Final (accepted) generated text |
| `global_score` | Overall confidence (0–1) |
| `token_scores` | Per-token confidence scores |
| `step_scores` | Per-REPL-step confidence (empty when `score_repl_steps=False`) |
| `accepted_on_attempt` | 1-indexed attempt number that was accepted |
| `low_confidence_positions` | Token indices below `min_confidence` |

---

## Configuration Reference

### SamplingParams

```python
from swiftllm import SamplingParams

params = SamplingParams(
    temperature=0.7,        # Sampling temperature (0 = greedy)
    top_p=0.9,              # Nucleus sampling
    top_k=50,               # Top-k sampling (-1 = disabled)
    min_p=0.0,              # Min-p filtering
    max_tokens=256,         # Maximum tokens to generate
    stop=["</s>"],          # Stop sequences
    presence_penalty=0.1,   # Presence penalty
    frequency_penalty=0.1,  # Frequency penalty
    repetition_penalty=1.1, # Repetition penalty
    n=1,                    # Number of output sequences
    best_of=1,              # Must be >= n
    seed=42,                # Random seed
    logprobs=5,             # Return top-5 token log-probs
)
```

### Phase 3 Inference Configs

```python
from swiftllm.config import (
    SelfConsistencyConfig, AnswerExtractor,
    RefinementConfig, StoppingCriterion, ImprovementMetric,
    VerificationConfig, ScoringStrategy,
    DisaggregatedServingConfig, DisaggregatedPolicy,
)

# Self-consistency
sc = SelfConsistencyConfig(
    num_samples=8,                       # chains to generate (≥ 2)
    extractor=AnswerExtractor.HEURISTIC, # HEURISTIC | AFTER_SENTINEL | LAST_LINE | XML_TAG
    answer_sentinel="The answer is",     # used by AFTER_SENTINEL
    answer_tag="answer",                 # used by XML_TAG
    temperature=0.8,                     # must be > 0
)

# Self-refinement
rf = RefinementConfig(
    max_rounds=3,
    min_improvement=0.05,
    stopping_criterion=StoppingCriterion.EITHER,  # MAX_ROUNDS | MIN_IMPROVEMENT | EITHER
    improvement_metric=ImprovementMetric.EDIT_DISTANCE, # EDIT_DISTANCE | ANY_CHANGE | EXTERNAL_SCORE
    critique_template=None,  # optional custom template with {output} placeholder
)

# Best-of-N verification
vc = VerificationConfig(
    num_candidates=8,
    scoring_strategy=ScoringStrategy.RULE_BASED,  # RULE_BASED | NEURAL | ENSEMBLE | SEQUENCE_LOG_PROB
    rule_weight=0.5,    # used in ENSEMBLE mode
    neural_weight=0.3,
    logprob_weight=0.2,
    neural_model=None,  # path to neural PRM for NEURAL / ENSEMBLE
)
```

### Phase 3 Model-Level Reasoning Configs

```python
from swiftllm.config import (
    RlmConfig, RlmMode,
    DenseVerificationConfig, VerificationStrategy,
)

# Recursive Language Model
rlm = RlmConfig(
    mode=RlmMode.REASONING,          # DISABLED | SHALLOW | REASONING | AGENTIC
    max_depth=3,                     # 0 = direct solve; paper recommends 2–4 for math/code
    enable_repl=True,                # symbolic REPL sandbox with variable binding
    var_binding_slots=32,            # number of soft key-value memory slots
    depth_hidden_size=None,          # None → d_model // 4 (set in Rust)
    early_exit_threshold=0.92,       # skip recursion when scheduler confidence ≥ threshold
    d_subproblem=None,               # None → d_model // 2 (set in Rust)
)

# Dense Verification Layer
dv = DenseVerificationConfig(
    strategy=VerificationStrategy.GATE_AND_REGEN,  # DISABLED | SCORE_ONLY | GATE | GATE_AND_REGEN
    num_verification_heads=8,        # cross-attention heads for draft ↔ REPL-trace attention
    min_confidence=0.80,             # global score threshold (0, 1]
    max_regen_attempts=3,            # max regeneration attempts (GATE_AND_REGEN only)
    score_repl_steps=True,           # also score REPL:VERIFY step annotations
)
```

### Phase 2 Training Configs

```python
from swiftllm.config import (
    GrpoConfig, CgarConfig, PrmConfig, PrmAggregation,
    LongRewardConfig, DenseAggregation,
)

grpo = GrpoConfig(
    group_size=8,                # rollout group size G (≥ 2)
    clip_eps=0.2,                # PPO clipping threshold ε
    kl_coeff=0.04,               # KL divergence penalty β
    correctness_weight=1.0,
    format_weight=0.2,
    length_penalty_weight=0.1,
    reference_model=None,        # defaults to same model as policy
)

cgar = CgarConfig(
    shallow_end=0.30,                    # training fraction ending shallow phase
    medium_end=0.60,                     # training fraction ending medium phase
    min_layers=None,                     # None → num_layers // 3
    max_layers=None,                     # None → num_layers
    enable_phased_specialisation=False,  # Jamba attention-lead / SSM-lead phases
    attention_lead_end=0.40,
)

prm = PrmConfig(
    aggregation=PrmAggregation.LAST_STEP,  # MIN | MEAN | PRODUCT | LAST_STEP | WEIGHTED_MEAN
    outcome_weight=0.5,
    prm_weight=0.5,
    step_separator="\n\n",
    neural_model=None,  # None → rule-based heuristic
)

lr_dense = LongRewardConfig(
    weight=0.1,                        # scalar applied to dense reward
    aggregation=DenseAggregation.MEAN, # MEAN | SUM | MAX | LAST
    normalise=True,                    # z-score normalise within batch
    reference_model=None,
)
```

### EngineConfig

```python
from swiftllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    download_dir="/data/models",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    dtype="float16",
    quantization="awq",
)
```

---

## Environment Variables

Every `SWIFTLLM_*` variable maps to a field in `EngineConfig` or `ServerConfig`. Set them in your shell profile, systemd unit, or Docker Compose file — they are read at startup and override coded defaults. Explicit constructor arguments always take final precedence.

### GPU & Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU VRAM for model weights + KV cache (0.0–1.0). Raise to ~0.95 on dedicated hosts; lower to 0.7 when sharing. |
| `SWIFTLLM_GPU_OVERHEAD_MB` | `0` | VRAM (in MB) to reserve for the OS and other processes. |
| `SWIFTLLM_NUM_GPU_LAYERS` | all | Number of layers to offload to GPU. `0` = CPU-only, `999` = all. |
| `SWIFTLLM_SWAP_SPACE` | `4.0` | CPU swap space in GiB for KV cache offloading. |
| `SWIFTLLM_CPU_OFFLOAD_GB` | `0.0` | Model weight gigabytes to keep on CPU RAM instead of GPU. |
| `SWIFTLLM_KV_CACHE_DTYPE` | `auto` | Data type for the KV cache. `fp8_e4m3`/`fp8_e5m2` halves memory. |
| `SWIFTLLM_BLOCK_SIZE` | `16` | Tokens per PagedAttention block. Allowed: `8`, `16`, `32`. |
| `SWIFTLLM_FLASH_ATTENTION` | `true` | Enable FlashAttention kernels. |
| `SWIFTLLM_ENFORCE_EAGER` | `false` | Disable CUDA graph capture; use eager execution. |
| `CUDA_VISIBLE_DEVICES` | (all) | Restrict which GPUs are visible, e.g. `0,2`. |

### Tensor Parallelism & Multi-GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_TENSOR_PARALLEL_SIZE` | `1` | GPUs for tensor parallelism. Must evenly divide attention heads. |
| `SWIFTLLM_PIPELINE_PARALLEL_SIZE` | `1` | Pipeline-parallel stages. |
| `NCCL_DEBUG` | — | NCCL logging level: `INFO`, `WARN`, `TRACE`. |
| `NCCL_P2P_DISABLE` | `0` | Set to `1` to disable GPU peer-to-peer on certain PCIe topologies. |

### Scheduling & Batching

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_MAX_NUM_SEQS` | `256` | Maximum concurrent sequences in a batch. |
| `SWIFTLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Maximum total tokens per forward pass. |
| `SWIFTLLM_MAX_PADDINGS` | `256` | Maximum padding tokens tolerated per batch. |
| `SWIFTLLM_SCHEDULER_POLICY` | `fcfs` | `fcfs`, `sjf`, or `priority`. |
| `SWIFTLLM_PREEMPTION_MODE` | `swap` | `swap` (KV cache to CPU) or `recompute`. |
| `SWIFTLLM_ENABLE_PREFIX_CACHING` | `false` | Reuse KV cache across requests sharing the same prefix. |
| `SWIFTLLM_ENABLE_CHUNKED_PREFILL` | `false` | Interleave prefill and decode; reduces time-to-first-token. |
| `SWIFTLLM_NUM_PARALLEL` | `1` | Parallel inference slots per model. |
| `SWIFTLLM_MAX_LOADED_MODELS` | `1` | Models held in GPU memory simultaneously. |
| `SWIFTLLM_KEEP_ALIVE` | `300` | Seconds a model stays loaded after last request. |

### Speculative Decoding

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_SPECULATIVE_MODEL` | — | Draft model for speculative decoding. |
| `SWIFTLLM_NUM_SPECULATIVE_TOKENS` | `5` | Tokens to draft per step. |
| `SWIFTLLM_SPECULATIVE_MAX_MODEL_LEN` | — | Override max sequence length for the draft model. |

### Model & Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_MODEL_DIR` | `~/.cache/swiftllm/models` | Default directory for downloaded models. |
| `SWIFTLLM_OFFLINE` | `false` | Set to `1` to disable all network downloads. |
| `SWIFTLLM_DTYPE` | `auto` | Weight data type: `auto`, `float16`, `bfloat16`, `float32`, `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`. |
| `SWIFTLLM_QUANTIZATION` | `none` | Quantization method: `none`, `awq`, `gptq`, `squeezellm`, `gguf`. |
| `SWIFTLLM_MAX_MODEL_LEN` | (model default) | Override the model's max sequence length. |
| `SWIFTLLM_TRUST_REMOTE_CODE` | `false` | Allow executing custom code from HuggingFace repos. |
| `SWIFTLLM_DEVICE` | `auto` | Device: `auto`, `cuda`, `cpu`, `metal`, `rocm`. |
| `SWIFTLLM_SEED` | `0` | Global random seed. |
| `HF_TOKEN` | — | HuggingFace API token for gated models. |

### LoRA

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_ENABLE_LORA` | `false` | Enable LoRA adapter support in the inference engine. |
| `SWIFTLLM_MAX_LORAS` | `1` | Maximum LoRA adapters loaded simultaneously. |
| `SWIFTLLM_MAX_LORA_RANK` | `16` | Maximum LoRA rank. |

### Server & Networking

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_HOST` | `0.0.0.0` | Bind address for the HTTP server. |
| `SWIFTLLM_PORT` | `8000` | Port for the HTTP server. |
| `SWIFTLLM_API_KEY` | — | Bearer-token API key. |
| `SWIFTLLM_CORS_ALLOW_ORIGINS` | `*` | Comma-separated allowed CORS origins. |
| `SWIFTLLM_SSL_CERTFILE` | — | Path to TLS certificate for HTTPS. |
| `SWIFTLLM_SSL_KEYFILE` | — | Path to TLS private key for HTTPS. |
| `SWIFTLLM_ROOT_PATH` | — | URL prefix for reverse-proxy deployments. |
| `SWIFTLLM_SERVED_MODEL_NAME` | — | Override model name in API responses. |
| `SWIFTLLM_MAX_LOG_LEN` | — | Truncate request/response logs to this many characters. |
| `SWIFTLLM_RESPONSE_ROLE` | `assistant` | Default role name in chat completion responses. |

### Build & CUDA

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_PATH` / `CUDA_HOME` | — | Path to CUDA toolkit. |
| `CUDACXX` | — | Path to `nvcc` binary. |
| `CMAKE_ARGS` | — | Extra CMake arguments for llama-cpp-python build. |

### Logging & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Rust log level: `trace`, `debug`, `info`, `warn`, `error`. |
| `SWIFTLLM_LOG_LEVEL` | — | Python log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

---

## CLI Commands

```bash
# Download a model
swiftllm download -m <model> [--download-dir <dir>] [--revision <rev>] [--token <hf-token>]

# Start the OpenAI-compatible server
swiftllm serve -m <model> --port 8000 [--api-key <key>] [--tensor-parallel-size 2]

# Standard generation
swiftllm generate -m <model> -p "Hello" --max-tokens 256

# Self-consistency generation (majority vote across N chains)
swiftllm generate -m <model> -p "Solve: 2x+4=10" --self-consistency 8 --temperature 0.8

# Multi-round refinement (up to R critique→revision cycles)
swiftllm generate -m <model> -p "Summarize the water cycle" --refinement-rounds 3

# Best-of-N: generate N, return the highest-scoring candidate
swiftllm generate -m <model> -p "Write a haiku" --best-of-n 8

# Recursive Language Model (up to 3 levels of recursive self-calling)
swiftllm generate -m <model> -p "Prove sqrt(2) is irrational." --rlm 3

# Dense Verification: cross-attention token/step confidence scoring
swiftllm generate -m <model> -p "Explain Gödel's theorems." \
  --dense-verification --dv-min-confidence 0.80 --dv-max-regen 3

# Interactive chat session
swiftllm chat -m <model> [--system "You are a helpful assistant"]

# Benchmark throughput
swiftllm benchmark -m <model> --num-prompts 100 --input-len 128 --output-len 128

# Model information
swiftllm info -m <model> [--json]

# Convert model format
swiftllm convert -i <path> -o <path> --format safetensors

# Standard fine-tuning
swiftllm train -m <model> --train-data <data> --method lora [--num-epochs 3]
swiftllm finetune -m <model> --train-data <data> --lora-r 16

# GRPO reinforcement learning training (Phase 2)
swiftllm grpo -m <model> --train-data <data> --group-size 8 [--enable-prm] [--long-reward-weight 0.1]

# Dataset ingestion — local files
swiftllm dataset -i ./docs/ -o train.jsonl                         # pretraining from a directory
swiftllm dataset -i ./src/ -o code.jsonl --format code --extensions .py,.rs
swiftllm dataset -i paper.pdf qa.csv ./notes/ -o sft.jsonl --format sft_completion
swiftllm dataset -i ./corpus/ -o /dev/null --stats-only            # dry-run statistics

# Dataset ingestion — HuggingFace Hub datasets
swiftllm dataset --hf-dataset tatsu-lab/alpaca -o alpaca.jsonl --format sft_completion
swiftllm dataset --hf-dataset HuggingFaceFW/fineweb --hf-subset sample-10BT \
  --hf-streaming --hf-max-samples 100000 -o fineweb.jsonl
swiftllm dataset --hf-dataset tatsu-lab/alpaca HuggingFaceH4/ultrachat_200k \
  --format sft_messages -o multi_hf.jsonl

# Dataset ingestion — local files + HuggingFace combined
swiftllm dataset -i ./my_docs/ --hf-dataset tatsu-lab/alpaca \
  --hf-max-samples 10000 --format sft_completion -o combined.jsonl
```

### Model Specifiers

The `-m` / `--model` flag accepts multiple formats:

| Format | Example |
|--------|---------|
| Local path | `/data/models/model.gguf` |
| HF repo ID | `meta-llama/Llama-2-7b-hf` |
| HF URL | `https://huggingface.co/org/repo/blob/main/file.gguf` |
| Repo:file shorthand | `org/repo:model.q4_k_m.gguf` |

---

## Architecture

```
+-----------------------------------------------------------------------------------+
|                           SwiftLLM Architecture (v2.2)                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  ┌─────────────────┐  ┌──────────────────────┐  ┌───────────────────────────┐   |
|  │  OpenAI API      │  │  Python SDK           │  │  CLI Interface            │   |
|  │  (auth, headers) │  │  LLM / AsyncLLM       │  │  serve/generate/train     │   |
|  │                  │  │                       │  │  finetune/grpo/dataset    │   |
|  └────────┬─────────┘  └──────────┬───────────┘  └──────────────┬────────────┘   |
|           │                       │                              │                |
|  ┌────────┴───────────────────────┴──────────────────────────────┴────────────┐  |
|  │                    Model Resolver & Downloader                              │  |
|  │        (HuggingFace Hub / Local Path / GGUF URL / Offline)                 │  |
|  └────────────────────────────────┬───────────────────────────────────────────┘  |
|                                   │                                               |
|  ┌────────────────────────────────┴───────────────────────────────────────────┐  |
|  │                  Phase 3 — Inference Enhancements                          │  |
|  │   ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │  |
|  │   │ Self-Consistency  │  │ Self-Refinement  │  │ Best-of-N Verification │   │  |
|  │   │  (Wang 2022)      │  │  (Madaan 2023)   │  │   + Dense Scoring      │   │  |
|  │   └──────────────────┘  └─────────────────┘  └────────────────────────┘   │  |
|  └────────────────────────────────┬───────────────────────────────────────────┘  |
|                                   │                                               |
|  ┌────────────────────────────────┴───────────────────────────────────────────┐  |
|  │               Disaggregated Serving (Phase 3)                              │  |
|  │   ┌────────────────────┐    ┌────────────────────┐   ┌──────────────────┐  │  |
|  │   │  Prefill Workers   │───▶│  KV Transfer       │──▶│  Decode Workers  │  │  |
|  │   │ (compute-bound)    │    │  (paged blocks)    │   │ (bandwidth-bound)│  │  |
|  │   └────────────────────┘    └────────────────────┘   └──────────────────┘  │  |
|  └────────────────────────────────┬───────────────────────────────────────────┘  |
|                                   │                                               |
|  ┌────────────────────────────────┴───────────────────────────────────────────┐  |
|  │              Core Inference Backend                                        │  |
|  │   [llama-cpp-python (GGUF)]         [Rust Engine (HF / SafeTensors)]       │  |
|  │   PagedAttention Memory Manager     Continuous Batching Scheduler          │  |
|  │   Speculative Decoding              Prefix Caching                         │  |
|  └────────────────────────────────┬───────────────────────────────────────────┘  |
|                                   │                                               |
|  ┌────────────────────────────────┴───────────────────────────────────────────┐  |
|  │              Phase 1 — Model Architectures                                 │  |
|  │   ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │  |
|  │   │  Standard        │  │  Mamba SSM      │  │  Jamba Hybrid          │   │  |
|  │   │  Attention Layers│  │  Layers (Phase 1)│  │  Attention + Mamba     │   │  |
|  │   └──────────────────┘  └─────────────────┘  │  + MoE FFN (Phase 1)   │   │  |
|  │                                               └────────────────────────┘   │  |
|  └────────────────────────────────┬───────────────────────────────────────────┘  |
|                                   │                                               |
|  ┌────────────────────────────────┴───────────────────────────────────────────┐  |
|  │              Phase 2 — Training Engine                                     │  |
|  │   ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │  |
|  │   │  GRPO    │  │  CGAR   │  │     PRM       │  │  LongR Dense Rewards  │  │  |
|  │   │  (RL)    │  │ Curriculum│  │ Step Rewards  │  │  Token-level NLL ΔIG  │  │  |
|  │   └──────────┘  └──────────┘  └──────────────┘  └──────────────────────┘  │  |
|  │   LoRA / QLoRA / Full Fine-Tuning                                           │  |
|  │   Muon / AdamW / SGD Optimizers    LR Schedulers   Checkpoint Management   │  |
|  │   ┌──────────────────────────────────────────────────────────────────────┐  │  |
|  │   │  Dataset Ingestion — .txt .md .py .rs .pdf .docx .csv .html .jsonl   │  │  |
|  │   │  pretraining | sft_messages | sft_completion | code  (auto-ingest)   │  │  |
|  │   └──────────────────────────────────────────────────────────────────────┘  │  |
|  └────────────────────────────────┬───────────────────────────────────────────┘  |
|                                   │                                               |
|  ┌────────────────────────────────┴───────────────────────────────────────────┐  |
|  │                         CUDA Kernels                                       │  |
|  └────────────────────────────────────────────────────────────────────────────┘  |
|                                                                                   |
|  ┌────────────────────────────────────────────────────────────────────────────┐  |
|  │  Install & Deploy: install.sh | airgap-bundle.sh | offline mode             │  |
|  └────────────────────────────────────────────────────────────────────────────┘  |
+-----------------------------------------------------------------------------------+
```

---

## Project Structure

```
swiftllm/
├── install.sh                        # Installer (GPU detection, venv, Rust build)
├── airgap-bundle.sh                  # Air-gap bundle creator (offline deploy)
│
├── python/swiftllm/                  # Python package
│   ├── __init__.py                   #   Public API & lazy training imports
│   ├── engine.py                     #   LLM / AsyncLLM / LLMEngine
│   │                                 #   + generate_with_self_consistency()
│   │                                 #   + generate_with_refinement()
│   │                                 #   + generate_best_of_n()
│   │                                 #   + generate_with_rlm()            ← Phase 3
│   │                                 #   + generate_with_dense_verification() ← Phase 3
│   ├── training.py                   #   Trainer / GrpoTrainer / fine_tune / prepare_dataset
│   ├── dataset.py                    #   DatasetIngester — txt/md/code/pdf/docx/csv→JSONL
│   ├── sampling.py                   #   Sampling strategies + SelfConsistencySampler
│   ├── config.py                     #   All config dataclasses (Phase 1–3)
│   │                                 #   GrpoConfig, CgarConfig, PrmConfig, LongRewardConfig
│   │                                 #   SelfConsistencyConfig, RefinementConfig
│   │                                 #   VerificationConfig, DisaggregatedServingConfig
│   │                                 #   RlmConfig, RlmMode                ← Phase 3
│   │                                 #   DenseVerificationConfig, VerificationStrategy ← Phase 3
│   ├── cli.py                        #   CLI: serve/generate/train/finetune/grpo/dataset/…
│   │                                 #   + --rlm DEPTH, --dense-verification flags ← Phase 3
│   └── model_resolver.py             #   HuggingFace / local / offline resolution
│
├── crates/
│   ├── swiftllm-core/                # Core engine
│   │   └── src/
│   │       ├── sampling/
│   │       │   ├── mod.rs            #   Sampling module root
│   │       │   ├── strategies.rs     #   Greedy, top-k, top-p, beam search
│   │       │   └── self_consistency.rs  # Phase 3: majority voting
│   │       ├── inference/
│   │       │   ├── mod.rs            #   Inference module root
│   │       │   ├── refinement.rs     #   Phase 3: self-refinement pipeline
│   │       │   └── verification.rs   #   Phase 3: Best-of-N dense verification
│   │       ├── serving/
│   │       │   ├── mod.rs            #   Serving module root
│   │       │   └── disaggregated.rs  #   Phase 3: disaggregated prefill/decode
│   │       ├── scheduler/            #   Continuous batching scheduler
│   │       ├── memory/               #   PagedAttention block manager
│   │       └── engine.rs             #   Core engine step loop
│   │
│   ├── swiftllm-models/              # Model loading & architectures
│   │   └── src/
│   │       ├── layers/
│   │       │   ├── mamba.rs          #   Phase 1: Mamba-3 SSM + MIMO scan
│   │       │   ├── moe.rs            #   Phase 1: LatentMoE + dynamic bias
│   │       │   ├── rlm.rs            #   Phase 3: RlmLayer + ReplState + VariableBindingTable
│   │       │   └── dense_verification.rs  # Phase 3: DenseVerificationLayer
│   │       ├── architectures/
│   │       │   └── jamba.rs          #   Phase 1: Jamba hybrid (Attn + Mamba + MoE)
│   │       └── …
│   │
│   ├── swiftllm-training/            # Training engine
│   │   └── src/
│   │       ├── grpo.rs               #   Phase 2: GRPO optimizer + rewards
│   │       ├── curriculum.rs         #   Phase 2: CgarScheduler + PhasedSpecialisation
│   │       ├── process_reward.rs     #   Phase 2: Process Reward Models (PRM)
│   │       ├── long_reward.rs        #   Phase 2: LongR dense token-level rewards
│   │       ├── config.rs             #   TrainingConfig (superset of Python's)
│   │       ├── trainer.rs            #   Training loop + curriculum integration
│   │       ├── optimizer.rs          #   AdamW, SGD, LR schedulers
│   │       ├── muon.rs               #   Muon optimizer
│   │       ├── fine_tuning.rs        #   LoRA, QLoRA, full fine-tuning
│   │       ├── data.rs               #   Data loading (JSONL, CSV, text)
│   │       └── metrics.rs            #   Training metrics & logging
│   │
│   ├── swiftllm-cuda/                # CUDA kernel bindings
│   └── swiftllm-server/              # HTTP server (OpenAI API, auth, security)
│       └── src/api/openai.rs         #   OpenAI-compatible chat completions endpoint
│
└── examples/
    ├── basic_inference.py            # Simple inference
    ├── streaming.py                  # Streaming generation
    ├── batch_processing.py           # High-throughput batch processing
    ├── dataset_ingestion.py          # Dataset ingestion — all formats, all output schemas
    ├── openai_server.py              # OpenAI API server
    ├── multi_gpu.py                  # Multi-GPU inference
    ├── fine_tuning.py                # LoRA and QLoRA fine-tuning
    ├── training.py                   # Full training with callbacks and config management
    ├── self_consistency.py           # Phase 3: self-consistency voting demo
    ├── grpo_training.py              # Phase 2: GRPO + CGAR + PRM + LongR training demo
    ├── rlm_inference.py              # Phase 3: RLM — 3 modes + variable-binding demo
    └── dense_verification_inference.py  # Phase 3: Dense Verification — 4 strategies
```

---

## Examples

See the [examples/](examples/) directory:

| File | What It Demonstrates |
|------|---------------------|
| [`basic_inference.py`](examples/basic_inference.py) | Simple LLM inference with SamplingParams |
| [`streaming.py`](examples/streaming.py) | Streaming token generation |
| [`batch_processing.py`](examples/batch_processing.py) | High-throughput batch inference |
| [`dataset_ingestion.py`](examples/dataset_ingestion.py) | **Dataset Ingestion**: convert dirs/files (.txt .md .py .rs .pdf .docx .csv …) to JSONL; all 4 output formats; auto-ingest demo in `fine_tune()` |
| [`openai_server.py`](examples/openai_server.py) | OpenAI-compatible API server |
| [`multi_gpu.py`](examples/multi_gpu.py) | Tensor-parallel multi-GPU inference |
| [`fine_tuning.py`](examples/fine_tuning.py) | LoRA and QLoRA fine-tuning |
| [`training.py`](examples/training.py) | Full training with callbacks and checkpoint management |
| [`self_consistency.py`](examples/self_consistency.py) | **Phase 3**: self-consistency voting — three demo modes including offline `SelfConsistencySampler.vote()` |
| [`grpo_training.py`](examples/grpo_training.py) | **Phase 2**: GRPO + CGAR + PRM + LongR — auto-generates synthetic math data, full config resolution, metric callback |
| [`rlm_inference.py`](examples/rlm_inference.py) | **Phase 3**: Recursive Language Model — SHALLOW / REASONING / AGENTIC modes, variable-binding demo, no-REPL variant |
| [`dense_verification_inference.py`](examples/dense_verification_inference.py) | **Phase 3**: Dense Verification — all four strategies, multi-prompt batch scoring, confidence calibration demo |

---

## OpenAI-Compatible Server

```bash
# Start with API key authentication
swiftllm serve -m /path/to/model.gguf --port 8000 --api-key sk-my-secret-key

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-secret-key" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Available endpoints
GET  /health                  # Health check
GET  /v1/models               # List loaded models
POST /v1/chat/completions     # Chat completions (streaming supported)
GET  /metrics                 # Prometheus + JSON metrics
```

---

## Model Resolver API

```python
from swiftllm import resolve_model

# Resolve a HuggingFace URL to a local path (downloads if needed)
path = resolve_model("https://huggingface.co/org/repo/blob/main/model.gguf")

# Resolve a repo:filename shorthand
path = resolve_model("org/repo:model.q4_k_m.gguf")

# Resolve a full repo (downloads all shards)
path = resolve_model("meta-llama/Llama-2-7b-hf")

# Local paths are validated and returned as-is
path = resolve_model("/data/models/my-model.gguf")

# Control download location and use gated-model token
path = resolve_model(
    "meta-llama/Llama-2-7b-hf",
    download_dir="/data/models",
    token="hf_...",
)
```

---

## Security

SwiftLLM includes built-in security features across the server, installer, and runtime:

**Server**
- **API Key Authentication** — Protect endpoints with `--api-key` flag
- **Input Validation** — Request size limits, parameter range checks, content length limits
- **Security Headers** — HSTS, X-Content-Type-Options, X-Frame-Options, X-Request-ID
- **Sanitized Errors** — Internal errors logged server-side only; never exposed to clients
- **Request Size Limits** — 10 MB maximum request body to prevent abuse
- **Safe JSON Serialization** — SSE streaming uses `serde_json` to prevent injection

**Installer & Runtime**
- **Supply Chain Verification** — SHA256 checksum verification for `rustup-init` downloads
- **Input Sanitization** — Platform tags and paths validated before use in shell commands
- **Path Traversal Protection** — Checkpoint and cache operations validate resolved paths stay within expected boundaries
- **UTF-8 Safety** — Data loading uses char-boundary-safe truncation to prevent panics on multi-byte input
- **Numeric Safety** — Optimizer arithmetic guards against division by zero and unbounded iteration

---

## Changelog

### v2.2.1-beta

**Security Audit & Hardening**

- **Critical**: Added `Drop` implementation for `Storage::Cuda` to properly call `swiftllm_cuda::free()` and prevent GPU memory leaks
- **Critical**: Enforced `Send`/`Sync` safety invariants via documentation for CUDA stream synchronization
- **Critical**: Added `check_cuda_last_error()` after all 10 CUDA kernel launches in `bindings.rs` to catch silent data corruption
- **High**: Replaced variable-time string comparisons with `subtle::ConstantTimeEq` for API key authentication to prevent timing attacks
- **High**: CORS configuration now reads from `ServerConfig.cors_origins` instead of hardcoded `Any`
- **High**: The `/metrics` endpoint is now authenticated
- **High**: Added comprehensive input validation to the legacy `/v1/completions` endpoint
- **High**: Changed default API server bind address from `0.0.0.0` to `127.0.0.1` and added security warnings for network exposure
- **High**: Hardened the install script by replacing `curl | sh` with a safer download-verify-execute pattern
- **Medium**: Fixed integer overflow in `engine.rs` block sizing by using `checked_mul` chain
- **Medium**: Corrected INT4 memory computation in `config.rs` with new `element_bit_width()` and `size_bytes_for_elements()`
- **Medium**: Fixed TOCTOU race condition in `Scheduler::is_empty()` by acquiring locks atomically
- **Medium**: Fixed UTF-8 string slicing panic in `PyGenerationOutput.__repr__` using char-safe truncation
- **Medium**: `swiftllm-server` now reads API keys via `SWIFTLLM_API_KEY` environment variable instead of CLI argument
- **Low**: Deprecated panicking `cuda_device_index()` method in favor of `try_cuda_device_index()`
- **Low**: Added bounds validation for `gpu_memory_utilization` CLI argument

---

### v2.1.0-beta

**Phase 1 — Hybrid Model Architectures** (`crates/swiftllm-models/`)

- **New**: `mamba.rs` — `MambaConfig`, `MambaLayer`, `MambaBlock`; selective SSM with full discretization (∆, A, B, C, D projections), hardware-aware parallel scan, 16 unit tests
- **New**: `moe.rs` — `MoeConfig`, `MoeLayer`; sparse top-k expert routing with load-balancing auxiliary loss, capacity enforcement, 14 tests
- **New**: `jamba.rs` — `JambaConfig`, `JambaLayer`; interleaved Attention + Mamba + MoE hybrid with configurable `attn_layer_offset`, 15 tests
- Updated `swiftllm-models/src/lib.rs` and `swiftllm-models/src/mod.rs` to re-export all Phase 1 public types

**Phase 2 — Training Enhancements** (`crates/swiftllm-training/`)

- **New**: `grpo.rs` — `GrpoConfig`, `GroupSamples`, `GrpoLoss`; group-relative advantage computation, PPO clipped policy gradient, KL divergence penalty, rule-based reward functions (correctness, format, length), 16 tests
- **New**: `curriculum.rs` — `CgarConfig`, `CgarScheduler`, `PhasedSpecialisationConfig`, `PhasedSpecialisationScheduler`, `CurriculumState`, `CurriculumTick`; smooth Hermite phase transitions, hybrid Jamba specialisation scheduling, 14 tests
- **New**: `process_reward.rs` — `RulePrm`, `NeuralPrm`, step boundary parsing, 5 aggregation strategies (Min/Mean/Product/LastStep/WeightedMean), PRM+outcome blending, 14 tests
- **New**: `long_reward.rs` — `LongRewardConfig`, `DenseAggregation`, token-level NLL relative information gain, batch normalisation, 12 tests
- Updated `trainer.rs` — integrates `CurriculumState`, per-step `apply_curriculum_lr()`
- Updated `config.rs` — `TrainingConfig` gains `num_layers`, `grpo`, `cgar`, `phased_spec`, `prm`, `long_reward_weight`
- Updated `lib.rs` — re-exports all Phase 2 public types

**Phase 3 — Inference Enhancements** (`crates/swiftllm-core/`)

- **New**: `sampling/self_consistency.rs` — `SelfConsistencyConfig`, `AnswerExtractor`, `ConsistencyCandidate`, majority voting with log-prob tiebreaking, 4 extractor strategies, 14 tests
- **New**: `inference/refinement.rs` — `RefinementConfig`, `StoppingCriterion`, `ImprovementMetric`, `RefinementPipeline`; `normalised_edit_distance()` O(min(m,n)) 2-row DP, 15 tests
- **New**: `inference/verification.rs` — `VerificationConfig`, `ScoringStrategy`, `ScoredCandidate`, `verify_and_rank()`, `best_of_n_by_logprob()`, 14 tests
- **New**: `serving/disaggregated.rs` — `DisaggregatedConfig`, `DisaggregatedScheduler`, `WorkerRole`, `WorkerSpec`, `SchedulingPolicy`, `KvTransferMetadata`, `optimal_worker_ratio()`, 15 tests
- Updated `lib.rs`, `sampling/mod.rs` — re-export all Phase 3 public types

**Phase 1 — Mamba-3 & LatentMoE Fixes** (`crates/swiftllm-models/src/layers/`)

- **New**: `mimo_scan_step_cpu()` in `mamba.rs` — MIMO multi-head scan; groups `d_inner` into `num_heads` heads; output step `y_head = H_head @ C` (GEMM per head vs. independent dot-products); converts decode from memory-bound to compute-bound; `MambaLayer::forward_step()` now dispatches to MIMO or standard scan
- **Fixed**: `Router::route()` — per-token `top_k_routing_cpu()` with real gate logits (was all-zeros)
- **Fixed**: `ExpertMlp::forward()` — `down_proj(silu(gate_proj(x)) * up_proj(x))` correct SwiGLU gating (was `down_proj(x)`)
- **Fixed**: `MoeLayer::forward()` → `&self`; dispatch loop over routing indices; `update_load_stats()` split off for training
- **Fixed**: `LatentMoeLayer::forward()` — now returns `Ok(expanded)` instead of zeros
- **Fixed**: `AttentionBlock`, `MambaBlock`, and `MambaBlock::forward_step()` in `jamba.rs` — now properly chain `norm → op → residual`

**Phase 3 — Model-Level Reasoning** (`crates/swiftllm-models/src/layers/`)

- **New**: `rlm.rs` (~500 lines) — `RlmConfig`, `RlmLayer`, `ReplState`, `ReplStep` (Assign/Compute/Verify/Recurse), `VariableBindingTable` (soft-attention key-value store), `RecursionScheduler` (complexity-classifier MLP with early-exit); `forward()` and `forward_with_repl()` APIs; 10+ tests
- **New**: `dense_verification.rs` (~450 lines) — `DenseVerificationConfig`, `DenseVerificationLayer`, `VerificationResult` (global/token/step scores, accepted flag, low-confidence positions), `embed_repl_trace()`, `cross_attention_cpu()`; `verify()` and `verify_and_correct()` APIs; 11 tests
- Updated `layers/mod.rs` — re-exports all new types

**Python API Updates**

- `config.py` — 14 new dataclasses: `SelfConsistencyConfig`, `RefinementConfig`, `VerificationConfig`, `DisaggregatedServingConfig`, `GrpoConfig`, `CgarConfig`, `PrmConfig`, `LongRewardConfig`, plus 7 new enums; `EngineConfig` gains 4 optional nested inference fields; full `from_dict()`/`to_dict()` round-trip support
  - **New** (this release): `RlmConfig`, `RlmMode`, `DenseVerificationConfig`, `VerificationStrategy`; `EngineConfig` gains `rlm` and `dense_verification` optional fields; `from_dict()` updated for both
- `training.py` — `TrainingConfig` gains Phase 2 fields; new `GrpoTrainer` class with CGAR layer scheduling (smooth Hermite); `grpo_train()` convenience function; JSON round-trip with enum deserialisation
- `sampling.py` — `SelfConsistencySampler` with 4 extractor strategies, majority voting, log-prob tiebreaking; `SelfConsistencyResult` dataclass
- `engine.py` — `LLM.generate_with_self_consistency()`, `LLM.generate_with_refinement()`, `LLM.generate_best_of_n()`; helper functions `_normalised_edit_distance()`, `_rule_score()`; `RefinementOutput`, `VerifiedOutput` dataclasses
  - **New** (this release): `LLM.generate_with_rlm()`, `LLM.generate_with_dense_verification()`; `RlmOutput`, `DenseVerificationOutput` dataclasses
- `__init__.py` — full re-export of all new types; lazy training set extended to `GrpoTrainer`, `grpo_train`; **new** exports: `RlmConfig`, `RlmMode`, `DenseVerificationConfig`, `VerificationStrategy`, `RlmOutput`, `DenseVerificationOutput`
- `cli.py` — `generate` command gains `--self-consistency`, `--refinement-rounds`, `--best-of-n` flags; new `grpo` subcommand with full CGAR/PRM/LongR flag set
  - **New** (this release): `generate` command gains `--rlm DEPTH`, `--rlm-no-repl`, `--dense-verification`, `--dv-min-confidence`, `--dv-max-regen`, `--dv-score-only` flags
- **New**: `examples/self_consistency.py` — three demo modes (basic SC, sentinel extractor, offline vote)
- **New**: `examples/grpo_training.py` — full GRPO + CGAR + PRM + LongR demo with synthetic math data
- **New** (this release): `examples/rlm_inference.py` — SHALLOW / REASONING / AGENTIC modes, variable-binding demo, no-REPL variant
- **New** (this release): `examples/dense_verification_inference.py` — all four strategies, multi-prompt batch scoring, confidence calibration demo

---

### v2.2.0-beta

**HuggingFace Dataset Support** (`python/swiftllm/dataset.py`, `training.py`, `cli.py`)

- **New**: `HuggingFaceSource` dataclass — describes one HuggingFace Hub dataset to pull into the ingestion pipeline
  - **Auto-detects** Alpaca (`instruction`/`output`), ShareGPT (`conversations` with `from`/`value`), OpenAI messages (`messages` with `role`/`content`), prompt+completion, Q&A, and plain `text` schemas — no field mapping needed for standard datasets
  - **Field overrides**: `text_field`, `prompt_field`, `completion_field`, `messages_field`, `instruction_field`, `input_field`, `output_field` for non-standard schemas
  - **Sampling controls**: `max_samples`, `shuffle`, `seed`
  - **Streaming mode**: avoids full dataset download for very large corpora (FineWeb, RedPajama, The Pile)
  - **`trust_remote_code`** / **`cache_dir`** pass-through to `load_dataset()`
  - ShareGPT `from`/`value` conversations normalised to OpenAI `role`/`content` format automatically
- **Updated**: `IngestionConfig` — new `hf_sources: List[HuggingFaceSource]` field; `input_paths` now optional (supply either `input_paths`, `hf_sources`, or both)
- **Updated**: `IngestionResult` — three new fields: `total_hf_rows`, `total_hf_chunks`, `hf_dataset_counts`; `summary()` now shows a separate HuggingFace section
- **Updated**: `DatasetIngester.ingest()` — processes local files then HF sources in sequence; SHA-256 dedup pool is shared so duplicates across sources are eliminated
- **Updated**: `ingest_dataset()` — gains `hf_sources` parameter; `input_paths` now defaults to `None` for HF-only use
- **Updated**: `Trainer._auto_ingest_if_needed()` — handles `train_data="hf:<dataset>"` shorthand and `config.hf_train_sources`
- **Updated**: `fine_tune()` — new parameters: `hf_dataset`, `hf_split`, `hf_subset`, `hf_max_samples`, `hf_streaming`
- **Updated**: `cli.py` — `swiftllm dataset` gains 14 new `--hf-*` flags; `--input` is now optional (omit when using only `--hf-dataset`); added Sources and HuggingFace options groups
- **Updated**: `__init__.py` — exports `HuggingFaceSource`
- **New optional dependency**: `pip install datasets` (`pip install swiftllm[hf]`)

**Dataset Ingestion** (`python/swiftllm/dataset.py`) *(original v2.2.0-beta additions)*

- **New**: `dataset.py` — full multi-format local file ingestion pipeline
  - **Formats read**: `.txt` `.md` `.rst` `.log` `.tex` · `.py` `.js` `.ts` `.rs` `.go` `.java` `.c` `.cpp` `.cs` `.rb` `.php` `.swift` `.kt` `.sql` `.yaml` `.toml` `.sh` and 20+ more code extensions · `.pdf` (pdfplumber → pypdf → PyPDF2 cascade) · `.docx` (python-docx) · `.html`/`.xml` (BeautifulSoup4 or regex fallback) · `.csv` · `.json` · `.jsonl`
  - **Output schemas**: `pretraining` · `sft_messages` · `sft_completion` · `code`
  - **Smart parsing**: CSV/JSONL auto-detects `prompt`/`completion`/`messages`/`text` columns and passes structured data through directly
  - **Code-aware chunking**: splits at `def`/`class`/`fn`/`func`/`impl` boundaries before falling back to character-level chunking
  - **SHA-256 deduplication**: skips exact-duplicate chunks across the entire run
  - **Size guard**: skips files above configurable `max_file_size_mb`
  - **Metadata attachment**: optional `_source` / `_ext` keys per record
- **New**: `prepare_dataset()` in `training.py` — convenience wrapper with full docstring
- **Updated**: `Trainer._auto_ingest_if_needed()` — fires automatically when `train_data` is a directory or list
- **Updated**: `fine_tune()` — `train_data` accepts `str | List[str]`; `dataset_format` parameter added
- **Updated**: `__init__.py` — exports all dataset classes and helpers
- **Updated**: `cli.py` — `swiftllm dataset` subcommand with full flag set
- **New**: `examples/dataset_ingestion.py` — 6 demo modes
- **Version bump**: `2.1.0-beta` → `2.2.0-beta`

---

### v2.0.1-beta

**Training UX Fixes**
- `Trainer.train()` now prints a visible `[SIMULATED]` banner at the start of every run
- `swiftllm train` and `swiftllm finetune` validate `--train-data` (and `--eval-data`) up front — missing paths raise `FileNotFoundError`, empty files raise `ValueError`
- Validation also covers path fields loaded from `--config` JSON

**Regression Coverage**
- Full regression matrix on Ubuntu 24.04 + CUDA 13.0 (RTX PRO 4000 Blackwell): install → download → generate → finetune → train → cleanup — all passing
- Checkpoint artifacts verified: `training_config.json` + `trainer_state.json` per checkpoint dir, `save_total_limit` rotation behaves as documented

---

### v2.0.0.2-alpha

**Regression Test Fixes**
- Added `fastapi>=0.100` and `uvicorn>=0.23` as `[serve]` optional dependencies
- `install.sh` installs the wheel with `[serve]` extras so the API server works out of the box
- `airgap-bundle.sh` now includes `fastapi` and `uvicorn` in offline wheel downloads

**CPU and ARM Wheel Support**
- CPU-only build is now the default — produces a portable wheel with zero CUDA dependencies
- Explicit `cpu` and `cuda` Cargo features; `cpu` is the default
- `airgap-bundle.sh --arch ARCH` flag auto-maps to the correct pip platform tag and rustup target triple

**Installer Portability Fixes**
- Replaced non-portable `grep -oP` with portable `sed` for CUDA version detection
- PEP 668 handling for externally-managed system Python
- SHA256 verification prefers `sha256sum` with `shasum` fallback
- `set -o pipefail` added to both scripts

**Rust Code Quality**
- Fixed 14 `partial_cmp().unwrap()` calls — replaced with `unwrap_or(Ordering::Equal)` to prevent NaN panics
- Added `checked_add` + bounds validation in GGUF loader
- Replaced `try_into().unwrap()` in SafeTensors header parser with proper `Result` propagation

---

### v2.0.0.1-alpha

**Air-Gapped / Offline Installation**
- New `airgap-bundle.sh` script; `install.sh --airgap` flag; `SWIFTLLM_OFFLINE=1` runtime mode

**Security Hardening**
- Fixed JSON injection in SSE streaming (`serde_json::json!()`)
- Fixed shell injection in `airgap-bundle.sh`
- SHA256 checksum verification for `rustup-init`
- Path traversal protection in `Trainer.resume_from_checkpoint()`

**Bug Fixes**
- Fixed use-after-move in `engine.rs`
- Fixed usize negation in `trainer.rs`
- Added missing `tempfile` dev-dependency to `swiftllm-training` Cargo.toml

---

### v2.0.0-alpha

**Training & Fine-Tuning**
- `swiftllm-training` Rust crate with LoRA/QLoRA/full fine-tuning
- Muon optimizer with Newton-Schulz orthogonalization
- AdamW and SGD with linear/cosine/constant LR schedulers
- Checkpoint save/load with configurable `save_total_limit`
- Python `Trainer` class with callbacks and early stopping
- CLI commands: `swiftllm train` and `swiftllm finetune`

**Inference Engine**
- Core engine `step()` with batch processing and token sampling
- Configurable EOS token ID per model
- Fast path for greedy sampling — zero allocation when no penalties are active

**Sampling Optimizations**
- Replaced O(n log n) full sort with O(n) quickselect for top-k and beam search
- Numerically stable `_log_softmax` across all Python samplers

**Server & API**
- API key authentication middleware
- Security headers: HSTS, X-Content-Type-Options, X-Frame-Options, X-Request-ID
- Request body size limit (10 MB)
- `/metrics` endpoint (JSON + Prometheus text format)
- SSE streaming for OpenAI-compatible streaming responses

---

### v1.0.0

- Initial release
- PagedAttention memory management
- Continuous batching with preemption (swap/recompute)
- Token sampling: greedy, temperature, top-k, top-p, min-p, repetition penalty
- OpenAI-compatible HTTP API
- Python SDK: `LLM`, `AsyncLLM`, `SamplingParams`
- CLI: serve, generate, benchmark, convert, info, chat, download
- HuggingFace model downloading and GGUF support via llama-cpp-python

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

Before submitting a PR:
1. Ensure all Rust tests pass: `cargo test --workspace`
2. Ensure Python syntax is clean: `python3 -m py_compile python/swiftllm/*.py`
3. Follow the existing file header format (project, file, path, author, date, USES/USED BY/SEE ALSO)
4. Add unit tests for any new Rust code (aim for ≥ 12 tests per new module)
5. Update `CLAUDE.md` memory if the change affects the project phase structure

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

SwiftLLM builds on ideas from:
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention and continuous batching
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF format and quantization
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Efficient attention kernels
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — Model architectures
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO reinforcement learning
- [Self-Refine](https://arxiv.org/abs/2303.17651) — Multi-round self-refinement (Madaan et al., 2023)
- [Self-Consistency](https://arxiv.org/abs/2203.11171) — Majority voting over reasoning chains (Wang et al., 2022)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Process Reward Models (Lightman et al., 2023)
- [Splitwise](https://arxiv.org/abs/2311.18677) / [DistServe](https://arxiv.org/abs/2401.09670) — Disaggregated prefill/decode serving
- [Jamba](https://arxiv.org/abs/2403.19887) — Hybrid Mamba + Attention + MoE architecture
- [Mamba](https://arxiv.org/abs/2312.00752) — Selective state space models (Gu & Dao, 2023)
- [Muon](https://arxiv.org/abs/2409.20325) — Newton-Schulz orthogonalized gradient optimizer
- "Architecting the Next-Generation Agentic Paradigm: A Hybrid Synthesis of Mamba-3, Mixture of Experts, Recursive Language Models, and Dense Verification" (2024) — RLM + Dense Verification architecture

<!--
    ------------------------------------------------------------------------------
    END OF FILE: README.md
    REPO PATH:   /swiftllm/README.md
    (c) 2026 SWIFTLLM | Apache 2.0 License
    ------------------------------------------------------------------------------
-->
