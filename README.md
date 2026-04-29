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

# SwiftLLM
![Logo](https://github.com/infrabrew/infrabrew.github.io/blob/master/swiftllm/assets/logo-mark-128.png?raw=true)
<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.1--beta-yellow.svg" alt="v2.0.1-beta">
  <img src="https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
</p>

**SwiftLLM** is a high-performance LLM inference, serving, and training engine built with Rust for maximum speed and efficiency. It features state-of-the-art memory management, continuous batching, multi-GPU support, and built-in LoRA/QLoRA fine-tuning.

## Key Features

- **High Throughput**: Continuous batching and efficient scheduling for maximum tokens/second
- **Memory Efficient**: PagedAttention for optimal KV cache management
- **Low Latency**: Optimized CUDA kernels and speculative decoding
- **Tensor Parallelism**: Scale to multiple GPUs seamlessly
- **OpenAI Compatible**: Drop-in replacement for OpenAI API with security hardening
- **Training & Fine-Tuning**: LoRA, QLoRA, and full fine-tuning with Muon, AdamW, and SGD optimizers
- **Python Friendly**: Easy-to-use Python API with async support
- **Multiple Formats**: Support for HuggingFace, GGUF, and SafeTensors
- **Model Downloading**: Download models from HuggingFace Hub by ID or URL
- **GGUF Inference**: Run quantized GGUF models on GPU via llama-cpp-python
- **Air-Gapped Install**: Bundle and deploy on networks with no internet access
- **Secure by Default**: API key authentication, input validation, and security headers

## Supported Models

| Architecture | Models |
|-------------|--------|
| **LLaMA** | LLaMA, LLaMA 2, LLaMA 3, Code Llama |
| **Mistral** | Mistral 7B, Mixtral 8x7B |
| **Qwen** | Qwen, Qwen 2, Qwen 3 |
| **Phi** | Phi-2, Phi-3 |
| **Falcon** | Falcon |
| **Gemma** | Gemma |

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

The wheel builds on any Linux or macOS host, on both Intel and ARM CPUs:

| OS        | x86_64                        | aarch64 / arm64                  |
| --------- | ----------------------------- | -------------------------------- |
| Linux     | `manylinux2014_x86_64`        | `manylinux2014_aarch64`          |
| macOS     | `macosx_10_15_x86_64` (≥10.15)| `macosx_11_0_arm64` (Apple Silicon) |

The wheel is Python-abi3 (`cp38-abi3`), meaning a single wheel works across
Python 3.8 – 3.12. CUDA is **opt-in** via the `cuda` cargo feature; the default
build (`./install.sh --cpu` or `./install.sh` on a host with no CUDA) produces
a portable CPU wheel with no CUDA toolkit dependency.

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
# On a CONNECTED machine — download everything into a portable archive
git clone https://github.com/swiftllm/swiftllm.git && cd swiftllm

# Basic bundle (source + all Python wheels + Rust installer)
./airgap-bundle.sh

# Include a model in the bundle
./airgap-bundle.sh --model "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"

# CPU-only wheels + custom output path
./airgap-bundle.sh --cpu -o /mnt/usb/swiftllm-bundle.tar.gz

# Cross-architecture bundle: build on x86_64, deploy to an ARM64 host
./airgap-bundle.sh --arch aarch64 -o swiftllm-bundle-arm64.tar.gz

# macOS Apple Silicon bundle (from either host)
./airgap-bundle.sh --arch arm64 --platform macosx_11_0_arm64
```

The `--arch` flag auto-selects the correct pip platform tag and rustup target
triple, so a bundle built on a developer laptop can target a remote ARM
server (AWS Graviton, Ampere, Raspberry Pi 4/5, Jetson, Apple Silicon).

Transfer the archive to the air-gapped host, then:

```bash
# On the AIR-GAPPED host
tar xzf swiftllm-airgap-bundle.tar.gz
cd swiftllm-airgap-bundle/swiftllm
./install.sh --airgap
```

To run in offline mode at runtime (skip all HuggingFace downloads):

```bash
export SWIFTLLM_OFFLINE=1
swiftllm generate -m /path/to/local/model.gguf -p "Hello"
```

### Manual Install

```bash
git clone https://github.com/swiftllm/swiftllm.git
cd swiftllm

# Build with Rust + Python
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

### Run a GGUF Model

```bash
# One-shot generation
swiftllm generate \
  -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  -p "What is the capital of France?" \
  --max-tokens 128

# Interactive chat
swiftllm chat \
  -m "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Use a local GGUF file directly
swiftllm generate -m /path/to/model.gguf -p "Hello world"
```

### Python API

```python
from swiftllm import LLM, SamplingParams

# Load a GGUF model (downloads automatically if not cached)
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf")

# Or from a local path
llm = LLM(model="/path/to/model.gguf")

# Generate text
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0].outputs[0].text)
```

### Model Resolver API

```python
from swiftllm import resolve_model

# Resolve a HuggingFace URL to a local path (downloads if needed)
path = resolve_model("https://huggingface.co/org/repo/blob/main/model.gguf")

# Resolve a repo:filename shorthand
path = resolve_model("org/repo:model.q4_k_m.gguf")

# Resolve a full repo
path = resolve_model("meta-llama/Llama-2-7b-hf")

# Local paths are validated and returned as-is
path = resolve_model("/data/models/my-model.gguf")

# Control download location
path = resolve_model("org/repo:model.gguf", download_dir="/data/models")
```

### Training & Fine-Tuning

#### Quick Fine-Tune with LoRA (CLI)

```bash
# LoRA fine-tuning (convenience command)
swiftllm finetune \
  -m meta-llama/Llama-2-7b-hf \
  --train-data ./data/train.jsonl \
  --lora-r 16 --lora-alpha 32 \
  --learning-rate 2e-4

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

#### Python Training API

```python
from swiftllm import Trainer, TrainingConfig, LoRAConfig

# LoRA fine-tuning
config = TrainingConfig(
    model="meta-llama/Llama-2-7b-hf",
    train_data="./data/train.jsonl",
    output_dir="./output",
    num_epochs=3,
    learning_rate=2e-4,
    lora=LoRAConfig(r=16, alpha=32),
)
trainer = Trainer(config)
trainer.train()

# Or use the convenience function
from swiftllm import fine_tune

trainer = fine_tune(
    model="meta-llama/Llama-2-7b-hf",
    train_data="data.jsonl",
    lora_r=16,
)
```

#### Supported Fine-Tuning Methods

| Method | Description | Memory |
|--------|-------------|--------|
| **LoRA** | Low-Rank Adaptation — trains small adapter matrices | Low |
| **QLoRA** | 4-bit quantized base model + LoRA adapters | Very Low |
| **Full** | Full parameter fine-tuning | High |

#### Optimizers

| Optimizer | Best For | Description |
|-----------|----------|-------------|
| **Muon** | Matrix-shaped params (linear layers) | Newton-Schulz orthogonalization on Nesterov momentum; faster convergence than Adam for >=2D weights. Auto-falls back to AdamW for 1D params (biases, norms). [arXiv:2409.20325](https://arxiv.org/abs/2409.20325) |
| **AdamW** | General purpose | Decoupled weight decay Adam; default for most fine-tuning |
| **SGD** | Large-batch training | SGD with optional Nesterov momentum |

#### Muon Optimizer (Rust API)

```rust
use swiftllm_training::{Muon, MuonConfig};
use swiftllm_training::Optimizer;

let mut opt = Muon::new(MuonConfig {
    lr: 0.02,              // LR for >=2D params (Muon path)
    momentum: 0.95,        // Nesterov momentum coefficient
    ns_steps: 5,           // Newton-Schulz iterations
    weight_decay: 0.0,     // Decoupled weight decay (Muon)
    adamw_lr: 3e-4,        // LR for 1D params (AdamW fallback)
    ..Default::default()
});

// Register shapes for explicit control (optional)
opt.set_shape("layer0.weight", 4096, 4096);

let mut param = vec![0.01f32; 4096 * 4096];
let grad = compute_gradient(&model, &batch);
opt.step(&mut param, &grad, "layer0.weight");
```

### OpenAI-Compatible Server

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
```

## Configuration

### Engine Configuration

```python
from swiftllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    download_dir="/data/models",          # Where to store downloaded models
    tensor_parallel_size=2,               # Use 2 GPUs
    gpu_memory_utilization=0.90,          # Use 90% of GPU memory
    max_model_len=4096,                   # Maximum sequence length
    dtype="float16",                      # Data type
    quantization="awq",                   # Quantization method
)
```

### Sampling Parameters

```python
from swiftllm import SamplingParams

params = SamplingParams(
    temperature=0.7,        # Sampling temperature
    top_p=0.9,              # Nucleus sampling
    top_k=50,               # Top-k sampling
    max_tokens=256,         # Maximum tokens to generate
    stop=["</s>"],          # Stop sequences
    presence_penalty=0.1,   # Presence penalty
    frequency_penalty=0.1,  # Frequency penalty
)
```

### Environment Variables

Every `SWIFTLLM_*` variable maps to a field in `EngineConfig` or `ServerConfig`. Set them in your shell profile, systemd unit, or Docker Compose file — they are read at startup and override coded defaults. Explicit constructor arguments always take final precedence.

#### GPU & Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU VRAM available for model weights + KV cache (0.0–1.0). Raise to ~0.95 on dedicated inference hosts; lower to 0.7 when sharing a GPU. |
| `SWIFTLLM_GPU_OVERHEAD_MB` | `0` | VRAM (in MB) to reserve for the OS, desktop, and other processes. Subtracted from usable memory before allocation. Useful when `GPU_MEMORY_UTILIZATION` alone isn't precise enough. |
| `SWIFTLLM_NUM_GPU_LAYERS` | all | Number of model layers to offload to GPU. Set to `0` for CPU-only, `999` for all layers. Partial offload saves VRAM at the cost of speed. |
| `SWIFTLLM_SWAP_SPACE` | `4.0` | CPU swap space in GiB for KV cache offloading when GPU memory is exhausted. |
| `SWIFTLLM_CPU_OFFLOAD_GB` | `0.0` | Amount of model weights (in GiB) to keep on CPU RAM instead of GPU. Trades latency for lower VRAM usage. |
| `SWIFTLLM_KV_CACHE_DTYPE` | `auto` | Data type for the KV cache. `auto` matches model dtype. Set to `fp8_e4m3` or `fp8_e5m2` to halve KV cache memory (with minor quality loss). |
| `SWIFTLLM_BLOCK_SIZE` | `16` | Number of tokens per block in PagedAttention. Allowed: `8`, `16`, `32`. Smaller blocks waste less memory on short sequences; larger blocks reduce overhead on long ones. |
| `SWIFTLLM_FLASH_ATTENTION` | `true` | Enable FlashAttention kernels. Disable for debugging or unsupported hardware (`false`). |
| `SWIFTLLM_ENFORCE_EAGER` | `false` | Disable CUDA graph capture; use eager execution. Set to `true` when CUDA graphs cause OOM or for easier profiling. |
| `CUDA_VISIBLE_DEVICES` | (all) | Standard CUDA variable. Restrict which GPUs are visible, e.g. `0,2`. |
| `PYTORCH_CUDA_ALLOC_CONF` | — | PyTorch allocator tuning, e.g. `expandable_segments:True,max_split_size_mb:512`. |

#### Tensor Parallelism & Multi-GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism (splits every layer across N GPUs). Must evenly divide attention heads. |
| `SWIFTLLM_PIPELINE_PARALLEL_SIZE` | `1` | Number of pipeline-parallel stages (splits layers sequentially across GPUs). Combine with tensor parallelism for very large models. |
| `NCCL_DEBUG` | — | NCCL logging: `INFO`, `WARN`, `TRACE`. Essential for debugging multi-GPU communication. |
| `NCCL_P2P_DISABLE` | `0` | Set to `1` to disable GPU peer-to-peer. Try `1` if you see hangs on certain PCIe topologies. |
| `NCCL_IB_DISABLE` | `0` | Set to `1` to disable InfiniBand for multi-node setups. |

#### Scheduling & Batching

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_MAX_NUM_SEQS` | `256` | Maximum number of concurrent sequences (requests) in a batch. Lower values reduce latency; higher values improve throughput. |
| `SWIFTLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Maximum total tokens processed in one forward pass (prefill + decode). Directly controls peak GPU compute per step. |
| `SWIFTLLM_MAX_PADDINGS` | `256` | Maximum padding tokens tolerated per batch. Padding wastes compute — keep low for variable-length workloads. |
| `SWIFTLLM_SCHEDULER_POLICY` | `fcfs` | Scheduling policy: `fcfs` (first-come-first-served), `sjf` (shortest-job-first), `priority`. |
| `SWIFTLLM_PREEMPTION_MODE` | `swap` | How to handle preemption when memory is full: `swap` (KV cache to CPU) or `recompute` (re-run prefill). |
| `SWIFTLLM_ENABLE_PREFIX_CACHING` | `false` | Reuse KV cache across requests sharing the same prompt prefix (system prompt, few-shot examples). Major speedup for chat-style workloads. |
| `SWIFTLLM_ENABLE_CHUNKED_PREFILL` | `false` | Interleave prefill and decode within the same batch. Reduces time-to-first-token for long prompts. |
| `SWIFTLLM_NUM_PARALLEL` | `1` | Number of parallel inference slots per loaded model. Each slot allocates its own KV cache. Increase for higher concurrent throughput. |
| `SWIFTLLM_MAX_LOADED_MODELS` | `1` | Maximum number of models held in GPU memory simultaneously. Excess models are evicted LRU. |
| `SWIFTLLM_KEEP_ALIVE` | `300` | Seconds a model stays loaded in memory after the last request. Set to `0` to unload immediately, `-1` to keep forever. |

#### Speculative Decoding

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_SPECULATIVE_MODEL` | — | Path or HuggingFace ID of the draft model for speculative decoding. Must share the same vocabulary as the main model. |
| `SWIFTLLM_NUM_SPECULATIVE_TOKENS` | `5` | Tokens to draft per step. Higher values increase potential speedup but waste compute on rejected tokens. |
| `SWIFTLLM_SPECULATIVE_MAX_MODEL_LEN` | — | Override max sequence length for the draft model if it differs from the main model. |

#### Model & Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_MODEL_DIR` | `~/.cache/swiftllm/models` | Default directory for downloaded models. All download/resolve calls use this as the cache root. |
| `SWIFTLLM_OFFLINE` | `false` | Set to `1` / `true` / `yes` to disable all network downloads (air-gapped / offline mode). Only local and cached models are used. |
| `SWIFTLLM_DTYPE` | `auto` | Data type for model weights: `auto`, `float16`, `bfloat16`, `float32`, `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`. |
| `SWIFTLLM_QUANTIZATION` | `none` | Quantization method: `none`, `awq`, `gptq`, `squeezellm`, `gguf`. |
| `SWIFTLLM_MAX_MODEL_LEN` | (model default) | Override the model's max sequence length. Lowering this reduces memory allocation. |
| `SWIFTLLM_TRUST_REMOTE_CODE` | `false` | Allow executing custom code from HuggingFace model repos (required by some architectures). |
| `SWIFTLLM_DEVICE` | `auto` | Device to run on: `auto`, `cuda`, `cpu`, `metal`, `rocm`. |
| `SWIFTLLM_SEED` | `0` | Global random seed for reproducibility. |
| `SWIFTLLM_MAX_PARALLEL_LOADING` | `1` | Number of parallel threads for loading model shards from disk. Increase on NVMe for faster startup. |
| `HF_TOKEN` | — | HuggingFace API token for accessing gated models (e.g. LLaMA, Gemma). |

#### LoRA

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_ENABLE_LORA` | `false` | Enable LoRA adapter support in the inference engine. |
| `SWIFTLLM_MAX_LORAS` | `1` | Maximum number of LoRA adapters that can be loaded simultaneously. |
| `SWIFTLLM_MAX_LORA_RANK` | `16` | Maximum LoRA rank supported. Higher ranks use more memory. |

#### Server & Networking

| Variable | Default | Description |
|----------|---------|-------------|
| `SWIFTLLM_HOST` | `0.0.0.0` | Bind address for the HTTP server. Use `127.0.0.1` to restrict to local access only. |
| `SWIFTLLM_PORT` | `8000` | Port for the HTTP server. |
| `SWIFTLLM_API_KEY` | — | API key for bearer-token authentication. When set, all requests must include `Authorization: Bearer <key>`. |
| `SWIFTLLM_CORS_ALLOW_ORIGINS` | `*` | Comma-separated list of allowed CORS origins. Set to specific origins in production. |
| `SWIFTLLM_SSL_CERTFILE` | — | Path to TLS certificate for HTTPS. |
| `SWIFTLLM_SSL_KEYFILE` | — | Path to TLS private key for HTTPS. |
| `SWIFTLLM_ROOT_PATH` | — | URL prefix / root path for reverse-proxy deployments (e.g. `/v1`). |
| `SWIFTLLM_SERVED_MODEL_NAME` | — | Override the model name returned in API responses (useful for A/B routing). |
| `SWIFTLLM_MAX_LOG_LEN` | — | Truncate request/response logs to this many characters. Prevents logging large prompts. |
| `SWIFTLLM_MAX_MODEL_LEN_LIMIT` | — | Hard server-side cap on `max_model_len` regardless of client request. |
| `SWIFTLLM_MAX_NUM_SEQS_LIMIT` | — | Hard server-side cap on concurrent sequences. |
| `SWIFTLLM_RESPONSE_ROLE` | `assistant` | Default role name in chat completion responses. |

#### Build & CUDA

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_PATH` / `CUDA_HOME` | — | Path to CUDA toolkit. Used by the Rust build to locate `libcuda`, `nvcc`, and headers. |
| `CUDACXX` | — | Path to `nvcc` binary. Set automatically by `install.sh` for llama-cpp-python GPU builds. |
| `CMAKE_ARGS` | — | Extra CMake arguments for llama-cpp-python build. Set to `-DGGML_CUDA=on` for GPU support. |

#### Logging & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Rust-side log level for all crates: `trace`, `debug`, `info`, `warn`, `error`. Use `swiftllm_server=debug` for per-crate control. |
| `SWIFTLLM_LOG_LEVEL` | — | Python-side log level override: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `SWIFTLLM_NO_USAGE_STATS` | `false` | Set to `1` to disable anonymous telemetry (if applicable). |

## CLI Commands

```bash
# Download a model
swiftllm download -m <model> [--download-dir <dir>]

# Start server
swiftllm serve -m <model> --port 8000 [--api-key <key>]

# Run inference
swiftllm generate -m <model> -p "Hello" --max-tokens 256

# Interactive chat
swiftllm chat -m <model>

# Benchmark
swiftllm benchmark -m <model> --num-prompts 100

# Model info
swiftllm info -m <model>

# Convert model format
swiftllm convert -i <path> -o <path> --format safetensors

# Train / fine-tune
swiftllm train -m <model> --train-data <data> --method lora
swiftllm finetune -m <model> --train-data <data> --lora-r 16
```

### Model Specifiers

The `-m` / `--model` flag accepts multiple formats:

| Format | Example | Description |
|--------|---------|-------------|
| Local path | `/data/models/model.gguf` | Use a model already on disk |
| HF repo ID | `meta-llama/Llama-2-7b-hf` | Download full repo |
| HF URL | `https://huggingface.co/org/repo/blob/main/file.gguf` | Download single file |
| Repo:file | `org/repo:model.q4_k_m.gguf` | Download single file (shorthand) |

## Architecture

```
+---------------------------------------------------------------------+
|                       SwiftLLM Architecture                         |
+---------------------------------------------------------------------+
|  +----------------+  +----------------+  +---------------------+    |
|  |  OpenAI API    |  |  Python SDK    |  |  CLI Interface      |    |
|  | (auth, headers)|  | (sync & async) |  | (serve/train/chat)  |    |
|  +-------+--------+  +-------+--------+  +--------+------------+    |
|          |                    |                     |                |
|  +-------+--------------------+---------------------+---------+     |
|  |          Model Resolver & Downloader                       |     |
|  |   (HuggingFace Hub / Local Path / GGUF URL / Offline)      |     |
|  +----------------------------+-------------------------------+     |
|                               |                                     |
|  +----------------------------+-------------------------------+     |
|  |              Inference Backend                              |     |
|  |    [llama-cpp-python (GGUF)]  [Rust Engine (HF/ST)]        |     |
|  +----------------------------+-------------------------------+     |
|                               |                                     |
|  +----------------------------+-------------------------------+     |
|  |          PagedAttention Memory Manager                      |     |
|  +----------------------------+-------------------------------+     |
|                               |                                     |
|  +----------------------------+-------------------------------+     |
|  |          Training & Fine-Tuning Engine                      |     |
|  |    [LoRA/QLoRA/Full]  [Muon/AdamW/SGD]  [LR Schedulers]    |     |
|  +----------------------------+-------------------------------+     |
|                               |                                     |
|  +----------------------------+-------------------------------+     |
|  |                    CUDA Kernels                             |     |
|  +------------------------------------------------------------+     |
|                                                                     |
|  +------------------------------------------------------------+     |
|  |  Install & Deploy: install.sh | airgap-bundle.sh | offline  |     |
|  +------------------------------------------------------------+     |
+---------------------------------------------------------------------+
```

## Multi-GPU Support

SwiftLLM supports tensor parallelism for large models:

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
)
```

## Examples

See the [examples/](examples/) directory for more:

- [basic_inference.py](examples/basic_inference.py) - Simple inference
- [streaming.py](examples/streaming.py) - Streaming generation
- [batch_processing.py](examples/batch_processing.py) - High-throughput batch processing
- [openai_server.py](examples/openai_server.py) - OpenAI API server
- [multi_gpu.py](examples/multi_gpu.py) - Multi-GPU inference
- [fine_tuning.py](examples/fine_tuning.py) - LoRA and QLoRA fine-tuning
- [training.py](examples/training.py) - Full training with callbacks and config management

## Security

SwiftLLM includes built-in security features across the server, installer, and runtime:

**Server**
- **API Key Authentication**: Protect endpoints with `--api-key` flag
- **Input Validation**: Request size limits, parameter range checks, content length limits
- **Security Headers**: HSTS, X-Content-Type-Options, X-Frame-Options, X-Request-ID
- **Sanitized Errors**: Internal errors are logged server-side but never exposed to clients
- **Request Size Limits**: 10 MB maximum request body to prevent abuse
- **Safe JSON Serialization**: SSE streaming uses `serde_json` to prevent injection

**Installer & Runtime**
- **Supply Chain Verification**: SHA256 checksum verification for `rustup-init` downloads
- **Input Sanitization**: Platform tags and paths are validated before use in shell commands
- **Path Traversal Protection**: Checkpoint and cache directory operations validate resolved paths stay within expected boundaries
- **UTF-8 Safety**: Data loading uses char-boundary-safe truncation to prevent panics on multi-byte input
- **Numeric Safety**: Optimizer arithmetic guards against division by zero and unbounded iteration

## Project Structure

```
swiftllm/
  src/lib.rs                    # PyO3 Python bindings
  install.sh                    # Installer (GPU detection, venv, build)
  airgap-bundle.sh              # Air-gap bundle creator (offline deploy)
  python/swiftllm/              # Python package
    engine.py                   #   LLM inference API
    training.py                 #   Training & fine-tuning API
    cli.py                      #   CLI (serve, train, finetune, chat, ...)
    model_resolver.py           #   HuggingFace / local / offline model resolution
    sampling.py                 #   Sampling strategies (top-k, top-p, beam, ...)
    config.py                   #   Configuration helpers
  crates/
    swiftllm-core/              # Core engine (scheduler, memory, sampling)
    swiftllm-models/            # Model loading and architectures
    swiftllm-cuda/              # CUDA kernel bindings
    swiftllm-server/            # HTTP server (OpenAI API, auth, security)
    swiftllm-training/          # Training engine (Rust)
      src/config.rs             #   Training configuration
      src/data.rs               #   Data loading (JSONL, CSV, text)
      src/optimizer.rs          #   AdamW, SGD, LR schedulers
      src/muon.rs               #   Muon optimizer (Newton-Schulz orthogonalization)
      src/fine_tuning.rs        #   LoRA, QLoRA, full fine-tuning
      src/metrics.rs            #   Training metrics & logging
      src/trainer.rs            #   Training loop & checkpointing
  examples/                     # Example scripts
```

## Changelog

### v2.0.1-beta

**Training UX Fixes**
- **Fix**: `Trainer.train()` now prints a visible `[SIMULATED]` banner at the start of every run, making it obvious that the current training loop is a stub (synthetic 100-step loss curve, no weights loaded, no gradients) rather than a real training pass — previously the only indication was an inline code comment, so users running `swiftllm train` / `swiftllm finetune` saw convincing metrics with no way to know real training wasn't wired up
- **Fix**: `swiftllm train` and `swiftllm finetune` now validate `--train-data` (and `--eval-data`) up front — missing paths raise `FileNotFoundError` with a helpful message, non-regular files (e.g. directories) are rejected, and empty files raise `ValueError`. Previously a bogus path silently "succeeded" all the way through the simulated loop and wrote a final checkpoint
- Validation also covers the path fields loaded from `--config` JSON so typos in saved configs fail fast

**Regression Coverage**
- Full regression matrix run on Ubuntu 24.04 + CUDA 13.0 (RTX PRO 4000 Blackwell): install → download → generate (18.61 tok/s) → finetune (LoRA) → train (LoRA / QLoRA / full) → cleanup — all passing
- Verified checkpoint artifacts written correctly: `training_config.json` + `trainer_state.json` per checkpoint dir, `save_total_limit` rotation behaves as documented

### v2.0.0.2-alpha

**Regression Test Fixes**
- **Fix**: Added `fastapi>=0.100` and `uvicorn>=0.23` as `[serve]` optional dependency in `pyproject.toml` — `swiftllm serve` previously failed on fresh installs with `Error: FastAPI and uvicorn are required for serving`
- **Fix**: `install.sh` now installs the wheel with `[serve]` extras so the API server works out of the box after installation
- **Fix**: `airgap-bundle.sh` now includes `fastapi` and `uvicorn` in the offline wheel downloads so `swiftllm serve` works in air-gapped environments
- Regression tested end-to-end on Ubuntu 24.04 (Python 3.12, RTX PRO 4000 Blackwell, CUDA 13.0): install → download model → generate → chat → serve + API call — all passing

**CPU and ARM Wheel Support**
- **New**: CPU-only build is now the default — `maturin build --release` (or `./install.sh --cpu`) produces a portable wheel with zero CUDA dependencies, buildable on any host including Apple Silicon, AWS Graviton, Raspberry Pi 4/5, Jetson, and Ampere ARM servers
- **New**: `swiftllm` top-level crate now exposes explicit `cpu` and `cuda` Cargo features; `cpu` is the default, CUDA is opt-in via `./install.sh --gpu` or `cargo build --features cuda`
- **New**: `swiftllm-core` default features changed from `["cuda"]` to `[]` — the CUDA code paths remain `#[cfg(feature = "cuda")]`-gated and are only compiled when explicitly enabled
- **New**: `airgap-bundle.sh --arch ARCH` flag (`x86_64`, `aarch64`, `arm64`) auto-maps to the correct pip platform tag (`manylinux2014_aarch64`, `macosx_11_0_arm64`, etc.) and rustup target triple — lets you build a cross-architecture bundle on a dev laptop and ship it to a remote ARM host
- **New**: `airgap-bundle.sh` now normalizes `arm64` → `aarch64` for rustup targets (Apple Silicon reports `arm64` from `uname -m`, but Rust's target triple is `aarch64-apple-darwin`)

**Installer Portability Fixes**
- **Fix**: Replaced non-portable `grep -oP 'release \K...'` (GNU PCRE) with portable `sed -n 's/.../\1/p'` for CUDA version detection — now works on macOS and BSD-based systems
- **Fix**: `install.sh` pip upgrade no longer swallows stderr silently; now fails loudly with a helpful message when `pip install --upgrade` fails
- **Fix**: PEP 668 handling — when `--no-venv` is used against an externally-managed system Python (Ubuntu 23.04+, Debian 12+), the installer detects the `EXTERNALLY-MANAGED` marker and automatically adds `--break-system-packages` with a warning recommending virtualenv
- **Fix**: `airgap-bundle.sh` SHA256 verification now prefers `sha256sum` (Linux coreutils) with `shasum` fallback (macOS) — previously only worked on hosts with macOS `shasum`
- **Fix**: Added `set -o pipefail` to both scripts so piped command failures propagate
- **Fix**: Added `trap 'rm -rf "$BUNDLE_DIR"' EXIT` to `airgap-bundle.sh` for guaranteed temp-dir cleanup
- **Fix**: Argument guards on `--venv`, `--model-dir`, `--model`, `-o`, `--platform`, `--arch` — missing values no longer silently consume the next flag
- **Fix**: Python version check now correctly handles `major > 3` (was `major >= 3 && minor >= 8`, which would have failed for future Python 4.x)
- **Fix**: `VENV_DIR` reference in verification step is now guarded when `--no-venv` is set
- **Fix**: rsync in `airgap-bundle.sh` now excludes `.env*`, `*.pem`, `*.key`, `*.log`, and `models/` to prevent accidentally bundling secrets or pre-existing model files

**Rust Code Quality**
- **Critical**: Fixed 14 `partial_cmp().unwrap()` calls across `sampling/mod.rs`, `sampling/strategies.rs`, and `execution/speculative.rs` — replaced with `unwrap_or(Ordering::Equal)` to prevent panics on NaN logits (could crash inference on numerically unstable models)
- **Critical**: Added `checked_add` + bounds validation in `gguf.rs` `load_weight()` and `load_weights()` — malformed GGUF files can no longer cause slice panics on mmap'd data
- **High**: Replaced `try_into().unwrap()` in `safetensors.rs` header parser with proper `Result` propagation
- Added `// SAFETY:` comments to all `unsafe` memory-map blocks in the loaders

**Documentation**
- New "Supported Platforms" table showing wheel tags for Linux/macOS × x86_64/aarch64
- Cross-architecture airgap bundle examples (AWS Graviton, Apple Silicon)
- Documented the new `cpu` / `cuda` Cargo features

### v2.0.0.1-alpha

**Air-Gapped / Offline Installation**
- New `airgap-bundle.sh` script to create portable install archives on a connected machine (bundles source, pip wheels, `rustup-init`, and optional models)
- `install.sh --airgap` flag for fully offline installation from a bundle
- Runtime offline mode via `SWIFTLLM_OFFLINE=1` — disables all HuggingFace downloads and uses local cache only
- `scan_local_cache()` walks the model cache directory for exact filename matches
- `download_file()` and `download_repo()` transparently fall back to cache in offline mode

**Security Hardening**
- **Critical**: Fixed JSON injection in SSE streaming — replaced raw `format!()` JSON construction with `serde_json::json!()` macro in `streaming.rs`
- **Critical**: Fixed shell injection in `airgap-bundle.sh` — model names are now passed via `sys.argv` instead of string interpolation into Python code
- **Critical**: Added SHA256 checksum verification for downloaded `rustup-init` binary in `airgap-bundle.sh`
- **High**: Fixed word-splitting vulnerability in `install.sh` — `AIRGAP_PIP_FLAGS` converted from string to bash array
- **High**: Added input validation for `--platform` tag and `--model-dir` path in installer scripts
- **High**: Capped Muon optimizer Newton-Schulz iterations at 20 to prevent runaway loops; added epsilon floor to AdamW bias correction divisors to prevent division by zero
- **Medium**: Tightened offline `download_repo()` directory matching from loose substring to exact name match, with symlink traversal protection
- **Medium**: Added path validation in `Trainer.resume_from_checkpoint()` to prevent directory traversal via `..`
- **Medium**: Fixed UTF-8 boundary-safe string truncation in `data.rs` — replaced byte-level slicing with `char_indices()` to prevent panics on multi-byte characters
- **Medium**: Added LoRA buffer size validation in `fine_tuning.rs` `merge_weights()` to prevent out-of-bounds indexing on malformed adapters

**Bug Fixes**
- Fixed use-after-move in `engine.rs` — `eos_token_id` now extracted before `config` is moved into struct
- Fixed usize negation in `trainer.rs` — cast to `f64` before negation for exponential decay calculation
- Fixed missing `mut` on `eval_data` parameter in `trainer.rs` evaluation loop
- Added missing `tempfile` dev-dependency to `swiftllm-training` Cargo.toml

### v2.0.0-alpha

**Training & Fine-Tuning**
- Added `swiftllm-training` Rust crate with full training infrastructure
- LoRA, QLoRA, and full fine-tuning support with configurable adapters
- Muon optimizer: Newton-Schulz orthogonalization on Nesterov momentum for fast convergence on matrix-shaped params, with automatic AdamW fallback for 1D params
- AdamW and SGD optimizers with linear/cosine/constant LR schedulers
- Dataset loading (JSONL, CSV, text) with instruction templates
- Training metrics tracking with rolling windows and perplexity
- Checkpoint save/load with configurable `save_total_limit`
- Python `Trainer` class with callbacks, early stopping, and checkpoint management
- CLI commands: `swiftllm train` and `swiftllm finetune`
- Training examples: `examples/fine_tuning.py`, `examples/training.py`

**Inference Engine**
- Implemented core engine `step()` with batch processing and token sampling
- Configurable EOS token ID per model (no longer hardcoded to 2)
- Eliminated redundant read-then-write lock upgrade in sampling hot path
- Fast path for greedy sampling — zero-allocation when no penalties are active

**Sampling Optimizations**
- Replaced O(n log n) full sort with O(n) quickselect (`select_nth_unstable_by`) for top-k, beam search, and `get_top_logprobs`
- Partial sort for `sample_top_n` — only the top N elements are fully sorted
- Python: numerically stable `_log_softmax` replaces `log(probs + eps)` across all samplers
- Python: `BeamSearchSampler` now maintains beam state across calls with proper expansion and pruning

**Scheduler**
- O(n) victim selection for preemption (was O(n log n) sort per step)

**Training Crate Improvements**
- Gradient clipping (`clip_grad_norm`) with global norm
- LoRA `transform_grad` now applies `alpha/r` scaling to adapter gradients
- AdamW bias correction uses `powf` instead of `powi` to avoid i32 overflow at large step counts
- Exported `clip_grad_norm` from crate root

**Server & API**
- API key authentication middleware (`--api-key` flag)
- Security headers: HSTS, X-Content-Type-Options, X-Frame-Options, X-Request-ID
- Request body size limit (10 MB)
- Input validation for chat completions (temperature, top_p, max_tokens, content length)
- Sanitized error responses — internal details logged server-side only
- `/metrics` endpoint with JSON and Prometheus text format (`Accept: text/plain`)
- SSE streaming helpers for OpenAI-compatible streaming responses

**Python API**
- Complete PyO3 bindings: `PyEngine`, `PyEngineConfig`, `PySamplingParams`, `PyGenerationOutput`, `PyRequestOutput`
- `EarlyStoppingConfig` for training with configurable patience and min_delta
- `Trainer.resume_from_checkpoint()` class method
- Checkpoint save/load with `save_total_limit` enforcement
- Throughput (tok/s) in training log output

**Added Airgap Support**
- All in One package setup bundle on a connected device.

### v1.0.0

- Initial release
- PagedAttention memory management with block allocator and copy-on-write
- Continuous batching scheduler with preemption (swap/recompute)
- Token sampling: greedy, temperature, top-k, top-p, min-p, repetition penalty
- OpenAI-compatible HTTP API (chat completions, completions, models)
- Python SDK: `LLM`, `AsyncLLM`, `SamplingParams`
- CLI: serve, generate, benchmark, convert, info, chat, download
- HuggingFace model downloading and resolution
- GGUF model support via llama-cpp-python

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

SwiftLLM builds on ideas from:
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention concept
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF format and quantization
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Efficient attention kernels
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model architectures

<!--
    ------------------------------------------------------------------------------
    END OF FILE: README.md
    REPO PATH:   /swiftllm/README.md
    (c) 2026 SWIFTLLM | Apache 2.0 License
    ------------------------------------------------------------------------------
-->
