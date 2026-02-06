# SwiftLLM

<p align="center">
  <img src="https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
</p>

**SwiftLLM** is a high-performance LLM inference and serving engine built with Rust for maximum speed and efficiency. It features state-of-the-art memory management, continuous batching, and multi-GPU support.

## Key Features

- **High Throughput**: Continuous batching and efficient scheduling for maximum tokens/second
- **Memory Efficient**: PagedAttention for optimal KV cache management
- **Low Latency**: Optimized CUDA kernels and speculative decoding
- **Tensor Parallelism**: Scale to multiple GPUs seamlessly
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Python Friendly**: Easy-to-use Python API with async support
- **Multiple Formats**: Support for HuggingFace, GGUF, and SafeTensors
- **Model Downloading**: Download models from HuggingFace Hub by ID or URL
- **GGUF Inference**: Run quantized GGUF models on GPU via llama-cpp-python

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
- Builds SwiftLLM from source
- Installs llama-cpp-python with GPU support (if available)

#### Installer Options

```bash
./install.sh --cpu          # CPU-only (skip GPU detection)
./install.sh --gpu          # Force GPU/CUDA build
./install.sh --venv ~/sllm  # Custom venv location
./install.sh --no-venv      # Install into current Python environment
./install.sh --model-dir /data/models  # Set model storage directory
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

### OpenAI-Compatible Server

```bash
swiftllm serve -m /path/to/model.gguf --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
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

| Variable | Description |
|----------|-------------|
| `SWIFTLLM_MODEL_DIR` | Default directory for downloaded models (overrides `~/.cache/swiftllm/models`) |
| `HF_TOKEN` | HuggingFace API token for accessing gated models |

## CLI Commands

```bash
# Download a model
swiftllm download -m <model> [--download-dir <dir>]

# Start server
swiftllm serve -m <model> --port 8000

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
+-----------------------------------------------------------------+
|                     SwiftLLM Architecture                       |
+-----------------------------------------------------------------+
|  +----------------+  +----------------+  +------------------+   |
|  |  OpenAI API    |  |  Python SDK    |  |  CLI Interface   |   |
|  +-------+--------+  +-------+--------+  +--------+---------+   |
|          |                    |                     |            |
|  +-------+--------------------+---------------------+--------+  |
|  |                Model Resolver & Downloader                 |  |
|  |         (HuggingFace Hub / Local Path / GGUF URL)          |  |
|  +----------------------------+-------------------------------+  |
|                               |                                  |
|  +----------------------------+-------------------------------+  |
|  |              Inference Backend                              |  |
|  |    [llama-cpp-python (GGUF)]  [Rust Engine (HF/ST)]        |  |
|  +----------------------------+-------------------------------+  |
|                               |                                  |
|  +----------------------------+-------------------------------+  |
|  |          PagedAttention Memory Manager                      |  |
|  +----------------------------+-------------------------------+  |
|                               |                                  |
|  +----------------------------+-------------------------------+  |
|  |                    CUDA Kernels                              |  |
|  +--------------------------------------------------------------+
+-----------------------------------------------------------------+
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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

SwiftLLM builds on ideas from:
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention concept
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF format and quantization
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Efficient attention kernels
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model architectures
