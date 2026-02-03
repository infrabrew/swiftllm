# SwiftLLM

<p align="center">
  <img src="https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
</p>

**SwiftLLM** is a high-performance LLM inference and serving engine built with Rust for maximum speed and efficiency. It features state-of-the-art memory management, continuous batching, and multi-GPU support.

## ✨ Key Features

- 🚀 **High Throughput**: Continuous batching and efficient scheduling for maximum tokens/second
- 💾 **Memory Efficient**: PagedAttention for optimal KV cache management
- ⚡ **Low Latency**: Optimized CUDA kernels and speculative decoding
- 🔄 **Tensor Parallelism**: Scale to multiple GPUs seamlessly
- 🌐 **OpenAI Compatible**: Drop-in replacement for OpenAI API
- 🐍 **Python Friendly**: Easy-to-use Python API with async support
- 📦 **Multiple Formats**: Support for HuggingFace, GGUF, and SafeTensors

## 🎯 Supported Models

| Architecture | Models |
|-------------|--------|
| **LLaMA** | LLaMA, LLaMA 2, LLaMA 3, Code Llama |
| **Mistral** | Mistral 7B, Mixtral 8x7B |
| **Qwen** | Qwen, Qwen 2 |
| **Phi** | Phi-2, Phi-3 |

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install swiftllm
```

### From Source

```bash
# Clone the repository
git clone https://github.com/swiftllm/swiftllm.git
cd swiftllm

# Build with Rust
cargo build --release

# Install Python package
pip install -e .
```

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Rust 1.70+ (for building from source)

## 🚀 Quick Start

### Python API

```python
from swiftllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate text
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

print(outputs[0].outputs[0].text)
```

### OpenAI-Compatible Server

```bash
# Start the server
swiftllm serve --model meta-llama/Llama-2-7b-hf --port 8000

# Use with OpenAI client
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```python
import asyncio
from swiftllm import AsyncLLM, SamplingParams

async def main():
    llm = AsyncLLM(model="meta-llama/Llama-2-7b-hf")

    async for output in llm.generate("Tell me a story:", SamplingParams()):
        print(output.outputs[0].text, end="", flush=True)

asyncio.run(main())
```

## ⚙️ Configuration

### Engine Configuration

```python
from swiftllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,        # Use 2 GPUs
    gpu_memory_utilization=0.90,   # Use 90% of GPU memory
    max_model_len=4096,            # Maximum sequence length
    dtype="float16",               # Data type
    quantization="awq",            # Quantization method
)
```

### Sampling Parameters

```python
from swiftllm import SamplingParams

params = SamplingParams(
    temperature=0.7,      # Sampling temperature
    top_p=0.9,            # Nucleus sampling
    top_k=50,             # Top-k sampling
    max_tokens=256,       # Maximum tokens to generate
    stop=["</s>"],        # Stop sequences
    presence_penalty=0.1, # Presence penalty
    frequency_penalty=0.1,# Frequency penalty
)
```

## 🔧 CLI Commands

```bash
# Start server
swiftllm serve --model <model> --port 8000

# Run inference
swiftllm generate --model <model> --prompt "Hello"

# Interactive chat
swiftllm chat --model <model>

# Benchmark
swiftllm benchmark --model <model> --num-prompts 100

# Model info
swiftllm info --model <model>

# Convert model format
swiftllm convert --input <path> --output <path> --format safetensors
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SwiftLLM Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  OpenAI API     │  │  Python SDK     │  │  CLI Interface  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│  ┌────────┴────────────────────┴────────────────────┴────────┐  │
│  │                 Scheduler (Continuous Batching)            │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               │                                  │
│  ┌────────────────────────────┴───────────────────────────────┐  │
│  │          PagedAttention Memory Manager                      │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               │                                  │
│  ┌────────────────────────────┴───────────────────────────────┐  │
│  │                    CUDA Kernels                              │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Performance

SwiftLLM achieves high throughput through:

- **PagedAttention**: Efficient KV cache management with minimal memory waste
- **Continuous Batching**: Dynamic request scheduling at iteration level
- **Speculative Decoding**: Accelerate generation with draft models
- **Optimized Kernels**: Custom CUDA kernels for attention and quantization

### Benchmarks

| Model | Hardware | Throughput | Latency (TTFT) |
|-------|----------|------------|----------------|
| Llama-2-7B | 1x A100 | 2000+ tok/s | <50ms |
| Llama-2-13B | 1x A100 | 1200+ tok/s | <80ms |
| Llama-2-70B | 4x A100 | 800+ tok/s | <150ms |

## 🔌 Multi-GPU Support

SwiftLLM supports tensor parallelism for large models:

```python
# Use 4 GPUs for a 70B model
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
)
```

## 📚 Examples

See the [examples/](examples/) directory for more:

- [basic_inference.py](examples/basic_inference.py) - Simple inference
- [streaming.py](examples/streaming.py) - Streaming generation
- [batch_processing.py](examples/batch_processing.py) - High-throughput batch processing
- [openai_server.py](examples/openai_server.py) - OpenAI API server
- [multi_gpu.py](examples/multi_gpu.py) - Multi-GPU inference

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

SwiftLLM builds on ideas from:
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention concept
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Efficient attention kernels
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model architectures
