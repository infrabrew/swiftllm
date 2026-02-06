#!/usr/bin/env python3
"""SwiftLLM Command Line Interface

This module provides the CLI for SwiftLLM, supporting:
- serve: Start the inference server
- generate: Run offline batch generation
- benchmark: Run performance benchmarks
- convert: Convert model formats
- info: Display model information
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Lazy imports to reduce startup time
def get_engine():
    from .engine import LLM, AsyncLLM
    return LLM, AsyncLLM

def get_config():
    from .config import SamplingParams, EngineConfig, ServerConfig
    return SamplingParams, EngineConfig, ServerConfig


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="swiftllm",
        description="SwiftLLM - High-performance LLM inference engine",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"SwiftLLM {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    _add_serve_args(serve_parser)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Run offline generation")
    _add_generate_args(generate_parser)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    _add_benchmark_args(bench_parser)

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    _add_convert_args(convert_parser)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    _add_info_args(info_parser)

    # Chat command (interactive)
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    _add_chat_args(chat_parser)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model from HuggingFace")
    _add_download_args(download_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    commands = {
        "serve": cmd_serve,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "convert": cmd_convert,
        "info": cmd_info,
        "chat": cmd_chat,
        "download": cmd_download,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _get_version() -> str:
    """Get the SwiftLLM version."""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "0.1.0"


def _add_serve_args(parser: argparse.ArgumentParser):
    """Add arguments for the serve command."""
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (default: 0.90)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Data type for model weights (default: auto)",
    )
    parser.add_argument(
        "--quantization", "-q",
        choices=["awq", "gptq", "squeezellm", None],
        default=None,
        help="Quantization method",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable automatic prefix caching",
    )
    parser.add_argument(
        "--speculative-model",
        default=None,
        help="Draft model for speculative decoding",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Number of speculative tokens (default: 5)",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for downloading models (default: ~/.cache/swiftllm/models, "
             "or set SWIFTLLM_MODEL_DIR env var)",
    )


def _add_generate_args(parser: argparse.ArgumentParser):
    """Add arguments for the generate command."""
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "-p", "--prompt",
        default=None,
        help="Input prompt",
    )
    parser.add_argument(
        "-f", "--file",
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1, disabled)",
    )
    parser.add_argument(
        "-n", "--num-sequences",
        type=int,
        default=1,
        help="Number of sequences to generate (default: 1)",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for downloading models (default: ~/.cache/swiftllm/models, "
             "or set SWIFTLLM_MODEL_DIR env var)",
    )


def _add_benchmark_args(parser: argparse.ArgumentParser):
    """Add arguments for the benchmark command."""
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Input prompt length (default: 128)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length (default: 128)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations (default: 3)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for downloading models (default: ~/.cache/swiftllm/models, "
             "or set SWIFTLLM_MODEL_DIR env var)",
    )


def _add_convert_args(parser: argparse.ArgumentParser):
    """Add arguments for the convert command."""
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input model path",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path",
    )
    parser.add_argument(
        "--format",
        choices=["safetensors", "gguf", "pytorch"],
        default="safetensors",
        help="Output format (default: safetensors)",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "int4", "fp8", "awq", "gptq"],
        default=None,
        help="Quantization to apply",
    )
    parser.add_argument(
        "--calibration-data",
        default=None,
        help="Calibration dataset for quantization",
    )


def _add_info_args(parser: argparse.ArgumentParser):
    """Add arguments for the info command."""
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for downloading models (default: ~/.cache/swiftllm/models, "
             "or set SWIFTLLM_MODEL_DIR env var)",
    )


def _add_chat_args(parser: argparse.ArgumentParser):
    """Add arguments for the chat command."""
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for downloading models (default: ~/.cache/swiftllm/models, "
             "or set SWIFTLLM_MODEL_DIR env var)",
    )


def _add_download_args(parser: argparse.ArgumentParser):
    """Add arguments for the download command."""
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="HuggingFace model ID (e.g., 'org/model'), "
             "HuggingFace URL (e.g., 'https://huggingface.co/org/repo/blob/main/file.gguf'), "
             "or repo:filename shorthand (e.g., 'org/repo:file.gguf')",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory to save the model (default: ~/.cache/swiftllm/models, "
             "or set SWIFTLLM_MODEL_DIR env var)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision to download (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token for gated models (or set HF_TOKEN env var)",
    )


def cmd_download(args: argparse.Namespace):
    """Download a model from HuggingFace."""
    from .model_resolver import resolve_model, is_local_path

    if is_local_path(args.model):
        print(f"'{args.model}' is a local path, nothing to download.")
        return

    print(f"Downloading: {args.model}")
    if args.download_dir:
        print(f"Destination: {args.download_dir}")

    local_path = resolve_model(
        model=args.model,
        download_dir=args.download_dir,
        token=args.token,
        revision=args.revision,
    )

    print(f"\nDone! Model saved to: {local_path}")

    # Show file size
    model_path = Path(local_path)
    if model_path.is_file():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        print(f"Size: {size_gb:.2f} GB")
    elif model_path.is_dir():
        total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        size_gb = total / (1024 ** 3)
        print(f"Total size: {size_gb:.2f} GB")


def cmd_serve(args: argparse.Namespace):
    """Start the inference server."""
    print(f"SwiftLLM Server v{_get_version()}")
    print(f"Loading model: {args.model}")
    print(f"Server will start on http://{args.host}:{args.port}")

    # Build command for Rust server
    # In production, this would call the Rust binary directly
    # For now, we'll start a simple Python server

    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        print("Error: FastAPI and uvicorn are required for serving.")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)

    LLM, _ = get_engine()
    SamplingParams, _, _ = get_config()

    # Initialize engine
    print("Initializing engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
    )

    app = FastAPI(title="SwiftLLM", version=_get_version())

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: float = 0.7
        max_tokens: int = 256
        stream: bool = False

    class ChatChoice(BaseModel):
        index: int
        message: ChatMessage
        finish_reason: str

    class ChatResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatChoice]

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": args.model, "object": "model"}]
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        # Build prompt from messages
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant:"

        params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        outputs = llm.generate([prompt], params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text

        return ChatResponse(
            id=f"chatcmpl-{outputs[0].request_id}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
        )

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_generate(args: argparse.Namespace):
    """Run offline generation."""
    LLM, _ = get_engine()
    SamplingParams, _, _ = get_config()

    # Get prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.file:
        with open(args.file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin
        print("Enter prompts (one per line, Ctrl+D to finish):")
        prompts = [line.strip() for line in sys.stdin if line.strip()]

    if not prompts:
        print("No prompts provided", file=sys.stderr)
        sys.exit(1)

    print(f"Generating {len(prompts)} prompt(s)...")

    # Initialize engine
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
    )

    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.num_sequences,
    )

    # Generate
    start_time = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start_time

    # Output results
    results = []
    for output in outputs:
        for completion in output.outputs:
            result = {
                "prompt": output.prompt,
                "generated_text": completion.text,
                "finish_reason": completion.finish_reason.value if completion.finish_reason else None,
            }
            results.append(result)

    if args.json:
        output_text = json.dumps(results, indent=2)
    else:
        output_text = ""
        for i, result in enumerate(results):
            output_text += f"\n{'='*60}\n"
            output_text += f"Prompt {i+1}: {result['prompt']}\n"
            output_text += f"{'='*60}\n"
            output_text += f"{result['generated_text']}\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print(output_text)

    # Print stats
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n--- Statistics ---")
    print(f"Time: {elapsed:.2f}s")
    print(f"Tokens: {total_tokens}")
    print(f"Throughput: {total_tokens/elapsed:.2f} tokens/s")


def cmd_benchmark(args: argparse.Namespace):
    """Run benchmarks."""
    print(f"Benchmarking model: {args.model}")
    print(f"Input length: {args.input_len}, Output length: {args.output_len}")
    print(f"Num prompts: {args.num_prompts}, Concurrency: {args.concurrency}")

    LLM, _ = get_engine()
    SamplingParams, _, _ = get_config()

    # Initialize engine
    print("\nInitializing engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir=args.download_dir,
    )

    # Generate random prompts of specified length
    tokenizer = llm.get_tokenizer()
    dummy_token = tokenizer.encode("hello")[0]
    prompts = [
        tokenizer.decode([dummy_token] * args.input_len)
        for _ in range(args.num_prompts)
    ]

    params = SamplingParams(
        temperature=0.0,  # Greedy for reproducibility
        max_tokens=args.output_len,
    )

    # Warmup
    print(f"\nWarmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        llm.generate(prompts[:1], params, use_tqdm=False)

    # Benchmark
    print("\nRunning benchmark...")
    start_time = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - start_time

    # Calculate metrics
    total_input_tokens = args.input_len * args.num_prompts
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = total_input_tokens + total_output_tokens

    results = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "total_time_s": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_s": total_tokens / elapsed,
        "latency_per_token_ms": (elapsed / total_output_tokens) * 1000,
        "requests_per_s": args.num_prompts / elapsed,
    }

    print("\n--- Benchmark Results ---")
    print(f"Total time: {results['total_time_s']:.2f}s")
    print(f"Throughput: {results['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"Latency per token: {results['latency_per_token_ms']:.2f} ms")
    print(f"Requests per second: {results['requests_per_s']:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_convert(args: argparse.Namespace):
    """Convert model format."""
    print(f"Converting model: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")

    if args.quantize:
        print(f"Quantization: {args.quantize}")

    # TODO: Implement actual conversion
    print("\nModel conversion not yet implemented in Python CLI.")
    print("Use the Rust CLI: swiftllm convert -i <input> -o <output>")


def cmd_info(args: argparse.Namespace):
    """Display model information."""
    from .model_resolver import resolve_model

    resolved = resolve_model(
        model=args.model,
        download_dir=args.download_dir,
    )
    model_path = Path(resolved)

    info = {
        "path": str(model_path),
    }

    # Try to load config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        info.update({
            "architecture": config.get("architectures", ["Unknown"])[0] if "architectures" in config else config.get("model_type", "Unknown"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads", config.get("num_attention_heads")),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rope_theta": config.get("rope_theta", 10000),
        })

        # Estimate parameters
        if all(k in info for k in ["hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"]):
            params = _estimate_params(info)
            info["estimated_params_b"] = params / 1e9

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"\nModel Information: {args.model}")
        print("=" * 60)
        for key, value in info.items():
            if value is not None:
                if key == "estimated_params_b":
                    print(f"  Estimated Parameters: {value:.2f}B")
                else:
                    print(f"  {key}: {value}")


def cmd_chat(args: argparse.Namespace):
    """Interactive chat mode."""
    LLM, _ = get_engine()
    SamplingParams, _, _ = get_config()

    print(f"SwiftLLM Chat - Loading {args.model}...")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
    )

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Chat history
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("\nChat started. Type 'quit' or 'exit' to end.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "user", "content": user_input})

        # Build prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"

        outputs = llm.generate([prompt], params, use_tqdm=False)
        response = outputs[0].outputs[0].text.strip()

        print(f"\nAssistant: {response}")
        messages.append({"role": "assistant", "content": response})

    print("\nGoodbye!")


def _estimate_params(info: dict) -> int:
    """Estimate model parameters from config."""
    hidden = info.get("hidden_size", 0)
    intermediate = info.get("intermediate_size", 0)
    layers = info.get("num_hidden_layers", 0)
    vocab = info.get("vocab_size", 0)
    heads = info.get("num_attention_heads", 0)
    kv_heads = info.get("num_key_value_heads", heads)
    head_dim = hidden // heads if heads else 0

    # Embedding
    embed = vocab * hidden

    # Attention per layer
    q_proj = hidden * heads * head_dim
    k_proj = hidden * kv_heads * head_dim
    v_proj = hidden * kv_heads * head_dim
    o_proj = heads * head_dim * hidden
    attn = q_proj + k_proj + v_proj + o_proj

    # MLP per layer
    mlp = 3 * hidden * intermediate  # gate, up, down

    # Norms
    norms = 2 * hidden

    # Per layer total
    per_layer = attn + mlp + norms

    # Final
    final_norm = hidden
    lm_head = hidden * vocab

    return embed + layers * per_layer + final_norm + lm_head


if __name__ == "__main__":
    main()
