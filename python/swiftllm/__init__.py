"""SwiftLLM - High-performance LLM Inference Engine

SwiftLLM is a fast and memory-efficient inference engine for large language models,
featuring PagedAttention, continuous batching, and multi-GPU support.

Example usage:
    >>> from swiftllm import LLM, SamplingParams
    >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
    >>> outputs = llm.generate(["Hello, how are you?"], SamplingParams(temperature=0.7))
    >>> print(outputs[0].outputs[0].text)
"""

from .engine import LLM, AsyncLLM, LLMEngine, RequestOutput, CompletionOutput
from .config import SamplingParams, EngineConfig, ServerConfig, LoRARequest
from .sampling import SamplingStrategy, create_sampler
from .model_resolver import resolve_model

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "LLM",
    "AsyncLLM",
    "LLMEngine",
    # Output types
    "RequestOutput",
    "CompletionOutput",
    # Configuration
    "SamplingParams",
    "EngineConfig",
    "ServerConfig",
    "LoRARequest",
    # Model resolution
    "resolve_model",
    # Sampling
    "SamplingStrategy",
    "create_sampler",
    # Version
    "__version__",
]


def version() -> str:
    """Get the SwiftLLM version."""
    return __version__
