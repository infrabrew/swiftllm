"""SwiftLLM Configuration Classes

This module provides configuration classes for the SwiftLLM inference engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path


class DataType(Enum):
    """Data type for model weights and computations."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    AUTO = "auto"


class QuantizationMethod(Enum):
    """Quantization method for model compression."""
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
    GGUF = "gguf"


class SchedulerPolicy(Enum):
    """Scheduling policy for request batching."""
    FCFS = "fcfs"           # First Come First Served
    SJF = "sjf"             # Shortest Job First
    PRIORITY = "priority"   # Priority-based


class PreemptionMode(Enum):
    """Mode for handling preemption."""
    SWAP = "swap"           # Swap to CPU memory
    RECOMPUTE = "recompute" # Recompute from beginning


@dataclass
class SamplingParams:
    """Parameters for text generation sampling.

    Attributes:
        temperature: Sampling temperature. Higher values produce more random outputs.
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling. Only consider top k tokens.
        min_p: Minimum probability threshold for sampling.
        max_tokens: Maximum number of tokens to generate.
        min_tokens: Minimum number of tokens to generate.
        stop: List of stop strings. Generation stops when any is encountered.
        stop_token_ids: List of token IDs that trigger stop.
        presence_penalty: Penalty for token presence in generated text.
        frequency_penalty: Penalty for token frequency in generated text.
        repetition_penalty: Multiplicative penalty for repetition.
        seed: Random seed for reproducibility.
        skip_special_tokens: Whether to skip special tokens in output.
        include_stop_str_in_output: Whether to include stop string in output.
        logprobs: Number of log probabilities to return per token.
        prompt_logprobs: Number of prompt log probabilities to return.
        best_of: Number of sequences to generate and return the best.
        n: Number of output sequences to return.
        use_beam_search: Whether to use beam search instead of sampling.
        length_penalty: Penalty for sequence length in beam search.
        early_stopping: Whether to stop beam search early.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 256
    min_tokens: int = 0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    skip_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    best_of: int = 1
    n: int = 1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False

    def __post_init__(self):
        """Validate sampling parameters."""
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {self.top_k}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.best_of < self.n:
            raise ValueError(f"best_of must be >= n, got best_of={self.best_of}, n={self.n}")
        if self.use_beam_search and self.temperature != 0:
            raise ValueError("temperature must be 0 when using beam search")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingParams":
        """Create SamplingParams from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "stop": self.stop,
            "stop_token_ids": self.stop_token_ids,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "skip_special_tokens": self.skip_special_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "best_of": self.best_of,
            "n": self.n,
            "use_beam_search": self.use_beam_search,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }


@dataclass
class EngineConfig:
    """Configuration for the SwiftLLM inference engine.

    Attributes:
        model: Path to the model or HuggingFace model ID.
        tokenizer: Path to tokenizer. Defaults to model path.
        dtype: Data type for model weights.
        quantization: Quantization method if any.
        max_model_len: Maximum sequence length for the model.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of pipeline parallel stages.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
        block_size: Block size for PagedAttention.
        swap_space: Swap space in GiB for CPU offloading.
        max_num_seqs: Maximum number of concurrent sequences.
        max_num_batched_tokens: Maximum tokens per batch.
        enable_prefix_caching: Enable automatic prefix caching.
        enable_chunked_prefill: Enable chunked prefill for long prompts.
        max_paddings: Maximum padding tokens allowed.
        scheduler_policy: Scheduling policy for requests.
        preemption_mode: Mode for handling preemption.
        trust_remote_code: Trust remote code from HuggingFace.
        download_dir: Directory for downloading models.
        seed: Random seed for reproducibility.
        device: Device to use ('cuda', 'cpu', 'auto').
    """
    model: str = ""
    tokenizer: Optional[str] = None
    dtype: DataType = DataType.AUTO
    quantization: QuantizationMethod = QuantizationMethod.NONE
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    swap_space: float = 4.0
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    max_paddings: int = 256
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.FCFS
    preemption_mode: PreemptionMode = PreemptionMode.SWAP
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    seed: int = 0
    device: str = "auto"

    # Speculative decoding
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5
    speculative_max_model_len: Optional[int] = None

    # LoRA
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    lora_dtype: Optional[DataType] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if self.block_size not in [8, 16, 32]:
            raise ValueError(f"block_size must be 8, 16, or 32, got {self.block_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.tokenizer is None:
            self.tokenizer = self.model

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        """Create EngineConfig from a dictionary."""
        # Convert string enums
        if "dtype" in d and isinstance(d["dtype"], str):
            d["dtype"] = DataType(d["dtype"])
        if "quantization" in d and isinstance(d["quantization"], str):
            d["quantization"] = QuantizationMethod(d["quantization"])
        if "scheduler_policy" in d and isinstance(d["scheduler_policy"], str):
            d["scheduler_policy"] = SchedulerPolicy(d["scheduler_policy"])
        if "preemption_mode" in d and isinstance(d["preemption_mode"], str):
            d["preemption_mode"] = PreemptionMode(d["preemption_mode"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Enum):
                value = value.value
            result[field_name] = value
        return result


@dataclass
class ServerConfig:
    """Configuration for the SwiftLLM HTTP server.

    Attributes:
        host: Host to bind to.
        port: Port to bind to.
        api_key: API key for authentication.
        root_path: Root path for the API.
        ssl_keyfile: Path to SSL key file.
        ssl_certfile: Path to SSL certificate file.
        cors_allow_origins: Allowed CORS origins.
        max_log_len: Maximum log length for requests.
        response_role: Default role for responses.
        served_model_name: Name to use for the served model.
    """
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    root_path: str = ""
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])
    max_log_len: Optional[int] = None
    response_role: str = "assistant"
    served_model_name: Optional[str] = None

    # Limits
    max_model_len_limit: Optional[int] = None
    max_num_seqs_limit: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServerConfig":
        """Create ServerConfig from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LoRARequest:
    """Request to use a specific LoRA adapter.

    Attributes:
        lora_name: Unique name for this LoRA adapter.
        lora_path: Path to the LoRA adapter weights.
        lora_local_path: Local path if different from lora_path.
    """
    lora_name: str
    lora_path: str
    lora_local_path: Optional[str] = None

    @property
    def lora_int_id(self) -> int:
        """Return a unique integer ID for this LoRA."""
        return hash(self.lora_name) & 0xFFFFFFFF
