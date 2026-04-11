# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      config.py
# PATH:      /python/swiftllm/config.py
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

"""SwiftLLM Configuration Classes

This module provides configuration classes for the SwiftLLM inference engine.

All configuration options can be overridden via environment variables with the
``SWIFTLLM_`` prefix.  For example, ``SWIFTLLM_GPU_MEMORY_UTILIZATION=0.95``
overrides ``EngineConfig.gpu_memory_utilization``.  Boolean values accept
``1/true/yes`` (case-insensitive).

See the README "Environment Variables" section for the full reference.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _env(name: str, default=None):
    """Read a SWIFTLLM_ environment variable (case-insensitive value)."""
    return os.environ.get(f"SWIFTLLM_{name}", default)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean SWIFTLLM_ environment variable."""
    val = _env(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Read an integer SWIFTLLM_ environment variable."""
    val = _env(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Read a float SWIFTLLM_ environment variable."""
    val = _env(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


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
        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be <= max_tokens ({self.max_tokens})"
            )
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
    dtype: DataType = field(default_factory=lambda: DataType(_env("DTYPE", "auto")))
    quantization: QuantizationMethod = field(default_factory=lambda: QuantizationMethod(_env("QUANTIZATION", "none")))
    max_model_len: Optional[int] = field(default_factory=lambda: _env_int("MAX_MODEL_LEN"))
    tensor_parallel_size: int = field(default_factory=lambda: _env_int("TENSOR_PARALLEL_SIZE", 1))
    pipeline_parallel_size: int = field(default_factory=lambda: _env_int("PIPELINE_PARALLEL_SIZE", 1))
    gpu_memory_utilization: float = field(default_factory=lambda: _env_float("GPU_MEMORY_UTILIZATION", 0.90))
    block_size: int = field(default_factory=lambda: _env_int("BLOCK_SIZE", 16))
    swap_space: float = field(default_factory=lambda: _env_float("SWAP_SPACE", 4.0))
    max_num_seqs: int = field(default_factory=lambda: _env_int("MAX_NUM_SEQS", 256))
    max_num_batched_tokens: Optional[int] = field(default_factory=lambda: _env_int("MAX_NUM_BATCHED_TOKENS"))
    enable_prefix_caching: bool = field(default_factory=lambda: _env_bool("ENABLE_PREFIX_CACHING"))
    enable_chunked_prefill: bool = field(default_factory=lambda: _env_bool("ENABLE_CHUNKED_PREFILL"))
    max_paddings: int = field(default_factory=lambda: _env_int("MAX_PADDINGS", 256))
    scheduler_policy: SchedulerPolicy = field(default_factory=lambda: SchedulerPolicy(_env("SCHEDULER_POLICY", "fcfs")))
    preemption_mode: PreemptionMode = field(default_factory=lambda: PreemptionMode(_env("PREEMPTION_MODE", "swap")))
    trust_remote_code: bool = field(default_factory=lambda: _env_bool("TRUST_REMOTE_CODE"))
    download_dir: Optional[str] = field(default_factory=lambda: _env("MODEL_DIR"))
    seed: int = field(default_factory=lambda: _env_int("SEED", 0))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    # Speculative decoding
    speculative_model: Optional[str] = field(default_factory=lambda: _env("SPECULATIVE_MODEL"))
    num_speculative_tokens: int = field(default_factory=lambda: _env_int("NUM_SPECULATIVE_TOKENS", 5))
    speculative_max_model_len: Optional[int] = field(default_factory=lambda: _env_int("SPECULATIVE_MAX_MODEL_LEN"))

    # LoRA
    enable_lora: bool = field(default_factory=lambda: _env_bool("ENABLE_LORA"))
    max_loras: int = field(default_factory=lambda: _env_int("MAX_LORAS", 1))
    max_lora_rank: int = field(default_factory=lambda: _env_int("MAX_LORA_RANK", 16))
    lora_dtype: Optional[DataType] = None

    # Performance tuning
    enforce_eager: bool = field(default_factory=lambda: _env_bool("ENFORCE_EAGER"))
    kv_cache_dtype: str = field(default_factory=lambda: _env("KV_CACHE_DTYPE", "auto"))
    num_gpu_layers: Optional[int] = field(default_factory=lambda: _env_int("NUM_GPU_LAYERS"))
    cpu_offload_gb: float = field(default_factory=lambda: _env_float("CPU_OFFLOAD_GB", 0.0))
    max_parallel_loading: int = field(default_factory=lambda: _env_int("MAX_PARALLEL_LOADING", 1))
    gpu_overhead_mb: int = field(default_factory=lambda: _env_int("GPU_OVERHEAD_MB", 0))
    flash_attention: bool = field(default_factory=lambda: _env_bool("FLASH_ATTENTION", True))
    keep_alive_secs: int = field(default_factory=lambda: _env_int("KEEP_ALIVE", 300))
    num_parallel_slots: int = field(default_factory=lambda: _env_int("NUM_PARALLEL", 1))
    max_loaded_models: int = field(default_factory=lambda: _env_int("MAX_LOADED_MODELS", 1))

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
        """Create EngineConfig from a dictionary (does not mutate input)."""
        d = dict(d)  # Avoid mutating caller's dict
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
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    api_key: Optional[str] = field(default_factory=lambda: _env("API_KEY"))
    root_path: str = field(default_factory=lambda: _env("ROOT_PATH", ""))
    ssl_keyfile: Optional[str] = field(default_factory=lambda: _env("SSL_KEYFILE"))
    ssl_certfile: Optional[str] = field(default_factory=lambda: _env("SSL_CERTFILE"))
    cors_allow_origins: List[str] = field(default_factory=lambda: (_env("CORS_ALLOW_ORIGINS", "*")).split(","))
    max_log_len: Optional[int] = field(default_factory=lambda: _env_int("MAX_LOG_LEN"))
    response_role: str = field(default_factory=lambda: _env("RESPONSE_ROLE", "assistant"))
    served_model_name: Optional[str] = field(default_factory=lambda: _env("SERVED_MODEL_NAME"))

    # Limits
    max_model_len_limit: Optional[int] = field(default_factory=lambda: _env_int("MAX_MODEL_LEN_LIMIT"))
    max_num_seqs_limit: Optional[int] = field(default_factory=lambda: _env_int("MAX_NUM_SEQS_LIMIT"))

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

# ------------------------------------------------------------------------------
# END OF FILE: config.py
# REPO PATH:   /swiftllm/python/swiftllm/config.py
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
