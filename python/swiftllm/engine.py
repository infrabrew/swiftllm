"""SwiftLLM Engine - High-level Python API

This module provides the main interface for running LLM inference with SwiftLLM.
"""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .config import EngineConfig, LoRARequest, SamplingParams


class FinishReason(Enum):
    """Reason for finishing generation."""
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass
class TokenLogprob:
    """Log probability information for a token."""
    token_id: int
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


@dataclass
class CompletionOutput:
    """Output for a single completion sequence.

    Attributes:
        index: Index of this output in the request.
        text: Generated text.
        token_ids: List of generated token IDs.
        cumulative_logprob: Cumulative log probability.
        logprobs: Per-token log probabilities if requested.
        finish_reason: Reason for finishing generation.
        stop_reason: The stop string or token that caused stop.
    """
    index: int
    text: str
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[TokenLogprob]] = None
    finish_reason: Optional[FinishReason] = None
    stop_reason: Optional[Union[int, str]] = None

    @property
    def finished(self) -> bool:
        """Check if generation is finished."""
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    """Output of a generation request.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: The input prompt.
        prompt_token_ids: Token IDs of the prompt.
        prompt_logprobs: Log probabilities of prompt tokens.
        outputs: List of completion outputs.
        finished: Whether all outputs are finished.
        metrics: Performance metrics.
    """
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    prompt_logprobs: Optional[List[TokenLogprob]] = None
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = False
    metrics: Optional[Dict[str, float]] = None

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt[:50] + '...' if self.prompt and len(self.prompt) > 50 else self.prompt!r}, "
            f"num_outputs={len(self.outputs)}, "
            f"finished={self.finished})"
        )


class LLMEngine:
    """Low-level LLM inference engine.

    This class provides the core functionality for running inference.
    For most use cases, use the higher-level `LLM` class instead.
    """

    def __init__(
        self,
        config: EngineConfig,
    ):
        """Initialize the LLM engine.

        Args:
            config: Engine configuration.
        """
        self.config = config
        self._initialized = False
        self._request_counter = 0
        self._pending_requests: Dict[str, Any] = {}

        # Lazy import to avoid startup overhead
        self._tokenizer = None
        self._model = None

    def _ensure_initialized(self):
        """Ensure the engine is initialized."""
        if self._initialized:
            return

        # Import heavy dependencies only when needed
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for tokenization. "
                "Install with: pip install transformers"
            )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer,
            trust_remote_code=self.config.trust_remote_code,
        )

        # TODO: Initialize actual Rust engine via PyO3
        # For now, we'll use a placeholder that demonstrates the API

        self._initialized = True

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        """Add a new request to the engine.

        Args:
            request_id: Unique identifier for the request.
            prompt: Text prompt (mutually exclusive with prompt_token_ids).
            prompt_token_ids: Token IDs of the prompt.
            sampling_params: Sampling parameters.
            lora_request: Optional LoRA adapter to use.
        """
        self._ensure_initialized()

        if prompt is None and prompt_token_ids is None:
            raise ValueError("Either prompt or prompt_token_ids must be provided")
        if prompt is not None and prompt_token_ids is not None:
            raise ValueError("Only one of prompt or prompt_token_ids should be provided")

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Tokenize if needed
        if prompt_token_ids is None:
            prompt_token_ids = self._tokenizer.encode(prompt)

        self._pending_requests[request_id] = {
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "lora_request": lora_request,
            "created_time": time.time(),
        }

    def abort_request(self, request_id: str) -> None:
        """Abort a pending request.

        Args:
            request_id: ID of the request to abort.
        """
        if request_id in self._pending_requests:
            del self._pending_requests[request_id]

    def step(self) -> List[RequestOutput]:
        """Run one step of the engine.

        Returns:
            List of request outputs that have new tokens.
        """
        self._ensure_initialized()

        # TODO: Implement actual engine step via Rust
        # This is a placeholder that simulates generation

        outputs = []
        completed_ids = []

        for request_id, request_data in self._pending_requests.items():
            # Simulate token generation
            prompt = request_data["prompt"]
            prompt_token_ids = request_data["prompt_token_ids"]
            params = request_data["sampling_params"]

            # For demo: generate a simple response
            response_text = " I'm SwiftLLM, ready to help!"
            response_tokens = self._tokenizer.encode(response_text)

            output = RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=response_text,
                        token_ids=response_tokens,
                        finish_reason=FinishReason.STOP,
                    )
                ],
                finished=True,
            )
            outputs.append(output)
            completed_ids.append(request_id)

        # Clean up completed requests
        for request_id in completed_ids:
            del self._pending_requests[request_id]

        return outputs

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return len(self._pending_requests)

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return len(self._pending_requests) > 0


class LLM:
    """High-level LLM interface for offline batched inference.

    This class provides a simple interface for running inference on a batch
    of prompts. For online serving, use the AsyncLLM class or the HTTP server.

    Example:
        >>> from swiftllm import LLM, SamplingParams
        >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
        >>> outputs = llm.generate(["Hello, how are you?"])
        >>> print(outputs[0].outputs[0].text)
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        swap_space: float = 4.0,
        trust_remote_code: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        """Initialize the LLM.

        Args:
            model: Path to the model or HuggingFace model ID.
            tokenizer: Path to tokenizer. Defaults to model path.
            dtype: Data type for model weights ('auto', 'float16', 'bfloat16', 'float32').
            quantization: Quantization method ('awq', 'gptq', 'squeezellm', None).
            max_model_len: Maximum sequence length for the model.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            swap_space: Swap space in GiB for CPU offloading.
            trust_remote_code: Trust remote code from HuggingFace.
            seed: Random seed for reproducibility.
            **kwargs: Additional engine configuration options.
        """
        from .config import DataType, QuantizationMethod

        # Build configuration
        dtype_enum = DataType(dtype) if dtype != "auto" else DataType.AUTO
        quant_enum = QuantizationMethod(quantization) if quantization else QuantizationMethod.NONE

        self.config = EngineConfig(
            model=model,
            tokenizer=tokenizer,
            dtype=dtype_enum,
            quantization=quant_enum,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            trust_remote_code=trust_remote_code,
            seed=seed,
            **{k: v for k, v in kwargs.items() if hasattr(EngineConfig, k)},
        )

        self._engine = LLMEngine(self.config)
        self._request_counter = 0

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"req-{self._request_counter}-{uuid.uuid4().hex[:8]}"

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt or list of prompts.
            sampling_params: Sampling parameters. Can be a single instance
                (applied to all prompts) or a list (one per prompt).
            use_tqdm: Whether to show a progress bar.
            lora_request: Optional LoRA adapter to use.

        Returns:
            List of RequestOutput objects, one per prompt.
        """
        # Normalize inputs
        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = [SamplingParams() for _ in prompts]
        elif isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params for _ in prompts]

        if len(sampling_params) != len(prompts):
            raise ValueError(
                f"Number of sampling params ({len(sampling_params)}) must match "
                f"number of prompts ({len(prompts)})"
            )

        # Add all requests
        request_ids = []
        for prompt, params in zip(prompts, sampling_params):
            request_id = self._generate_request_id()
            request_ids.append(request_id)
            self._engine.add_request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=params,
                lora_request=lora_request,
            )

        # Optionally show progress bar
        if use_tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(prompts), desc="Generating")
            except ImportError:
                pbar = None
        else:
            pbar = None

        # Run engine until all requests complete
        outputs: Dict[str, RequestOutput] = {}
        while self._engine.has_unfinished_requests():
            step_outputs = self._engine.step()
            for output in step_outputs:
                outputs[output.request_id] = output
                if pbar is not None and output.finished:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Return outputs in original order
        return [outputs[rid] for rid in request_ids]

    def encode(
        self,
        prompts: Union[str, List[str]],
    ) -> List[List[int]]:
        """Tokenize prompts into token IDs.

        Args:
            prompts: A single prompt or list of prompts.

        Returns:
            List of token ID lists.
        """
        self._engine._ensure_initialized()

        if isinstance(prompts, str):
            prompts = [prompts]

        return [self._engine._tokenizer.encode(p) for p in prompts]

    def get_tokenizer(self):
        """Get the tokenizer used by this LLM."""
        self._engine._ensure_initialized()
        return self._engine._tokenizer


class AsyncLLM:
    """Asynchronous LLM interface for online serving.

    This class provides an async interface for running inference,
    suitable for use in async web servers.

    Example:
        >>> from swiftllm import AsyncLLM, SamplingParams
        >>> llm = AsyncLLM(model="meta-llama/Llama-2-7b-hf")
        >>> async for output in llm.generate("Hello"):
        ...     print(output.outputs[0].text)
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the async LLM.

        Args:
            model: Path to the model or HuggingFace model ID.
            tokenizer: Path to tokenizer. Defaults to model path.
            **kwargs: Additional engine configuration options.
        """
        self.config = EngineConfig(
            model=model,
            tokenizer=tokenizer,
            **{k: v for k, v in kwargs.items() if hasattr(EngineConfig, k)},
        )
        self._engine = LLMEngine(self.config)
        self._request_counter = 0
        self._background_task: Optional[asyncio.Task] = None

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"async-req-{self._request_counter}-{uuid.uuid4().hex[:8]}"

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Generate completions for a prompt asynchronously.

        This method yields RequestOutput objects as tokens are generated,
        allowing for streaming responses.

        Args:
            prompt: The input prompt.
            sampling_params: Sampling parameters.
            request_id: Optional custom request ID.
            lora_request: Optional LoRA adapter to use.

        Yields:
            RequestOutput objects with incremental completions.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if request_id is None:
            request_id = self._generate_request_id()

        self._engine.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        # Poll for results
        while True:
            await asyncio.sleep(0.001)  # Yield control to event loop
            outputs = self._engine.step()

            for output in outputs:
                if output.request_id == request_id:
                    yield output
                    if output.finished:
                        return

    async def abort(self, request_id: str) -> None:
        """Abort a pending request.

        Args:
            request_id: ID of the request to abort.
        """
        self._engine.abort_request(request_id)


def create_engine(config: Union[EngineConfig, Dict[str, Any]]) -> LLMEngine:
    """Create an LLM engine from configuration.

    Args:
        config: Engine configuration or dictionary.

    Returns:
        Initialized LLMEngine.
    """
    if isinstance(config, dict):
        config = EngineConfig.from_dict(config)
    return LLMEngine(config)


# Convenience function for quick inference
def generate(
    model: str,
    prompts: Union[str, List[str]],
    sampling_params: Optional[SamplingParams] = None,
    **kwargs,
) -> List[RequestOutput]:
    """Quick generation without explicitly creating an LLM instance.

    Args:
        model: Path to the model or HuggingFace model ID.
        prompts: A single prompt or list of prompts.
        sampling_params: Sampling parameters.
        **kwargs: Additional LLM configuration options.

    Returns:
        List of RequestOutput objects.
    """
    llm = LLM(model=model, **kwargs)
    return llm.generate(prompts, sampling_params)
