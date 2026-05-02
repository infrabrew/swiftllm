# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      engine.py
# PATH:      /python/swiftllm/engine.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/config.py     EngineConfig, SamplingParams, SelfConsistencyConfig,
#                                   RefinementConfig, VerificationConfig, DisaggregatedServingConfig
#   - python/swiftllm/sampling.py   SelfConsistencySampler, SelfConsistencyResult
# USED BY:
#   - python/swiftllm/__init__.py   LLM, AsyncLLM, LLMEngine re-exports
#   - python/swiftllm/cli.py        cmd_generate, cmd_serve, cmd_chat
# SEE ALSO:
#   - crates/swiftllm-core/src/inference/refinement.rs    Rust RefinementPipeline
#   - crates/swiftllm-core/src/inference/verification.rs  Rust verify_and_rank / best_of_n_by_logprob
#   - crates/swiftllm-core/src/sampling/self_consistency.rs  Rust self_consistency_vote
#   - crates/swiftllm-core/src/serving/disaggregated.rs   Rust DisaggregatedScheduler
#   - crates/swiftllm-core/src/engine.rs                  Rust Engine integration point
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

from .config import (
    EngineConfig,
    LoRARequest,
    SamplingParams,
    SelfConsistencyConfig,
    RefinementConfig,
    VerificationConfig,
    DisaggregatedServingConfig,
    ScoringStrategy,
)


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
            f"prompt={repr(self.prompt[:50] + '...') if self.prompt and len(self.prompt) > 50 else repr(self.prompt)}, "
            f"num_outputs={len(self.outputs)}, "
            f"finished={self.finished})"
        )


def _is_gguf_model(model_path: str) -> bool:
    """Check if the model path points to a GGUF file."""
    return model_path.lower().endswith(".gguf")


def _normalised_edit_distance(s: str, t: str) -> float:
    """Compute normalised Levenshtein edit distance between s and t.

    Uses O(min(m, n)) space with a two-row rolling DP table.
    Returns 0.0 when the strings are identical, 1.0 when they share nothing.
    Mirrors ``normalised_edit_distance()`` in
    ``crates/swiftllm-core/src/inference/refinement.rs``.
    """
    m, n = len(s), len(t)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    if m < n:
        s, t, m, n = t, s, n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[n] / max(m, n)


def _rule_score(text: str) -> float:
    """Heuristic quality score for a candidate text (0.0–1.0).

    Mirrors ``rule_score()`` in
    ``crates/swiftllm-core/src/inference/verification.rs``.
    Rewards:
    - Non-empty output         (+0.2)
    - Moderate length          (+0.2, optimal 50–500 tokens)
    - Structured reasoning     (+0.2, contains numbered steps or bullet points)
    - Definitive conclusion    (+0.2, ends with a clear statement)
    - No self-correction flags (+0.2, e.g. no "I was wrong" / "let me correct")
    """
    if not text.strip():
        return 0.0
    score = 0.2
    words = len(text.split())
    if 20 <= words <= 500:
        score += 0.2
    import re
    if re.search(r"(\d+[\.\)]\s|\*\s|-\s)", text):
        score += 0.2
    if re.search(r"\.\s*$", text.strip()):
        score += 0.2
    if not re.search(r"(i was wrong|let me correct|actually,|wait,)", text.lower()):
        score += 0.2
    return min(score, 1.0)


@dataclass
class RefinementOutput:
    """Output from LLM.generate_with_refinement().

    Attributes:
        prompt: The original prompt.
        initial_output: The first generated response.
        final_output: The response after all refinement rounds.
        rounds: List of per-round detail dicts with keys
            round, critique, revised, improvement_score.
        num_rounds_used: Actual number of refinement rounds executed.
    """
    prompt: str
    initial_output: str
    final_output: str
    rounds: List[Dict[str, Any]]
    num_rounds_used: int


@dataclass
class VerifiedOutput:
    """Output from LLM.generate_best_of_n().

    Attributes:
        prompt: The original prompt.
        best_text: Highest-scoring candidate text.
        best_score: Combined score of the best candidate.
        candidates: All candidates sorted by score (best first).
    """
    prompt: str
    best_text: str
    best_score: float
    candidates: List[Dict[str, Any]]


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
        self._is_gguf = False

    def _ensure_initialized(self):
        """Ensure the engine is initialized."""
        if self._initialized:
            return

        # Resolve model path (download from HuggingFace if needed)
        from .model_resolver import resolve_model

        original_model = self.config.model
        resolved_path = resolve_model(
            model=self.config.model,
            download_dir=self.config.download_dir,
        )
        self.config.model = resolved_path

        if self.config.tokenizer is None or self.config.tokenizer == original_model:
            self.config.tokenizer = resolved_path
        else:
            self.config.tokenizer = resolve_model(
                model=self.config.tokenizer,
                download_dir=self.config.download_dir,
            )

        # Check if this is a GGUF model
        if _is_gguf_model(resolved_path):
            self._init_gguf(resolved_path)
        else:
            self._init_transformers(resolved_path)

        self._initialized = True

    def _init_gguf(self, model_path: str):
        """Initialize with llama-cpp-python for GGUF models."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python\n"
                "For CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        self._is_gguf = True

        # Determine GPU layers
        n_gpu_layers = -1  # offload all layers to GPU by default
        if self.config.device == "cpu":
            n_gpu_layers = 0

        # Context length
        n_ctx = self.config.max_model_len or 4096

        print(f"Loading GGUF model: {model_path}")
        print(f"  GPU layers: {'all' if n_gpu_layers == -1 else n_gpu_layers}")
        print(f"  Context length: {n_ctx}")

        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        # llama-cpp-python has its own tokenizer built in
        self._tokenizer = None
        print("GGUF model loaded successfully!")

    def _init_transformers(self, model_path: str):
        """Initialize with transformers tokenizer for non-GGUF models."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for tokenization. "
                "Install with: pip install transformers"
            )

        self._is_gguf = False
        if self.config.trust_remote_code:
            import logging
            logging.getLogger(__name__).warning(
                "trust_remote_code is enabled — remote code from HuggingFace will be executed"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer,
            trust_remote_code=self.config.trust_remote_code,
        )

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

        # Tokenize if needed (for non-GGUF models)
        if prompt_token_ids is None and not self._is_gguf:
            prompt_token_ids = self._tokenizer.encode(prompt)
        elif prompt_token_ids is None:
            prompt_token_ids = []  # GGUF handles tokenization internally

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

        if self._is_gguf:
            return self._step_gguf()
        else:
            return self._step_placeholder()

    def _step_gguf(self) -> List[RequestOutput]:
        """Run inference step using llama-cpp-python."""
        outputs = []
        completed_ids = []

        for request_id, request_data in self._pending_requests.items():
            prompt = request_data["prompt"]
            prompt_token_ids = request_data["prompt_token_ids"]
            params = request_data["sampling_params"]

            try:
                # Build llama-cpp generation kwargs
                kwargs = {
                    "max_tokens": params.max_tokens,
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                }

                if params.top_k > 0:
                    kwargs["top_k"] = params.top_k

                if params.stop:
                    kwargs["stop"] = params.stop

                if params.frequency_penalty != 0.0:
                    kwargs["frequency_penalty"] = params.frequency_penalty

                if params.presence_penalty != 0.0:
                    kwargs["presence_penalty"] = params.presence_penalty

                if params.repetition_penalty != 1.0:
                    kwargs["repeat_penalty"] = params.repetition_penalty

                # Run generation
                result = self._model(prompt, **kwargs)

                response_text = result["choices"][0]["text"]
                finish = result["choices"][0].get("finish_reason", "stop")

                if finish == "length":
                    finish_reason = FinishReason.LENGTH
                else:
                    finish_reason = FinishReason.STOP

                # Get token count from usage
                completion_tokens = result.get("usage", {}).get("completion_tokens", 0)

                output = RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=response_text,
                            token_ids=list(range(completion_tokens)),
                            finish_reason=finish_reason,
                        )
                    ],
                    finished=True,
                    metrics={
                        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": completion_tokens,
                    },
                )
            except Exception:
                output = RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[],
                            finish_reason=FinishReason.ABORT,
                        )
                    ],
                    finished=True,
                )
            outputs.append(output)
            completed_ids.append(request_id)

        for request_id in completed_ids:
            del self._pending_requests[request_id]

        return outputs

    def _step_placeholder(self) -> List[RequestOutput]:
        """Placeholder step for non-GGUF models (until Rust backend is ready)."""
        outputs = []
        completed_ids = []

        for request_id, request_data in self._pending_requests.items():
            prompt = request_data["prompt"]
            prompt_token_ids = request_data["prompt_token_ids"]
            params = request_data["sampling_params"]

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
        download_dir: Optional[str] = None,
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
            download_dir: Directory for downloading models. Defaults to ~/.cache/swiftllm/models.
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
            download_dir=download_dir,
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

    def generate_with_self_consistency(
        self,
        prompts: Union[str, List[str]],
        config: Optional[SelfConsistencyConfig] = None,
        base_params: Optional[SamplingParams] = None,
    ) -> List["SelfConsistencyResult"]:
        """Generate answers using self-consistency majority voting.

        For each prompt, generates ``config.num_samples`` independent reasoning
        chains at ``config.temperature`` and returns the plurality-majority
        answer.  Mirrors ``self_consistency_vote()`` in
        ``crates/swiftllm-core/src/sampling/self_consistency.rs``.

        Args:
            prompts: Single prompt or list of prompts.
            config: Self-consistency configuration; falls back to
                ``self.config.self_consistency`` if None.
            base_params: Base sampling params (temperature is overridden by
                ``config.temperature``; other fields are kept).

        Returns:
            List of SelfConsistencyResult objects, one per prompt.

        Raises:
            ValueError: If no SelfConsistencyConfig is provided or configured.

        Example::

            from swiftllm import LLM
            from swiftllm.config import SelfConsistencyConfig

            llm = LLM(model="meta-llama/Llama-2-7b-hf")
            results = llm.generate_with_self_consistency(
                "What is 12 × 15?",
                config=SelfConsistencyConfig(num_samples=8),
            )
            print(results[0].answer, f"({results[0].vote_fraction:.0%} agreement)")
        """
        from .sampling import SelfConsistencySampler

        cfg = config or self.config.self_consistency
        if cfg is None:
            raise ValueError(
                "No SelfConsistencyConfig provided. Pass config= or set "
                "EngineConfig.self_consistency."
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        # Build sampling params for individual samples
        if base_params is None:
            base_params = SamplingParams()
        sample_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=base_params.top_p,
            top_k=base_params.top_k,
            min_p=base_params.min_p,
            max_tokens=base_params.max_tokens,
            stop=base_params.stop,
            stop_token_ids=base_params.stop_token_ids,
            n=1,
        )

        sampler = SelfConsistencySampler(cfg)
        results = []

        for prompt in prompts:
            # Generate all samples
            raw_texts: List[str] = []
            log_probs_list: List[Optional[float]] = []

            for _ in range(cfg.num_samples):
                outputs = self.generate([prompt], sample_params, use_tqdm=False)
                completion = outputs[0].outputs[0]
                raw_texts.append(completion.text)
                log_probs_list.append(completion.cumulative_logprob)

            result = sampler.vote(raw_texts, log_probs=log_probs_list)
            results.append(result)

        return results

    def generate_with_refinement(
        self,
        prompts: Union[str, List[str]],
        config: Optional[RefinementConfig] = None,
        base_params: Optional[SamplingParams] = None,
        critique_fn: Optional[Callable[[str], str]] = None,
    ) -> List["RefinementOutput"]:
        """Generate answers using iterative self-refinement.

        For each prompt, generates an initial response and then iteratively
        critiques and refines it until a stopping criterion is reached.  Mirrors
        ``RefinementPipeline::refine()`` in
        ``crates/swiftllm-core/src/inference/refinement.rs``.

        Args:
            prompts: Single prompt or list of prompts.
            config: Refinement configuration; falls back to
                ``self.config.refinement`` if None.
            base_params: Base sampling parameters for generation.
            critique_fn: Optional external critique function
                ``(current_output: str) -> critique_prompt: str``.
                If None, uses a generic self-critique template.

        Returns:
            List of RefinementOutput objects, one per prompt.

        Raises:
            ValueError: If no RefinementConfig is provided or configured.
        """
        cfg = config or self.config.refinement
        if cfg is None:
            raise ValueError(
                "No RefinementConfig provided. Pass config= or set "
                "EngineConfig.refinement."
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        if base_params is None:
            base_params = SamplingParams()

        def _default_critique(prompt: str, output: str) -> str:
            template = cfg.critique_template or (
                "Review the following answer and provide a critique "
                "identifying specific errors or improvements:\n\n"
                "Answer: {output}\n\nCritique:"
            )
            return template.format(output=output)

        results = []

        for prompt in prompts:
            # Initial generation
            initial = self.generate([prompt], base_params, use_tqdm=False)[0].outputs[0].text
            current = initial
            rounds = []

            for round_idx in range(cfg.max_rounds):
                critique_prompt = (
                    critique_fn(current) if critique_fn
                    else _default_critique(prompt, current)
                )
                critique_out = self.generate(
                    [critique_prompt], base_params, use_tqdm=False
                )[0].outputs[0].text

                refine_prompt = (
                    f"Given this critique:\n{critique_out}\n\n"
                    f"Revise your answer:\n{current}\n\nRevised answer:"
                )
                revised = self.generate(
                    [refine_prompt], base_params, use_tqdm=False
                )[0].outputs[0].text

                # Measure improvement (normalised edit distance)
                improvement = _normalised_edit_distance(current, revised)
                rounds.append({
                    "round": round_idx + 1,
                    "critique": critique_out,
                    "revised": revised,
                    "improvement_score": improvement,
                })
                current = revised

                # Check stopping criterion
                from .config import StoppingCriterion, ImprovementMetric
                if cfg.stopping_criterion in (
                    StoppingCriterion.MIN_IMPROVEMENT, StoppingCriterion.EITHER
                ):
                    if improvement < cfg.min_improvement:
                        break

            results.append(RefinementOutput(
                prompt=prompt,
                initial_output=initial,
                final_output=current,
                rounds=rounds,
                num_rounds_used=len(rounds),
            ))

        return results

    def generate_best_of_n(
        self,
        prompts: Union[str, List[str]],
        config: Optional[VerificationConfig] = None,
        base_params: Optional[SamplingParams] = None,
    ) -> List["VerifiedOutput"]:
        """Generate N candidates and select the best via dense verification.

        Generates ``config.num_candidates`` responses per prompt and ranks
        them using the configured scoring strategy.  Mirrors
        ``verify_and_rank()`` and ``best_of_n_by_logprob()`` in
        ``crates/swiftllm-core/src/inference/verification.rs``.

        Args:
            prompts: Single prompt or list of prompts.
            config: Verification configuration; falls back to
                ``self.config.verification`` if None.
            base_params: Base sampling parameters.

        Returns:
            List of VerifiedOutput objects with ranked candidates, one per prompt.

        Raises:
            ValueError: If no VerificationConfig is provided or configured.
        """
        cfg = config or self.config.verification
        if cfg is None:
            raise ValueError(
                "No VerificationConfig provided. Pass config= or set "
                "EngineConfig.verification."
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        if base_params is None:
            base_params = SamplingParams(temperature=0.8)

        results = []

        for prompt in prompts:
            candidates: List[Dict[str, Any]] = []

            for _ in range(cfg.num_candidates):
                output = self.generate([prompt], base_params, use_tqdm=False)[0].outputs[0]
                score = _rule_score(output.text)
                candidates.append({
                    "text": output.text,
                    "rule_score": score,
                    "logprob": output.cumulative_logprob or 0.0,
                    "token_ids": output.token_ids,
                })

            # Compute combined score based on strategy
            for c in candidates:
                if cfg.scoring_strategy == ScoringStrategy.RULE_BASED:
                    c["score"] = c["rule_score"]
                elif cfg.scoring_strategy == ScoringStrategy.SEQUENCE_LOG_PROB:
                    c["score"] = c["logprob"]
                else:  # ENSEMBLE or NEURAL (neural falls back to rule-based for now)
                    total = cfg.rule_weight + cfg.neural_weight + cfg.logprob_weight
                    w_rule = cfg.rule_weight / total
                    w_lp = cfg.logprob_weight / total
                    c["score"] = w_rule * c["rule_score"] + w_lp * (c["logprob"] or 0.0)

            candidates.sort(key=lambda c: c["score"], reverse=True)

            results.append(VerifiedOutput(
                prompt=prompt,
                best_text=candidates[0]["text"],
                best_score=candidates[0]["score"],
                candidates=candidates,
            ))

        return results

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

        if self._engine._is_gguf:
            return [self._engine._model.tokenize(p.encode()) for p in prompts]

        return [self._engine._tokenizer.encode(p) for p in prompts]

    def get_tokenizer(self):
        """Get the tokenizer used by this LLM."""
        self._engine._ensure_initialized()
        if self._engine._is_gguf:
            return self._engine._model
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

# ------------------------------------------------------------------------------
# END OF FILE: engine.py
# REPO PATH:   /swiftllm/python/swiftllm/engine.py
# INTEGRATES:  config.py · sampling.py · __init__.py · cli.py
#              Rust: refinement.rs · verification.rs · self_consistency.rs
#              Rust: disaggregated.rs · engine.rs
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
