"""Type stubs for SwiftLLM Rust core bindings.

These stubs provide type hints for the PyO3-generated Rust bindings.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

class RustEngine:
    """Low-level Rust inference engine."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Rust engine with configuration."""
        ...

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> None:
        """Add a request to the engine."""
        ...

    def abort_request(self, request_id: str) -> None:
        """Abort a pending request."""
        ...

    def step(self) -> List[Dict[str, Any]]:
        """Run one step of the engine and return outputs."""
        ...

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        ...

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        ...


class RustBlockManager:
    """Rust block manager for PagedAttention."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the block manager."""
        ...

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        """Allocate blocks for a sequence."""
        ...

    def free(self, seq_id: int) -> None:
        """Free blocks for a sequence."""
        ...

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if blocks can be allocated."""
        ...

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get the block table for a sequence."""
        ...


class RustScheduler:
    """Rust scheduler for continuous batching."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the scheduler."""
        ...

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> None:
        """Add a request to the scheduler."""
        ...

    def schedule(self) -> Tuple[List[str], List[str], List[str]]:
        """Schedule the next batch.

        Returns:
            Tuple of (scheduled_ids, preempted_ids, swapped_in_ids).
        """
        ...

    def update(self, outputs: List[Dict[str, Any]]) -> None:
        """Update scheduler state with generation outputs."""
        ...


class RustSampler:
    """Rust token sampler."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the sampler."""
        ...

    def sample(
        self,
        logits: List[float],
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample a token from logits.

        Returns:
            Tuple of (token_id, log_probability).
        """
        ...


def cuda_available() -> bool:
    """Check if CUDA is available."""
    ...


def cuda_device_count() -> int:
    """Get the number of CUDA devices."""
    ...


def cuda_get_device_properties(device_id: int) -> Dict[str, Any]:
    """Get properties of a CUDA device."""
    ...


def cuda_synchronize() -> None:
    """Synchronize all CUDA devices."""
    ...
