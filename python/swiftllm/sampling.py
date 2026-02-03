"""SwiftLLM Sampling Strategies

This module provides various sampling strategies for token generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class Sampler(ABC):
    """Abstract base class for token samplers."""

    @abstractmethod
    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample a token from logits.

        Args:
            logits: Log probabilities for each token in the vocabulary.
            token_ids: Previously generated token IDs (for penalties).

        Returns:
            Tuple of (sampled_token_id, log_probability).
        """
        pass


class GreedySampler(Sampler):
    """Greedy sampling - always select the highest probability token."""

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Select the token with highest probability."""
        token_id = int(np.argmax(logits))
        # Compute log probability (log softmax)
        log_probs = logits - np.logaddexp.reduce(logits)
        return token_id, float(log_probs[token_id])


class TemperatureSampler(Sampler):
    """Temperature-scaled sampling."""

    def __init__(self, temperature: float = 1.0):
        """Initialize temperature sampler.

        Args:
            temperature: Sampling temperature. Higher = more random.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample with temperature scaling."""
        scaled_logits = logits / self.temperature
        probs = _softmax(scaled_logits)
        token_id = int(np.random.choice(len(probs), p=probs))
        log_probs = np.log(probs + 1e-10)
        return token_id, float(log_probs[token_id])


class TopKSampler(Sampler):
    """Top-K sampling - sample from the K most likely tokens."""

    def __init__(self, k: int, temperature: float = 1.0):
        """Initialize top-k sampler.

        Args:
            k: Number of top tokens to consider.
            temperature: Sampling temperature.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.temperature = temperature

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample from top-k tokens."""
        scaled_logits = logits / self.temperature if self.temperature != 1.0 else logits

        # Get top-k indices
        top_k_indices = np.argpartition(scaled_logits, -self.k)[-self.k:]
        top_k_logits = scaled_logits[top_k_indices]

        # Compute probabilities over top-k only
        top_k_probs = _softmax(top_k_logits)

        # Sample from top-k
        idx = int(np.random.choice(len(top_k_probs), p=top_k_probs))
        token_id = int(top_k_indices[idx])

        # Compute original log probability
        log_probs = scaled_logits - np.logaddexp.reduce(scaled_logits)
        return token_id, float(log_probs[token_id])


class TopPSampler(Sampler):
    """Top-P (nucleus) sampling - sample from the smallest set of tokens
    whose cumulative probability exceeds p."""

    def __init__(self, p: float, temperature: float = 1.0):
        """Initialize top-p sampler.

        Args:
            p: Cumulative probability threshold.
            temperature: Sampling temperature.
        """
        if not 0 < p <= 1:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p
        self.temperature = temperature

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample from nucleus of probability mass."""
        scaled_logits = logits / self.temperature if self.temperature != 1.0 else logits
        probs = _softmax(scaled_logits)

        # Sort by probability descending
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cutoff index
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = int(np.searchsorted(cumsum, self.p)) + 1
        cutoff_idx = min(cutoff_idx, len(probs))

        # Select from nucleus
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()  # Renormalize

        # Sample
        idx = int(np.random.choice(len(nucleus_probs), p=nucleus_probs))
        token_id = int(nucleus_indices[idx])

        log_probs = np.log(probs + 1e-10)
        return token_id, float(log_probs[token_id])


class MinPSampler(Sampler):
    """Min-P sampling - sample from tokens with probability >= min_p * max_prob."""

    def __init__(self, min_p: float, temperature: float = 1.0):
        """Initialize min-p sampler.

        Args:
            min_p: Minimum probability threshold (relative to max).
            temperature: Sampling temperature.
        """
        if not 0 <= min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {min_p}")
        self.min_p = min_p
        self.temperature = temperature

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample from tokens above min probability threshold."""
        scaled_logits = logits / self.temperature if self.temperature != 1.0 else logits
        probs = _softmax(scaled_logits)

        # Find threshold
        max_prob = np.max(probs)
        threshold = max_prob * self.min_p

        # Filter tokens
        mask = probs >= threshold
        if not np.any(mask):
            # Fallback to greedy if no tokens pass
            token_id = int(np.argmax(probs))
        else:
            filtered_probs = np.where(mask, probs, 0)
            filtered_probs = filtered_probs / filtered_probs.sum()
            token_id = int(np.random.choice(len(filtered_probs), p=filtered_probs))

        log_probs = np.log(probs + 1e-10)
        return token_id, float(log_probs[token_id])


class BeamSearchSampler(Sampler):
    """Beam search decoding (not a true sampler, but follows the interface)."""

    def __init__(self, beam_width: int, length_penalty: float = 1.0):
        """Initialize beam search.

        Args:
            beam_width: Number of beams to maintain.
            length_penalty: Penalty for sequence length.
        """
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self._beams: List[Tuple[List[int], float]] = []

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Return best token from beam search.

        Note: This is a simplified version. Full beam search requires
        maintaining multiple sequences.
        """
        # For single-step, just return the best token
        token_id = int(np.argmax(logits))
        log_probs = logits - np.logaddexp.reduce(logits)
        return token_id, float(log_probs[token_id])


@dataclass
class SamplingStrategy:
    """Combined sampling strategy with penalties and constraints.

    This class combines multiple sampling methods with repetition penalties
    and other generation constraints.
    """

    temperature: float = 1.0
    top_k: int = -1  # -1 means disabled
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        """Set random seed if provided."""
        if self.seed is not None:
            np.random.seed(self.seed)

    def apply_penalties(
        self,
        logits: np.ndarray,
        token_ids: List[int],
    ) -> np.ndarray:
        """Apply repetition and frequency penalties to logits.

        Args:
            logits: Original logits.
            token_ids: Previously generated tokens.

        Returns:
            Modified logits with penalties applied.
        """
        if not token_ids:
            return logits

        logits = logits.copy()

        # Count token frequencies
        token_counts: Dict[int, int] = {}
        for tid in token_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for token_id, count in token_counts.items():
            if token_id >= len(logits):
                continue

            # Repetition penalty (multiplicative)
            if self.repetition_penalty != 1.0:
                if logits[token_id] > 0:
                    logits[token_id] /= self.repetition_penalty
                else:
                    logits[token_id] *= self.repetition_penalty

            # Presence penalty (additive, based on presence)
            if self.presence_penalty != 0:
                logits[token_id] -= self.presence_penalty

            # Frequency penalty (additive, based on count)
            if self.frequency_penalty != 0:
                logits[token_id] -= self.frequency_penalty * count

        return logits

    def sample(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample a token using the configured strategy.

        Args:
            logits: Log probabilities for vocabulary.
            token_ids: Previously generated tokens.

        Returns:
            Tuple of (token_id, log_probability).
        """
        # Apply penalties
        if token_ids:
            logits = self.apply_penalties(logits, token_ids)

        # Apply temperature
        if self.temperature == 0:
            # Greedy
            sampler = GreedySampler()
            return sampler(logits, token_ids)

        scaled_logits = logits / self.temperature
        probs = _softmax(scaled_logits)

        # Apply top-k
        if self.top_k > 0:
            top_k_indices = np.argpartition(probs, -self.top_k)[-self.top_k:]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_k_indices] = True
            probs = np.where(mask, probs, 0)

        # Apply top-p
        if self.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            cutoff_idx = int(np.searchsorted(cumsum, self.top_p)) + 1
            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_indices[:cutoff_idx]] = True
            probs = np.where(mask, probs, 0)

        # Apply min-p
        if self.min_p > 0:
            max_prob = np.max(probs)
            threshold = max_prob * self.min_p
            probs = np.where(probs >= threshold, probs, 0)

        # Renormalize
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            # Fallback to uniform over non-zero original probs
            probs = _softmax(logits)

        # Sample
        token_id = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[token_id] + 1e-10))

        return token_id, log_prob


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for array x."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum()


def create_sampler(
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    min_p: float = 0.0,
    **kwargs,
) -> SamplingStrategy:
    """Create a sampling strategy from parameters.

    Args:
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter (-1 to disable).
        top_p: Top-p (nucleus) sampling parameter.
        min_p: Min-p sampling parameter.
        **kwargs: Additional parameters (penalties, etc.).

    Returns:
        Configured SamplingStrategy.
    """
    return SamplingStrategy(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        **kwargs,
    )
