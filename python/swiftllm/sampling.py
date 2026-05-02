# ==============================================================================
# PROJECT:   SWIFTLLM
# FILE:      sampling.py
# PATH:      /python/swiftllm/sampling.py
# AUTHOR:    Peter A. Aldrich Jr.
# DATE:      2026
# ------------------------------------------------------------------------------
# USES:
#   - python/swiftllm/config.py   SelfConsistencyConfig, AnswerExtractor
# USED BY:
#   - python/swiftllm/__init__.py   SelfConsistencySampler re-export
#   - python/swiftllm/engine.py     generate_with_self_consistency()
# SEE ALSO:
#   - crates/swiftllm-core/src/sampling/self_consistency.rs  Rust implementation
#   - crates/swiftllm-core/src/sampling/mod.rs               Rust module root
#   - crates/swiftllm-core/src/inference/verification.rs     downstream verifier
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

"""SwiftLLM Sampling Strategies

This module provides various sampling strategies for token generation,
including the Phase 3 self-consistency majority voting sampler.
"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import AnswerExtractor, SelfConsistencyConfig


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
        # Numerically stable log softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
        log_probs = _log_softmax(logits)
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
        log_probs = _log_softmax(scaled_logits)
        probs = np.exp(log_probs)
        token_id = int(np.random.choice(len(probs), p=probs))
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

        # Log probability from renormalized top-k distribution
        top_k_log_probs = _log_softmax(top_k_logits)
        return token_id, float(top_k_log_probs[idx])


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
        log_probs = _log_softmax(scaled_logits)
        probs = np.exp(log_probs)

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
        log_probs = _log_softmax(scaled_logits)
        probs = np.exp(log_probs)

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

        return token_id, float(log_probs[token_id])


class BeamSearchSampler(Sampler):
    """Beam search decoding with persistent beam state across calls."""

    def __init__(self, beam_width: int, length_penalty: float = 1.0):
        """Initialize beam search.

        Args:
            beam_width: Number of beams to maintain.
            length_penalty: Penalty for sequence length.
        """
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        # Each beam: (token_ids, cumulative_log_prob)
        self._beams: List[Tuple[List[int], float]] = []
        self._step: int = 0

    def reset(self):
        """Reset beam state for a new sequence."""
        self._beams = []
        self._step = 0

    def __call__(
        self,
        logits: np.ndarray,
        token_ids: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Advance beam search by one step and return the best beam's latest token.

        Maintains beam state across calls. Call reset() between sequences.
        """
        log_probs = _log_softmax(logits)
        self._step += 1

        if not self._beams:
            # First step — seed beams from top-k of initial logits
            top_indices = np.argpartition(log_probs, -self.beam_width)[-self.beam_width:]
            self._beams = [
                ([int(idx)], float(log_probs[idx]))
                for idx in top_indices
            ]
        else:
            # Expand each beam with all vocab tokens, keep top beam_width overall
            candidates: List[Tuple[List[int], float]] = []
            for beam_tokens, beam_score in self._beams:
                for idx in np.argpartition(log_probs, -self.beam_width)[-self.beam_width:]:
                    idx = int(idx)
                    new_score = beam_score + float(log_probs[idx])
                    # Length-normalized score
                    length = len(beam_tokens) + 1
                    norm_score = new_score / (length ** self.length_penalty)
                    candidates.append((beam_tokens + [idx], new_score))

            # Keep top beam_width by normalized score
            candidates.sort(
                key=lambda c: c[1] / (len(c[0]) ** self.length_penalty),
                reverse=True,
            )
            self._beams = candidates[:self.beam_width]

        # Return the latest token of the best beam
        best_beam = self._beams[0]
        return best_beam[0][-1], best_beam[1] / (len(best_beam[0]) ** self.length_penalty)


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
        """Initialize per-instance RNG to avoid mutating global state."""
        self._rng = np.random.default_rng(self.seed)

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
        token_id = int(self._rng.choice(len(probs), p=probs))
        # Use log of the filtered+renormalized probability (avoids log(0))
        log_prob = float(np.log(probs[token_id])) if probs[token_id] > 0 else float('-inf')

        return token_id, log_prob


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for array x (numerically stable)."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum()


def _log_softmax(x: np.ndarray) -> np.ndarray:
    """Compute log-softmax (numerically stable, avoids log(0))."""
    x_max = np.max(x)
    shifted = x - x_max
    return shifted - np.log(np.sum(np.exp(shifted)))


@dataclass
class SelfConsistencyResult:
    """Result from SelfConsistencySampler.

    Attributes:
        answer: The plurality-majority answer string.
        vote_fraction: Fraction of samples that agreed on the majority answer.
        all_answers: All extracted answer strings (one per sample).
        raw_outputs: Raw generated texts before answer extraction.
    """
    answer: Optional[str]
    vote_fraction: float
    all_answers: List[Optional[str]]
    raw_outputs: List[str]


class SelfConsistencySampler:
    """Self-consistency majority voting over multiple independent generations.

    Generates ``config.num_samples`` reasoning chains from the same prompt,
    extracts an answer from each using the configured ``AnswerExtractor``,
    and returns the plurality-majority answer.  Ties are broken by the
    average log-probability of samples that produced each candidate answer.

    This mirrors ``self_consistency_vote()`` in
    ``crates/swiftllm-core/src/sampling/self_consistency.rs``.

    Example::

        from swiftllm.sampling import SelfConsistencySampler
        from swiftllm.config import SelfConsistencyConfig, AnswerExtractor

        cfg = SelfConsistencyConfig(
            num_samples=8,
            extractor=AnswerExtractor.HEURISTIC,
            temperature=0.8,
        )
        sampler = SelfConsistencySampler(cfg)
        result = sampler.vote(raw_outputs=["...chain 1...", "...chain 2...", ...])
        print(result.answer, result.vote_fraction)
    """

    def __init__(self, config: SelfConsistencyConfig):
        self.config = config

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from a single generated text.

        Args:
            text: The full generated reasoning chain.

        Returns:
            The extracted answer string, or None if extraction failed.
        """
        ext = self.config.extractor
        if ext == AnswerExtractor.HEURISTIC:
            return self._heuristic_extract(text)
        elif ext == AnswerExtractor.AFTER_SENTINEL:
            return self._sentinel_extract(text, self.config.answer_sentinel)
        elif ext == AnswerExtractor.LAST_LINE:
            return self._last_line_extract(text)
        elif ext == AnswerExtractor.XML_TAG:
            return self._xml_tag_extract(text, self.config.answer_tag)
        return None

    def vote(
        self,
        raw_outputs: List[str],
        log_probs: Optional[List[float]] = None,
    ) -> SelfConsistencyResult:
        """Apply majority voting over a list of generated texts.

        Args:
            raw_outputs: Generated texts (one per sample).
            log_probs: Cumulative log-probabilities per sample for tiebreaking.
                If None, ties are broken by index (first encountered wins).

        Returns:
            SelfConsistencyResult with the majority answer and vote statistics.
        """
        all_answers = [self.extract_answer(t) for t in raw_outputs]
        non_null = [a for a in all_answers if a is not None]

        if not non_null:
            return SelfConsistencyResult(
                answer=None,
                vote_fraction=0.0,
                all_answers=all_answers,
                raw_outputs=raw_outputs,
            )

        # Count votes
        counts = Counter(non_null)
        max_count = max(counts.values())
        candidates = [a for a, c in counts.items() if c == max_count]

        # Tiebreak by mean log-prob of samples that produced each candidate
        if len(candidates) > 1 and log_probs is not None:
            best_answer = None
            best_score = float("-inf")
            for cand in candidates:
                scores = [
                    log_probs[i]
                    for i, a in enumerate(all_answers)
                    if a == cand and log_probs[i] is not None
                ]
                mean_score = sum(scores) / len(scores) if scores else float("-inf")
                if mean_score > best_score:
                    best_score = mean_score
                    best_answer = cand
        else:
            best_answer = candidates[0]

        return SelfConsistencyResult(
            answer=best_answer,
            vote_fraction=max_count / len(non_null),
            all_answers=all_answers,
            raw_outputs=raw_outputs,
        )

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Strip, lowercase, remove trailing punctuation."""
        return text.strip().lower().rstrip(".,;:!?")

    def _heuristic_extract(self, text: str) -> Optional[str]:
        """Look for a number or boxed expression near the end of the text."""
        import re
        # Try \\boxed{...}
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return self._normalise(boxed[-1])
        # Look for "the answer is <X>" (case-insensitive)
        sentinel_match = re.search(
            r"the answer is\s+([^\s.,;:!?\n]+)", text, re.IGNORECASE
        )
        if sentinel_match:
            return self._normalise(sentinel_match.group(1))
        # Fall back to last number
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return self._normalise(numbers[-1]) if numbers else None

    def _sentinel_extract(self, text: str, sentinel: str) -> Optional[str]:
        """Return the text that follows the sentinel string."""
        lower = text.lower()
        idx = lower.rfind(sentinel.lower())
        if idx == -1:
            return None
        remainder = text[idx + len(sentinel):].strip()
        # Take up to the first newline or sentence end
        import re
        match = re.match(r"([^\n.!?]+)", remainder)
        return self._normalise(match.group(1)) if match else self._normalise(remainder[:80])

    @staticmethod
    def _last_line_extract(text: str) -> Optional[str]:
        """Return the last non-empty line."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def _xml_tag_extract(text: str, tag: str) -> Optional[str]:
        """Return content inside <tag>…</tag>."""
        import re
        matches = re.findall(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
        return matches[-1].strip() if matches else None


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

# ------------------------------------------------------------------------------
# END OF FILE: sampling.py
# REPO PATH:   /swiftllm/python/swiftllm/sampling.py
# INTEGRATES:  config.py · engine.py · __init__.py
#              Rust: self_consistency.rs · strategies.rs · mod.rs
# (c) 2026 SWIFTLLM | Apache 2.0 License
# ------------------------------------------------------------------------------
