"""
Consensus: Self-consistency voting strategies.

Provides ABC-based strategies for selecting the best answer from multiple
sampled reasoning paths. Implements majority vote and value-weighted vote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import SampledPath, ConsensusResult


class ConsensusStrategy(ABC):
    """Abstract base class for consensus/voting strategies."""

    @abstractmethod
    def vote(self, paths: list[SampledPath]) -> ConsensusResult:
        """
        Select the best answer from a list of sampled paths.

        Args:
            paths: Sampled reasoning paths with answers and values.

        Returns:
            ConsensusResult with the winning answer and confidence.
        """


class MajorityVote(ConsensusStrategy):
    """Simple majority voting -- each path gets one vote."""

    def vote(self, paths: list[SampledPath]) -> ConsensusResult:
        counts: dict[str, int] = {}
        for p in paths:
            if p.answer:
                counts[p.answer] = counts.get(p.answer, 0) + 1
        if not counts:
            return ConsensusResult(
                answer="",
                confidence=0.0,
                distribution={},
                paths_used=len(paths),
            )
        winner = max(counts, key=counts.get)  # type: ignore[arg-type]
        total = sum(counts.values())
        return ConsensusResult(
            answer=winner,
            confidence=counts[winner] / total,
            distribution=counts,
            paths_used=len(paths),
        )


class WeightedVote(ConsensusStrategy):
    """Value-weighted voting -- paths contribute their value as weight."""

    def vote(self, paths: list[SampledPath]) -> ConsensusResult:
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for p in paths:
            if p.answer:
                scores[p.answer] = scores.get(p.answer, 0.0) + p.value
                counts[p.answer] = counts.get(p.answer, 0) + 1
        if not scores:
            return ConsensusResult(
                answer="",
                confidence=0.0,
                distribution={},
                paths_used=len(paths),
            )
        winner = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        return ConsensusResult(
            answer=winner,
            confidence=scores[winner] / total if total > 0 else 0.0,
            distribution=counts,
            paths_used=len(paths),
        )


__all__ = [
    "ConsensusStrategy",
    "MajorityVote",
    "WeightedVote",
]
