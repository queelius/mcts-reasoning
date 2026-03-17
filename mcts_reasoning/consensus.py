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


class NormalizedVote(ConsensusStrategy):
    """Majority vote with LLM-based answer normalization.

    Before voting, sends all unique answers to an LLM and asks it to
    group semantically equivalent answers under a canonical form.
    E.g., "Knight", "KNIGHT", "A is a knight" all become "A is a knight".
    """

    def __init__(self, provider):
        """
        Args:
            provider: An LLMProvider used for normalization.
        """
        self.provider = provider

    def _normalize_answers(self, answers: list[str]) -> dict[str, str]:
        """Ask the LLM to map each answer to a canonical form.

        Returns dict mapping original -> canonical.
        """
        if not answers:
            return {}

        unique = sorted(set(answers))
        if len(unique) == 1:
            return {unique[0]: unique[0]}

        from .types import Message

        prompt_text = (
            "I have multiple answers to the same question. "
            "Some may be semantically equivalent but worded differently.\n\n"
            "Answers:\n"
            + "\n".join(f"- {a}" for a in unique)
            + "\n\n"
            "Group equivalent answers and pick one canonical form for each group. "
            "Reply with ONLY lines of the format:\n"
            "original -> canonical\n\n"
            "Example:\n"
            "Knight -> A is a knight\n"
            "KNIGHT -> A is a knight\n"
            "A is a knight -> A is a knight\n"
            "A is a knave -> A is a knave\n"
        )

        try:
            response = self.provider.generate(
                [Message(role="user", content=prompt_text)],
                max_tokens=500,
                temperature=0.0,
            )
            mapping = {}
            for line in response.strip().split("\n"):
                if "->" in line:
                    parts = line.split("->", 1)
                    original = parts[0].strip()
                    canonical = parts[1].strip()
                    mapping[original] = canonical
            # Fill in any answers the LLM missed
            for a in unique:
                if a not in mapping:
                    mapping[a] = a
            return mapping
        except Exception:
            # Fallback: no normalization
            return {a: a for a in unique}

    def vote(self, paths: list[SampledPath]) -> ConsensusResult:
        raw_answers = [p.answer for p in paths if p.answer]
        mapping = self._normalize_answers(raw_answers)

        counts: dict[str, int] = {}
        for p in paths:
            if p.answer:
                canonical = mapping.get(p.answer, p.answer)
                counts[canonical] = counts.get(canonical, 0) + 1

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


__all__ = [
    "ConsensusStrategy",
    "MajorityVote",
    "WeightedVote",
    "NormalizedVote",
]
