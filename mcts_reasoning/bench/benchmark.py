"""
Benchmark: Abstract base and Problem dataclass.

A Benchmark provides a collection of Problems to evaluate a Solver against.
Problems carry all metadata needed to measure accuracy by domain/difficulty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Problem:
    """A single benchmark problem with ground-truth answer."""

    question: str
    ground_truth: str
    domain: str = "general"
    difficulty: str = "medium"  # easy | medium | hard
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    @abstractmethod
    def problems(self) -> list[Problem]:
        """Return all problems in this benchmark."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""
