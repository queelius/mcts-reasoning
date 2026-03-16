"""
PromptOptimizer: Abstract base for prompt optimization (v0.7).

Placeholder that defines the interface; concrete implementations will search
for better PromptStrategy configurations given a budget of evaluations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PromptOptimizer(ABC):
    """
    Abstract base class for prompt optimizers.

    v0.7 feature: search for a better PromptStrategy for a given benchmark
    within a fixed evaluation budget.
    """

    @abstractmethod
    def optimize(self, base_strategy, benchmark, solver_factory, budget: int):
        """
        Search for an improved PromptStrategy.

        Args:
            base_strategy: Starting PromptStrategy to improve from.
            benchmark: Benchmark to evaluate candidate strategies against.
            solver_factory: Callable(strategy) -> Solver used to evaluate
                candidate strategies.
            budget: Maximum number of solver evaluations allowed.

        Returns:
            A (PromptStrategy, BenchReport) tuple with the best strategy found
            and its evaluation report.
        """
