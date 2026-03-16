"""
Solver: Abstractions for solving benchmark problems.

BaselineSolver: single-pass LLM call (no search).
MCTSSolver: full MCTS search.

Both accept a Problem and return a SolverResult.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .benchmark import Problem


@dataclass
class SolverResult:
    """Result of running a Solver on a single Problem."""

    answer: Optional[str]
    correct: bool
    score: float
    time_seconds: float
    metadata: dict = field(default_factory=dict)


class Solver(ABC):
    """Abstract base class for problem solvers."""

    @abstractmethod
    def solve(self, problem: Problem) -> SolverResult:
        """Solve a problem, returning a SolverResult."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name."""


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------


def _answers_match(answer: Optional[str], ground_truth: str) -> bool:
    """Case-insensitive, stripped string equality check."""
    if answer is None:
        return False
    return answer.strip().lower() == ground_truth.strip().lower()


def _extract_answer_from_text(text: str) -> Optional[str]:
    """Extract the text after ANSWER: marker if present."""
    marker = "ANSWER:"
    idx = text.upper().find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker) :].strip()


# ---------------------------------------------------------------------------
# Concrete Solvers
# ---------------------------------------------------------------------------


class BaselineSolver(Solver):
    """
    Single-pass LLM call (no tree search).

    Calls the provider once and checks whether the response contains an
    ANSWER: marker.  If not, the raw response is used as-is.
    """

    def __init__(self, provider, prompt_strategy=None):
        """
        Args:
            provider: LLMProvider instance with .generate() method.
            prompt_strategy: PromptStrategy to build messages.  Defaults to
                StepByStepPrompt with MarkerTerminalDetector.
        """
        self._provider = provider
        self._prompt_strategy = prompt_strategy or self._default_prompt_strategy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_prompt_strategy():
        from ..prompt import StepByStepPrompt
        from ..terminal import MarkerTerminalDetector

        return StepByStepPrompt(terminal_detector=MarkerTerminalDetector())

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"baseline-{self._provider.get_name()}"

    def solve(self, problem: Problem) -> SolverResult:
        from ..types import State

        start = time.time()
        messages = self._prompt_strategy.format(problem.question, State(""), n=1)
        response = self._provider.generate(messages)
        continuations = self._prompt_strategy.parse(response, n=1)
        raw = continuations[0] if continuations else response

        answer = _extract_answer_from_text(raw) or raw.strip() or None
        elapsed = time.time() - start
        correct = _answers_match(answer, problem.ground_truth)
        return SolverResult(
            answer=answer,
            correct=correct,
            score=1.0 if correct else 0.0,
            time_seconds=elapsed,
        )


class MCTSSolver(Solver):
    """
    Full MCTS search.

    Uses LLMGenerator + MCTS.search(), returning the highest-scored terminal
    state as the answer.
    """

    def __init__(
        self,
        provider,
        prompt_strategy=None,
        evaluator=None,
        terminal_detector=None,
        simulations: int = 10,
        exploration_constant: float = 1.414,
        max_children_per_node: int = 3,
        max_rollout_depth: int = 5,
    ):
        """
        Args:
            provider: LLMProvider instance.
            prompt_strategy: PromptStrategy (default: StepByStepPrompt).
            evaluator: Evaluator instance (default: MockEvaluator with score=0.8).
            terminal_detector: TerminalDetector (default: MarkerTerminalDetector).
            simulations: Number of MCTS simulations.
            exploration_constant: UCB1 exploration constant.
            max_children_per_node: Branching factor.
            max_rollout_depth: Maximum rollout depth.
        """
        self._provider = provider
        self._prompt_strategy = prompt_strategy
        self._evaluator = evaluator
        self._terminal_detector = terminal_detector
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.max_children_per_node = max_children_per_node
        self.max_rollout_depth = max_rollout_depth

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"mcts-{self.simulations}sim"

    def solve(self, problem: Problem) -> SolverResult:
        from ..evaluator import MockEvaluator
        from ..generator import LLMGenerator
        from ..mcts import MCTS
        from ..terminal import MarkerTerminalDetector

        terminal_detector = self._terminal_detector or MarkerTerminalDetector()
        evaluator = self._evaluator or MockEvaluator()

        gen = LLMGenerator(
            provider=self._provider,
            prompt_strategy=self._prompt_strategy,
            terminal_detector=terminal_detector,
        )
        mcts = MCTS(
            generator=gen,
            evaluator=evaluator,
            terminal_detector=terminal_detector,
            exploration_constant=self.exploration_constant,
            max_children_per_node=self.max_children_per_node,
            max_rollout_depth=self.max_rollout_depth,
        )

        start = time.time()
        search_state = mcts.search(problem.question, simulations=self.simulations)
        elapsed = time.time() - start

        terminals = search_state.terminal_states
        best = max(terminals, key=lambda t: t["score"]) if terminals else None
        answer = best["answer"] if best else None
        score = best["score"] if best else 0.0
        correct = _answers_match(answer, problem.ground_truth)

        return SolverResult(
            answer=answer,
            correct=correct,
            score=score,
            time_seconds=elapsed,
            metadata={
                "simulations": self.simulations,
                "terminals": len(terminals),
            },
        )
