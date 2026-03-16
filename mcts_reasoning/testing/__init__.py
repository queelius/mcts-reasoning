"""
Testing utilities for MCTS-Reasoning.

Provides mock implementations of core interfaces for deterministic testing.
"""

from __future__ import annotations

from ..types import Continuation, Evaluation, State, extend_state


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.calls: list = []

    def generate(
        self,
        messages: list | str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    def is_available(self) -> bool:
        return True

    @classmethod
    def detect(cls, base_url: str | None = None) -> bool:
        return False

    def get_name(self) -> str:
        return "Mock"

    def get_provider_name(self) -> str:
        return "Mock"


class MockGenerator:
    """Mock generator for deterministic testing."""

    def __init__(
        self,
        responses: list[str] | None = None,
        terminal_at: int = 3,
    ):
        self.responses = responses or ["Thinking step..."]
        self.call_count = 0
        self.terminal_at = terminal_at

    def generate(
        self,
        question: str,
        state: str | State,
        n: int = 1,
    ) -> list[Continuation]:
        results = []
        for _ in range(n):
            text = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            is_terminal = self.call_count >= self.terminal_at
            answer = "42" if is_terminal else None
            if is_terminal:
                text = f"{text} ANSWER: 42"
            new_state = extend_state(State(state), text)
            results.append(
                Continuation(
                    text=new_state,
                    is_terminal=is_terminal,
                    answer=answer,
                )
            )
        return results

    def extract_answer(self, text: str) -> str | None:
        """Extract answer -- for backward compat with mcts.py."""
        from ..terminal import MarkerTerminalDetector

        result = MarkerTerminalDetector().is_terminal(text)
        return result.answer

    def is_terminal(self, text: str) -> bool:
        """Check terminal -- for backward compat with mcts.py."""
        from ..terminal import MarkerTerminalDetector

        result = MarkerTerminalDetector().is_terminal(text)
        return result.is_terminal


class MockEvaluator:
    """Mock evaluator for deterministic testing."""

    def __init__(self, score: float = 0.8):
        self.score = score
        self.call_count = 0

    def evaluate(
        self,
        question: str,
        state: str,
        answer: str,
    ) -> Evaluation:
        self.call_count += 1
        return Evaluation(score=self.score, explanation="mock")
