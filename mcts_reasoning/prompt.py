"""
Prompt Strategy: Builds messages for LLM reasoning continuations.

Separates prompt construction from generation so that:
1. Prompts can be swapped independently of the generator
2. Few-shot examples can be injected via decorator pattern
3. Diverse generation uses the same strategy with n>1
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .terminal import TerminalDetector
from .types import Message, State


class PromptStrategy(ABC):
    """Abstract base class for prompt strategies."""

    @abstractmethod
    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        """
        Build messages for the LLM.

        Args:
            question: The original question being solved
            state: Current reasoning state
            n: Number of continuations to request (1=single, >1=diverse)

        Returns:
            List of Message dicts with role/content pairs
        """

    @abstractmethod
    def parse(self, response: str, n: int = 1) -> list[str]:
        """
        Extract continuation strings from the LLM response.

        Args:
            response: Raw LLM response text
            n: Number of continuations expected

        Returns:
            List of continuation strings
        """


class StepByStepPrompt(PromptStrategy):
    """
    Default prompt strategy that asks the LLM to reason step by step.

    For n=1, asks for a single clear next step.
    For n>1, asks for N different continuations separated by markers.
    """

    def __init__(self, terminal_detector: TerminalDetector):
        self.terminal_detector = terminal_detector

    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        terminal_instruction = self.terminal_detector.format_instruction()

        system_content = "You are solving a problem step by step."

        if n == 1:
            user_content = (
                f"Question: {question}\n\n"
                f"Reasoning so far:\n{state}\n\n"
                f"Continue the reasoning with ONE clear next step.\n"
                f"- Think carefully about what would be most helpful next\n"
                f"- {terminal_instruction}\n"
                f"- If not done, continue reasoning toward the solution\n\n"
                f"Your next step:"
            )
        else:
            user_content = (
                f"Question: {question}\n\n"
                f"Reasoning so far:\n{state}\n\n"
                f"Generate {n} DIFFERENT possible next steps. "
                f"Each should explore a meaningfully different approach or direction.\n\n"
                f"For each continuation:\n"
                f"- {terminal_instruction}\n"
                f"- Otherwise, show the next reasoning step\n\n"
                f"Format your response as:\n"
                f"--- CONTINUATION 1 ---\n"
                f"[your first continuation]\n\n"
                f"--- CONTINUATION 2 ---\n"
                f"[your second continuation]\n\n"
                f"(etc.)"
            )

        return [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]

    def parse(self, response: str, n: int = 1) -> list[str]:
        if n == 1:
            return [response]

        # Split on continuation markers
        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)
        continuations = [p.strip() for p in parts if p.strip()]

        # Fall back to whole response if markers not found
        if not continuations:
            return [response]

        return continuations


class StrictAnswerPrompt(PromptStrategy):
    """
    Stronger prompt strategy that enforces answer format.

    Key differences from StepByStepPrompt:
    - System prompt explicitly forbids hedging or deferring answers
    - Answer format is specified with examples
    - Shorter, more directive instructions
    """

    def __init__(self, terminal_detector: TerminalDetector):
        self.terminal_detector = terminal_detector

    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        terminal_instruction = self.terminal_detector.format_instruction()

        system_content = (
            "You are a precise reasoning assistant. "
            "You solve problems step by step. "
            "When you have enough information to answer, you MUST give a definitive answer immediately.\n\n"
            "RULES:\n"
            "- Each response is exactly ONE reasoning step (1-3 sentences)\n"
            "- Never say 'I need more information' or 'let me think more' as your answer\n"
            "- Never write 'ANSWER:' followed by a hedge or placeholder\n"
            f"- {terminal_instruction}\n"
            "- Your answer after ANSWER: must be a SHORT, DIRECT response (1-5 words)\n"
            "- Examples of good answers: 'ANSWER: A is a knight', 'ANSWER: 42', 'ANSWER: True'\n"
            "- Examples of BAD answers: 'ANSWER: Let me think...', 'ANSWER: The answer will be provided later'\n"
        )

        if n == 1:
            user_content = (
                f"Question: {question}\n\n"
                f"Reasoning so far:\n{state}\n\n"
                f"Write the next reasoning step (1-3 sentences). "
                f"If you can determine the answer, write it as {self.terminal_detector.format_instruction()}"
            )
        else:
            user_content = (
                f"Question: {question}\n\n"
                f"Reasoning so far:\n{state}\n\n"
                f"Generate {n} DIFFERENT possible next steps.\n"
                f"Format:\n"
                f"--- CONTINUATION 1 ---\n[step]\n\n"
                f"--- CONTINUATION 2 ---\n[step]\n"
            )

        return [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]

    def parse(self, response: str, n: int = 1) -> list[str]:
        if n == 1:
            return [response]

        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)
        continuations = [p.strip() for p in parts if p.strip()]

        if not continuations:
            return [response]

        return continuations


# ---------------------------------------------------------------------------
# Example / Few-shot support
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """A single problem/solution example for few-shot prompting."""

    problem: str
    solution: str
    reasoning: str | None = None


@runtime_checkable
class ExampleSource(Protocol):
    """Protocol for finding similar examples."""

    def find_similar(self, question: str, k: int) -> list[Example]: ...


class StaticExampleSource:
    """Returns the first k examples from a fixed list."""

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def find_similar(self, question: str, k: int) -> list[Example]:
        return self.examples[:k]


class FewShotPrompt(PromptStrategy):
    """
    Decorator that prepends few-shot examples to any base PromptStrategy.

    Examples are retrieved from an ExampleSource and inserted as
    user/assistant message pairs before the base strategy's messages.
    """

    def __init__(
        self,
        base: PromptStrategy,
        examples: ExampleSource,
        k: int = 3,
    ):
        self.base = base
        self.examples = examples
        self.k = k

    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        similar = self.examples.find_similar(question, self.k)
        base_messages = self.base.format(question, state, n)

        # Prepend examples as user/assistant pairs
        example_messages: list[Message] = []
        for ex in similar:
            example_messages.append(
                Message(role="user", content=f"Problem: {ex.problem}")
            )
            solution_text = (
                ex.reasoning + "\n" + ex.solution if ex.reasoning else ex.solution
            )
            example_messages.append(Message(role="assistant", content=solution_text))

        return example_messages + base_messages

    def parse(self, response: str, n: int = 1) -> list[str]:
        return self.base.parse(response, n)
