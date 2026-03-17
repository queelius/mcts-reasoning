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
            "You are a precise reasoning assistant that works ONE SMALL STEP at a time.\n\n"
            "CRITICAL RULES:\n"
            "- Write ONLY ONE reasoning step per response\n"
            "- Each step must be exactly 1-2 sentences\n"
            "- Do NOT solve the entire problem at once\n"
            "- Do NOT skip ahead to the answer unless this step's logic directly yields it\n"
            "- Make ONE deduction, test ONE assumption, or check ONE case per step\n"
            f"- {terminal_instruction}\n"
            "- Your answer after ANSWER: must be SHORT and DIRECT (1-5 words)\n"
            "- BAD: solving the whole problem in one step. GOOD: one careful deduction.\n"
        )

        if n == 1:
            user_content = (
                f"Question: {question}\n\n"
                f"Reasoning so far:\n{state}\n\n"
                f"What is the single next logical step? (1-2 sentences only, ONE deduction)"
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


class IncrementalReasoningPrompt(PromptStrategy):
    """Forces very small reasoning steps with explicit state tracking.

    Best performing strategy in experiments. Each response is limited to
    exactly one logical deduction or calculation, preventing models from
    solving entire problems in a single step.
    """

    def __init__(self, terminal_detector: TerminalDetector):
        self.terminal_detector = terminal_detector

    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        terminal_instruction = self.terminal_detector.format_instruction()
        system = (
            "You reason by taking tiny, careful steps. Each response contains:\n"
            "1. EXACTLY ONE logical deduction or calculation\n"
            "2. A brief statement of what you now know\n\n"
            "CONSTRAINTS:\n"
            "- Maximum 2 sentences per response\n"
            "- Each step must make progress toward the answer\n"
            "- Do not repeat reasoning already done\n"
            "- Do not try to solve the whole problem at once\n"
            f"- When you can definitively answer the original question: {terminal_instruction}\n"
            "- ANSWER: must respond to the ORIGINAL question, not a sub-question\n"
        )
        if n == 1:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Next single deduction:"
            )
        else:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Provide {n} different possible next deductions.\n"
                f"Format: --- CONTINUATION 1 ---\n[deduction]\n--- CONTINUATION 2 ---\n[deduction]"
            )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

    def parse(self, response: str, n: int = 1) -> list[str]:
        if n == 1:
            return [response]
        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)
        continuations = [p.strip() for p in parts if p.strip()]
        return continuations if continuations else [response]


class AssumptionTestingPrompt(PromptStrategy):
    """Explicitly asks the LLM to state and test assumptions.

    Each response follows ASSUME/THEN/CHECK pattern. Particularly effective
    for logic puzzles where case analysis is needed.
    """

    def __init__(self, terminal_detector: TerminalDetector):
        self.terminal_detector = terminal_detector

    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        terminal_instruction = self.terminal_detector.format_instruction()
        system = (
            "You solve problems by explicitly stating and testing assumptions.\n\n"
            "Each response must follow this pattern:\n"
            "- ASSUME: [state one specific assumption]\n"
            "- THEN: [derive one consequence from that assumption]\n"
            "- CHECK: [verify if the consequence is consistent or contradictory]\n\n"
            "If CHECK reveals a contradiction, say CONTRADICTION and try a different assumption next time.\n"
            "If CHECK is consistent AND you have enough to answer the original question:\n"
            f"{terminal_instruction}\n"
            "ANSWER: must respond to the ORIGINAL question directly (1-5 words).\n"
        )
        if n == 1:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Next assumption to test:"
            )
        else:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Provide {n} different assumptions to test.\n"
                f"Format: --- CONTINUATION 1 ---\n[assumption test]\n--- CONTINUATION 2 ---\n[assumption test]"
            )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

    def parse(self, response: str, n: int = 1) -> list[str]:
        if n == 1:
            return [response]
        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)
        continuations = [p.strip() for p in parts if p.strip()]
        return continuations if continuations else [response]


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
