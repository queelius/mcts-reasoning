"""
Reasoning Actions: the action space for MCTS.

Each action is a different type of reasoning move the LLM can make.
Instead of one prompt strategy for the whole search, the LLM gets a
different prompt depending on which action MCTS selects. This means
the tree branches represent genuinely different reasoning strategies,
not just random LLM variation.

The actions constrain reasoning granularity naturally (each action type
produces a focused response), so no artificial token limits are needed.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .terminal import TerminalDetector, MarkerTerminalDetector
from .types import Continuation, Message, State, extend_state
from .generator import Generator


class ReasoningAction(ABC):
    """A type of reasoning move the LLM can make at a tree node."""

    @abstractmethod
    def prompt(self, question: str, state: State) -> list[Message]:
        """Build the prompt for this action type."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this action (e.g., 'deduce', 'assume')."""


class DecomposeAction(ReasoningAction):
    """Break the problem into sub-problems."""

    @property
    def name(self) -> str:
        return "decompose"

    def prompt(self, question: str, state: State) -> list[Message]:
        return [
            Message(role="system", content=(
                "You break problems into smaller, manageable sub-problems. "
                "List the key things that need to be figured out, in order. "
                "Be specific and concise."
            )),
            Message(role="user", content=(
                f"Problem: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"What are the key sub-problems to solve? List them briefly."
            )),
        ]


class DeduceAction(ReasoningAction):
    """Make one logical deduction from what is known."""

    @property
    def name(self) -> str:
        return "deduce"

    def prompt(self, question: str, state: State) -> list[Message]:
        return [
            Message(role="system", content=(
                "You make careful logical deductions. "
                "Given the current state of reasoning, derive exactly ONE new conclusion. "
                "State what you conclude and why."
            )),
            Message(role="user", content=(
                f"Problem: {question}\n\n"
                f"What we know so far:\n{state}\n\n"
                f"What is one thing that logically follows from what we know?"
            )),
        ]


class AssumeAction(ReasoningAction):
    """State a specific assumption and derive its consequences."""

    @property
    def name(self) -> str:
        return "assume"

    def prompt(self, question: str, state: State) -> list[Message]:
        return [
            Message(role="system", content=(
                "You test assumptions by stating them explicitly and deriving consequences. "
                "Pick ONE specific assumption to test. State it clearly, then derive "
                "what would follow if it were true. Note any contradictions."
            )),
            Message(role="user", content=(
                f"Problem: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"State one assumption to test and derive its consequences."
            )),
        ]


class VerifyAction(ReasoningAction):
    """Check the current reasoning for contradictions or errors."""

    @property
    def name(self) -> str:
        return "verify"

    def prompt(self, question: str, state: State) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are a careful checker. Review the reasoning so far and look for: "
                "contradictions, invalid steps, unstated assumptions, or arithmetic errors. "
                "If everything checks out, say so. If you find a problem, identify it precisely."
            )),
            Message(role="user", content=(
                f"Problem: {question}\n\n"
                f"Reasoning to check:\n{state}\n\n"
                f"Is this reasoning correct so far? Check for errors."
            )),
        ]


class CalculateAction(ReasoningAction):
    """Perform one arithmetic or algebraic calculation step."""

    @property
    def name(self) -> str:
        return "calculate"

    def prompt(self, question: str, state: State) -> list[Message]:
        return [
            Message(role="system", content=(
                "You perform careful calculations, one step at a time. "
                "Show your work clearly. Do ONE calculation, not the whole problem."
            )),
            Message(role="user", content=(
                f"Problem: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Perform the next calculation step."
            )),
        ]


class ConcludeAction(ReasoningAction):
    """State the final answer based on the reasoning so far."""

    def __init__(self, terminal_detector: Optional[TerminalDetector] = None):
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    @property
    def name(self) -> str:
        return "conclude"

    def prompt(self, question: str, state: State) -> list[Message]:
        instruction = self.terminal_detector.format_instruction()
        return [
            Message(role="system", content=(
                "Based on the reasoning so far, determine if you can give a definitive answer "
                "to the ORIGINAL question (not a sub-question). "
                "If the reasoning is sufficient, state your answer clearly. "
                f"{instruction}\n\n"
                "CRITICAL: If you do NOT have enough reasoning to answer definitively, "
                "do NOT write 'ANSWER:'. Instead, continue reasoning with the next logical step. "
                "Only write ANSWER: when you are CERTAIN of the answer."
            )),
            Message(role="user", content=(
                f"Original question: {question}\n\n"
                f"Full reasoning:\n{state}\n\n"
                f"Can you answer the original question definitively? "
                f"If yes, give the answer. If not, take the next reasoning step instead."
            )),
        ]


# ─── Pre-built action sets ──────────────────────────────────────

def logic_actions(terminal_detector: Optional[TerminalDetector] = None) -> list[ReasoningAction]:
    """Action set for logic puzzles (knights/knaves, etc.)."""
    return [
        AssumeAction(),
        DeduceAction(),
        VerifyAction(),
        ConcludeAction(terminal_detector),
    ]


def math_actions(terminal_detector: Optional[TerminalDetector] = None) -> list[ReasoningAction]:
    """Action set for math/arithmetic problems."""
    return [
        DecomposeAction(),
        CalculateAction(),
        VerifyAction(),
        ConcludeAction(terminal_detector),
    ]


def general_actions(terminal_detector: Optional[TerminalDetector] = None) -> list[ReasoningAction]:
    """General-purpose action set."""
    return [
        DecomposeAction(),
        DeduceAction(),
        AssumeAction(),
        CalculateAction(),
        VerifyAction(),
        ConcludeAction(terminal_detector),
    ]


# ─── Action-Aware Generator ─────────────────────────────────────

class ActionGenerator(Generator):
    """Generator that selects reasoning actions from an action space.

    Instead of one fixed prompt strategy, this generator chooses
    which type of reasoning move to make at each step. MCTS explores
    different sequences of actions (ASSUME -> DEDUCE -> VERIFY vs
    CALCULATE -> VERIFY -> CONCLUDE, etc.).

    Action selection: random by default. The MCTS tree's UCB1 naturally
    favors branches (action sequences) that lead to good outcomes.
    """

    def __init__(
        self,
        provider,
        actions: list[ReasoningAction],
        terminal_detector: Optional[TerminalDetector] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.actions = actions
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _select_action(self, state: str) -> ReasoningAction:
        """Select an action, weighting CONCLUDE lower for shallow reasoning."""
        # Count how many reasoning steps have been done (rough heuristic: count action tags)
        depth = state.count("[deduce]") + state.count("[assume]") + state.count("[calculate]") + state.count("[decompose]") + state.count("[verify]")

        weights = []
        for action in self.actions:
            if action.name == "conclude":
                # Low probability early, increasing with depth
                weights.append(min(depth * 0.2, 1.0))
            else:
                weights.append(1.0)

        # Ensure at least some weight on conclude
        total = sum(weights)
        if total == 0:
            return random.choice(self.actions)

        return random.choices(self.actions, weights=weights, k=1)[0]

    def generate(self, question: str, state: str, n: int = 1) -> list[Continuation]:
        """Generate n continuations, each using a selected action."""
        results = []
        selected_actions = [self._select_action(state) for _ in range(n)]

        for action in selected_actions:
            messages = action.prompt(question, State(state))
            response = self.provider.generate(
                messages, self.max_tokens, self.temperature,
            )

            # Check for terminal state
            check = self.terminal_detector.is_terminal(response)
            # Tag the continuation with which action produced it
            tagged_response = f"[{action.name}] {response}"
            new_state = extend_state(State(state), tagged_response)

            results.append(Continuation(
                text=new_state,
                is_terminal=check.is_terminal,
                answer=check.answer,
            ))

        return results
