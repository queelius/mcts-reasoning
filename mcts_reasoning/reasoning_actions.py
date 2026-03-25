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
from typing import Optional, TYPE_CHECKING

from .terminal import TerminalDetector, MarkerTerminalDetector
from .types import Continuation, Message, State, extend_state
from .generator import Generator

if TYPE_CHECKING:
    from .node import Node


class ReasoningAction(ABC):
    """A type of reasoning move the LLM can make at a tree node."""

    @abstractmethod
    def prompt(self, question: str, state: State, node: "Node | None" = None) -> list[Message]:
        """Build the prompt for this action type.

        Args:
            question: The original question.
            state: The current reasoning state (linear path from root).
            node: The current Node in the tree. Actions that need structural
                  context (siblings, ancestors, depth) can inspect this.
                  None for backward compat with callers that don't have a node.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this action (e.g., 'deduce', 'assume')."""


class DecomposeAction(ReasoningAction):
    """Break the problem into sub-problems."""

    @property
    def name(self) -> str:
        return "decompose"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are solving a problem. Think about what sub-problems need to be solved. "
                "Write in first person as if you are working through this yourself."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My work so far:\n{state}\n\n"
                f"Let me break this into sub-problems:"
            )),
        ]


class DeduceAction(ReasoningAction):
    """Make one logical deduction from what is known."""

    @property
    def name(self) -> str:
        return "deduce"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are solving a problem step by step. "
                "Make exactly ONE logical deduction from what you know so far. "
                "Write in first person."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My reasoning so far:\n{state}\n\n"
                f"From what I know, I can deduce:"
            )),
        ]


class AssumeAction(ReasoningAction):
    """State a specific assumption and derive its consequences."""

    @property
    def name(self) -> str:
        return "assume"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are solving a problem by testing assumptions. "
                "Pick ONE specific assumption, state it, and work out what follows. "
                "If you hit a contradiction, say so clearly. Write in first person."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My work so far:\n{state}\n\n"
                f"Let me assume:"
            )),
        ]


class VerifyAction(ReasoningAction):
    """Check the current reasoning for contradictions or errors."""

    @property
    def name(self) -> str:
        return "verify"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are checking your own reasoning for mistakes. "
                "Look for contradictions, invalid steps, or arithmetic errors. "
                "Write in first person. Be honest about any problems you find."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My reasoning so far:\n{state}\n\n"
                f"Let me check my work:"
            )),
        ]


class CalculateAction(ReasoningAction):
    """Perform one arithmetic or algebraic calculation step."""

    @property
    def name(self) -> str:
        return "calculate"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are doing a calculation. Perform ONE step, show your work. "
                "Write in first person."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My work so far:\n{state}\n\n"
                f"Next calculation:"
            )),
        ]


class ConcludeAction(ReasoningAction):
    """State the final answer based on the reasoning so far."""

    def __init__(self, terminal_detector: Optional[TerminalDetector] = None):
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    @property
    def name(self) -> str:
        return "conclude"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        instruction = self.terminal_detector.format_instruction()
        return [
            Message(role="system", content=(
                "You are solving a problem. Based on your reasoning, decide if you have "
                "enough to give a definitive answer to the ORIGINAL question. "
                f"{instruction}\n\n"
                "CRITICAL: Only write ANSWER: if you are CERTAIN. "
                "If not ready, continue reasoning instead. Write in first person."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My reasoning so far:\n{state}\n\n"
                f"Do I have enough to answer? If yes, I conclude:"
            )),
        ]


# ─── Meta-actions (need tree context) ────────────────────────────

class SummarizeAction(ReasoningAction):
    """Condense the reasoning chain so far into a compact state.

    Reads the full ancestor chain and produces a summary that preserves
    key conclusions while reducing text. Useful for long chains where
    the accumulated state is getting noisy.
    """

    @property
    def name(self) -> str:
        return "summarize"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        return [
            Message(role="system", content=(
                "You are solving a problem. Summarize what you have established so far. "
                "Keep all conclusions, contradictions found, and key results. "
                "Drop repetition and verbose explanations. Write in first person."
            )),
            Message(role="user", content=(
                f"I need to solve: {question}\n\n"
                f"My full reasoning so far:\n{state}\n\n"
                f"Let me summarize what I know:"
            )),
        ]


class CompareAction(ReasoningAction):
    """Compare sibling branches to find agreement and disagreement.

    Needs the Node to inspect siblings. If no siblings exist or no node
    is provided, falls back to reviewing the current path.
    """

    @property
    def name(self) -> str:
        return "compare"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        # Collect sibling summaries if we have the node
        sibling_info = ""
        if node and node.parent:
            siblings = [ch for ch in node.parent.children if ch is not node]
            if siblings:
                sibling_texts = []
                for i, sib in enumerate(siblings):
                    # Get the last meaningful line of each sibling's state
                    lines = str(sib.state).split("\n")
                    meaningful = [l.strip() for l in lines if l.strip()][-3:]
                    snippet = " ".join(meaningful)[:200]
                    answer = f" (concluded: {sib.answer})" if sib.answer else ""
                    sibling_texts.append(f"Branch {i+1} (v={sib.value:.2f}, n={sib.visits}){answer}: {snippet}")
                sibling_info = "\n".join(sibling_texts)

        if sibling_info:
            return [
                Message(role="system", content=(
                    "You are solving a problem and have explored multiple approaches. "
                    "Compare them to see what they agree on and where they diverge. "
                    "Write in first person."
                )),
                Message(role="user", content=(
                    f"I need to solve: {question}\n\n"
                    f"My current approach:\n{state}\n\n"
                    f"Other approaches I tried:\n{sibling_info}\n\n"
                    f"Comparing these approaches, I notice:"
                )),
            ]
        else:
            return [
                Message(role="system", content=(
                    "You are solving a problem. Identify what you are most confident about "
                    "and what remains uncertain. Write in first person."
                )),
                Message(role="user", content=(
                    f"I need to solve: {question}\n\n"
                    f"My reasoning:\n{state}\n\n"
                    f"What am I most confident about? What is still uncertain?"
                )),
            ]


class RefineAction(ReasoningAction):
    """Take an existing answer and improve it with more careful reasoning.

    Useful when a branch has reached an answer but with low confidence.
    """

    @property
    def name(self) -> str:
        return "refine"

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
        # Find the most recent answer in the path
        existing_answer = None
        if node:
            current = node
            while current:
                if current.answer:
                    existing_answer = current.answer
                    break
                current = current.parent

        if existing_answer:
            return [
                Message(role="system", content=(
                    "You are solving a problem and reached a preliminary answer. "
                    "Re-examine it critically. If it holds up, confirm it. "
                    "If you find a flaw, correct it. Write in first person."
                )),
                Message(role="user", content=(
                    f"I need to solve: {question}\n\n"
                    f"My reasoning so far:\n{state}\n\n"
                    f"My preliminary answer: {existing_answer}\n\n"
                    f"Let me double-check this:"
                )),
            ]
        else:
            return [
                Message(role="system", content=(
                    "You are solving a problem. Try to draw a conclusion from "
                    "your reasoning so far. Write in first person."
                )),
                Message(role="user", content=(
                    f"I need to solve: {question}\n\n"
                    f"My reasoning:\n{state}\n\n"
                    f"Can I draw a conclusion from this?"
                )),
            ]


# ─── Pre-built action sets ──────────────────────────────────────

def logic_actions(terminal_detector: Optional[TerminalDetector] = None) -> list[ReasoningAction]:
    """Action set for logic puzzles (knights/knaves, etc.)."""
    return [
        AssumeAction(),
        DeduceAction(),
        VerifyAction(),
        CompareAction(),
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
    """Full action set including meta-actions."""
    return [
        DecomposeAction(),
        DeduceAction(),
        AssumeAction(),
        CalculateAction(),
        VerifyAction(),
        CompareAction(),
        SummarizeAction(),
        RefineAction(),
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

    # Default transition probabilities between action types.
    # Produces coherent chains like ASSUME -> DEDUCE -> VERIFY -> CONCLUDE
    # instead of random incoherent sequences.
    DEFAULT_TRANSITIONS: dict[str | None, dict[str, float]] = {
        None:        {"assume": 0.4, "deduce": 0.3, "decompose": 0.2, "compare": 0.1},
        "assume":    {"deduce": 0.5, "verify": 0.3, "assume": 0.2},
        "deduce":    {"verify": 0.4, "deduce": 0.3, "conclude": 0.2, "compare": 0.1},
        "verify":    {"conclude": 0.4, "deduce": 0.3, "assume": 0.2, "compare": 0.1},
        "compare":   {"conclude": 0.4, "deduce": 0.3, "assume": 0.3},
        "conclude":  {"assume": 0.5, "deduce": 0.3, "verify": 0.2},
        "calculate": {"calculate": 0.3, "verify": 0.3, "conclude": 0.2, "deduce": 0.2},
        "decompose": {"assume": 0.3, "calculate": 0.3, "deduce": 0.3, "decompose": 0.1},
        "summarize": {"conclude": 0.4, "deduce": 0.3, "verify": 0.3},
        "refine":    {"conclude": 0.5, "verify": 0.3, "deduce": 0.2},
    }

    def _select_action(self, state: str) -> ReasoningAction:
        """Select next action based on transition probabilities from previous action."""
        # Find the last action tag in the state
        prev_action = None
        for tag in ["[deduce]", "[assume]", "[verify]", "[compare]", "[conclude]",
                     "[calculate]", "[decompose]", "[summarize]", "[refine]"]:
            if tag in state:
                # Find the LAST occurrence
                idx = state.rfind(tag)
                if prev_action is None or idx > state.rfind(f"[{prev_action}]"):
                    prev_action = tag[1:-1]  # strip brackets

        # Get transition probabilities
        transitions = self.DEFAULT_TRANSITIONS.get(prev_action, self.DEFAULT_TRANSITIONS[None])

        # Filter to actions we actually have
        available = {a.name: a for a in self.actions}
        weights = []
        candidates = []
        for action in self.actions:
            w = transitions.get(action.name, 0.1)  # small default weight for unlisted
            weights.append(w)
            candidates.append(action)

        return random.choices(candidates, weights=weights, k=1)[0]

    def generate(self, question: str, state: str, n: int = 1, node: Node | None = None) -> list[Continuation]:
        """Generate n continuations, each using a selected action.

        Args:
            question: The original question.
            state: Current reasoning state.
            n: Number of continuations.
            node: Optional tree node for meta-actions that need structural context.
        """
        results = []
        selected_actions = [self._select_action(state) for _ in range(n)]

        for action in selected_actions:
            messages = action.prompt(question, State(state), node=node)
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
