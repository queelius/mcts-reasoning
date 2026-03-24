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

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
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

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
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

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
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

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
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

    def prompt(self, question: str, state: State, node: Node | None = None) -> list[Message]:
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
                "You are a precise summarizer. Given a chain of reasoning steps, "
                "produce a concise summary that preserves:\n"
                "- All conclusions reached so far\n"
                "- Any contradictions or dead ends found\n"
                "- Key intermediate results needed for future reasoning\n\n"
                "Discard: repetition, hedging, false starts, verbose explanations.\n"
                "The summary should be a self-contained starting point for further reasoning."
            )),
            Message(role="user", content=(
                f"Question: {question}\n\n"
                f"Full reasoning chain:\n{state}\n\n"
                f"Summarize the key findings and conclusions so far:"
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
                    "You compare different reasoning paths that branched from the same point. "
                    "Identify what they agree on, where they diverge, and which path seems more promising."
                )),
                Message(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Current path:\n{state}\n\n"
                    f"Alternative branches from the same parent:\n{sibling_info}\n\n"
                    f"Compare these paths. What do they agree on? Where do they diverge?"
                )),
            ]
        else:
            # No siblings — fall back to self-review
            return [
                Message(role="system", content=(
                    "Review the reasoning so far. Identify the strongest conclusions "
                    "and any areas of uncertainty."
                )),
                Message(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Reasoning:\n{state}\n\n"
                    f"What are the strongest conclusions so far? What is still uncertain?"
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
                    "You have a preliminary answer. Re-examine it critically. "
                    "Check the reasoning that led to it. If it holds up, confirm it "
                    "with a clearer explanation. If you find a flaw, derive the correct answer."
                )),
                Message(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Reasoning so far:\n{state}\n\n"
                    f"Preliminary answer: {existing_answer}\n\n"
                    f"Is this answer correct? Verify it carefully."
                )),
            ]
        else:
            return [
                Message(role="system", content=(
                    "Review the reasoning so far and try to draw a conclusion. "
                    "If the reasoning is sufficient, state the answer. "
                    "If not, identify what's missing."
                )),
                Message(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Reasoning:\n{state}\n\n"
                    f"Can we draw a conclusion from this?"
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
