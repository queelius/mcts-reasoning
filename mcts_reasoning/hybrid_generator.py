"""
Hybrid Generator: CoT solutions at root, incremental refinement deeper.

Architecture:
1. Shallow nodes (depth 0-1): Generate complete chain-of-thought solutions
   (same quality as CoT baseline). Decompose into step chains.
2. Deep nodes (depth 2+): Use targeted reasoning actions (VERIFY, REFINE,
   ASSUME alternative) guided by UCB to refine promising branches.

This gives:
- CoT-quality reasoning for free (no prompting disadvantage vs baseline)
- Tree structure for inspection, voting, and backpropagation
- Incremental exploration where UCB says it's worth looking
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from .types import Continuation, Message, State, extend_state
from .terminal import TerminalDetector, MarkerTerminalDetector
from .generator import Generator

if TYPE_CHECKING:
    from .node import Node


class HybridGenerator(Generator):
    """Broad CoT solutions at root, focused actions deeper in the tree.

    Args:
        provider: LLM provider
        terminal_detector: How to detect answers
        cot_depth: Maximum depth for full CoT generation (default 1).
            Nodes at depth <= cot_depth get complete solutions.
            Deeper nodes get targeted actions.
        max_tokens_cot: Token budget for complete solutions (generous)
        max_tokens_action: Token budget for incremental actions (tighter)
    """

    def __init__(
        self,
        provider,
        terminal_detector: Optional[TerminalDetector] = None,
        cot_depth: int = 1,
        max_tokens_cot: int = 1000,
        max_tokens_action: int = 400,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.cot_depth = cot_depth
        self.max_tokens_cot = max_tokens_cot
        self.max_tokens_action = max_tokens_action
        self.temperature = temperature

    def generate(self, question: str, state: str, n: int = 1,
                 node: "Node | None" = None) -> list[Continuation]:
        depth = node.depth if node else 0

        if depth <= self.cot_depth:
            return self._generate_cot(question, State(state), n)
        else:
            return self._generate_action(question, State(state), n, node)

    def _generate_cot(self, question: str, state: State, n: int) -> list[Continuation]:
        """Generate a complete chain-of-thought solution (same as CoT baseline)."""
        results = []
        for _ in range(n):
            messages = [
                Message(role="system", content=(
                    "You are a careful math and logic assistant. "
                    "Think step by step. Show all your work. "
                    "When you reach the final answer, write: ANSWER: <your answer>"
                )),
                Message(role="user", content=question),
            ]
            response = self.provider.generate(
                messages, self.max_tokens_cot, self.temperature,
            )
            check = self.terminal_detector.is_terminal(response)
            new_state = extend_state(state, response)
            results.append(Continuation(
                text=new_state,
                is_terminal=check.is_terminal,
                answer=check.answer,
            ))
        return results

    def _generate_action(self, question: str, state: State, n: int,
                         node: "Node | None") -> list[Continuation]:
        """Generate targeted refinement at deeper nodes."""
        results = []
        actions = self._pick_actions(state, node, n)

        for action_type in actions:
            messages = self._build_action_prompt(action_type, question, state, node)
            response = self.provider.generate(
                messages, self.max_tokens_action, self.temperature,
            )
            check = self.terminal_detector.is_terminal(response)
            tagged = f"[{action_type}] {response}"
            new_state = extend_state(state, tagged)
            results.append(Continuation(
                text=new_state,
                is_terminal=check.is_terminal,
                answer=check.answer,
            ))
        return results

    def _pick_actions(self, state: State, node: "Node | None", n: int) -> list[str]:
        """Choose which actions to take based on context."""
        import random

        # If parent reached an answer, try VERIFY or REFINE
        if node and node.answer:
            return random.choices(["verify", "refine"], k=n)

        # If siblings have answers, try COMPARE
        if node and node.parent:
            sibling_answers = [c.answer for c in node.parent.children if c.answer and c is not node]
            if sibling_answers:
                return random.choices(["compare", "verify", "try_alternative"], k=n)

        # Default: try alternative approach or verify existing reasoning
        return random.choices(["try_alternative", "verify", "calculate"], k=n)

    def _build_action_prompt(self, action: str, question: str, state: State,
                             node: "Node | None") -> list[Message]:
        """Build prompt for a specific action type."""
        # Use assistant role for state (avoids reviewer mode)
        system_prompts = {
            "verify": (
                f"You are solving: {question}\n\n"
                "Check the reasoning for errors. If correct, confirm with ANSWER: <answer>. "
                "If wrong, explain the error and give the correct answer."
            ),
            "refine": (
                f"You are solving: {question}\n\n"
                "The reasoning reached an answer but may not be correct. "
                "Re-derive the answer carefully. Write ANSWER: <answer> when done."
            ),
            "try_alternative": (
                f"You are solving: {question}\n\n"
                "Try a completely different approach than what was tried before. "
                "Show your work step by step. Write ANSWER: <answer> when done."
            ),
            "compare": (
                f"You are solving: {question}\n\n"
                "Multiple approaches have been tried. Compare them and determine "
                "which answer is most likely correct. Write ANSWER: <answer>."
            ),
            "calculate": (
                f"You are solving: {question}\n\n"
                "Perform the calculation carefully, step by step. "
                "Double-check each step. Write ANSWER: <answer> when done."
            ),
        }

        return [
            Message(role="system", content=system_prompts.get(action, system_prompts["verify"])),
            Message(role="assistant", content=str(state)),
            Message(role="user", content=f"[{action}]"),
        ]
