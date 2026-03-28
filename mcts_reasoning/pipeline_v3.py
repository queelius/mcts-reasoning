"""
v3 Pipeline: fixes the extraction loss problem and explores intermediate rewards.

Changes from v2:
1. No extraction for initial solutions. Each CoT response = one node. Zero loss.
2. Reward signal options beyond self-consistency:
   - Verification: "does this solution satisfy constraints?" (per-terminal)
   - Process reward: "is the reasoning at this node sound?" (per-node, not just terminal)
3. Exploration generates from the state at a selected node, not from root.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

from .node import Node
from .types import Message, State, SearchState, extend_state
from .terminal import TerminalDetector, MarkerTerminalDetector
from .evaluator import Evaluator


@dataclass
class PipelineV3Config:
    n_solutions: int = 8
    n_explore: int = 4            # exploration rounds after initial solutions
    max_tokens_cot: int = 800
    max_tokens_explore: int = 600
    temperature: float = 0.9
    exploration_constant: float = 1.414


class PipelineV3:
    """v3: no extraction loss, intermediate reward option."""

    def __init__(self, provider, evaluator: Evaluator,
                 terminal_detector: Optional[TerminalDetector] = None,
                 config: Optional[PipelineV3Config] = None,
                 on_phase: Optional[Callable] = None):
        self.provider = provider
        self.evaluator = evaluator
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.config = config or PipelineV3Config()
        self.on_phase = on_phase

    def run(self, question: str) -> SearchState:
        cfg = self.config
        root = Node(state=State(f"Question: {question}"))

        state = SearchState(
            root=root, question=question,
            exploration_constant=cfg.exploration_constant,
        )

        # Phase 1: Generate N complete CoT solutions
        # Each is ONE node. Use LLM judge to extract the answer (not regex).
        self._notify("generate", state)
        for _ in range(cfg.n_solutions):
            resp = self._generate_cot(question)
            answer = self._extract_answer(question, resp)
            root.add_child(
                state=State(resp),  # full response preserved
                is_terminal=answer is not None,
                answer=answer,
            )

        # Phase 2: Score all terminals, backpropagate
        self._notify("score", state)
        self._score_and_backprop(state)

        # Phase 3: UCB-guided exploration
        for i in range(cfg.n_explore):
            self._notify("explore", state)
            self._explore(state)
            self._score_and_backprop(state)

        state.simulations_run = cfg.n_solutions + cfg.n_explore
        return state

    def best_answer(self, state: SearchState) -> tuple[str | None, float]:
        """Pick best answer by highest-value terminal."""
        best = None
        best_val = -1.0
        for node in self._iter_terminals(state.root):
            if node.answer and node.value > best_val:
                best_val = node.value
                best = node.answer
        return best, best_val

    def _generate_cot(self, question: str) -> str:
        """One complete CoT solution. Same prompt as baseline."""
        return self.provider.generate([
            Message(role="system", content=(
                "You are a careful problem solver. Think step by step. "
                "Show all your work. Give your final answer clearly."
            )),
            Message(role="user", content=question),
        ], self.config.max_tokens_cot, self.config.temperature)

    def _extract_answer(self, question: str, response: str) -> str | None:
        """Use LLM to extract the final answer. Not regex."""
        resp = self.provider.generate([
            Message(role="system", content=(
                "Extract the final answer from this response. "
                "Reply with ONLY the answer, nothing else. "
                "If no clear answer, reply NO_ANSWER."
            )),
            Message(role="user", content=(
                f"Question: {question}\n\nResponse:\n{response}\n\nFinal answer:"
            )),
        ], max_tokens=100, temperature=0.0)
        answer = resp.strip()
        if "NO_ANSWER" in answer.upper() or not answer:
            return None
        return answer

    def _score_and_backprop(self, state: SearchState) -> None:
        """Score unscored terminals, backpropagate."""
        for node in self._iter_terminals(state.root):
            if node.visits > 0:
                continue
            if node.answer:
                ev = self.evaluator.evaluate(state.question, str(node.state), node.answer)
                score = ev.score
                state.terminal_states.append({
                    "answer": node.answer, "score": score,
                    "state": str(node.state)[:200],
                })
            else:
                score = 0.0
            # Backpropagate
            current = node
            while current is not None:
                current.visits += 1
                current.value = (current.value * (current.visits - 1) + score) / current.visits
                current = current.parent

    def _explore(self, state: SearchState) -> None:
        """UCB-guided exploration: select a promising node, generate from it."""
        node = self._ucb_select(state.root)
        if node is None:
            return

        # Generate a new solution that continues from this node's reasoning
        resp = self.provider.generate([
            Message(role="system", content=(
                f"You are solving: {state.question}\n\n"
                "Build on the reasoning below. Try a different approach or "
                "correct any errors you notice. Think step by step."
            )),
            Message(role="assistant", content=str(node.state)),
            Message(role="user", content="Continue with an alternative approach:"),
        ], self.config.max_tokens_explore, self.config.temperature)

        answer = self._extract_answer(state.question, resp)
        new_state = extend_state(node.state, resp)
        node.add_child(
            state=new_state,
            is_terminal=answer is not None,
            answer=answer,
        )

    def _ucb_select(self, root: Node) -> Node | None:
        """Select node for exploration via UCB1."""
        c = self.config.exploration_constant
        best = None
        best_score = -1.0

        stack = [root]
        while stack:
            node = stack.pop()
            if not node.is_terminal and (not node.children or len(node.children) < 3):
                if node.visits == 0:
                    return node
                ucb = node.value + c * math.sqrt(math.log(root.visits + 1) / (node.visits + 1))
                if ucb > best_score:
                    best_score = ucb
                    best = node
            stack.extend(node.children)

        return best

    def _iter_terminals(self, root: Node) -> list[Node]:
        terminals = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_terminal:
                terminals.append(node)
            stack.extend(node.children)
        return terminals

    def _notify(self, phase, state):
        if self.on_phase:
            self.on_phase(phase, state)
