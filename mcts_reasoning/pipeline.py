"""
v2 Pipeline: natural CoT generation, LLM extraction, tree-guided exploration.

Phase 1: Generate N complete CoT solutions (unconstrained, same as baseline)
Phase 2: LLM extracts tree structure from each response
Phase 3: Merge into a single tree (N root-to-leaf paths)
Phase 4: Score terminals, backpropagate
Phase 5: UCB-guided exploration at promising branches
Phase 6: Iterate (phases 4-5, not back to phase 1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .node import Node
from .types import Message, State, SearchState, Continuation, extend_state
from .terminal import TerminalDetector, MarkerTerminalDetector
from .extract import LLMExtractor, ExtractedStep
from .evaluator import Evaluator
from .tree_viz import render_tree
import math
import random


@dataclass
class PipelineConfig:
    """Configuration for the v2 pipeline."""
    n_solutions: int = 4           # Phase 1: how many CoT solutions to generate
    max_tokens_cot: int = 1000     # Token budget for CoT generation
    max_tokens_explore: int = 500  # Token budget for exploration rollouts
    temperature_cot: float = 0.9   # High temp for diverse solutions
    temperature_explore: float = 0.7
    exploration_constant: float = 1.414  # UCB1 c parameter
    explore_iterations: int = 4    # Phase 5-6: how many exploration rounds
    max_explore_depth: int = 3     # How deep to rollout from a selected node


class Pipeline:
    """The v2 MCTS pipeline.

    Usage:
        pipeline = Pipeline(provider, evaluator)
        state = pipeline.run("What is gcd(36, 46)?")
        print(state.root.count_nodes(), "nodes")
        print(pipeline.best_answer(state))
    """

    def __init__(
        self,
        provider,
        evaluator: Evaluator,
        terminal_detector: Optional[TerminalDetector] = None,
        config: Optional[PipelineConfig] = None,
        on_phase: Optional[Callable] = None,
    ):
        self.provider = provider
        self.evaluator = evaluator
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.config = config or PipelineConfig()
        self.extractor = LLMExtractor(provider, self.terminal_detector)
        self.on_phase = on_phase  # callback(phase_name, state)

    def run(self, question: str) -> SearchState:
        """Run the full pipeline."""
        cfg = self.config

        # Phase 1: Generate N natural CoT solutions
        self._notify("generate", None)
        solutions = self._generate_solutions(question, cfg.n_solutions)

        # Phase 2: Extract structure from each
        self._notify("extract", None)
        root = Node(state=State(f"Question: {question}"))
        for i, sol in enumerate(solutions):
            steps = self.extractor.extract(question, sol)
            self._attach_path(root, steps, sol)

        state = SearchState(
            root=root,
            question=question,
            exploration_constant=cfg.exploration_constant,
        )

        # Phase 3-4: Score terminals and backpropagate
        self._notify("score", state)
        self._score_and_backprop(state)

        # Phase 5-6: Explore and iterate
        for i in range(cfg.explore_iterations):
            self._notify("explore", state)
            self._explore_round(state)
            self._score_and_backprop(state)

        state.simulations_run = cfg.n_solutions + cfg.explore_iterations
        return state

    def best_answer(self, state: SearchState) -> tuple[str | None, float]:
        """Extract the best answer from the tree via weighted vote."""
        terminals = self._find_terminals(state.root)
        if not terminals:
            return None, 0.0

        # Weighted vote by node value
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for node in terminals:
            if node.answer:
                scores[node.answer] = scores.get(node.answer, 0.0) + node.value
                counts[node.answer] = counts.get(node.answer, 0) + 1

        if not scores:
            return None, 0.0

        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[winner] / total if total > 0 else 0.0
        return winner, confidence

    # ─── Phase 1: Generate ────────────────────────────────────

    def _generate_solutions(self, question: str, n: int) -> list[str]:
        """Generate N complete CoT solutions. No constraints, natural prompting."""
        solutions = []
        for _ in range(n):
            messages = [
                Message(role="system", content=(
                    "You are a careful problem solver. Think step by step. "
                    "Show all your work. Give your final answer clearly."
                )),
                Message(role="user", content=question),
            ]
            resp = self.provider.generate(
                messages, self.config.max_tokens_cot, self.config.temperature_cot,
            )
            solutions.append(resp)
        return solutions

    # ─── Phase 2: Extract and attach ──────────────────────────

    def _attach_path(self, root: Node, steps: list[ExtractedStep], raw_solution: str) -> None:
        """Attach extracted steps as a path in the tree.

        Steps form a chain from root. Branch points (where parent_idx
        goes backward) create new branches from the appropriate ancestor.
        """
        if not steps:
            # No structure extracted: attach raw solution as single child
            check = self.terminal_detector.is_terminal(raw_solution)
            root.add_child(
                state=extend_state(root.state, raw_solution),
                is_terminal=check.is_terminal,
                answer=check.answer,
            )
            return

        # Map step indices to nodes
        step_nodes: dict[int, Node] = {0: root}
        current = root

        for i, step in enumerate(steps):
            # Find the parent node for this step
            parent_idx = step.parent_idx
            if parent_idx in step_nodes:
                parent = step_nodes[parent_idx]
            elif parent_idx == 0 or parent_idx == -1:
                parent = root
            else:
                parent = current  # fallback: chain from previous

            state_text = f"[{step.step_type}] {step.text}"
            child = parent.add_child(
                state=extend_state(parent.state, state_text),
                is_terminal=step.is_terminal,
                answer=step.answer,
            )
            step_nodes[i + 1] = child  # steps are 1-indexed in the extractor
            current = child

    # ─── Phase 3-4: Score and backpropagate ───────────────────

    def _score_and_backprop(self, state: SearchState) -> None:
        """Score all unscored terminals and backpropagate."""
        terminals = self._find_terminals(state.root)

        for node in terminals:
            if node.visits > 0:
                continue  # already scored

            if node.answer:
                evaluation = self.evaluator.evaluate(
                    state.question, str(node.state), node.answer,
                )
                score = evaluation.score
                state.terminal_states.append({
                    "answer": node.answer,
                    "score": score,
                    "state": str(node.state)[:200],
                })
            else:
                score = 0.0

            # Backpropagate
            current = node
            while current is not None:
                current.visits += 1
                current.value = (
                    (current.value * (current.visits - 1) + score) / current.visits
                )
                current = current.parent

    # ─── Phase 5: Explore ─────────────────────────────────────

    def _explore_round(self, state: SearchState) -> None:
        """One round of UCB-guided exploration.

        Select a promising non-terminal node, generate a continuation
        from that point, extract structure, attach.
        """
        # Select node via UCB1
        node = self._ucb_select(state.root)
        if node is None:
            return

        # Generate a continuation from this node's state
        messages = [
            Message(role="system", content=(
                f"You are solving: {state.question}\n\n"
                "Continue the reasoning from where it left off. "
                "Try a different approach than what was already tried. "
                "Think step by step. Give your final answer clearly."
            )),
            Message(role="assistant", content=str(node.state)),
            Message(role="user", content="[continue]"),
        ]
        resp = self.provider.generate(
            messages, self.config.max_tokens_explore, self.config.temperature_explore,
        )

        # Extract and attach
        steps = self.extractor.extract(state.question, resp)
        self._attach_path(node, steps, resp)

    def _ucb_select(self, root: Node) -> Node | None:
        """Select a non-terminal leaf node using UCB1."""
        best_node = None
        best_score = -1.0
        c = self.config.exploration_constant

        # Find all non-terminal leaves (or nodes with room for more children)
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_terminal:
                continue
            if not node.children:
                # Leaf node: candidate for exploration
                if node.visits == 0:
                    return node  # unvisited, explore immediately
                ucb = node.value + c * math.sqrt(
                    math.log(root.visits + 1) / (node.visits + 1)
                )
                if ucb > best_score:
                    best_score = ucb
                    best_node = node
            else:
                stack.extend(node.children)
                # Also consider this node if it has few children
                if len(node.children) < 3 and not node.is_terminal:
                    ucb = node.value + c * math.sqrt(
                        math.log(root.visits + 1) / (node.visits + 1)
                    )
                    if ucb > best_score:
                        best_score = ucb
                        best_node = node

        return best_node

    # ─── Helpers ──────────────────────────────────────────────

    def _find_terminals(self, root: Node) -> list[Node]:
        """Iterative terminal finder."""
        terminals = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_terminal:
                terminals.append(node)
            stack.extend(node.children)
        return terminals

    def _notify(self, phase: str, state: SearchState | None) -> None:
        if self.on_phase:
            self.on_phase(phase, state)
