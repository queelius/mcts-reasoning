"""
MCTS: Monte Carlo Tree Search for reasoning.

Stateless implementation: search() creates a fresh SearchState each time;
continue_search(state, n) accepts an existing SearchState and extends it.
No mutable search state lives on the MCTS object itself.

Implements the spec from paper/main.tex with:
- UCB1 selection (Definition 4.1)
- On-demand expansion via Generator (no ActionSpace indirection)
- Tree-building rollouts (Algorithm 4, Remark 4.4)
- Backpropagation from terminal nodes (Algorithm 5)
- Terminal-only evaluation for cost efficiency
- on_simulation callback for observability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .generator import Generator
from .evaluator import Evaluator
from .node import Node
from .terminal import MarkerTerminalDetector, TerminalDetector
from .types import SearchState, State


# ---------------------------------------------------------------------------
# Legacy SearchResult -- kept for backward compatibility with test_v2 etc.
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Result of MCTS search (legacy wrapper around SearchState)."""

    best_answer: Optional[str]
    confidence: float
    root: Node
    simulations: int
    terminal_states: List[Dict[str, Any]] = field(default_factory=list)
    _cached_stats: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get search statistics (cached for efficiency)."""
        if self._cached_stats is not None:
            return self._cached_stats

        self._cached_stats = {
            "total_nodes": self.root.count_nodes(),
            "max_depth": self.root.max_depth(),
            "simulations": self.simulations,
            "terminal_states_found": len(self.terminal_states),
            "best_answer": self.best_answer,
            "confidence": self.confidence,
        }
        return self._cached_stats

    def invalidate_stats_cache(self):
        """Invalidate cached stats (call after modifying tree)."""
        self._cached_stats = None


# Type alias for the on_simulation callback signature.
OnSimulationCallback = Callable[[int, str, Node, SearchState], None]


class MCTS:
    """
    Monte Carlo Tree Search for LLM reasoning.

    **Stateless**: search() and continue_search() do NOT mutate ``self``.
    All mutable search state lives on the returned :class:`SearchState`.

    Usage::

        generator = LLMGenerator(llm)
        evaluator = LLMEvaluator(llm)

        mcts = MCTS(generator, evaluator)
        state = mcts.search("What is 2+2?", simulations=50)

        # Continue later
        state = mcts.continue_search(state, simulations=50)
    """

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        terminal_detector: Optional[TerminalDetector] = None,
        exploration_constant: float = 1.414,
        max_children_per_node: int = 3,
        max_rollout_depth: int = 5,
        on_simulation: Optional[OnSimulationCallback] = None,
        # Legacy kwargs -- silently accepted so old code doesn't break.
        action_space: Any = None,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.exploration_constant = exploration_constant
        self.max_children_per_node = max_children_per_node
        self.max_rollout_depth = max_rollout_depth
        self.on_simulation = on_simulation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, question: str, simulations: int = 10) -> SearchState:
        """Run a fresh MCTS search, returning a :class:`SearchState`."""
        initial_state = State(
            f"Question: {question}\n\nLet me solve this step by step."
        )
        root = Node(state=initial_state)
        search_state = SearchState(
            root=root,
            question=question,
            exploration_constant=self.exploration_constant,
            max_children_per_node=self.max_children_per_node,
            max_rollout_depth=self.max_rollout_depth,
        )
        return self._run_simulations(search_state, simulations)

    def continue_search(self, state: SearchState, simulations: int = 10) -> SearchState:
        """Run more simulations on an existing :class:`SearchState`."""
        return self._run_simulations(state, simulations)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _run_simulations(
        self, search_state: SearchState, simulations: int
    ) -> SearchState:
        base = search_state.simulations_run
        for sim in range(simulations):
            self._simulate(search_state, base + sim + 1)
        search_state.simulations_run += simulations
        return search_state

    def _simulate(self, state: SearchState, sim_number: int) -> None:
        # 1. SELECT: UCB1 traversal to expandable leaf
        node = self._select(
            state.root, state.exploration_constant, state.max_children_per_node
        )
        self._fire_callback(sim_number, "select", node, state)

        # 2. EXPAND: generate continuation, add child
        if not node.is_terminal:
            child = self._expand(state.question, node, state.max_children_per_node)
            if child is not None:
                node = child
        self._fire_callback(sim_number, "expand", node, state)

        # 3. ROLLOUT: keep generating until terminal or max depth
        rollout_node = self._rollout(
            state.question,
            node,
            state.max_children_per_node,
            state.max_rollout_depth,
        )
        self._fire_callback(sim_number, "rollout", rollout_node, state)

        # 4. BACKPROP: evaluate if terminal, propagate score
        if rollout_node.is_terminal and rollout_node.answer:
            evaluation = self.evaluator.evaluate(
                state.question,
                str(rollout_node.state),
                rollout_node.answer,
            )
            state.terminal_states.append(
                {
                    "answer": rollout_node.answer,
                    "score": evaluation.score,
                    "state": str(rollout_node.state)[:200],
                }
            )
            self._backpropagate(rollout_node, evaluation.score)
        else:
            self._backpropagate(rollout_node, 0.0)
        self._fire_callback(sim_number, "backprop", rollout_node, state)

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    @staticmethod
    def _select(root: Node, exploration_constant: float, max_children: int) -> Node:
        """UCB1 walk from *root* to an expandable or terminal leaf (iterative)."""
        node = root
        while True:
            if node.is_terminal:
                return node

            # Expandable: has room for more children
            if len(node.children) < max_children:
                return node

            # All slots filled -- descend via best UCB1 child
            if not node.children:
                return node

            best = node.best_child(exploration_constant)
            if best is None:
                return node
            node = best

    def _expand(self, question: str, node: Node, max_children: int) -> Optional[Node]:
        """Generate one continuation and attach it as a child of *node*."""
        if len(node.children) >= max_children:
            return None

        continuations = self.generator.generate(question, str(node.state), n=1)
        if not continuations:
            return None

        cont = continuations[0]
        child = node.add_child(
            state=str(cont.text),
            is_terminal=cont.is_terminal,
            answer=cont.answer,
        )
        return child

    def _rollout(
        self,
        question: str,
        node: Node,
        max_children: int,
        max_depth: int,
    ) -> Node:
        """Tree-building rollout: generate until terminal or *max_depth* (iterative)."""
        current = node
        depth = 0
        while depth < max_depth and not current.is_terminal:
            # Respect branching-factor bound during rollout
            if len(current.children) >= max_children:
                break

            continuations = self.generator.generate(question, str(current.state), n=1)
            if not continuations:
                break

            cont = continuations[0]
            child = current.add_child(
                state=str(cont.text),
                is_terminal=cont.is_terminal,
                answer=cont.answer,
            )
            current = child
            depth += 1
        return current

    @staticmethod
    def _backpropagate(node: Node, score: float) -> None:
        """Walk from *node* to root, updating visits and running-average value (iterative)."""
        current: Optional[Node] = node
        while current is not None:
            current.visits += 1
            # Running average: new_avg = (old_avg * (n-1) + score) / n
            if current.visits == 1:
                current.value = score
            else:
                current.value = (
                    current.value * (current.visits - 1) + score
                ) / current.visits
            current = current.parent

    def _fire_callback(
        self,
        sim_number: int,
        phase: str,
        node: Node,
        state: SearchState,
    ) -> None:
        if self.on_simulation is not None:
            self.on_simulation(sim_number, phase, node, state)

    # ------------------------------------------------------------------
    # Convenience: build a legacy SearchResult from a SearchState
    # ------------------------------------------------------------------

    @staticmethod
    def to_search_result(state: SearchState) -> SearchResult:
        """Convert a :class:`SearchState` into a legacy :class:`SearchResult`."""
        best_answer: Optional[str] = None
        confidence: float = 0.0

        if state.terminal_states:
            best = max(state.terminal_states, key=lambda t: t["score"])
            best_answer = best["answer"]
            confidence = best["score"]

        return SearchResult(
            best_answer=best_answer,
            confidence=confidence,
            root=state.root,
            simulations=state.simulations_run,
            terminal_states=state.terminal_states,
        )
