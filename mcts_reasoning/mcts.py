"""
MCTS: Monte Carlo Tree Search for reasoning.

Implements the spec from paper/main.tex with:
- UCB1 selection (Definition 4.1)
- State-dependent action spaces (Definition 3.5)
- On-demand expansion with branching factor bound B (Definition 3.7)
- Tree-building rollouts (Algorithm 4, Remark 4.4)
- Backpropagation from terminal nodes (Algorithm 5)
- Terminal-only evaluation for cost efficiency
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import random
import json
from pathlib import Path

from .node import Node
from .generator import Generator
from .evaluator import Evaluator
from .actions import ActionSpace, DefaultActionSpace


@dataclass
class SearchResult:
    """Result of MCTS search."""

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

        def count_nodes(node: Node) -> int:
            return 1 + sum(count_nodes(c) for c in node.children)

        def max_depth(node: Node) -> int:
            if not node.children:
                return 0
            return 1 + max(max_depth(c) for c in node.children)

        self._cached_stats = {
            "total_nodes": count_nodes(self.root),
            "max_depth": max_depth(self.root),
            "simulations": self.simulations,
            "terminal_states_found": len(self.terminal_states),
            "best_answer": self.best_answer,
            "confidence": self.confidence,
        }
        return self._cached_stats

    def invalidate_stats_cache(self):
        """Invalidate cached stats (call after modifying tree)."""
        self._cached_stats = None


class MCTS:
    """
    Monte Carlo Tree Search for LLM reasoning.

    Usage:
        generator = LLMGenerator(llm)
        evaluator = LLMEvaluator(llm)

        mcts = MCTS(generator, evaluator)
        result = mcts.search("What is 2+2?", simulations=50)

        print(result.best_answer)
        print(result.confidence)

    With custom action space:
        action_space = ExtendedActionSpace(generator=generator, llm=llm)
        mcts = MCTS(generator, evaluator, action_space=action_space)
    """

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        exploration_constant: float = 1.414,
        max_children_per_node: int = 3,
        max_rollout_depth: int = 10,
        action_space: Optional[ActionSpace] = None,
    ):
        """
        Initialize MCTS.

        Args:
            generator: Produces reasoning continuations
            evaluator: Scores terminal states
            exploration_constant: UCB1 exploration parameter (higher = more exploration)
            max_children_per_node: Maximum branching factor (B in the spec)
            max_rollout_depth: Maximum steps in a rollout (D in the spec)
            action_space: Defines available actions per state (default: CONTINUE only)
        """
        self.generator = generator
        self.evaluator = evaluator
        self.exploration_constant = exploration_constant
        self.max_children_per_node = max_children_per_node
        self.max_rollout_depth = max_rollout_depth

        # Action space: defaults to CONTINUE only via DefaultActionSpace
        self.action_space = action_space or DefaultActionSpace(generator=generator)

        # Search state
        self.root: Optional[Node] = None
        self.question: str = ""
        self.terminal_states: List[Dict[str, Any]] = []

    def search(self, question: str, simulations: int = 100) -> SearchResult:
        """
        Run MCTS search.

        Args:
            question: The question to solve
            simulations: Number of simulations to run

        Returns:
            SearchResult with best answer and statistics
        """
        self.question = question
        self.terminal_states = []

        # Initialize root with the question as the initial state
        initial_state = f"Question: {question}\n\nLet me solve this step by step."
        self.root = Node(state=initial_state)

        # Run simulations
        for i in range(simulations):
            self._simulate()

        # Find best answer
        best_answer, confidence = self._get_best_answer()

        return SearchResult(
            best_answer=best_answer,
            confidence=confidence,
            root=self.root,
            simulations=simulations,
            terminal_states=self.terminal_states,
        )

    def _simulate(self):
        """Run one MCTS simulation: select, expand, rollout, backpropagate."""
        # Selection: traverse tree to find promising leaf
        node = self._select(self.root)

        # If terminal, just evaluate and backpropagate
        if node.is_terminal:
            reward = self._evaluate_terminal(node)
            self._backpropagate(node, reward)
            return

        # Expansion: add one new child
        child = self._expand(node)

        if child is None:
            # No expansion possible, evaluate current node
            reward = 0.0  # Non-terminal leaf gets low reward
            self._backpropagate(node, reward)
            return

        # Rollout: continue from child until terminal (tree-building)
        terminal_node, reward = self._rollout(child)

        # Backpropagate from terminal node (per spec Algorithm 5)
        self._backpropagate(terminal_node, reward)

    def _select(self, node: Node) -> Node:
        """
        Select a node to expand using UCB1.

        Traverses tree until finding a node that:
        1. Is terminal, OR
        2. Has untried continuations (can be expanded), OR
        3. Has fewer children than max_children_per_node
        """
        while True:
            # If terminal, return it
            if node.is_terminal:
                return node

            # If can expand (has untried continuations or room for more children)
            if self._can_expand(node):
                return node

            # If no children, must expand
            if not node.children:
                return node

            # Otherwise, select best child and continue
            node = node.best_child(self.exploration_constant)

            if node is None:
                # Shouldn't happen, but safety fallback
                return self.root

        return node

    def _can_expand(self, node: Node) -> bool:
        """Check if node can be expanded with a new child."""
        # Already at max children
        if len(node.children) >= self.max_children_per_node:
            return False

        # Has untried continuations cached
        if node.has_untried_continuations():
            return True

        # Hasn't generated continuations yet
        if node._continuations is None:
            return True

        return False

    def _expand(self, node: Node) -> Optional[Node]:
        """Expand node by adding one child."""
        # Generate continuations if not cached
        if node._continuations is None:
            continuations = self.generator.generate(
                self.question,
                node.state,
                n=self.max_children_per_node,
            )
            node.set_continuations([c.text for c in continuations])

            # Store terminal info for each continuation
            node._continuation_info = {
                c.text: (c.is_terminal, c.answer) for c in continuations
            }

        # Get next untried continuation
        next_state = node.get_next_continuation()
        if next_state is None:
            return None

        # Check if this continuation is terminal
        is_terminal = False
        answer = None
        if (
            hasattr(node, "_continuation_info")
            and next_state in node._continuation_info
        ):
            is_terminal, answer = node._continuation_info[next_state]

        # Create child node
        child = node.add_child(
            state=next_state,
            is_terminal=is_terminal,
            answer=answer,
        )

        return child

    def _rollout(self, node: Node) -> tuple:
        """
        Tree-building rollout from node until terminal state.

        Per spec (Algorithm 4, Remark 4.4): During rollout, we ADD nodes to the
        tree rather than just simulating. This preserves the reasoning trace
        for future exploration.

        Uses ActionSpace to select and apply actions at each step.

        Returns:
            (terminal_node, reward) - terminal_node is added to tree
        """
        current_node = node
        depth = 0

        # If already terminal, evaluate immediately
        if current_node.is_terminal:
            reward = self._evaluate_terminal(current_node)
            return current_node, reward

        # Rollout until terminal or max depth (adding nodes to tree)
        while depth < self.max_rollout_depth:
            # Get available actions for current state
            actions = self.action_space.get_actions(
                current_node.state,
                is_terminal=current_node.is_terminal,
            )

            if not actions:
                # No actions available (shouldn't happen for non-terminal)
                break

            # Select action (for canonical case, there's only CONTINUE)
            # For extended action spaces, could use UCB or random selection
            action = random.choice(actions) if len(actions) > 1 else actions[0]

            # Apply action to get new state
            result = action.apply(self.question, current_node.state)
            depth += 1

            # Create and add child node to tree (tree-building rollout)
            child_node = current_node.add_child(
                state=result.new_state,
                is_terminal=result.is_terminal,
                answer=result.answer,
            )

            if result.is_terminal:
                # Reached terminal state - evaluate and return
                reward = self._evaluate_terminal(child_node)
                return child_node, reward

            # Continue rollout from this new node
            current_node = child_node

        # Didn't reach terminal - return last node with low reward
        return current_node, 0.1

    def _evaluate_terminal(self, node: Node) -> float:
        """Evaluate a terminal node."""
        evaluation = self.evaluator.evaluate(
            self.question,
            node.state,
            node.answer or "",
        )

        # Record this terminal state
        self.terminal_states.append(
            {
                "state": node.state,
                "answer": node.answer,
                "score": evaluation.score,
                "depth": node.depth,
            }
        )

        return evaluation.score

    def _backpropagate(self, node: Node, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_best_answer(self) -> tuple:
        """Get the best answer from terminal states found."""
        if not self.terminal_states:
            # No terminal states found - try to get best leaf
            return self._get_best_leaf_answer()

        # Find highest scoring terminal state
        best = max(self.terminal_states, key=lambda x: x["score"])
        return best["answer"], best["score"]

    def _get_best_leaf_answer(self) -> tuple:
        """Fallback: get answer from best leaf node."""

        def find_best_leaf(node: Node) -> Node:
            if not node.children:
                return node
            best_child = node.highest_value_child()
            if best_child is None:
                return node
            return find_best_leaf(best_child)

        best_leaf = find_best_leaf(self.root)

        # Try to extract any answer from the state
        answer = self.generator.extract_answer(best_leaf.state)
        confidence = best_leaf.average_value if best_leaf.visits > 0 else 0.0

        return answer, confidence

    def get_tree_visualization(self, max_depth: int = 3) -> str:
        """Get a text visualization of the search tree."""
        lines = []

        def visualize(node: Node, prefix: str = "", is_last: bool = True):
            # Node representation
            marker = "└── " if is_last else "├── "
            state_preview = node.state.split("\n")[-1][:40]  # Last line, truncated
            if node.is_terminal:
                state_preview = f"[TERMINAL: {node.answer}]"

            stats = f"(v={node.visits}, avg={node.average_value:.2f})"
            lines.append(f"{prefix}{marker}{state_preview} {stats}")

            # Recurse to children
            if node.depth < max_depth:
                child_prefix = prefix + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    is_last_child = i == len(node.children) - 1
                    visualize(child, child_prefix, is_last_child)

        visualize(self.root, "", True)
        return "\n".join(lines)

    # ========== Serialization ==========

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the MCTS state to a dictionary.

        Note: Generator and evaluator are not serialized. They must be
        provided when loading via from_dict() or load().
        """
        if self.root is None:
            raise ValueError("Cannot serialize MCTS without a root node")

        return {
            "version": "0.4.0",
            "question": self.question,
            "exploration_constant": self.exploration_constant,
            "max_children_per_node": self.max_children_per_node,
            "max_rollout_depth": self.max_rollout_depth,
            "terminal_states": self.terminal_states,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        generator: Generator,
        evaluator: Evaluator,
        action_space: Optional["ActionSpace"] = None,
    ) -> "MCTS":
        """
        Deserialize MCTS state from a dictionary.

        Args:
            data: Dictionary from to_dict()
            generator: Generator instance (required for continued search)
            evaluator: Evaluator instance (required for continued search)
            action_space: Optional custom action space

        Returns:
            MCTS instance with restored tree state
        """
        mcts = cls(
            generator=generator,
            evaluator=evaluator,
            exploration_constant=data.get("exploration_constant", 1.414),
            max_children_per_node=data.get("max_children_per_node", 3),
            max_rollout_depth=data.get("max_rollout_depth", 10),
            action_space=action_space,
        )

        mcts.question = data.get("question", "")
        mcts.terminal_states = data.get("terminal_states", [])
        mcts.root = Node.from_dict(data["root"])

        return mcts

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        generator: Generator,
        evaluator: Evaluator,
        action_space: Optional["ActionSpace"] = None,
    ) -> "MCTS":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data, generator, evaluator, action_space)

    def save(self, path: str) -> None:
        """
        Save MCTS tree state to a file.

        Args:
            path: File path to save to (will be created/overwritten)

        Example:
            result = mcts.search("What is 2+2?", simulations=50)
            mcts.save("tree.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(
        cls,
        path: str,
        generator: Generator,
        evaluator: Evaluator,
        action_space: Optional["ActionSpace"] = None,
    ) -> "MCTS":
        """
        Load MCTS tree state from a file.

        Args:
            path: File path to load from
            generator: Generator instance (required for continued search)
            evaluator: Evaluator instance (required for continued search)
            action_space: Optional custom action space

        Returns:
            MCTS instance with restored tree state

        Example:
            mcts = MCTS.load("tree.json", generator, evaluator)
            result = mcts.continue_search(simulations=50)
        """
        with open(path, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str, generator, evaluator, action_space)

    # ========== Continued Search ==========

    def continue_search(self, simulations: int = 100) -> SearchResult:
        """
        Continue search from existing tree state.

        Use this to add more simulations to an existing search,
        either after initial search() or after loading a saved tree.

        Args:
            simulations: Additional simulations to run

        Returns:
            SearchResult with updated statistics

        Example:
            result = mcts.search("What is 2+2?", simulations=50)
            # Later, add more simulations
            result = mcts.continue_search(simulations=50)
        """
        if self.root is None:
            raise ValueError(
                "Cannot continue search without existing tree. Use search() first."
            )

        # Run additional simulations
        for _ in range(simulations):
            self._simulate()

        # Get updated best answer
        best_answer, confidence = self._get_best_answer()

        # Count total simulations from root visits
        total_simulations = self.root.visits

        return SearchResult(
            best_answer=best_answer,
            confidence=confidence,
            root=self.root,
            simulations=total_simulations,
            terminal_states=self.terminal_states,
        )
