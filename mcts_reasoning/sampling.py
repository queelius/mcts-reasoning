"""
Sampling: Extract paths from the MCTS search tree.

Provides strategies for sampling solution paths from a completed search:
- value: Paths ordered by average value (quality)
- visits: Paths ordered by visit count (confidence)
- diverse: Paths maximizing diversity (exploration)
- topk: Top-k terminal states by evaluation score

Example usage:
    result = mcts.search(question, simulations=100)
    sampler = PathSampler(result.root)

    # Get top 5 paths by value
    paths = sampler.sample(n=5, strategy="value")

    # Get diverse paths
    diverse_paths = sampler.sample(n=3, strategy="diverse")

    # Get all terminal states
    terminals = sampler.get_terminals()
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Callable
from enum import Enum

from .node import Node


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    VALUE = "value"       # Order by average value
    VISITS = "visits"     # Order by visit count
    DIVERSE = "diverse"   # Maximize path diversity
    TOPK = "topk"         # Top-k by terminal score


@dataclass
class SampledPath:
    """A sampled reasoning path from the tree."""
    nodes: List[Node]
    terminal: Optional[Node]  # Terminal node if path reaches one
    value: float              # Average value of terminal or leaf
    visits: int               # Visit count of terminal or leaf
    answer: Optional[str]     # Extracted answer if terminal

    @property
    def depth(self) -> int:
        """Path depth (number of steps)."""
        return len(self.nodes) - 1  # Exclude root

    @property
    def reasoning(self) -> str:
        """Full reasoning trace as string."""
        if self.terminal:
            return self.terminal.state
        elif self.nodes:
            return self.nodes[-1].state
        return ""

    @property
    def steps(self) -> List[str]:
        """Individual reasoning steps (excluding root)."""
        result = []
        for i, node in enumerate(self.nodes):
            if i == 0:
                continue  # Skip root
            prev_state = self.nodes[i - 1].state
            step = node.state[len(prev_state):].strip()
            if step:
                result.append(step)
        return result


class PathSampler:
    """
    Sample paths from an MCTS search tree.

    Provides multiple sampling strategies for extracting
    solution paths of varying quality and diversity.
    """

    def __init__(self, root: Node):
        """
        Initialize sampler with a search tree root.

        Args:
            root: Root node of the completed search tree
        """
        self.root = root
        self._terminals: Optional[List[Node]] = None

    def get_terminals(self) -> List[Node]:
        """Get all terminal nodes in the tree."""
        if self._terminals is None:
            self._terminals = self._find_terminals(self.root)
        return self._terminals

    def _find_terminals(self, node: Node) -> List[Node]:
        """Recursively find all terminal nodes."""
        terminals = []
        if node.is_terminal:
            terminals.append(node)
        for child in node.children:
            terminals.extend(self._find_terminals(child))
        return terminals

    def sample(
        self,
        n: int = 5,
        strategy: str = "value",
        include_non_terminal: bool = False,
    ) -> List[SampledPath]:
        """
        Sample n paths from the tree.

        Args:
            n: Number of paths to sample
            strategy: Sampling strategy ("value", "visits", "diverse", "topk")
            include_non_terminal: If True, include non-terminal leaf paths

        Returns:
            List of SampledPath objects
        """
        strategy_enum = SamplingStrategy(strategy)

        if strategy_enum == SamplingStrategy.VALUE:
            return self._sample_by_value(n, include_non_terminal)
        elif strategy_enum == SamplingStrategy.VISITS:
            return self._sample_by_visits(n, include_non_terminal)
        elif strategy_enum == SamplingStrategy.DIVERSE:
            return self._sample_diverse(n, include_non_terminal)
        elif strategy_enum == SamplingStrategy.TOPK:
            return self._sample_topk(n)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_candidate_nodes(self, include_non_terminal: bool) -> List[Node]:
        """Get candidate nodes for sampling."""
        terminals = self.get_terminals()
        if include_non_terminal:
            # Also include high-value leaf nodes
            leaves = self._find_leaves(self.root)
            # Combine without duplicates using id()
            seen = set(id(t) for t in terminals)
            candidates = list(terminals)
            for leaf in leaves:
                if id(leaf) not in seen:
                    candidates.append(leaf)
                    seen.add(id(leaf))
        else:
            candidates = terminals
        return candidates

    def _find_leaves(self, node: Node) -> List[Node]:
        """Find all leaf nodes (no children)."""
        leaves = []
        if not node.children:
            leaves.append(node)
        for child in node.children:
            leaves.extend(self._find_leaves(child))
        return leaves

    def _node_to_path(self, node: Node) -> SampledPath:
        """Convert a node to a SampledPath."""
        path_nodes = node.path_from_root()
        return SampledPath(
            nodes=path_nodes,
            terminal=node if node.is_terminal else None,
            value=node.average_value,
            visits=node.visits,
            answer=node.answer if node.is_terminal else None,
        )

    def _sample_by_value(
        self,
        n: int,
        include_non_terminal: bool,
    ) -> List[SampledPath]:
        """Sample paths ordered by average value."""
        candidates = self._get_candidate_nodes(include_non_terminal)
        # Sort by value descending
        sorted_nodes = sorted(
            candidates,
            key=lambda x: x.average_value,
            reverse=True,
        )
        return [self._node_to_path(node) for node in sorted_nodes[:n]]

    def _sample_by_visits(
        self,
        n: int,
        include_non_terminal: bool,
    ) -> List[SampledPath]:
        """Sample paths ordered by visit count."""
        candidates = self._get_candidate_nodes(include_non_terminal)
        # Sort by visits descending
        sorted_nodes = sorted(
            candidates,
            key=lambda x: x.visits,
            reverse=True,
        )
        return [self._node_to_path(node) for node in sorted_nodes[:n]]

    def _sample_topk(self, n: int) -> List[SampledPath]:
        """Sample top-k terminal states by value."""
        terminals = self.get_terminals()
        sorted_nodes = sorted(
            terminals,
            key=lambda x: x.average_value,
            reverse=True,
        )
        return [self._node_to_path(node) for node in sorted_nodes[:n]]

    def _sample_diverse(
        self,
        n: int,
        include_non_terminal: bool,
    ) -> List[SampledPath]:
        """
        Sample diverse paths using greedy selection.

        Selects paths that maximize diversity by choosing paths
        that differ most from already selected paths.
        """
        candidates = self._get_candidate_nodes(include_non_terminal)
        if not candidates:
            return []

        # Convert to paths with precomputed step sets for efficiency
        path_data = []
        for node in candidates:
            path = self._node_to_path(node)
            step_set = frozenset(path.steps)
            path_data.append((path, step_set))

        # Greedy diverse selection
        selected: List[SampledPath] = []
        selected_steps: List[frozenset] = []

        # Start with highest value path
        path_data.sort(key=lambda x: x[0].value, reverse=True)
        selected.append(path_data[0][0])
        selected_steps.append(path_data[0][1])
        path_data = path_data[1:]

        while len(selected) < n and path_data:
            # Find path most different from selected paths
            best_idx = 0
            best_diversity = -1

            for i, (path, steps) in enumerate(path_data):
                # Calculate minimum Jaccard distance to any selected path
                min_similarity = 1.0
                for sel_steps in selected_steps:
                    if not steps and not sel_steps:
                        similarity = 1.0
                    elif not steps or not sel_steps:
                        similarity = 0.0
                    else:
                        intersection = len(steps & sel_steps)
                        union = len(steps | sel_steps)
                        similarity = intersection / union if union > 0 else 0.0
                    min_similarity = min(min_similarity, similarity)

                diversity = 1.0 - min_similarity
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_idx = i

            selected.append(path_data[best_idx][0])
            selected_steps.append(path_data[best_idx][1])
            path_data.pop(best_idx)

        return selected

    def get_answer_distribution(self) -> dict:
        """
        Get distribution of answers in terminal states.

        Returns:
            Dict mapping answers to (count, total_value)
        """
        terminals = self.get_terminals()
        distribution = {}

        for terminal in terminals:
            answer = terminal.answer
            if answer not in distribution:
                distribution[answer] = {"count": 0, "total_value": 0.0, "nodes": []}
            distribution[answer]["count"] += 1
            distribution[answer]["total_value"] += terminal.average_value
            distribution[answer]["nodes"].append(terminal)

        # Calculate average value per answer
        for answer in distribution:
            count = distribution[answer]["count"]
            distribution[answer]["avg_value"] = (
                distribution[answer]["total_value"] / count
            )

        return distribution

    def consistency_score(self) -> float:
        """
        Calculate consistency of answers across terminal states.

        Returns:
            Score from 0 (all different) to 1 (all same)
        """
        terminals = self.get_terminals()
        if len(terminals) <= 1:
            return 1.0

        answers = [t.answer for t in terminals]
        if not any(answers):
            return 0.0

        # Find most common answer
        from collections import Counter
        counter = Counter(a for a in answers if a is not None)
        if not counter:
            return 0.0

        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(terminals)
