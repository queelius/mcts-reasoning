"""
Sampling: Extract paths from the MCTS search tree.

Provides ABC-based strategies for sampling solution paths:
- ValueSampling: Paths ordered by average value (quality)
- VisitSampling: Paths ordered by visit count (confidence)
- DiverseSampling: Paths maximizing diversity (exploration)
- TopKSampling: Top-k terminal states by evaluation score

Example usage:
    result = mcts.search(question, simulations=100)

    # ABC-based usage
    from mcts_reasoning.sampling import PathSampler, ValueSampling
    sampler = PathSampler(result.root, strategy=ValueSampling())
    paths = sampler.sample(n=5)

    # Convenience string-based usage (backward compat)
    sampler = PathSampler(result.root)
    paths = sampler.sample(n=5, strategy="value")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

from .node import Node
from .types import SampledPath


# ---------------------------------------------------------------------------
# Iterative tree helpers
# ---------------------------------------------------------------------------


def _find_terminals(root: Node) -> list[Node]:
    """Find all terminal nodes via iterative stack-based DFS."""
    terminals: list[Node] = []
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal:
            terminals.append(node)
        stack.extend(node.children)
    return terminals


def _find_leaves(root: Node) -> list[Node]:
    """Find all leaf nodes (no children) via iterative stack-based DFS."""
    leaves: list[Node] = []
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if not node.children:
            leaves.append(node)
        else:
            stack.extend(node.children)
    return leaves


def _node_to_sampled_path(node: Node) -> SampledPath:
    """Convert a Node to a SampledPath."""
    path_nodes = node.path_from_root()
    return SampledPath(
        nodes=path_nodes,
        answer=node.answer if node.is_terminal else None,
        value=node.average_value,
        visits=node.visits,
    )


# ---------------------------------------------------------------------------
# Sampling strategy ABC and implementations
# ---------------------------------------------------------------------------


class SamplingStrategy(ABC):
    """Abstract base class for path sampling strategies."""

    @abstractmethod
    def sample(self, root: Node, n: int) -> list[SampledPath]:
        """
        Sample n paths from the tree rooted at root.

        Args:
            root: Root node of the search tree.
            n: Maximum number of paths to return.

        Returns:
            List of SampledPath objects, up to n.
        """


class ValueSampling(SamplingStrategy):
    """Sort terminal nodes by average value (descending), return top n."""

    def sample(self, root: Node, n: int) -> list[SampledPath]:
        terminals = _find_terminals(root)
        sorted_nodes = sorted(terminals, key=lambda nd: nd.average_value, reverse=True)
        return [_node_to_sampled_path(nd) for nd in sorted_nodes[:n]]


class VisitSampling(SamplingStrategy):
    """Sort terminal nodes by visit count (descending), return top n."""

    def sample(self, root: Node, n: int) -> list[SampledPath]:
        terminals = _find_terminals(root)
        sorted_nodes = sorted(terminals, key=lambda nd: nd.visits, reverse=True)
        return [_node_to_sampled_path(nd) for nd in sorted_nodes[:n]]


class TopKSampling(SamplingStrategy):
    """Top-k terminal states by value -- same ranking as ValueSampling."""

    def sample(self, root: Node, n: int) -> list[SampledPath]:
        terminals = _find_terminals(root)
        sorted_nodes = sorted(terminals, key=lambda nd: nd.average_value, reverse=True)
        return [_node_to_sampled_path(nd) for nd in sorted_nodes[:n]]


class DiverseSampling(SamplingStrategy):
    """
    Greedy diverse selection.

    Picks one representative per unique answer first (highest value
    within that answer), then fills remaining slots by Jaccard distance.
    """

    def sample(self, root: Node, n: int) -> list[SampledPath]:
        terminals = _find_terminals(root)
        if not terminals:
            return []

        # Build paths with precomputed step sets
        path_data: list[tuple[SampledPath, frozenset[str]]] = []
        for node in terminals:
            path = _node_to_sampled_path(node)
            step_set = frozenset(self._path_steps(path))
            path_data.append((path, step_set))

        # Greedy diverse selection using index marking
        selected: list[SampledPath] = []
        selected_steps: list[frozenset[str]] = []
        used: set[int] = set()

        # Start with highest value path
        path_data.sort(key=lambda x: x[0].value, reverse=True)
        selected.append(path_data[0][0])
        selected_steps.append(path_data[0][1])
        used.add(0)

        while len(selected) < n and len(used) < len(path_data):
            best_idx = -1
            best_diversity = -1.0

            for i, (path, steps) in enumerate(path_data):
                if i in used:
                    continue

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

            if best_idx >= 0:
                selected.append(path_data[best_idx][0])
                selected_steps.append(path_data[best_idx][1])
                used.add(best_idx)

        return selected

    @staticmethod
    def _path_steps(path: SampledPath) -> list[str]:
        """Extract individual reasoning steps from a path."""
        result: list[str] = []
        nodes = path.nodes
        for i in range(1, len(nodes)):
            prev_state = nodes[i - 1].state
            step = nodes[i].state[len(prev_state) :].strip()
            if step:
                result.append(step)
        return result


# ---------------------------------------------------------------------------
# Strategy registry for string-based dispatch (backward compat)
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type[SamplingStrategy]] = {
    "value": ValueSampling,
    "visits": VisitSampling,
    "diverse": DiverseSampling,
    "topk": TopKSampling,
}


def _resolve_strategy(strategy: str | SamplingStrategy | None) -> SamplingStrategy:
    """Resolve a strategy from a string name or instance."""
    if strategy is None:
        return ValueSampling()
    if isinstance(strategy, SamplingStrategy):
        return strategy
    if isinstance(strategy, str):
        if strategy not in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: {list(_STRATEGY_REGISTRY.keys())}"
            )
        return _STRATEGY_REGISTRY[strategy]()
    raise TypeError(f"Expected str or SamplingStrategy, got {type(strategy)}")


# ---------------------------------------------------------------------------
# PathSampler facade
# ---------------------------------------------------------------------------


class PathSampler:
    """
    Convenience facade for sampling paths from an MCTS search tree.

    Supports both ABC-based and string-based strategy dispatch.
    """

    def __init__(
        self,
        root: Node,
        strategy: SamplingStrategy | None = None,
        consensus: object | None = None,
    ):
        from .consensus import MajorityVote

        self.root = root
        self.strategy = strategy or ValueSampling()
        self.consensus = consensus or MajorityVote()
        self._terminals: list[Node] | None = None

    def get_terminals(self) -> list[Node]:
        """Get all terminal nodes in the tree (cached)."""
        if self._terminals is None:
            self._terminals = _find_terminals(self.root)
        return self._terminals

    def sample(
        self,
        n: int = 5,
        strategy: str | SamplingStrategy | None = None,
        include_non_terminal: bool = False,
    ) -> list[SampledPath]:
        """
        Sample n paths from the tree.

        Args:
            n: Number of paths to sample.
            strategy: Override strategy (string name or SamplingStrategy instance).
                      Uses the instance default if None.
            include_non_terminal: If True, include non-terminal leaf paths.
                                  Only applies to value/visits string strategies.
        """
        resolved = _resolve_strategy(strategy) if strategy else self.strategy

        if include_non_terminal and isinstance(
            resolved, (ValueSampling, VisitSampling)
        ):
            return self._sample_with_non_terminal(n, resolved)

        return resolved.sample(self.root, n)

    def _sample_with_non_terminal(
        self, n: int, strategy: SamplingStrategy
    ) -> list[SampledPath]:
        """Sample including non-terminal leaf nodes."""
        terminals = self.get_terminals()
        leaves = _find_leaves(self.root)

        # Combine without duplicates
        seen = set(id(t) for t in terminals)
        candidates = list(terminals)
        for leaf in leaves:
            if id(leaf) not in seen:
                candidates.append(leaf)
                seen.add(id(leaf))

        paths = [_node_to_sampled_path(nd) for nd in candidates]

        if isinstance(strategy, VisitSampling):
            paths.sort(key=lambda p: p.visits, reverse=True)
        else:
            paths.sort(key=lambda p: p.value, reverse=True)

        return paths[:n]

    def vote(self) -> object:
        """Run consensus voting on sampled paths."""
        paths = self.sample()
        return self.consensus.vote(paths)  # type: ignore[attr-defined]

    def get_answer_distribution(self) -> dict:
        """
        Get distribution of answers in terminal states.

        Returns:
            Dict mapping answers to {count, total_value, avg_value, nodes}.
        """
        terminals = self.get_terminals()
        distribution: dict = {}

        for terminal in terminals:
            answer = terminal.answer
            if answer not in distribution:
                distribution[answer] = {
                    "count": 0,
                    "total_value": 0.0,
                    "nodes": [],
                }
            distribution[answer]["count"] += 1
            distribution[answer]["total_value"] += terminal.average_value
            distribution[answer]["nodes"].append(terminal)

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
            Score from 0 (all different) to 1 (all same).
        """
        terminals = self.get_terminals()
        if len(terminals) <= 1:
            return 1.0

        answers = [t.answer for t in terminals]
        if not any(answers):
            return 0.0

        counter = Counter(a for a in answers if a is not None)
        if not counter:
            return 0.0

        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(terminals)

    # ------------------------------------------------------------------
    # Legacy self-consistency API (kept for backward compat)
    # ------------------------------------------------------------------

    def self_consistency_vote(
        self,
        weighted: bool = True,
        normalize_answers: bool = True,
    ) -> dict:
        """
        Apply self-consistency voting to select the best answer.

        Args:
            weighted: If True, weight votes by path value.
            normalize_answers: If True, normalize answers before comparing.

        Returns:
            Dict with answer, confidence, votes, weighted_votes, total_votes.
        """
        terminals = self.get_terminals()
        if not terminals:
            return {
                "answer": None,
                "confidence": 0.0,
                "votes": {},
                "weighted_votes": {},
                "total_votes": 0,
            }

        def normalize(answer: Optional[str]) -> Optional[str]:
            if answer is None:
                return None
            if not normalize_answers:
                return answer
            result = answer.lower().strip()
            result = result.replace("$", "").replace(",", "")
            return result

        votes = Counter()
        weighted_votes: dict[str, float] = {}

        for terminal in terminals:
            answer = normalize(terminal.answer)
            if answer is not None:
                votes[answer] += 1
                if answer not in weighted_votes:
                    weighted_votes[answer] = 0.0
                weighted_votes[answer] += terminal.average_value

        if not votes:
            return {
                "answer": None,
                "confidence": 0.0,
                "votes": {},
                "weighted_votes": {},
                "total_votes": 0,
            }

        total_votes = sum(votes.values())

        if weighted:
            winner = max(weighted_votes.keys(), key=lambda a: weighted_votes[a])
            total_weight = sum(weighted_votes.values())
            confidence = (
                weighted_votes[winner] / total_weight if total_weight > 0 else 0.0
            )
        else:
            winner = votes.most_common(1)[0][0]
            confidence = votes[winner] / total_votes

        return {
            "answer": winner,
            "confidence": confidence,
            "votes": dict(votes),
            "weighted_votes": weighted_votes,
            "total_votes": total_votes,
        }

    def majority_vote(self) -> tuple:
        """Simple majority voting. Returns (answer, confidence)."""
        result = self.self_consistency_vote(weighted=False)
        return result["answer"], result["confidence"]

    def weighted_vote(self) -> tuple:
        """Value-weighted voting. Returns (answer, confidence)."""
        result = self.self_consistency_vote(weighted=True)
        return result["answer"], result["confidence"]


__all__ = [
    "SamplingStrategy",
    "ValueSampling",
    "VisitSampling",
    "DiverseSampling",
    "TopKSampling",
    "PathSampler",
]
