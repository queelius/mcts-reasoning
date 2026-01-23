"""
MCTS Node: Represents a state in the reasoning tree.

Each node contains:
- A reasoning state (text generated so far)
- Cached children (generated continuations)
- Statistics for UCB1 (visits, value)
"""

from __future__ import annotations
import math
import json
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field


@dataclass
class Node:
    """A node in the MCTS reasoning tree."""

    state: str
    parent: Optional[Node] = None
    children: List[Node] = field(default_factory=list)

    # UCB1 statistics
    visits: int = 0
    value: float = 0.0

    # Cached continuations (generated on demand)
    _continuations: Optional[List[str]] = field(default=None, repr=False)
    _continuation_index: int = field(default=0, repr=False)

    # Terminal status
    is_terminal: bool = False
    answer: Optional[str] = None  # Extracted answer if terminal

    @property
    def is_leaf(self) -> bool:
        """Node has no children yet."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Node has no parent."""
        return self.parent is None

    @property
    def depth(self) -> int:
        """Depth from root."""
        if self.is_root:
            return 0
        return self.parent.depth + 1

    @property
    def average_value(self) -> float:
        """Average value (for selection)."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """
        UCB1 score for node selection.

        UCB1 = average_value + c * sqrt(ln(parent_visits) / visits)

        Balances exploitation (high average value) with
        exploration (low visits relative to parent).
        """
        if self.visits == 0:
            return float("inf")  # Unvisited nodes have highest priority

        if self.parent is None or self.parent.visits == 0:
            return self.average_value

        exploitation = self.average_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def has_untried_continuations(self) -> bool:
        """Check if there are cached continuations we haven't expanded yet."""
        if self._continuations is None:
            return True  # Haven't generated any yet
        return self._continuation_index < len(self._continuations)

    def get_next_continuation(self) -> Optional[str]:
        """Get the next untried continuation from cache."""
        if self._continuations is None:
            return None
        if self._continuation_index >= len(self._continuations):
            return None

        continuation = self._continuations[self._continuation_index]
        self._continuation_index += 1
        return continuation

    def set_continuations(self, continuations: List[str]):
        """Set the cached continuations for this node."""
        self._continuations = continuations
        self._continuation_index = 0

    def add_child(
        self, state: str, is_terminal: bool = False, answer: Optional[str] = None
    ) -> Node:
        """Add a child node with the given state."""
        child = Node(
            state=state,
            parent=self,
            is_terminal=is_terminal,
            answer=answer,
        )
        self.children.append(child)
        return child

    def best_child(self, exploration_constant: float = 1.414) -> Optional[Node]:
        """Select best child by UCB1."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def most_visited_child(self) -> Optional[Node]:
        """Select most visited child (for final answer selection)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

    def highest_value_child(self) -> Optional[Node]:
        """Select highest average value child."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.average_value)

    def path_from_root(self) -> List[Node]:
        """Get path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        state_preview = self.state[:50] + "..." if len(self.state) > 50 else self.state
        return f"Node(visits={self.visits}, value={self.value:.2f}, state='{state_preview}')"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize node and its subtree to a dictionary.

        Parent references are not included (reconstructed on load).
        Continuation cache is preserved for resuming expansion.
        """
        return {
            "state": self.state,
            "visits": self.visits,
            "value": self.value,
            "is_terminal": self.is_terminal,
            "answer": self.answer,
            # Continuation cache for resuming
            "_continuations": self._continuations,
            "_continuation_index": self._continuation_index,
            "_continuation_info": getattr(self, "_continuation_info", None),
            # Recursively serialize children
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional[Node] = None) -> Node:
        """
        Deserialize node and its subtree from a dictionary.

        Args:
            data: Dictionary from to_dict()
            parent: Parent node (for reconstructing parent references)

        Returns:
            Reconstructed Node with full subtree
        """
        node = cls(
            state=data["state"],
            parent=parent,
            visits=data.get("visits", 0),
            value=data.get("value", 0.0),
            is_terminal=data.get("is_terminal", False),
            answer=data.get("answer"),
        )

        # Restore continuation cache
        node._continuations = data.get("_continuations")
        node._continuation_index = data.get("_continuation_index", 0)
        if data.get("_continuation_info"):
            node._continuation_info = data["_continuation_info"]

        # Recursively deserialize children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, parent=node)
            node.children.append(child)

        return node

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Node:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
