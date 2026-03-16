"""
Core data types for MCTS-Reasoning.

Canonical value objects shared across modules. These are plain data
containers with no behavioral dependencies on Generator, Evaluator, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType, TypedDict

if TYPE_CHECKING:
    from .node import Node


class Message(TypedDict):
    """A single message in an LLM conversation."""

    role: str
    content: str


State = NewType("State", str)
"""A reasoning state -- the accumulated text so far."""


def extend_state(state: State, continuation: str) -> State:
    """Append a continuation to a state, separated by a blank line."""
    return State(f"{state}\n\n{continuation}")


@dataclass
class Continuation:
    """A single reasoning continuation produced by a Generator."""

    text: State
    is_terminal: bool = False
    answer: str | None = None


@dataclass
class Evaluation:
    """Result of evaluating a terminal state."""

    score: float
    explanation: str = ""


@dataclass
class TerminalCheck:
    """Result of checking whether a state is terminal."""

    is_terminal: bool
    answer: str | None = None


@dataclass
class SampledPath:
    """A sampled reasoning path extracted from the search tree."""

    nodes: list
    answer: str | None = None
    value: float = 0.0
    visits: int = 0


@dataclass
class ConsensusResult:
    """Result of self-consistency voting across multiple paths."""

    answer: str
    confidence: float
    distribution: dict[str, int] = field(default_factory=dict)
    paths_used: int = 0


@dataclass
class SearchState:
    """Serialisable snapshot of an MCTS search in progress."""

    root: "Node"
    question: str
    terminal_states: list[dict] = field(default_factory=list)
    simulations_run: int = 0
    exploration_constant: float = 1.414
    max_children_per_node: int = 3
    max_rollout_depth: int = 5

    def save(self, path: str) -> None:
        """Persist this search state to a JSON file."""
        import json
        from pathlib import Path as FilePath

        data = {
            "question": self.question,
            "simulations_run": self.simulations_run,
            "exploration_constant": self.exploration_constant,
            "max_children_per_node": self.max_children_per_node,
            "max_rollout_depth": self.max_rollout_depth,
            "terminal_states": self.terminal_states,
            "root": self.root.to_dict(),
        }
        FilePath(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "SearchState":
        """Load a search state from a JSON file."""
        import json
        from pathlib import Path as FilePath

        from .node import Node

        data = json.loads(FilePath(path).read_text())
        return cls(
            root=Node.from_dict(data["root"]),
            question=data["question"],
            simulations_run=data["simulations_run"],
            terminal_states=data.get("terminal_states", []),
            exploration_constant=data.get("exploration_constant", 1.414),
            max_children_per_node=data.get("max_children_per_node", 3),
            max_rollout_depth=data.get("max_rollout_depth", 5),
        )
