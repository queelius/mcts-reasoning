"""
Stream-based data structures for Unix-style piping.

Data flows through commands as typed streams, not just text.
Commands can consume and produce different stream types.
"""

import json
from typing import Any, List, Dict, Iterator, Union, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from ..core import MCTS, MCTSNode
from ..sampling import SampledPath


class Stream(ABC):
    """
    Base class for data streams flowing through pipes.

    Streams are iterables that can be consumed by commands.
    Commands transform streams into other streams.
    """

    @abstractmethod
    def items(self) -> Iterator[Any]:
        """Iterate over stream items."""
        pass

    @abstractmethod
    def to_text(self) -> str:
        """Convert stream to text representation."""
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Convert stream to JSON representation."""
        pass

    def __iter__(self):
        return self.items()


class TextStream(Stream):
    """Stream of text lines."""

    def __init__(self, lines: Union[str, List[str]]):
        if isinstance(lines, str):
            self.lines = lines.splitlines()
        else:
            self.lines = list(lines)

    def items(self) -> Iterator[str]:
        return iter(self.lines)

    def to_text(self) -> str:
        return '\n'.join(self.lines)

    def to_json(self) -> str:
        return json.dumps({'lines': self.lines}, indent=2)

    def __len__(self):
        return len(self.lines)


class MCTSStream(Stream):
    """
    Stream containing an MCTS tree.

    The fundamental data structure - most commands operate on this.
    """

    def __init__(self, mcts: MCTS, metadata: Optional[Dict] = None):
        self.mcts = mcts
        self.metadata = metadata or {}

    def items(self) -> Iterator[MCTS]:
        """Single-item stream containing the MCTS tree."""
        yield self.mcts

    def to_text(self) -> str:
        """Human-readable text representation."""
        if not self.mcts.root:
            return "Empty MCTS tree"

        stats = self.mcts.stats
        lines = [
            f"MCTS Tree ({self.metadata.get('description', 'unnamed')})",
            f"  Nodes: {stats['total_nodes']}",
            f"  Max depth: {stats['max_depth']}",
            f"  Root visits: {stats['root_visits']}",
            f"  Best value: {stats['best_value']:.3f}",
        ]

        if 'question' in stats.get('metadata', {}):
            lines.insert(1, f"  Question: {stats['metadata']['question']}")

        return '\n'.join(lines)

    def to_json(self) -> str:
        """JSON representation of the tree."""
        data = self.mcts.to_json()
        data['stream_metadata'] = self.metadata
        return json.dumps(data, indent=2)


class PathStream(Stream):
    """
    Stream of sampled paths from an MCTS tree.

    Produced by sampling commands, consumed by analysis/filter commands.
    """

    def __init__(self, paths: List[SampledPath], source_mcts: Optional[MCTS] = None):
        self.paths = paths
        self.source_mcts = source_mcts

    def items(self) -> Iterator[SampledPath]:
        return iter(self.paths)

    def to_text(self) -> str:
        """Human-readable representation."""
        if not self.paths:
            return "No paths"

        lines = [f"{len(self.paths)} sampled paths:"]

        for i, path in enumerate(self.paths, 1):
            avg_value = path.total_value / max(path.total_visits, 1)
            lines.append(
                f"  {i}. length={path.length}, "
                f"value={avg_value:.3f}, "
                f"visits={path.total_visits}"
            )

        return '\n'.join(lines)

    def to_json(self) -> str:
        """JSON representation."""
        return json.dumps({
            'num_paths': len(self.paths),
            'paths': [p.to_dict() for p in self.paths]
        }, indent=2)

    def __len__(self):
        return len(self.paths)


class SolutionStream(Stream):
    """
    Stream of solutions (final states/answers).

    The end result - what users typically want.
    """

    def __init__(self, solutions: List[str], metadata: Optional[List[Dict]] = None):
        self.solutions = solutions
        self.metadata = metadata or [{} for _ in solutions]

    def items(self) -> Iterator[str]:
        return iter(self.solutions)

    def to_text(self) -> str:
        """Human-readable solutions."""
        if not self.solutions:
            return "No solutions"

        if len(self.solutions) == 1:
            return self.solutions[0]

        lines = [f"{len(self.solutions)} solutions:"]
        for i, sol in enumerate(self.solutions, 1):
            # Truncate long solutions
            display = sol if len(sol) < 200 else sol[:197] + "..."
            lines.append(f"\n--- Solution {i} ---")
            lines.append(display)

        return '\n'.join(lines)

    def to_json(self) -> str:
        """JSON representation."""
        return json.dumps({
            'num_solutions': len(self.solutions),
            'solutions': [
                {'text': sol, 'metadata': meta}
                for sol, meta in zip(self.solutions, self.metadata)
            ]
        }, indent=2)

    def __len__(self):
        return len(self.solutions)


class StatsStream(Stream):
    """Stream of statistics/analysis results."""

    def __init__(self, stats: Dict[str, Any]):
        self.stats = stats

    def items(self) -> Iterator[Dict]:
        yield self.stats

    def to_text(self) -> str:
        """Human-readable stats."""
        lines = []

        def format_value(v):
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)

        def format_dict(d, indent=0):
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    format_dict(value, indent + 1)
                else:
                    lines.append(f"{prefix}{key}: {format_value(value)}")

        format_dict(self.stats)
        return '\n'.join(lines)

    def to_json(self) -> str:
        """JSON representation."""
        return json.dumps(self.stats, indent=2)


def stream_from_json(json_str: str) -> Stream:
    """
    Reconstruct a stream from JSON.

    Used for loading from files or receiving piped JSON data.
    """
    data = json.loads(json_str)

    # Detect stream type
    if 'root' in data and 'config' in data:
        # MCTSStream
        from ..reasoning import ReasoningMCTS
        mcts = ReasoningMCTS.from_json(data)
        metadata = data.get('stream_metadata', {})
        return MCTSStream(mcts, metadata)

    elif 'paths' in data:
        # PathStream
        # Note: Can't fully reconstruct without MCTS reference
        # For now, return as JSON text
        return TextStream([json.dumps(data, indent=2)])

    elif 'solutions' in data:
        # SolutionStream
        solutions = [s['text'] for s in data['solutions']]
        metadata = [s.get('metadata', {}) for s in data['solutions']]
        return SolutionStream(solutions, metadata)

    elif 'lines' in data:
        # TextStream
        return TextStream(data['lines'])

    else:
        # Generic stats/dict
        return StatsStream(data)


def stream_from_file(filepath: str) -> Stream:
    """Load a stream from a file."""
    from pathlib import Path
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    content = path.read_text()

    # Try JSON first
    try:
        return stream_from_json(content)
    except json.JSONDecodeError:
        # Fall back to text
        return TextStream(content)
