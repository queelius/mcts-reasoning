"""
Built-in benchmarks for MCTS-Reasoning.

Available benchmarks:
  knights    – KnightsAndKnaves logic puzzles (15 problems)
  arithmetic – ArithmeticChains multi-step arithmetic (20 problems)

Use get_benchmark(name) to instantiate by name.
"""

from __future__ import annotations

from .knights import KnightsAndKnaves
from .arithmetic import ArithmeticChains


_REGISTRY: dict[str, type] = {
    "knights": KnightsAndKnaves,
    "arithmetic": ArithmeticChains,
}


def get_benchmark(name: str):
    """
    Instantiate a benchmark by name.

    Args:
        name: Benchmark identifier.  One of "knights", "arithmetic".

    Returns:
        Benchmark instance.

    Raises:
        KeyError: If *name* is not a registered benchmark.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}") from None
    return cls()


__all__ = [
    "KnightsAndKnaves",
    "ArithmeticChains",
    "get_benchmark",
]
