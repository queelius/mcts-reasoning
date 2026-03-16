"""
ArithmeticChains benchmark.

Generates 20 multi-step arithmetic problems programmatically.
Ground truth is computed directly via Python, so it is exact.

Problems are structured as chains of 3-5 sequential arithmetic operations
(+, -, *, //) applied to moderately-sized integers, making them tractable
yet non-trivial for mental arithmetic or single-pass LLM evaluation.

Difficulties:
  easy   - 2-step chains, small numbers
  medium - 3-4 step chains, moderate numbers
  hard   - 5-step chains, larger numbers with mixed operations
"""

from __future__ import annotations

from ..benchmark import Benchmark, Problem


class ArithmeticChains(Benchmark):
    """Programmatically generated multi-step arithmetic chains."""

    @property
    def name(self) -> str:
        return "arithmetic_chains"

    def problems(self) -> list[Problem]:
        return _PROBLEMS


# ---------------------------------------------------------------------------
# Problem generation helpers
# ---------------------------------------------------------------------------
# Each problem is specified as a (description, ground_truth_int, difficulty)
# triple.  Ground truth is pre-computed to avoid any code-execution concerns.
# ---------------------------------------------------------------------------


def _p(description: str, ground_truth: int, difficulty: str) -> Problem:
    """Convenience constructor for an arithmetic Problem."""
    return Problem(
        question=f"Calculate: {description}. What is the result?",
        ground_truth=str(ground_truth),
        domain="arithmetic",
        difficulty=difficulty,
        metadata={"expression": description},
    )


_PROBLEMS: list[Problem] = [
    # ============================= EASY (7) ================================
    # 2-step chains, small numbers
    _p("(3 + 7) * 4", 40, "easy"),
    _p("(15 - 6) * 3", 27, "easy"),
    _p("(8 + 12) // 4", 5, "easy"),
    _p("(20 - 5) + 9", 24, "easy"),
    _p("(6 * 7) - 10", 32, "easy"),
    _p("(18 // 3) * 5", 30, "easy"),
    _p("(9 + 16) - 8", 17, "easy"),
    # ============================ MEDIUM (7) ================================
    # 3-4 step chains, moderate numbers
    _p("(5 + 3) * 4 - 7", 25, "medium"),
    _p("(12 * 3 + 6) // 2", 21, "medium"),
    _p("(100 - 37) * 2 + 13", 139, "medium"),
    _p("(7 * 8) - (5 * 6)", 26, "medium"),
    _p("(50 // 5 + 3) * 7", 91, "medium"),
    _p("(48 // 4) * 3 - 11", 25, "medium"),
    _p("(9 + 11) * (6 - 2)", 80, "medium"),
    # ============================== HARD (6) ================================
    # 5-step chains, larger numbers, mixed operations
    _p("(17 * 3 + 9) * 2 - 14", 106, "hard"),
    _p("(144 // 12 + 8) * 5 - 20", 80, "hard"),
    _p("(23 * 4 - 15) // 3 + 11", 36, "hard"),
    _p("(7 * 6 + 5) * (8 - 3)", 235, "hard"),
    _p("(256 // 8 - 7) * 9 + 14", 239, "hard"),
    _p("(13 + 7) * 4 - (18 // 3)", 74, "hard"),
]
