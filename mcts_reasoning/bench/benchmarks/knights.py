"""
KnightsAndKnaves benchmark.

15 hand-crafted logic puzzles with two characters (A and B).
Knights always tell the truth; knaves always lie.
All puzzles are verified to have exactly one solution by exhaustive
truth-table enumeration over the four (A-type, B-type) assignments.

Difficulties:
  easy   – direct single-statement deduction (no case analysis needed)
  medium – one round of case analysis required
  hard   – multiple interlocking constraints
"""

from __future__ import annotations

from ..benchmark import Benchmark, Problem


class KnightsAndKnaves(Benchmark):
    """Hand-crafted Knights and Knaves logic puzzles."""

    @property
    def name(self) -> str:
        return "knights_and_knaves"

    def problems(self) -> list[Problem]:
        return _PROBLEMS


# ---------------------------------------------------------------------------
# Verified problem set
# All 15 puzzles were verified with an exhaustive truth-table check.
# Ground truth format: "A is a <type>, B is a <type>"
# ---------------------------------------------------------------------------

_PROBLEMS: list[Problem] = [
    # ============================= EASY =====================================
    Problem(
        question=(
            "A says: 'A and B are the same type.' "
            "B says: 'A is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knight",
        domain="logic",
        difficulty="easy",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'Both of us are knaves.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knight",
        domain="logic",
        difficulty="easy",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'At least one of us is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knave",
        domain="logic",
        difficulty="easy",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'We are both knights.' "
            "B says: 'A is lying.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knight",
        domain="logic",
        difficulty="easy",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'Exactly one of us is a knight.' "
            "B says: 'A is a knight.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knave",
        domain="logic",
        difficulty="easy",
        metadata={"verified": True},
    ),
    # ============================ MEDIUM ====================================
    Problem(
        question=(
            "A says: 'B is a knave.' "
            "B says: 'A and B are the same type.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knave",
        domain="logic",
        difficulty="medium",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'B is a knave and I am a knight.' "
            "B says: 'A is a knight.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knave",
        domain="logic",
        difficulty="medium",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'At least one of us is a knight.' "
            "B says: 'A is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knave",
        domain="logic",
        difficulty="medium",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'I am a knight or B is a knight.' "
            "B says: 'A is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knave",
        domain="logic",
        difficulty="medium",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'I am a knight and B is a knight.' "
            "B says: 'A is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knight",
        domain="logic",
        difficulty="medium",
        metadata={"verified": True},
    ),
    # ============================== HARD ====================================
    Problem(
        question=(
            "A says: 'I am a knight if and only if B is a knight.' "
            "B says: 'A and B are not the same type.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knight",
        domain="logic",
        difficulty="hard",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'B is a knight.' "
            "B says: 'I am a knave or A is a knight.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knight",
        domain="logic",
        difficulty="hard",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'I am a knave and B is a knight.' "
            "B says: 'A is not a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knave, B is a knave",
        domain="logic",
        difficulty="hard",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'Exactly one of us is a knight.' "
            "B says: 'A is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knave",
        domain="logic",
        difficulty="hard",
        metadata={"verified": True},
    ),
    Problem(
        question=(
            "A says: 'I am a knight if and only if B is a knave.' "
            "B says: 'A is a knave.' "
            "What are A and B?"
        ),
        ground_truth="A is a knight, B is a knave",
        domain="logic",
        difficulty="hard",
        metadata={"verified": True},
    ),
]
