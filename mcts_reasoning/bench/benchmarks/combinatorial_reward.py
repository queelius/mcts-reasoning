"""
Honest reward signals for combinatorial problems.

These evaluators do NOT use ground truth. They use:
1. The problem constraints (public, part of the problem statement)
2. LLM-as-judge for parsing candidate answers
3. Classical constraint checking on the parsed answer

The ground truth solution is used ONLY for benchmarking (measuring
whether MCTS got the right answer), never for the reward signal.
"""

from __future__ import annotations

from ...evaluator import Evaluator, Evaluation
from ...types import Message
from .combinatorial_verifier import (
    NAMES, SLOTS, COLORS,
    verify_seating, verify_assignment, verify_coloring,
)


class CombinatorialRewardEvaluator(Evaluator):
    """Reward signal from constraint verification. No ground truth used.

    Pipeline:
    1. LLM parses candidate answer from natural text
    2. Classical solver checks parsed answer against problem constraints
    3. Score = 1.0 if valid, 0.0 if invalid

    The constraints come from problem.metadata, which is public
    information (it's in the problem statement).
    """

    def __init__(self, provider, problem_metadata: dict):
        self.provider = provider
        self.meta = problem_metadata

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        ptype = self.meta["type"]

        if ptype == "seating":
            return self._eval_seating(state)
        elif ptype == "assignment":
            return self._eval_assignment(state)
        elif ptype == "coloring":
            return self._eval_coloring(state)

        return Evaluation(score=0.0, explanation="Unknown problem type")

    def _parse_with_llm(self, response: str, instruction: str) -> str:
        """Use LLM to extract structured answer from natural text."""
        return self.provider.generate(
            [
                Message(role="system", content=instruction),
                Message(role="user", content=response),
            ],
            max_tokens=100,
            temperature=0.0,
        )

    def _eval_seating(self, state: str) -> Evaluation:
        n = self.meta["n"]
        names = NAMES[:n]
        constraints = self.meta["constraints"]

        parsed = self._parse_with_llm(
            state,
            f"Extract the seating order. People: {', '.join(names)}. "
            f"Reply with ONLY the names in order, comma-separated.",
        )

        order = [name.strip() for name in parsed.strip().split(",")]
        if len(order) != n or set(order) != set(names):
            return Evaluation(score=0.0, explanation=f"Parse failed: {parsed[:50]}")

        valid = verify_seating(names, constraints, order)
        return Evaluation(
            score=1.0 if valid else 0.0,
            explanation=f"{'Valid' if valid else 'Invalid'}: {', '.join(order)}",
        )

    def _eval_assignment(self, state: str) -> Evaluation:
        n = self.meta["n"]
        items = NAMES[:n]
        slots = SLOTS[:n]
        constraints = self.meta["constraints"]

        parsed = self._parse_with_llm(
            state,
            f"Extract the assignment. People: {', '.join(items)}. "
            f"Days: {', '.join(slots)}. Reply with lines like: Alice: Monday",
        )

        assignment = {}
        for line in parsed.strip().split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                person = parts[0].strip()
                day = parts[1].strip()
                if person in items and day in slots:
                    assignment[person] = day

        if len(assignment) != n:
            return Evaluation(score=0.0, explanation=f"Parse failed: {parsed[:50]}")

        valid = verify_assignment(items, slots, constraints, assignment)
        return Evaluation(
            score=1.0 if valid else 0.0,
            explanation=f"{'Valid' if valid else 'Invalid'}: {assignment}",
        )

    def _eval_coloring(self, state: str) -> Evaluation:
        n = self.meta["n_nodes"]
        names = NAMES[:n]
        colors = COLORS[:self.meta["n_colors"]]
        edges = self.meta["edges"]

        parsed = self._parse_with_llm(
            state,
            f"Extract the color assignment. People: {', '.join(names)}. "
            f"Colors: {', '.join(colors)}. Reply with lines like: Alice: red",
        )

        assignment = {}
        for line in parsed.strip().split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                person = parts[0].strip()
                color = parts[1].strip().lower()
                if person in names and color in colors:
                    assignment[person] = color

        if len(assignment) != n:
            return Evaluation(score=0.0, explanation=f"Parse failed: {parsed[:50]}")

        valid = verify_coloring(edges, assignment)
        return Evaluation(
            score=1.0 if valid else 0.0,
            explanation=f"{'Valid' if valid else 'Invalid'}: {assignment}",
        )
