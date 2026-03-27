"""
Combinatorial constraint problems: easy to generate, easy to verify, hard to solve.

Problems are generated with a classical solver that provides the ground truth.
An LLM-as-judge verifier checks whether a candidate solution matches.

Problem types:
- Seating arrangements with constraints ("A can't sit next to B")
- Scheduling with conflicts ("Class X and Y can't overlap")
- Assignment problems ("Match N items to N slots with constraints")
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from ..benchmark import Benchmark, Problem


# ─── Classical solvers ────────────────────────────────────────

def solve_seating(n_people: int, constraints: list[tuple[int, int]]) -> list[int] | None:
    """Brute-force solve a seating arrangement. Returns a valid permutation or None."""
    for perm in itertools.permutations(range(n_people)):
        valid = True
        for a, b in constraints:
            # "a and b can't sit next to each other"
            ia = perm.index(a)
            ib = perm.index(b)
            if abs(ia - ib) == 1:
                valid = False
                break
        if valid:
            return list(perm)
    return None


def solve_assignment(n: int, constraints: list[tuple[int, int]]) -> dict[int, int] | None:
    """Solve: assign n items to n slots. constraints = [(item, forbidden_slot), ...]."""
    def backtrack(item, assignment, used_slots):
        if item == n:
            return dict(assignment)
        for slot in range(n):
            if slot in used_slots:
                continue
            if (item, slot) in constraint_set:
                continue
            assignment[item] = slot
            used_slots.add(slot)
            result = backtrack(item + 1, assignment, used_slots)
            if result is not None:
                return result
            del assignment[item]
            used_slots.remove(slot)
        return None

    constraint_set = set(constraints)
    return backtrack(0, {}, set())


def solve_coloring(n_nodes: int, edges: list[tuple[int, int]], n_colors: int) -> dict[int, int] | None:
    """Graph coloring: assign colors to nodes so no adjacent nodes share a color."""
    def backtrack(node, assignment):
        if node == n_nodes:
            return dict(assignment)
        for color in range(n_colors):
            # Check if any neighbor has this color
            conflict = False
            for a, b in edges:
                if a == node and b in assignment and assignment[b] == color:
                    conflict = True
                    break
                if b == node and a in assignment and assignment[a] == color:
                    conflict = True
                    break
            if not conflict:
                assignment[node] = color
                result = backtrack(node + 1, assignment)
                if result is not None:
                    return result
                del assignment[node]
        return None

    return backtrack(0, {})


# ─── Problem generators ──────────────────────────────────────

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
COLORS = ["red", "blue", "green", "yellow"]
SLOTS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ROOMS = ["Room A", "Room B", "Room C", "Room D"]


def generate_seating_problem(n: int = 4, n_constraints: int = 2) -> Problem | None:
    """Generate a seating arrangement problem with verifiable solution."""
    names = NAMES[:n]
    # Generate random constraints
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if n_constraints > len(all_pairs):
        n_constraints = len(all_pairs)
    constraint_pairs = random.sample(all_pairs, n_constraints)

    solution = solve_seating(n, constraint_pairs)
    if solution is None:
        return None  # overconstrained, skip

    # Format as natural language
    constraint_text = []
    for a, b in constraint_pairs:
        constraint_text.append(f"{names[a]} cannot sit next to {names[b]}")

    question = (
        f"{n} people need to sit in a row: {', '.join(names)}.\n"
        f"Constraints:\n"
        + "\n".join(f"- {c}" for c in constraint_text)
        + "\n\nFind a valid seating arrangement (left to right)."
    )

    answer_names = [names[i] for i in solution]
    ground_truth = ", ".join(answer_names)

    return Problem(
        question=question,
        ground_truth=ground_truth,
        domain="combinatorial",
        difficulty="medium" if n <= 4 else "hard",
        metadata={
            "type": "seating",
            "n": n,
            "n_constraints": n_constraints,
            "constraints": [(names[a], names[b]) for a, b in constraint_pairs],
            "solution_perm": solution,
        },
    )


def generate_assignment_problem(n: int = 4, n_constraints: int = 3) -> Problem | None:
    """Generate an assignment problem (items to slots with forbidden pairs)."""
    items = NAMES[:n]
    slots = SLOTS[:n]

    all_pairs = [(i, j) for i in range(n) for j in range(n)]
    if n_constraints > len(all_pairs):
        n_constraints = len(all_pairs)
    constraints = random.sample(all_pairs, n_constraints)

    solution = solve_assignment(n, constraints)
    if solution is None:
        return None

    constraint_text = []
    for item, slot in constraints:
        constraint_text.append(f"{items[item]} cannot be assigned to {slots[slot]}")

    question = (
        f"Assign each person to a different day:\n"
        f"People: {', '.join(items)}\n"
        f"Days: {', '.join(slots)}\n"
        f"Constraints:\n"
        + "\n".join(f"- {c}" for c in constraint_text)
        + "\n\nFind a valid assignment."
    )

    ground_truth = "; ".join(f"{items[i]}: {slots[solution[i]]}" for i in range(n))

    return Problem(
        question=question,
        ground_truth=ground_truth,
        domain="combinatorial",
        difficulty="medium" if n <= 4 else "hard",
        metadata={
            "type": "assignment",
            "n": n,
            "n_constraints": n_constraints,
            "solution": {items[i]: slots[solution[i]] for i in range(n)},
        },
    )


def generate_coloring_problem(n_nodes: int = 4, edge_prob: float = 0.4,
                              n_colors: int = 3) -> Problem | None:
    """Generate a graph coloring problem."""
    names = NAMES[:n_nodes]
    colors = COLORS[:n_colors]

    # Random graph
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                edges.append((i, j))

    if not edges:
        return None  # trivial

    solution = solve_coloring(n_nodes, edges, n_colors)
    if solution is None:
        return None

    edge_text = [f"{names[a]} and {names[b]} are connected" for a, b in edges]

    question = (
        f"Color each person with one of: {', '.join(colors)}.\n"
        f"Connected people cannot share a color.\n"
        f"Connections:\n"
        + "\n".join(f"- {e}" for e in edge_text)
        + "\n\nFind a valid coloring."
    )

    ground_truth = "; ".join(f"{names[i]}: {colors[solution[i]]}" for i in range(n_nodes))

    return Problem(
        question=question,
        ground_truth=ground_truth,
        domain="combinatorial",
        difficulty="easy" if n_nodes <= 3 else ("medium" if n_nodes <= 5 else "hard"),
        metadata={
            "type": "coloring",
            "n_nodes": n_nodes,
            "n_colors": n_colors,
            "edges": [(names[a], names[b]) for a, b in edges],
            "solution": {names[i]: colors[solution[i]] for i in range(n_nodes)},
        },
    )


# ─── LLM-as-judge verifier ───────────────────────────────────

def verify_combinatorial_answer(provider, problem: Problem, candidate_answer: str) -> bool:
    """Use an LLM to check if a candidate answer satisfies the constraints.

    Verification is easier than generation: the LLM just needs to check
    each constraint against the proposed solution.
    """
    from ...types import Message

    messages = [
        Message(role="system", content=(
            "You verify whether a proposed solution satisfies all constraints. "
            "Check EACH constraint one by one. "
            "Reply with ONLY 'VALID' or 'INVALID: <which constraint is violated>'."
        )),
        Message(role="user", content=(
            f"Problem:\n{problem.question}\n\n"
            f"Proposed solution:\n{candidate_answer}\n\n"
            f"Known correct solution:\n{problem.ground_truth}\n\n"
            f"Does the proposed solution satisfy ALL constraints? "
            f"(It doesn't need to match the known solution exactly, "
            f"just satisfy all the constraints.)"
        )),
    ]

    resp = provider.generate(messages, max_tokens=200, temperature=0.0)
    return "VALID" in resp.upper() and "INVALID" not in resp.upper()


# ─── Benchmark ────────────────────────────────────────────────

class CombinatorialBenchmark(Benchmark):
    """Combinatorial constraint satisfaction problems."""

    def __init__(self, n_problems: int = 15, difficulty: str = "mixed"):
        self._n = n_problems
        self._difficulty = difficulty
        self._problems: list[Problem] | None = None

    @property
    def name(self) -> str:
        return "combinatorial"

    def problems(self) -> list[Problem]:
        if self._problems is not None:
            return self._problems

        self._problems = []
        generators = [
            lambda: generate_seating_problem(4, 2),
            lambda: generate_seating_problem(5, 3),
            lambda: generate_assignment_problem(4, 3),
            lambda: generate_assignment_problem(4, 4),
            lambda: generate_coloring_problem(4, 0.5, 3),
            lambda: generate_coloring_problem(5, 0.4, 3),
        ]

        attempts = 0
        while len(self._problems) < self._n and attempts < self._n * 10:
            gen = random.choice(generators)
            p = gen()
            if p is not None:
                self._problems.append(p)
            attempts += 1

        return self._problems
