"""
Classical verifiers for combinatorial constraint problems.

The LLM extracts the structured answer. The classical algorithm verifies it.
This gives a perfectly reliable reward signal: no LLM-as-judge errors.
"""

from __future__ import annotations

from ...types import Message

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
COLORS = ["red", "blue", "green", "yellow"]
SLOTS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def verify_seating(names: list[str], constraints: list[tuple[str, str]],
                   candidate: list[str]) -> bool:
    """Check if a seating arrangement satisfies adjacency constraints."""
    if set(candidate) != set(names):
        return False  # must use all names exactly once
    if len(candidate) != len(set(candidate)):
        return False  # no duplicates
    for a, b in constraints:
        if a in candidate and b in candidate:
            ia = candidate.index(a)
            ib = candidate.index(b)
            if abs(ia - ib) == 1:
                return False  # a and b are adjacent, constraint violated
    return True


def verify_assignment(items: list[str], slots: list[str],
                      constraints: list[tuple[str, str]],
                      candidate: dict[str, str]) -> bool:
    """Check if an assignment satisfies forbidden-pair constraints."""
    if set(candidate.keys()) != set(items):
        return False
    if set(candidate.values()) != set(slots):
        return False  # must use all slots
    for item, forbidden_slot in constraints:
        if candidate.get(item) == forbidden_slot:
            return False
    return True


def verify_coloring(edges: list[tuple[str, str]], candidate: dict[str, str]) -> bool:
    """Check if a coloring satisfies adjacency constraints."""
    for a, b in edges:
        if a in candidate and b in candidate:
            if candidate[a] == candidate[b]:
                return False
    return True


def parse_and_verify(provider, problem_metadata: dict, question: str,
                     response: str) -> tuple[bool, str | None]:
    """Use LLM to parse the answer, then classical solver to verify.

    Returns (is_valid, extracted_answer_string).
    """
    ptype = problem_metadata["type"]

    if ptype == "seating":
        return _parse_and_verify_seating(provider, problem_metadata, question, response)
    elif ptype == "assignment":
        return _parse_and_verify_assignment(provider, problem_metadata, question, response)
    elif ptype == "coloring":
        return _parse_and_verify_coloring(provider, problem_metadata, question, response)
    return False, None


def _parse_and_verify_seating(provider, meta, question, response):
    names = [c[0] for c in meta["constraints"]]
    all_names = list(set(names + [c[1] for c in meta["constraints"]]))
    # Extend with any names from the problem (up to meta["n"])
    pass  # NAMES, SLOTS, COLORS are module-level constants
    all_names = NAMES[:meta["n"]]

    resp = provider.generate([
        Message(role="system", content=(
            "Extract the seating order from this response. "
            f"The people are: {', '.join(all_names)}. "
            "Reply with ONLY the names in order, comma-separated. "
            "Example: Alice, Bob, Carol, Dave"
        )),
        Message(role="user", content=response),
    ], max_tokens=50, temperature=0.0)

    parsed = [name.strip() for name in resp.strip().split(",")]
    if len(parsed) != len(all_names):
        return False, resp.strip()

    valid = verify_seating(all_names, meta["constraints"], parsed)
    return valid, ", ".join(parsed)


def _parse_and_verify_assignment(provider, meta, question, response):
    n = meta["n"]
    pass  # NAMES, SLOTS, COLORS are module-level constants, SLOTS
    items = NAMES[:n]
    slots = SLOTS[:n]

    resp = provider.generate([
        Message(role="system", content=(
            "Extract the assignment from this response. "
            f"People: {', '.join(items)}. Days: {', '.join(slots)}. "
            "Reply with ONLY lines like: Alice: Monday"
        )),
        Message(role="user", content=response),
    ], max_tokens=100, temperature=0.0)

    assignment = {}
    for line in resp.strip().split("\n"):
        if ":" in line:
            parts = line.split(":", 1)
            person = parts[0].strip()
            day = parts[1].strip()
            if person in items and day in slots:
                assignment[person] = day

    if len(assignment) != n:
        return False, resp.strip()

    valid = verify_assignment(items, slots, meta["constraints"], assignment)
    return valid, "; ".join(f"{k}: {v}" for k, v in assignment.items())


def _parse_and_verify_coloring(provider, meta, question, response):
    n = meta["n_nodes"]
    pass  # NAMES, SLOTS, COLORS are module-level constants, COLORS
    names = NAMES[:n]
    colors = COLORS[:meta["n_colors"]]

    resp = provider.generate([
        Message(role="system", content=(
            "Extract the color assignment from this response. "
            f"People: {', '.join(names)}. Colors: {', '.join(colors)}. "
            "Reply with ONLY lines like: Alice: red"
        )),
        Message(role="user", content=response),
    ], max_tokens=100, temperature=0.0)

    assignment = {}
    for line in resp.strip().split("\n"):
        if ":" in line:
            parts = line.split(":", 1)
            person = parts[0].strip()
            color = parts[1].strip().lower()
            if person in names and color in colors:
                assignment[person] = color

    if len(assignment) != n:
        return False, resp.strip()

    valid = verify_coloring(meta["edges"], assignment)
    return valid, "; ".join(f"{k}: {v}" for k, v in assignment.items())


