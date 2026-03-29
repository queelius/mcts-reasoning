"""
Strong CoT baselines for fair comparison with MCTS.

Each baseline gets the same token budget as the MCTS pipeline.
The only difference: how the budget is allocated.

- MCTS: some budget on initial solutions, rest on UCB-guided exploration
- CoT baselines: all budget on independent solutions, then pick the best

Baselines:
1. CoT-SC: N solutions, majority vote (self-consistency)
2. CoT-BestOfN: N solutions, pick the one with highest verifier score
3. CoT-USC: N solutions, LLM reads all and picks the most consistent
"""

from __future__ import annotations

from typing import Optional

from .types import Message
from .evaluator import Evaluator
from .judge import LLMJudge


def cot_generate(provider, question: str, n: int,
                 max_tokens: int = 800, temperature: float = 0.9) -> list[str]:
    """Generate N independent CoT solutions. Same prompt quality as MCTS."""
    solutions = []
    for _ in range(n):
        resp = provider.generate([
            Message(role="system", content=(
                "You are a careful problem solver. Think step by step. "
                "Show all your work. Give your final answer clearly."
            )),
            Message(role="user", content=question),
        ], max_tokens, temperature)
        solutions.append(resp)
    return solutions


def cot_self_consistency(provider, question: str, solutions: list[str],
                         judge: Optional[LLMJudge] = None) -> str | None:
    """Self-consistency: extract answers, group by equivalence, pick majority."""
    if judge is None:
        judge = LLMJudge(provider)

    answers = []
    for sol in solutions:
        ans = judge.extract_answer(question, sol)
        if ans:
            answers.append((ans, sol))

    if not answers:
        return None

    # Group by equivalence
    groups: list[list[tuple[str, str]]] = []
    for ans, sol in answers:
        placed = False
        for group in groups:
            if judge.compare(question, ans, group[0][0]):
                group.append((ans, sol))
                placed = True
                break
        if not placed:
            groups.append([(ans, sol)])

    # Majority
    groups.sort(key=len, reverse=True)
    return groups[0][0][0]  # answer from the largest group


def cot_best_of_n(provider, question: str, solutions: list[str],
                  evaluator: Evaluator) -> tuple[str | None, str | None]:
    """Best-of-N: score each solution with the evaluator, pick the highest.

    Uses the SAME evaluator as MCTS. This isolates the search strategy
    as the only variable.

    Returns (best_answer, best_solution).
    """
    judge = LLMJudge(provider)
    best_score = -1.0
    best_answer = None
    best_solution = None

    for sol in solutions:
        answer = judge.extract_answer(question, sol)
        if not answer:
            continue
        ev = evaluator.evaluate(question, sol, answer)
        if ev.score > best_score:
            best_score = ev.score
            best_answer = answer
            best_solution = sol

    return best_answer, best_solution


def cot_universal_sc(provider, question: str, solutions: list[str]) -> str | None:
    """Universal Self-Consistency: LLM reads ALL solutions and picks the best.

    Instead of extracting and comparing individual answers, the LLM
    sees every solution and makes one judgment call. This is the
    strongest single-call aggregation.
    """
    if not solutions:
        return None

    solutions_text = "\n\n---\n\n".join(
        f"Solution {i+1}:\n{sol}" for i, sol in enumerate(solutions)
    )

    resp = provider.generate([
        Message(role="system", content=(
            "You are given multiple solutions to the same problem. "
            "Read all of them carefully. Determine which answer appears "
            "most correct based on the quality of reasoning. "
            "Reply with ONLY the final answer (the content, not the solution number)."
        )),
        Message(role="user", content=(
            f"Problem: {question}\n\n"
            f"{solutions_text}\n\n"
            f"Which answer is most likely correct? Give ONLY the answer:"
        )),
    ], max_tokens=200, temperature=0.0)

    return resp.strip() if resp.strip() else None
