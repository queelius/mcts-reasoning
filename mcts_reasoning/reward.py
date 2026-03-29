"""
Reward signal generation: from hand-crafted to LLM-generated verifiers.

Three levels of reward signal:
1. Hand-crafted: human writes verify() (classical verifier, perfect signal)
2. LLM-generated: LLM writes verify() code given the problem (inspectable, deterministic)
3. LLM-reasoned: LLM generates multi-faceted verifier with partial credit

The key insight: the LLM generates the verifier ONCE per problem.
The verifier runs MANY times (once per terminal). This amortizes the
cost and produces a deterministic, inspectable reward signal.

Security: LLM-generated code runs in a restricted namespace with only
safe built-in functions. No file I/O, no imports, no network access.
"""

from __future__ import annotations

import textwrap
from typing import Optional

from .types import Message, State, Evaluation
from .evaluator import Evaluator


# Safe built-ins for executing LLM-generated verifiers
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "frozenset": frozenset, "int": int,
    "isinstance": isinstance, "len": len, "list": list,
    "map": map, "max": max, "min": min, "range": range,
    "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    "print": lambda *a, **kw: None,  # no-op print
}


def _safe_exec(code: str) -> dict:
    """Execute code in a restricted namespace. Returns the namespace."""
    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    compiled = compile(code, "<verifier>", "exec")
    exec(compiled, namespace)  # noqa: S102 - intentional, sandboxed exec
    return namespace


class GeneratedVerifierEvaluator(Evaluator):
    """LLM generates a Python verifier function, then uses it for scoring.

    Phase 1: LLM reads the problem, generates verify(candidate) -> float
    Phase 2: For each terminal, extract the candidate answer and run verify()

    The generated code runs in a sandbox with only safe built-in functions.
    No file I/O, no imports, no network, no os access.
    """

    def __init__(self, provider, question: str, max_retries: int = 2):
        self.provider = provider
        self.question = question
        self._verify_fn: Optional[callable] = None
        self._verify_code: Optional[str] = None
        self._max_retries = max_retries
        self._generate_verifier()

    def _generate_verifier(self) -> None:
        """Ask the LLM to write a verification function for this problem."""
        prompt = Message(role="user", content=textwrap.dedent(f"""\
            Write a Python function that verifies a candidate solution to this problem:

            {self.question}

            Requirements:
            - Function signature: def verify(candidate: str) -> float
            - Input: a string containing the candidate solution in natural language
            - Output: a float between 0.0 and 1.0
              - 1.0 = all constraints satisfied
              - 0.0 = no constraints satisfied or unparseable
              - Partial values for partial satisfaction (e.g., 3/5 constraints met = 0.6)
            - The function must parse the candidate string to extract the answer
            - Be generous in parsing (handle various formats)
            - Do NOT use import statements (only built-in functions available)
            - Do NOT use open(), os, sys, or any I/O

            Output ONLY the Python code. No explanation, no markdown fences."""))

        for attempt in range(self._max_retries):
            resp = self.provider.generate(
                [
                    Message(role="system", content=(
                        "You write Python verification functions. "
                        "Output ONLY valid Python code. No markdown, no explanation. "
                        "Use only built-in functions, no imports."
                    )),
                    prompt,
                ],
                max_tokens=1000,
                temperature=0.0,
            )

            code = self._clean_code(resp)
            fn = self._try_compile(code)
            if fn is not None:
                self._verify_fn = fn
                self._verify_code = code
                return

        # Fallback: always return 0.5
        self._verify_fn = lambda candidate: 0.5
        self._verify_code = "# Failed to generate verifier\ndef verify(candidate): return 0.5"

    def _clean_code(self, resp: str) -> str:
        """Strip markdown fences and other non-code content."""
        lines = resp.strip().split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)

    def _try_compile(self, code: str) -> Optional[callable]:
        """Compile in sandbox, extract verify(), smoke test."""
        try:
            namespace = _safe_exec(code)
            if "verify" in namespace and callable(namespace["verify"]):
                result = namespace["verify"]("test input")
                if isinstance(result, (int, float)):
                    return namespace["verify"]
        except Exception:
            pass
        return None

    def evaluate(self, question: str, state: str | State, answer: str) -> Evaluation:
        if self._verify_fn is None:
            return Evaluation(score=0.5, explanation="No verifier available")

        try:
            score = float(self._verify_fn(str(state)))
            score = max(0.0, min(1.0, score))
            return Evaluation(score=score, explanation=f"Verifier score: {score:.2f}")
        except Exception as e:
            return Evaluation(score=0.0, explanation=f"Verifier error: {e}")

    @property
    def code(self) -> str:
        """The generated verifier code (for inspection)."""
        return self._verify_code or ""


class ProcessRewardEvaluator(Evaluator):
    """LLM evaluates the REASONING PROCESS, not just the answer.

    Scores intermediate reasoning quality:
    - Are the logical steps sound?
    - Are calculations correct?
    - Are assumptions stated and tested?
    - Does the conclusion follow from the premises?

    This gives UCB1 signal about WHICH REASONING PATHS are productive,
    not just which answers are right.
    """

    def __init__(self, provider):
        self.provider = provider

    def evaluate(self, question: str, state: str | State, answer: str) -> Evaluation:
        resp = self.provider.generate([
            Message(role="system", content=(
                "You evaluate reasoning quality on a scale of 0.0 to 1.0. "
                "Score based on:\n"
                "- Are the logical steps valid? (0.3)\n"
                "- Are calculations correct? (0.3)\n"
                "- Are assumptions clearly stated? (0.2)\n"
                "- Does the conclusion follow? (0.2)\n\n"
                "Reply with ONLY a number between 0.0 and 1.0."
            )),
            Message(role="user", content=(
                f"Problem: {question}\n\n"
                f"Reasoning:\n{state}\n\n"
                f"Answer given: {answer}\n\n"
                f"Reasoning quality score (0.0-1.0):"
            )),
        ], max_tokens=20, temperature=0.0)

        try:
            score = float(resp.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            score = 0.5

        return Evaluation(score=score, explanation=f"Process score: {score:.2f}")


class CompositeRewardEvaluator(Evaluator):
    """Combines multiple reward signals with weights.

    Example: 0.5 * generated_verifier + 0.3 * process_reward + 0.2 * consistency
    """

    def __init__(self, evaluators: list[tuple[float, Evaluator]]):
        """Args: list of (weight, evaluator) pairs. Weights should sum to 1.0."""
        self.evaluators = evaluators

    def evaluate(self, question: str, state: str | State, answer: str) -> Evaluation:
        total_score = 0.0
        explanations = []
        for weight, ev in self.evaluators:
            result = ev.evaluate(question, state, answer)
            total_score += weight * result.score
            explanations.append(f"{ev.__class__.__name__}={result.score:.2f}")

        return Evaluation(
            score=total_score,
            explanation="; ".join(explanations),
        )
