"""
LLM-as-judge: use the LLM for all evaluation tasks.

Verification is easier than generation. The same model that struggles
to produce a correct solution can reliably check whether a candidate
solution is correct.

Three capabilities:
1. Extract: "What is the final answer in this text?"
2. Verify: "Does this solution satisfy the constraints?"
3. Compare: "Do these two answers mean the same thing?"
"""

from __future__ import annotations

from .types import Message, State, Evaluation
from .evaluator import Evaluator


class LLMJudge:
    """LLM-based judge for answer extraction, verification, and comparison.

    Use this instead of regex, string matching, or terminal detectors
    for evaluating LLM outputs.
    """

    def __init__(self, provider, max_tokens: int = 200, temperature: float = 0.0):
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

    def extract_answer(self, question: str, response: str) -> str | None:
        """Extract the final answer from a CoT response.

        No regex. Asks the LLM to identify the answer.
        """
        resp = self.provider.generate(
            [
                Message(role="system", content=(
                    "Extract the final answer from this response. "
                    "Reply with ONLY the answer itself, nothing else. "
                    "If no clear answer is given, reply with: NO_ANSWER"
                )),
                Message(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Response:\n{response}\n\n"
                    f"The final answer is:"
                )),
            ],
            max_tokens=100,
            temperature=self.temperature,
        )
        answer = resp.strip()
        if "NO_ANSWER" in answer.upper():
            return None
        return answer

    def verify(self, question: str, candidate: str, constraints: str | None = None) -> bool:
        """Check if a candidate solution satisfies the problem constraints.

        Args:
            question: The original problem statement (contains constraints).
            candidate: The proposed solution.
            constraints: Optional explicit constraint list.
        """
        constraint_text = f"\nExplicit constraints:\n{constraints}" if constraints else ""

        resp = self.provider.generate(
            [
                Message(role="system", content=(
                    "You verify solutions. Check EACH constraint one by one. "
                    "Reply with ONLY 'VALID' or 'INVALID'."
                )),
                Message(role="user", content=(
                    f"Problem:\n{question}{constraint_text}\n\n"
                    f"Proposed solution:\n{candidate}\n\n"
                    f"Does this solution satisfy ALL constraints?"
                )),
            ],
            max_tokens=20,
            temperature=self.temperature,
        )
        return "VALID" in resp.upper() and "INVALID" not in resp.upper()

    def compare(self, question: str, answer_a: str, answer_b: str) -> bool:
        """Check if two answers are semantically equivalent.

        Handles format differences: "Alice, Bob, Carol" vs
        "Alice: Monday; Bob: Tuesday" vs natural language descriptions.
        """
        resp = self.provider.generate(
            [
                Message(role="system", content=(
                    "Do these two answers to the same question mean the same thing? "
                    "They may be formatted differently but semantically equivalent. "
                    "Reply with ONLY 'YES' or 'NO'."
                )),
                Message(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Answer A: {answer_a}\n\n"
                    f"Answer B: {answer_b}\n\n"
                    f"Same answer?"
                )),
            ],
            max_tokens=10,
            temperature=self.temperature,
        )
        return "YES" in resp.upper()

    def score(self, question: str, response: str, ground_truth: str) -> float:
        """Score a response against ground truth. Returns 0.0 to 1.0.

        Combines answer extraction, comparison, and verification.
        """
        answer = self.extract_answer(question, response)
        if answer is None:
            return 0.0
        if self.compare(question, answer, ground_truth):
            return 1.0
        # Partial credit: check if the reasoning is on the right track
        return 0.0


class JudgeEvaluator(Evaluator):
    """Evaluator that uses LLMJudge for scoring.

    Drop-in replacement for ProcessEvaluator or GroundTruthEvaluator
    that works on any problem format.
    """

    def __init__(self, provider, ground_truth: str | None = None):
        self.judge = LLMJudge(provider)
        self.ground_truth = ground_truth

    def evaluate(self, question: str, state: str | State, answer: str) -> Evaluation:
        if self.ground_truth:
            # Compare against known answer
            match = self.judge.compare(question, answer, self.ground_truth)
            score = 1.0 if match else 0.0
            return Evaluation(score=score, explanation=f"Match: {match}")
        else:
            # No ground truth: verify against constraints in the question
            valid = self.judge.verify(question, answer)
            score = 1.0 if valid else 0.0
            return Evaluation(score=score, explanation=f"Valid: {valid}")
