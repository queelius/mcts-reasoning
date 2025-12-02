"""
Evaluator: Scores terminal reasoning states.

The evaluator is called ONLY on terminal states (those with ANSWER:).
This keeps costs low - we don't evaluate every intermediate step.

Key responsibilities:
1. Score the quality of a complete solution (0 to 1)
2. Optionally verify correctness (for domains where that's possible)

Available Evaluators:
    LLMEvaluator: Uses LLM-as-judge to score solutions. Best for open-ended
        problems without a known correct answer.

    GroundTruthEvaluator: Compares against a known correct answer with
        normalization and optional partial credit.

    NumericEvaluator: For math problems - compares numeric values with
        tolerance. Handles fractions, scientific notation, and provides
        partial credit based on relative error.

    ProcessEvaluator: Evaluates reasoning quality using heuristics:
        step structure, calculations, verification, logical connectives.
        Can combine with an answer evaluator.

    CompositeEvaluator: Combines multiple evaluators with weighted scores.

    MockEvaluator: For testing - returns configurable default score.

Example usage:
    # For math benchmarks
    evaluator = NumericEvaluator(ground_truth=42.0, rel_tol=0.01)

    # For evaluating both answer and process
    evaluator = ProcessEvaluator(
        answer_evaluator=NumericEvaluator(ground_truth=42.0),
        answer_weight=0.7,
        process_weight=0.3,
    )

    # For LLM-based evaluation
    evaluator = LLMEvaluator(llm=my_llm, temperature=0.1)
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass
import re


@dataclass
class Evaluation:
    """Result of evaluating a terminal state."""
    score: float  # 0 to 1
    reasoning: Optional[str] = None  # Evaluator's explanation (for debugging)
    is_correct: Optional[bool] = None  # If ground truth available


class Evaluator(ABC):
    """Abstract base class for solution evaluators."""

    @abstractmethod
    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """
        Evaluate a terminal state.

        Args:
            question: The original question
            state: Full reasoning trace
            answer: Extracted answer (from ANSWER: marker)

        Returns:
            Evaluation with score 0-1
        """
        pass


class LLMEvaluator(Evaluator):
    """Evaluator that uses LLM-as-a-judge to score solutions."""

    DEFAULT_PROMPT = """Evaluate this solution to the given question.

Question: {question}

Solution and reasoning:
{state}

Final answer: {answer}

Rate the quality of this solution from 0 to 1:
- 0.0 = Completely wrong or nonsensical
- 0.3 = Has major errors but shows some understanding
- 0.5 = Partially correct or incomplete
- 0.7 = Mostly correct with minor issues
- 1.0 = Fully correct and well-reasoned

Consider:
1. Is the final answer correct?
2. Is the reasoning sound?
3. Are there any logical errors?

Respond with ONLY a number between 0 and 1 (e.g., 0.8):"""

    def __init__(
        self,
        llm,
        prompt_template: Optional[str] = None,
        temperature: float = 0.1,  # Low temperature for consistent scoring
    ):
        """
        Initialize the LLM evaluator.

        Args:
            llm: LLM provider with .generate(prompt, **kwargs) method
            prompt_template: Custom evaluation prompt
            temperature: Sampling temperature (low for consistency)
        """
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.temperature = temperature

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """Evaluate a terminal state using LLM-as-judge."""
        prompt = self.prompt_template.format(
            question=question,
            state=state[-2000:],  # Truncate long states
            answer=answer,
        )

        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=50,
        )

        score = self._parse_score(response)

        return Evaluation(
            score=score,
            reasoning=response.strip(),
        )

    def _parse_score(self, response: str) -> float:
        """Parse a score from the LLM response."""
        # Try to find a decimal number
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            # Clamp to 0-1
            return max(0.0, min(1.0, score))

        # Default to middle score if parsing fails
        return 0.5


class GroundTruthEvaluator(Evaluator):
    """Evaluator that checks against a known correct answer."""

    def __init__(
        self,
        ground_truth: str,
        normalize_fn: Optional[Callable[[str], str]] = None,
        partial_credit: bool = True,
    ):
        """
        Initialize ground truth evaluator.

        Args:
            ground_truth: The correct answer
            normalize_fn: Function to normalize answers for comparison
            partial_credit: If True, give partial credit for close answers
        """
        self.ground_truth = ground_truth
        self.normalize_fn = normalize_fn or self._default_normalize
        self.partial_credit = partial_credit

    def _default_normalize(self, s: str) -> str:
        """Default normalization: lowercase, strip whitespace/punctuation."""
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = re.sub(r'\s+', ' ', s)  # Normalize whitespace
        return s

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """Evaluate by comparing to ground truth."""
        normalized_answer = self.normalize_fn(answer)
        normalized_truth = self.normalize_fn(self.ground_truth)

        is_correct = normalized_answer == normalized_truth

        if is_correct:
            score = 1.0
        elif self.partial_credit:
            # Simple partial credit: check if answer is contained in truth or vice versa
            if normalized_answer in normalized_truth or normalized_truth in normalized_answer:
                score = 0.7
            else:
                score = 0.0
        else:
            score = 0.0

        return Evaluation(
            score=score,
            is_correct=is_correct,
            reasoning=f"Expected '{self.ground_truth}', got '{answer}'",
        )


class CompositeEvaluator(Evaluator):
    """Combines multiple evaluators with weighted scores."""

    def __init__(self, evaluators: list, weights: Optional[list] = None):
        """
        Initialize composite evaluator.

        Args:
            evaluators: List of (evaluator, weight) tuples or just evaluators
            weights: Weights for each evaluator (default: equal weights)
        """
        self.evaluators = evaluators
        if weights is None:
            self.weights = [1.0 / len(evaluators)] * len(evaluators)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """Evaluate using all evaluators and combine scores."""
        total_score = 0.0
        reasonings = []

        for evaluator, weight in zip(self.evaluators, self.weights):
            result = evaluator.evaluate(question, state, answer)
            total_score += result.score * weight
            if result.reasoning:
                reasonings.append(result.reasoning)

        return Evaluation(
            score=total_score,
            reasoning=" | ".join(reasonings) if reasonings else None,
        )


class NumericEvaluator(Evaluator):
    """
    Evaluator for numeric answers with tolerance comparison.

    Useful for math problems where exact string matching is too strict.
    Supports absolute and relative tolerance.
    """

    def __init__(
        self,
        ground_truth: float,
        abs_tol: float = 1e-9,
        rel_tol: float = 1e-5,
        partial_credit_factor: float = 0.5,
    ):
        """
        Initialize numeric evaluator.

        Args:
            ground_truth: The correct numeric answer
            abs_tol: Absolute tolerance for comparison
            rel_tol: Relative tolerance for comparison
            partial_credit_factor: Factor to apply for "close" answers
        """
        self.ground_truth = ground_truth
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.partial_credit_factor = partial_credit_factor

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text."""
        # Try to parse the whole thing as a number first
        try:
            return float(text.strip())
        except ValueError:
            pass

        # Try fractions first (must check before integers to catch "1/2" not just "1")
        fraction_match = re.search(r'[-+]?\d+/\d+', text)
        if fraction_match:
            found = fraction_match.group()
            try:
                num, denom = found.split('/')
                return float(num) / float(denom)
            except (ValueError, ZeroDivisionError):
                pass

        # Try scientific notation and decimals
        number_match = re.search(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
        if number_match:
            try:
                return float(number_match.group())
            except ValueError:
                pass

        return None

    def _is_close(self, a: float, b: float) -> bool:
        """Check if two numbers are close within tolerance."""
        return abs(a - b) <= max(self.rel_tol * max(abs(a), abs(b)), self.abs_tol)

    def _relative_error(self, answer: float, truth: float) -> float:
        """Calculate relative error."""
        if truth == 0:
            return abs(answer) if answer != 0 else 0.0
        return abs(answer - truth) / abs(truth)

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """Evaluate by comparing numeric values."""
        extracted = self._extract_number(answer)

        if extracted is None:
            return Evaluation(
                score=0.0,
                is_correct=False,
                reasoning=f"Could not extract number from '{answer}'",
            )

        is_correct = self._is_close(extracted, self.ground_truth)

        if is_correct:
            score = 1.0
        else:
            # Give partial credit based on relative error
            rel_error = self._relative_error(extracted, self.ground_truth)
            if rel_error < 0.1:  # Within 10%
                score = 0.8
            elif rel_error < 0.5:  # Within 50%
                score = self.partial_credit_factor
            else:
                score = 0.0

        return Evaluation(
            score=score,
            is_correct=is_correct,
            reasoning=f"Expected {self.ground_truth}, got {extracted} (error: {self._relative_error(extracted, self.ground_truth):.2%})",
        )


class ProcessEvaluator(Evaluator):
    """
    Evaluator that scores the reasoning process, not just the answer.

    Uses heuristics to assess reasoning quality:
    - Presence of step-by-step structure
    - Mathematical notation usage
    - Self-verification statements
    - Logical connectives
    """

    def __init__(
        self,
        answer_evaluator: Optional[Evaluator] = None,
        answer_weight: float = 0.7,
        process_weight: float = 0.3,
    ):
        """
        Initialize process evaluator.

        Args:
            answer_evaluator: Evaluator for the answer (optional)
            answer_weight: Weight for answer score (0-1)
            process_weight: Weight for process score (0-1)
        """
        self.answer_evaluator = answer_evaluator
        self.answer_weight = answer_weight
        self.process_weight = process_weight

    def _score_process(self, state: str) -> tuple:
        """
        Score the reasoning process.

        Returns:
            (score, list of reasons)
        """
        score = 0.0
        reasons = []

        # Check for step structure
        step_patterns = [
            r'[Ss]tep\s*\d',
            r'\d+\.\s+\w',
            r'[Ff]irst[,:]',
            r'[Nn]ext[,:]',
            r'[Tt]hen[,:]',
            r'[Ff]inally[,:]',
        ]
        if any(re.search(p, state) for p in step_patterns):
            score += 0.25
            reasons.append("Has step structure")

        # Check for mathematical reasoning
        math_patterns = [
            r'=',
            r'\d+\s*[+\-*/]\s*\d+',
            r'therefore',
            r'thus',
            r'hence',
        ]
        if any(re.search(p, state, re.IGNORECASE) for p in math_patterns):
            score += 0.25
            reasons.append("Shows calculations")

        # Check for verification/checking
        verify_patterns = [
            r'[Ll]et.?s (check|verify)',
            r'[Cc]heck:',
            r'[Vv]erif',
            r'[Cc]onfirm',
            r'[Dd]ouble.?check',
        ]
        if any(re.search(p, state) for p in verify_patterns):
            score += 0.25
            reasons.append("Includes verification")

        # Check for logical structure
        logic_patterns = [
            r'[Bb]ecause',
            r'[Ss]ince',
            r'[Tt]herefore',
            r'[Ii]f.*then',
            r'[Gg]iven that',
        ]
        if any(re.search(p, state) for p in logic_patterns):
            score += 0.25
            reasons.append("Has logical structure")

        return score, reasons

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """Evaluate both process and answer."""
        process_score, process_reasons = self._score_process(state)

        if self.answer_evaluator:
            answer_result = self.answer_evaluator.evaluate(question, state, answer)
            answer_score = answer_result.score
            is_correct = answer_result.is_correct
        else:
            answer_score = 0.5  # Neutral if no answer evaluator
            is_correct = None

        total_score = (
            self.answer_weight * answer_score +
            self.process_weight * process_score
        )

        reasoning = f"Process: {process_score:.2f} ({', '.join(process_reasons) or 'minimal structure'})"
        if self.answer_evaluator:
            reasoning += f" | Answer: {answer_score:.2f}"

        return Evaluation(
            score=total_score,
            is_correct=is_correct,
            reasoning=reasoning,
        )


class MockEvaluator(Evaluator):
    """Mock evaluator for testing."""

    def __init__(self, default_score: float = 0.8):
        """Initialize with a default score to return."""
        self.default_score = default_score
        self.call_count = 0

    def evaluate(self, question: str, state: str, answer: str) -> Evaluation:
        """Return the default score."""
        self.call_count += 1

        # Give higher scores to answers that look like numbers (for math problems)
        if answer and answer.strip().isdigit():
            score = self.default_score
        else:
            score = self.default_score * 0.8

        return Evaluation(
            score=score,
            reasoning=f"Mock evaluation (call #{self.call_count})",
        )
