"""
Solution Detection and Finalization

LLM-as-a-judge for detecting complete solutions and creating polished final answers.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolutionJudgment:
    """Result of LLM-as-a-judge solution evaluation."""

    is_solution: bool
    confidence: float  # 0-1
    reasoning: str
    needs_refinement: bool = False

    def __repr__(self):
        status = "✓ SOLUTION" if self.is_solution else "✗ INCOMPLETE"
        return f"{status} (confidence={self.confidence:.2f})"


class SolutionDetector:
    """
    LLM-as-a-judge for detecting complete solutions.

    More sophisticated than pattern matching - understands if the reasoning
    actually provides a satisfactory answer to the original question.
    """

    def __init__(self, llm_provider, threshold: float = 0.7):
        """
        Initialize solution detector.

        Args:
            llm_provider: LLM provider for judgment
            threshold: Confidence threshold for considering something a solution (0-1)
        """
        self.llm = llm_provider
        self.threshold = threshold
        self._judgment_count = 0

    def judge_solution(self, state: str, original_question: str,
                       temperature: float = 0.1) -> SolutionJudgment:
        """
        Judge whether the state contains a satisfactory solution.

        Args:
            state: Current reasoning state
            original_question: Original problem/question
            temperature: LLM temperature (low for consistency)

        Returns:
            SolutionJudgment with evaluation results
        """
        self._judgment_count += 1

        prompt = f"""You are an expert judge evaluating whether reasoning provides a complete solution.

Original Question:
{original_question}

Current Reasoning State:
{state[-2000:]}  # Last 2000 chars for context

Task: Determine if this reasoning provides a COMPLETE and SATISFACTORY answer to the original question.

Evaluation Criteria:
1. Does it directly answer the question asked?
2. Is the answer clear and unambiguous?
3. Is the reasoning logically sound?
4. Are all necessary steps present?
5. Would a human consider this a complete solution?

Respond in this exact format:
VERDICT: [SOLUTION or INCOMPLETE]
CONFIDENCE: [0.0 to 1.0]
REASONING: [One sentence explanation]
REFINEMENT_NEEDED: [YES or NO]

Your response:"""

        try:
            response = self.llm.generate(prompt, max_tokens=150, temperature=temperature)

            # Parse response
            is_solution = "VERDICT: SOLUTION" in response.upper()

            # Extract confidence
            confidence = 0.5  # Default
            if "CONFIDENCE:" in response.upper():
                try:
                    conf_str = response.upper().split("CONFIDENCE:")[1].split()[0]
                    confidence = float(conf_str.strip())
                except:
                    pass

            # Extract reasoning
            reasoning = "Unknown"
            if "REASONING:" in response.upper():
                reasoning = response.split("REASONING:")[1].split("REFINEMENT")[0].strip()

            # Extract refinement need
            needs_refinement = "REFINEMENT_NEEDED: YES" in response.upper()

            judgment = SolutionJudgment(
                is_solution=is_solution and confidence >= self.threshold,
                confidence=confidence,
                reasoning=reasoning,
                needs_refinement=needs_refinement
            )

            logger.info(f"Solution judgment #{self._judgment_count}: {judgment}")
            return judgment

        except Exception as e:
            logger.error(f"Solution judgment failed: {e}")
            # Fallback to pattern-based detection
            return SolutionJudgment(
                is_solution=self._pattern_based_detection(state),
                confidence=0.3,
                reasoning="Fallback to pattern matching due to LLM error"
            )

    def _pattern_based_detection(self, state: str) -> bool:
        """Fallback pattern-based solution detection."""
        import re
        patterns = [
            r'\bfinal answer:',
            r'\bthe answer is\s+',
            r'\btherefore,?\s+the answer',
            r'\bin conclusion,?\s+',
            r'\bQED\b',
            r'##\s*answer'
        ]

        for pattern in patterns:
            if re.search(pattern, state.lower(), re.IGNORECASE):
                return True
        return False


class SolutionFinalizer:
    """
    Creates polished final answers from reasoning contexts.

    Takes accumulated reasoning (potentially summarized) and the original problem,
    then generates a clean, well-formatted final answer.
    """

    def __init__(self, llm_provider):
        """
        Initialize solution finalizer.

        Args:
            llm_provider: LLM provider for finalization
        """
        self.llm = llm_provider
        self._finalization_count = 0

    def finalize_solution(self, state: str, original_question: str,
                         style: str = "clear", temperature: float = 0.3) -> str:
        """
        Create a polished final answer from reasoning state.

        Args:
            state: Current reasoning state (may be summarized)
            original_question: Original problem
            style: Answer style ("clear", "formal", "concise", "detailed")
            temperature: LLM temperature

        Returns:
            Polished final answer
        """
        self._finalization_count += 1

        style_guidance = {
            "clear": "Clear and accessible to general audiences",
            "formal": "Formal and rigorous, suitable for academic contexts",
            "concise": "Brief and to-the-point",
            "detailed": "Comprehensive with full explanations"
        }

        style_desc = style_guidance.get(style, style_guidance["clear"])

        prompt = f"""You are finalizing a solution to present to the user.

Original Question:
{original_question}

Reasoning Context:
{state}

Task: Create a polished FINAL ANSWER that:
1. Directly answers the original question
2. Is {style_desc}
3. Includes the key insights from the reasoning
4. Presents the answer in a well-formatted way
5. States the answer clearly and definitively

Format your response as:

## Final Answer

[Your polished answer here]

## Key Reasoning

[1-3 bullet points of key insights]

Now provide the finalized solution:"""

        try:
            final_answer = self.llm.generate(prompt, max_tokens=500, temperature=temperature)

            logger.info(f"Solution finalized (#{self._finalization_count})")

            # Add finalization marker for easy detection
            finalized = f"[SOLUTION FINALIZED]\n\n{final_answer}"
            return finalized

        except Exception as e:
            logger.error(f"Solution finalization failed: {e}")
            # Fallback: just add a marker to the existing state
            return f"[SOLUTION FINALIZED]\n\nBased on the reasoning above:\n\n{state[-1000:]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get finalization statistics."""
        return {
            'finalization_count': self._finalization_count
        }


def is_finalized_solution(state: str) -> bool:
    """Check if a state has been marked as a finalized solution."""
    return "[SOLUTION FINALIZED]" in state


def should_attempt_finalization(state: str, detector: SolutionDetector,
                                original_question: str) -> bool:
    """
    Determine if we should attempt to finalize this state as a solution.

    Args:
        state: Current reasoning state
        detector: SolutionDetector instance
        original_question: Original problem

    Returns:
        True if finalization should be attempted
    """
    # Don't re-finalize
    if is_finalized_solution(state):
        return False

    # Use LLM-as-a-judge
    judgment = detector.judge_solution(state, original_question)

    return judgment.is_solution


__all__ = [
    'SolutionJudgment',
    'SolutionDetector',
    'SolutionFinalizer',
    'is_finalized_solution',
    'should_attempt_finalization'
]
