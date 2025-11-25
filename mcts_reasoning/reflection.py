"""
Reflection and Critique: Self-Improvement Through Self-Evaluation

Enables the LLM to critique its own reasoning and suggest improvements,
creating a feedback loop for higher quality reasoning.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Critique:
    """LLM's critique of reasoning."""

    quality_score: float  # 0-1
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    needs_refinement: bool
    reasoning: str = ""


class ReflectionCritic:
    """
    Uses LLM to critique reasoning and suggest improvements.

    This creates a reflection loop where the system evaluates its own
    reasoning and iteratively improves it.
    """

    def __init__(self, llm, temperature: float = 0.3):
        """
        Initialize reflection critic.

        Args:
            llm: LLM provider for critique
            temperature: Temperature for critique (0.2-0.4 recommended)
        """
        self.llm = llm
        self.temperature = temperature
        self._critique_count = 0
        self._critiques_history = []

    def critique(self, reasoning: str, original_question: str,
                 criteria: Optional[List[str]] = None) -> Critique:
        """
        Critique a piece of reasoning.

        Args:
            reasoning: The reasoning to critique
            original_question: Original problem being solved
            criteria: Optional custom criteria (uses defaults if None)

        Returns:
            Critique with scores, strengths, weaknesses, suggestions
        """
        if criteria is None:
            criteria = [
                "Logical soundness - Are all steps logically valid?",
                "Completeness - Does it fully address the problem?",
                "Clarity - Is the reasoning easy to follow?",
                "Correctness - Are the conclusions correct?",
                "Efficiency - Is the approach efficient/elegant?"
            ]

        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))

        # Truncate reasoning for token efficiency
        reasoning_preview = reasoning[-2000:] if len(reasoning) > 2000 else reasoning

        prompt = f"""You are a critical reasoning evaluator. Analyze the reasoning below and provide constructive critique.

Original Problem:
{original_question}

Reasoning to Evaluate:
{reasoning_preview}

Evaluation Criteria:
{criteria_text}

Task: Critically evaluate this reasoning.

Respond in this EXACT format:
QUALITY_SCORE: [0.0 to 1.0]
STRENGTHS:
- [strength 1]
- [strength 2]
WEAKNESSES:
- [weakness 1]
- [weakness 2]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
NEEDS_REFINEMENT: [YES or NO]
REASONING: [One sentence overall assessment]"""

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=400,
                temperature=self.temperature
            )

            critique = self._parse_critique(response)

            self._critique_count += 1
            self._critiques_history.append({
                'quality_score': critique.quality_score,
                'needs_refinement': critique.needs_refinement,
                'weaknesses_count': len(critique.weaknesses),
                'suggestions_count': len(critique.suggestions)
            })

            logger.info(f"Reflection critique #{self._critique_count}: "
                       f"quality={critique.quality_score:.2f}, "
                       f"needs_refinement={critique.needs_refinement}")
            logger.debug(f"  Weaknesses: {len(critique.weaknesses)}, "
                        f"Suggestions: {len(critique.suggestions)}")

            return critique

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            # Fallback to neutral critique
            return Critique(
                quality_score=0.5,
                strengths=[],
                weaknesses=["Critique failed"],
                suggestions=[],
                needs_refinement=False,
                reasoning="Critique system encountered error"
            )

    def _parse_critique(self, response: str) -> Critique:
        """Parse LLM response into Critique."""
        lines = response.strip().split('\n')

        quality_score = 0.5
        strengths = []
        weaknesses = []
        suggestions = []
        needs_refinement = False
        reasoning = ""

        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("QUALITY_SCORE:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    quality_score = float(score_text)
                except:
                    quality_score = 0.5

            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"

            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"

            elif line.startswith("SUGGESTIONS:"):
                current_section = "suggestions"

            elif line.startswith("NEEDS_REFINEMENT:"):
                current_section = None
                needs_refinement = "YES" in line.upper()

            elif line.startswith("REASONING:"):
                current_section = None
                reasoning = line.split(":", 1)[1].strip()

            elif line.startswith("-") and current_section:
                item = line.lstrip("- ").strip()
                if item:
                    if current_section == "strengths":
                        strengths.append(item)
                    elif current_section == "weaknesses":
                        weaknesses.append(item)
                    elif current_section == "suggestions":
                        suggestions.append(item)

        return Critique(
            quality_score=quality_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            needs_refinement=needs_refinement,
            reasoning=reasoning
        )

    def refine(self, reasoning: str, critique: Critique,
               original_question: str) -> str:
        """
        Create refined version of reasoning based on critique.

        Args:
            reasoning: Original reasoning
            critique: Critique with suggestions
            original_question: Original problem

        Returns:
            Refined reasoning
        """
        weaknesses_text = "\n".join(f"- {w}" for w in critique.weaknesses)
        suggestions_text = "\n".join(f"- {s}" for s in critique.suggestions)

        prompt = f"""You are improving reasoning based on critical feedback.

Original Problem:
{original_question}

Previous Reasoning:
{reasoning[-1500:]}

Critique Identified These Issues:
{weaknesses_text}

Suggestions for Improvement:
{suggestions_text}

Task: Create an IMPROVED version of the reasoning that addresses these issues and implements the suggestions.

Present your improved reasoning clearly and concisely.
"""

        try:
            refined = self.llm.generate(
                prompt,
                max_tokens=500,
                temperature=self.temperature
            )

            # Mark as refined
            result = f"[REFINED REASONING]\n\n{refined}"

            logger.info(f"Refined reasoning based on {len(critique.suggestions)} suggestions")
            return result

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return reasoning  # Return original if refinement fails

    def get_stats(self) -> Dict[str, Any]:
        """Get critique statistics."""
        if not self._critiques_history:
            return {
                'critique_count': 0,
                'average_quality': 0.0,
                'refinement_rate': 0.0
            }

        avg_quality = sum(c['quality_score'] for c in self._critiques_history) / len(self._critiques_history)
        refinement_rate = sum(1 for c in self._critiques_history if c['needs_refinement']) / len(self._critiques_history)

        return {
            'critique_count': self._critique_count,
            'average_quality': avg_quality,
            'refinement_rate': refinement_rate,
            'recent_critiques': self._critiques_history[-5:]
        }


class ReflectiveRefinementLoop:
    """
    Implements iterative refinement loop: reason → critique → refine → repeat.

    This can be used standalone or integrated into MCTS.
    """

    def __init__(self, llm, max_iterations: int = 3,
                 quality_threshold: float = 0.8):
        """
        Initialize reflective refinement loop.

        Args:
            llm: LLM provider
            max_iterations: Maximum refinement iterations
            quality_threshold: Stop if quality exceeds this
        """
        self.critic = ReflectionCritic(llm)
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def refine_iteratively(self, initial_reasoning: str,
                          original_question: str) -> tuple:
        """
        Iteratively refine reasoning until quality threshold or max iterations.

        Args:
            initial_reasoning: Starting reasoning
            original_question: Problem being solved

        Returns:
            (final_reasoning, critiques_history)
        """
        current_reasoning = initial_reasoning
        critiques = []

        for iteration in range(self.max_iterations):
            # Critique current reasoning
            critique = self.critic.critique(current_reasoning, original_question)
            critiques.append(critique)

            logger.info(f"Refinement iteration {iteration+1}: "
                       f"quality={critique.quality_score:.2f}")

            # Check if quality is sufficient
            if critique.quality_score >= self.quality_threshold:
                logger.info(f"Quality threshold reached ({critique.quality_score:.2f} >= {self.quality_threshold})")
                break

            # Check if refinement needed
            if not critique.needs_refinement:
                logger.info("Critique indicates no further refinement needed")
                break

            # Refine
            current_reasoning = self.critic.refine(
                current_reasoning, critique, original_question
            )

        return current_reasoning, critiques


__all__ = ['ReflectionCritic', 'Critique', 'ReflectiveRefinementLoop']
