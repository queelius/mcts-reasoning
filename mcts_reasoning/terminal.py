"""
Terminal Detection: Determines when reasoning is complete.

Terminal detection is separate from action generation because:
1. Different problems may use different completion signals
2. The detector informs prompt construction (format instructions)
3. Enables pluggable completion strategies (marker, LLM-judge, confidence)
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable
from dataclasses import dataclass
import re


@dataclass
class TerminalCheck:
    """Result of checking if a state is terminal."""
    is_terminal: bool
    answer: Optional[str] = None
    confidence: float = 1.0  # For probabilistic detectors


@runtime_checkable
class TerminalDetector(Protocol):
    """
    Detects when a reasoning state represents a complete answer.

    Implementations can use different strategies:
    - Marker-based: Look for "ANSWER:" or similar markers
    - LLM-as-judge: Ask LLM if reasoning is complete
    - Confidence-based: Check if answer confidence exceeds threshold
    """

    def check(self, state: str) -> TerminalCheck:
        """
        Check if state is terminal.

        Args:
            state: Current reasoning state

        Returns:
            TerminalCheck with is_terminal flag and extracted answer
        """
        ...

    def format_instruction(self) -> str:
        """
        Return instruction for LLM on how to signal completion.

        This is included in prompts so the LLM knows the expected format.
        """
        ...


class MarkerTerminalDetector:
    """
    Detect terminal states by looking for a marker string.

    The canonical approach: LLM outputs "ANSWER: <answer>" when done.
    """

    def __init__(self, marker: str = "ANSWER:"):
        self.marker = marker

    def check(self, state: str) -> TerminalCheck:
        """Check if state contains the answer marker."""
        if self.marker not in state:
            return TerminalCheck(is_terminal=False)

        answer = self._extract_answer(state)
        return TerminalCheck(
            is_terminal=True,
            answer=answer,
            confidence=1.0,
        )

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from text containing the marker."""
        # Try to find structured answer
        pattern = rf'{re.escape(self.marker)}\s*(.+?)(?:\n\n|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: everything after marker
        idx = text.find(self.marker)
        if idx >= 0:
            return text[idx + len(self.marker):].strip()

        return None

    def format_instruction(self) -> str:
        """Instruction for marker-based completion."""
        return f"When you reach a final answer, clearly state: {self.marker} <your answer>"


class BoxedTerminalDetector:
    """
    Detect terminal states by looking for \\boxed{} (LaTeX style).

    Common in mathematical reasoning benchmarks.
    """

    def check(self, state: str) -> TerminalCheck:
        """Check if state contains a boxed answer."""
        # Match \boxed{...} allowing nested braces
        match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', state)
        if not match:
            return TerminalCheck(is_terminal=False)

        return TerminalCheck(
            is_terminal=True,
            answer=match.group(1).strip(),
            confidence=1.0,
        )

    def format_instruction(self) -> str:
        """Instruction for boxed completion."""
        return "When you reach a final answer, write it as: \\boxed{your answer}"


class MultiMarkerTerminalDetector:
    """
    Detect terminal states using multiple possible markers.

    Useful when different LLMs or prompts use different conventions.
    """

    def __init__(self, markers: Optional[list] = None):
        self.markers = markers or ["ANSWER:", "FINAL ANSWER:", "\\boxed{"]

    def check(self, state: str) -> TerminalCheck:
        """Check if state contains any of the markers."""
        for marker in self.markers:
            if marker == "\\boxed{":
                # Special handling for boxed
                detector = BoxedTerminalDetector()
                result = detector.check(state)
                if result.is_terminal:
                    return result
            elif marker in state:
                # Use marker detector
                detector = MarkerTerminalDetector(marker)
                return detector.check(state)

        return TerminalCheck(is_terminal=False)

    def format_instruction(self) -> str:
        """Instruction listing all accepted formats."""
        return f"When you reach a final answer, use one of: {', '.join(self.markers)}"


# =============================================================================
# EXTENSIONS (for future consideration)
# =============================================================================

class LLMJudgeTerminalDetector:
    """
    Use LLM to judge if reasoning is complete.

    NOTE: This is an EXTENSION concept. More expensive but can detect
    implicit completions that lack explicit markers.
    """

    JUDGE_PROMPT = """Analyze this reasoning trace and determine if it has reached a complete, final answer.

Reasoning trace:
{state}

Questions to consider:
1. Has the reasoning reached a definitive conclusion?
2. Is there a clear answer that addresses the original question?
3. Would continuing add meaningful value?

Respond with:
COMPLETE: <yes/no>
ANSWER: <extracted answer if complete, or "none">
CONFIDENCE: <0.0-1.0>"""

    def __init__(self, llm, confidence_threshold: float = 0.8):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def check(self, state: str) -> TerminalCheck:
        """Ask LLM to judge if state is terminal."""
        prompt = self.JUDGE_PROMPT.format(state=state)
        response = self.llm.generate(prompt, temperature=0.1, max_tokens=100)

        # Parse response
        is_complete = "COMPLETE: yes" in response.lower()
        confidence = self._extract_confidence(response)

        if is_complete and confidence >= self.confidence_threshold:
            answer = self._extract_answer(response)
            return TerminalCheck(
                is_terminal=True,
                answer=answer,
                confidence=confidence,
            )

        return TerminalCheck(is_terminal=False, confidence=confidence)

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response."""
        match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from judge response."""
        match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response)
        if match:
            answer = match.group(1).strip()
            if answer.lower() != "none":
                return answer
        return None

    def format_instruction(self) -> str:
        """No specific format needed for LLM-judge."""
        return "Reason until you reach a complete answer."
