"""
Generator: Produces reasoning continuations.

The generator takes a current reasoning state and produces one or more
possible continuations. Terminal detection is delegated to a TerminalDetector.

Key responsibilities:
1. Generate diverse continuations from a state
2. Delegate terminal detection to TerminalDetector
3. Support multiple continuation strategies (single, diverse)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import re

from .terminal import TerminalDetector, MarkerTerminalDetector

if TYPE_CHECKING:
    pass


# Marker that signals a complete answer (default)
ANSWER_MARKER = "ANSWER:"


@dataclass
class Continuation:
    """A single reasoning continuation."""

    text: str
    is_terminal: bool
    answer: Optional[str] = None  # Extracted answer if terminal


class Generator(ABC):
    """Abstract base class for reasoning generators."""

    def __init__(self, terminal_detector: Optional[TerminalDetector] = None):
        """
        Initialize generator with optional terminal detector.

        Args:
            terminal_detector: Detector for terminal states (default: MarkerTerminalDetector)
        """
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    @abstractmethod
    def generate(self, question: str, state: str, n: int = 1) -> List[Continuation]:
        """
        Generate n continuations from the current state.

        Args:
            question: The original question being solved
            state: Current reasoning state
            n: Number of diverse continuations to generate

        Returns:
            List of Continuation objects
        """
        pass

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer using the terminal detector."""
        result = self.terminal_detector.check(text)
        return result.answer

    def is_terminal(self, text: str) -> bool:
        """Check if text contains a terminal answer using the detector."""
        result = self.terminal_detector.check(text)
        return result.is_terminal


class LLMGenerator(Generator):
    """Generator that uses an LLM to produce reasoning continuations."""

    DEFAULT_PROMPT_TEMPLATE = """You are solving a problem step by step.

Question: {question}

Reasoning so far:
{state}

Continue the reasoning with ONE clear next step.
- Think carefully about what would be most helpful next
- {terminal_instruction}
- If not done, continue reasoning toward the solution

Your next step:"""

    DEFAULT_DIVERSE_PROMPT_TEMPLATE = """You are solving a problem step by step.

Question: {question}

Reasoning so far:
{state}

Generate {n} DIFFERENT possible next steps. Each should explore a meaningfully different approach or direction.

For each continuation:
- {terminal_instruction}
- Otherwise, show the next reasoning step

Format your response as:
--- CONTINUATION 1 ---
[your first continuation]

--- CONTINUATION 2 ---
[your second continuation]

(etc.)"""

    def __init__(
        self,
        llm,
        prompt_template: Optional[str] = None,
        diverse_prompt_template: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        terminal_detector: Optional[TerminalDetector] = None,
    ):
        """
        Initialize the LLM generator.

        Args:
            llm: LLM provider with .generate(prompt, **kwargs) method
            prompt_template: Custom prompt for single continuation
            diverse_prompt_template: Custom prompt for multiple continuations
            temperature: Sampling temperature
            max_tokens: Max tokens per continuation
            terminal_detector: Detector for terminal states
        """
        super().__init__(terminal_detector)
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Build prompts with terminal instruction from detector
        terminal_instruction = self.terminal_detector.format_instruction()
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE.format(
            question="{question}",
            state="{state}",
            terminal_instruction=terminal_instruction,
        )
        self.diverse_prompt_template = (
            diverse_prompt_template
            or self.DEFAULT_DIVERSE_PROMPT_TEMPLATE.format(
                question="{question}",
                state="{state}",
                n="{n}",
                terminal_instruction=terminal_instruction,
            )
        )

    def generate(self, question: str, state: str, n: int = 1) -> List[Continuation]:
        """Generate n continuations from the current state."""
        if n == 1:
            return [self._generate_single(question, state)]
        else:
            return self._generate_diverse(question, state, n)

    def _generate_single(self, question: str, state: str) -> Continuation:
        """Generate a single continuation."""
        prompt = self.prompt_template.format(question=question, state=state)

        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Build the new state by appending the response
        new_state = f"{state}\n\n{response.strip()}"

        return Continuation(
            text=new_state,
            is_terminal=self.is_terminal(response),
            answer=self.extract_answer(response),
        )

    def _generate_diverse(
        self, question: str, state: str, n: int
    ) -> List[Continuation]:
        """Generate n diverse continuations."""
        prompt = self.diverse_prompt_template.format(
            question=question,
            state=state,
            n=n,
        )

        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens * n,  # More tokens for multiple continuations
        )

        # Parse the response into separate continuations
        continuations = self._parse_diverse_response(response, state)

        # If parsing failed, fall back to single generation multiple times
        if len(continuations) < n:
            for _ in range(n - len(continuations)):
                continuations.append(self._generate_single(question, state))

        return continuations[:n]

    def _parse_diverse_response(
        self, response: str, base_state: str
    ) -> List[Continuation]:
        """Parse a diverse response into separate continuations."""
        continuations = []

        # Split by continuation markers
        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            new_state = f"{base_state}\n\n{part}"
            continuations.append(
                Continuation(
                    text=new_state,
                    is_terminal=self.is_terminal(part),
                    answer=self.extract_answer(part),
                )
            )

        return continuations


class MockGenerator(Generator):
    """Mock generator for testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        terminal_detector: Optional[TerminalDetector] = None,
    ):
        """
        Initialize mock generator.

        Args:
            responses: List of canned responses to return in order.
                       If None, generates simple numbered responses.
            terminal_detector: Detector for terminal states
        """
        super().__init__(terminal_detector)
        self.responses = responses or []
        self.call_count = 0

    def generate(self, question: str, state: str, n: int = 1) -> List[Continuation]:
        """Generate mock continuations."""
        continuations = []

        for i in range(n):
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
            else:
                # Default: after a few steps, produce an answer
                depth = state.count("Step")
                if depth >= 2:
                    response = f"Step {depth + 1}: Therefore, ANSWER: 4"
                else:
                    response = f"Step {depth + 1}: Analyzing the problem..."

            self.call_count += 1

            new_state = f"{state}\n\n{response}"
            continuations.append(
                Continuation(
                    text=new_state,
                    is_terminal=self.is_terminal(response),
                    answer=self.extract_answer(response),
                )
            )

        return continuations
