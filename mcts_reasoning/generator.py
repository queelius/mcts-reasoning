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
from typing import Optional, List

from .types import State, Continuation, extend_state
from .terminal import TerminalDetector, MarkerTerminalDetector
from .prompt import PromptStrategy, StepByStepPrompt


# Marker that signals a complete answer (default)
ANSWER_MARKER = "ANSWER:"


class Generator(ABC):
    """Abstract base class for reasoning generators."""

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

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer using a marker detector (convenience method)."""
        result = MarkerTerminalDetector().is_terminal(text)
        return result.answer

    def is_terminal(self, text: str) -> bool:
        """Check if text contains a terminal answer (convenience method)."""
        result = MarkerTerminalDetector().is_terminal(text)
        return result.is_terminal


class LLMGenerator(Generator):
    """Generator that uses an LLM to produce reasoning continuations."""

    def __init__(
        self,
        provider=None,
        prompt_strategy: Optional[PromptStrategy] = None,
        terminal_detector: Optional[TerminalDetector] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        # Legacy kwargs -- silently accepted for backward compat
        llm=None,
        prompt_template=None,
        diverse_prompt_template=None,
    ):
        """
        Initialize the LLM generator.

        Args:
            provider: LLM provider with .generate(messages, ...) method
            prompt_strategy: Strategy for building prompts (default: StepByStepPrompt)
            terminal_detector: Detector for terminal states (default: MarkerTerminalDetector)
            max_tokens: Max tokens per continuation
            temperature: Sampling temperature

        Legacy kwargs (deprecated, for backward compat):
            llm: Alias for provider
            prompt_template: Ignored (use prompt_strategy)
            diverse_prompt_template: Ignored (use prompt_strategy)
        """
        # Support legacy `llm=` kwarg
        self.provider = provider if provider is not None else llm
        if self.provider is None:
            raise TypeError("LLMGenerator requires a provider (or llm=) argument")

        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.prompt_strategy = prompt_strategy or StepByStepPrompt(
            terminal_detector=self.terminal_detector,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, state: str, n: int = 1) -> List[Continuation]:
        """Generate n continuations from the current state."""
        messages = self.prompt_strategy.format(question, State(state), n)
        response = self.provider.generate(
            messages,
            self.max_tokens,
            self.temperature,
        )
        raw_continuations = self.prompt_strategy.parse(response, n)

        results = []
        for text in raw_continuations:
            check = self.terminal_detector.is_terminal(text)
            new_state = extend_state(State(state), text)
            results.append(
                Continuation(
                    text=new_state,
                    is_terminal=check.is_terminal,
                    answer=check.answer,
                )
            )
        return results

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer using the terminal detector."""
        result = self.terminal_detector.is_terminal(text)
        return result.answer

    def is_terminal(self, text: str) -> bool:
        """Check if text contains a terminal answer using the detector."""
        result = self.terminal_detector.is_terminal(text)
        return result.is_terminal


# ---------------------------------------------------------------------------
# Backward compatibility re-exports
# ---------------------------------------------------------------------------
# Old code imports MockGenerator from generator.py; re-export from testing
from .testing import MockGenerator  # noqa: E402,F401
