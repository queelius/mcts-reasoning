"""
Actions: State-dependent operations in the MCTS search.

In MCTS, actions transform states. For LLM reasoning:
- Most actions are prompts that generate the next reasoning step
- Different states may have different available actions
- The CONTINUE action (generate next step) is always available for non-terminal states

This abstraction allows extensions like:
- COMPRESS: Summarize long reasoning traces
- TOOL_CALL: Invoke external tools (calculator, search, etc.)
- VERIFY: Ask LLM to verify current reasoning
"""

from typing import List, Optional, Protocol, runtime_checkable, TYPE_CHECKING
from dataclasses import dataclass
import re

if TYPE_CHECKING:
    from .generator import Generator


@dataclass
class ActionResult:
    """Result of applying an action to a state."""

    new_state: str
    is_terminal: bool
    answer: Optional[str] = None


@runtime_checkable
class Action(Protocol):
    """
    An action that can be applied to a reasoning state.

    Actions are the edges in the MCTS tree. Each action transforms
    the current state into a new state.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this action type."""
        ...

    def apply(self, question: str, state: str) -> ActionResult:
        """
        Apply this action to produce a new state.

        Args:
            question: The original question being solved
            state: Current reasoning state

        Returns:
            ActionResult with new state and terminal info
        """
        ...


class ContinueAction:
    """
    The default action: generate the next reasoning step.

    This action wraps a Generator to produce continuations. It is available
    in all non-terminal states and is the core of MCTS-based reasoning.

    Can be used in two modes:
    1. With a Generator (preferred for MCTS integration)
    2. With a raw LLM (standalone usage)
    """

    DEFAULT_PROMPT = """You are solving a problem step by step.

Question: {question}

Reasoning so far:
{state}

Continue the reasoning with ONE clear next step.
- Think carefully about what would be most helpful next
- If you have reached a final answer, clearly state: ANSWER: <your answer>
- If not done, continue reasoning toward the solution

Your next step:"""

    ANSWER_MARKER = "ANSWER:"

    def __init__(
        self,
        generator: Optional["Generator"] = None,
        *,
        # Fallback params for standalone usage without Generator
        llm=None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize ContinueAction.

        Args:
            generator: Generator to use for continuations (preferred)
            llm: Raw LLM provider (fallback if no generator)
            prompt_template: Custom prompt template (only used with raw LLM)
            temperature: Sampling temperature (only used with raw LLM)
            max_tokens: Max tokens (only used with raw LLM)
        """
        self.generator = generator
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "CONTINUE"

    def apply(self, question: str, state: str) -> ActionResult:
        """Generate the next reasoning step."""
        if self.generator is not None:
            return self._apply_with_generator(question, state)
        elif self.llm is not None:
            return self._apply_with_llm(question, state)
        else:
            raise ValueError("ContinueAction requires either a generator or llm")

    def _apply_with_generator(self, question: str, state: str) -> ActionResult:
        """Apply using Generator (preferred path)."""
        continuations = self.generator.generate(question, state, n=1)
        if not continuations:
            # No continuation generated - return unchanged state
            return ActionResult(state, is_terminal=False)

        cont = continuations[0]
        return ActionResult(
            new_state=cont.text,
            is_terminal=cont.is_terminal,
            answer=cont.answer,
        )

    def _apply_with_llm(self, question: str, state: str) -> ActionResult:
        """Apply using raw LLM (fallback path)."""
        prompt = self.prompt_template.format(question=question, state=state)

        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        new_state = f"{state}\n\n{response.strip()}"
        is_terminal = self.ANSWER_MARKER in response
        answer = self._extract_answer(response) if is_terminal else None

        return ActionResult(
            new_state=new_state,
            is_terminal=is_terminal,
            answer=answer,
        )

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from text containing ANSWER: marker."""
        match = re.search(rf"{self.ANSWER_MARKER}\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        idx = text.find(self.ANSWER_MARKER)
        if idx >= 0:
            return text[idx + len(self.ANSWER_MARKER) :].strip()
        return None


@runtime_checkable
class ActionSpace(Protocol):
    """
    Defines available actions for each state.

    In standard MCTS for games, different positions have different legal moves.
    Similarly, different reasoning states may have different available actions.
    """

    def get_actions(self, state: str, is_terminal: bool) -> List[Action]:
        """
        Get available actions for a state.

        Args:
            state: Current reasoning state
            is_terminal: Whether state is terminal

        Returns:
            List of available actions (empty if terminal)
        """
        ...


class DefaultActionSpace:
    """
    Default action space: only CONTINUE is available.

    This is the canonical MCTS-reasoning action space where
    the only action is to generate the next reasoning step.
    """

    def __init__(
        self,
        generator: Optional["Generator"] = None,
        continue_action: Optional[ContinueAction] = None,
    ):
        """
        Initialize default action space.

        Args:
            generator: Generator to use for CONTINUE action
            continue_action: Custom ContinueAction (overrides generator)
        """
        if continue_action is not None:
            self.continue_action = continue_action
        elif generator is not None:
            self.continue_action = ContinueAction(generator=generator)
        else:
            # Will fail at apply time if no generator/llm provided
            self.continue_action = ContinueAction()

    def get_actions(self, state: str, is_terminal: bool) -> List[Action]:
        """Return [CONTINUE] for non-terminal states, [] for terminal."""
        if is_terminal:
            return []
        return [self.continue_action]


# =============================================================================
# EXTENSIONS (documented but not part of canonical implementation)
# See paper Section 8 (Extensions)
# =============================================================================


class CompressAction:
    """
    COMPRESS action: Summarize a long reasoning trace.

    When the reasoning trace exceeds a threshold, this action
    creates a compressed representation. Subsequent actions
    only need the compressed state, not the full history.

    NOTE: This is an EXTENSION, not part of the canonical implementation.
    """

    COMPRESS_PROMPT = """Summarize the following reasoning trace concisely,
preserving key insights, intermediate results, and the current direction of reasoning.

Question: {question}

Full reasoning trace:
{state}

Provide a compressed summary that captures the essential progress made:"""

    def __init__(
        self,
        llm,
        threshold: int = 2000,  # Characters before compression available
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        """
        Initialize CompressAction.

        Args:
            llm: LLM provider for compression
            threshold: Min state length before COMPRESS is available
            temperature: Sampling temperature
            max_tokens: Max tokens for compressed output
        """
        self.llm = llm
        self.threshold = threshold
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "COMPRESS"

    def is_available(self, state: str) -> bool:
        """COMPRESS is only available for long traces."""
        return len(state) > self.threshold

    def apply(self, question: str, state: str) -> ActionResult:
        """Compress the reasoning trace."""
        prompt = self.COMPRESS_PROMPT.format(question=question, state=state)

        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Mark this as a compressed state
        new_state = f"[COMPRESSED REASONING]\n{response.strip()}\n[/COMPRESSED]"

        return ActionResult(
            new_state=new_state,
            is_terminal=False,  # Compression never terminates
            answer=None,
        )


class ExtendedActionSpace:
    """
    Extended action space with CONTINUE and COMPRESS.

    NOTE: This is an EXTENSION example, not part of canonical implementation.
    """

    def __init__(
        self,
        generator: Optional["Generator"] = None,
        llm=None,
        continue_action: Optional[ContinueAction] = None,
        compress_action: Optional[CompressAction] = None,
        compress_threshold: int = 2000,
    ):
        """
        Initialize extended action space.

        Args:
            generator: Generator for CONTINUE action
            llm: LLM for COMPRESS action (and CONTINUE fallback)
            continue_action: Custom ContinueAction
            compress_action: Custom CompressAction
            compress_threshold: Threshold for COMPRESS availability
        """
        # Set up CONTINUE action
        if continue_action is not None:
            self.continue_action = continue_action
        elif generator is not None:
            self.continue_action = ContinueAction(generator=generator)
        elif llm is not None:
            self.continue_action = ContinueAction(llm=llm)
        else:
            raise ValueError("ExtendedActionSpace requires generator or llm")

        # Set up COMPRESS action
        if compress_action is not None:
            self.compress_action = compress_action
        elif llm is not None:
            self.compress_action = CompressAction(llm=llm, threshold=compress_threshold)
        else:
            # COMPRESS won't be available without LLM
            self.compress_action = None

    def get_actions(self, state: str, is_terminal: bool) -> List[Action]:
        """Return available actions based on state."""
        if is_terminal:
            return []

        actions: List[Action] = [self.continue_action]

        if self.compress_action is not None and self.compress_action.is_available(
            state
        ):
            actions.append(self.compress_action)

        return actions
