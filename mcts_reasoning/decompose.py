"""
Solution Decomposition: let the LLM solve freely, then structure the output.

Instead of constraining the LLM to write one step at a time (which it fights),
we let it write a complete solution, then decompose that solution into a chain
of nodes. The tree structure is imposed post-hoc, not as a prompt constraint.

Two types of tree growth:
- Depth (decomposition): complete solution → chain of step nodes
- Breadth (alternatives): different solution approaches → sibling chains

This approach:
1. Gets better LLM output (unconstrained generation)
2. Creates richer trees (each step is a separate node for fine-grained eval)
3. Enables parallelism (independent solution attempts can run concurrently)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Message, State, Continuation, extend_state
from .terminal import TerminalDetector, MarkerTerminalDetector
from .generator import Generator


@dataclass
class DecomposedStep:
    """One step in a decomposed solution."""
    text: str
    is_terminal: bool = False
    answer: str | None = None


class SolutionDecomposer(ABC):
    """Takes a complete solution and breaks it into sequential steps."""

    @abstractmethod
    def decompose(self, question: str, solution: str) -> list[DecomposedStep]:
        """Split a complete solution into individual reasoning steps."""


class LLMDecomposer(SolutionDecomposer):
    """Uses the LLM itself to decompose a solution into steps."""

    def __init__(self, provider, terminal_detector: TerminalDetector | None = None):
        self.provider = provider
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    def decompose(self, question: str, solution: str) -> list[DecomposedStep]:
        messages = [
            Message(role="system", content=(
                "You rewrite solutions as a sequence of individual reasoning steps.\n\n"
                "RULES:\n"
                "- Each step must be exactly ONE logical deduction, calculation, or conclusion\n"
                "- Steps must be self-contained (understandable without reading other steps)\n"
                "- Preserve ALL reasoning from the original solution\n"
                "- Do not add new reasoning not in the original\n"
                "- Number each step: STEP 1:, STEP 2:, etc.\n"
                "- The final step should contain the answer if one was reached\n"
            )),
            Message(role="user", content=(
                f"Question: {question}\n\n"
                f"Complete solution:\n{solution}\n\n"
                f"Rewrite as numbered steps (STEP 1:, STEP 2:, ...):"
            )),
        ]

        response = self.provider.generate(messages, max_tokens=2000, temperature=0.0)
        return self._parse_steps(response)

    def _parse_steps(self, response: str) -> list[DecomposedStep]:
        """Parse STEP N: formatted response into DecomposedStep objects."""
        import re
        # Split on STEP N: pattern
        parts = re.split(r"STEP\s+\d+\s*:", response)
        steps = []
        for part in parts:
            text = part.strip()
            if not text:
                continue
            check = self.terminal_detector.is_terminal(text)
            steps.append(DecomposedStep(
                text=text,
                is_terminal=check.is_terminal,
                answer=check.answer,
            ))
        # If no STEP markers found, treat entire response as one step
        if not steps:
            check = self.terminal_detector.is_terminal(response)
            steps.append(DecomposedStep(
                text=response,
                is_terminal=check.is_terminal,
                answer=check.answer,
            ))
        return steps


class SentenceDecomposer(SolutionDecomposer):
    """Simple decomposer: splits on sentence boundaries. No LLM call needed."""

    def __init__(self, terminal_detector: TerminalDetector | None = None):
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    def decompose(self, question: str, solution: str) -> list[DecomposedStep]:
        import re
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', solution)
        # Group into logical steps (2-3 sentences per step)
        steps = []
        buffer = []
        for sent in sentences:
            buffer.append(sent.strip())
            if len(buffer) >= 2 or sent.strip().endswith((':', '?')):
                text = ' '.join(buffer)
                check = self.terminal_detector.is_terminal(text)
                steps.append(DecomposedStep(
                    text=text,
                    is_terminal=check.is_terminal,
                    answer=check.answer,
                ))
                buffer = []
        if buffer:
            text = ' '.join(buffer)
            check = self.terminal_detector.is_terminal(text)
            steps.append(DecomposedStep(
                text=text,
                is_terminal=check.is_terminal,
                answer=check.answer,
            ))
        return steps if steps else [DecomposedStep(text=solution)]


class DecomposeGenerator(Generator):
    """Generator that asks for complete solutions, then decomposes into node chains.

    The generate() method:
    1. Asks the LLM for a complete solution (unconstrained)
    2. Decomposes the solution into steps
    3. Returns the FIRST step as the continuation (MCTS adds it as a child)
    4. Stores remaining steps so the next expansion continues the chain

    When MCTS calls generate() again on a node that has pending steps,
    it returns the next pending step instead of calling the LLM.
    This creates the chain: node -> step1 -> step2 -> step3 (terminal).
    """

    def __init__(
        self,
        provider,
        decomposer: SolutionDecomposer | None = None,
        terminal_detector: TerminalDetector | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        n_alternatives: int = 1,
    ):
        self.provider = provider
        self.decomposer = decomposer or LLMDecomposer(provider, terminal_detector)
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_alternatives = n_alternatives
        # Cache: state_hash -> remaining steps
        self._pending_steps: dict[int, list[DecomposedStep]] = {}

    def _get_complete_solution(self, question: str, state: State) -> str:
        """Ask the LLM for a complete, unconstrained solution."""
        terminal_instruction = self.terminal_detector.format_instruction()
        messages = [
            Message(role="system", content=(
                "You are a careful reasoning assistant. Solve the problem completely. "
                "Show your full reasoning step by step, then give your final answer.\n"
                f"{terminal_instruction}\n"
                "Your answer after ANSWER: should be short and direct (1-5 words)."
            )),
            Message(role="user", content=(
                f"Question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Continue solving. Show your full reasoning."
            )),
        ]
        return self.provider.generate(
            messages, max_tokens=self.max_tokens, temperature=self.temperature,
        )

    def generate(self, question: str, state: str, n: int = 1) -> list[Continuation]:
        """Generate continuations by solving completely then decomposing.

        If there are pending steps from a previous decomposition, return
        the next pending step. Otherwise, get a new complete solution.
        """
        state_key = hash(state)
        results = []

        for _ in range(n):
            # Check for pending steps from a previous decomposition
            if state_key in self._pending_steps and self._pending_steps[state_key]:
                step = self._pending_steps[state_key].pop(0)
                new_state = extend_state(State(state), step.text)
                results.append(Continuation(
                    text=new_state,
                    is_terminal=step.is_terminal,
                    answer=step.answer,
                ))
                # Update state_key for next pending step lookup
                state_key = hash(str(new_state))
                continue

            # Get a complete solution
            solution = self._get_complete_solution(question, State(state))

            # Decompose into steps
            steps = self.decomposer.decompose(question, solution)

            if not steps:
                # Fallback: use the raw solution as one step
                check = self.terminal_detector.is_terminal(solution)
                results.append(Continuation(
                    text=extend_state(State(state), solution),
                    is_terminal=check.is_terminal,
                    answer=check.answer,
                ))
                continue

            # Return the first step as the continuation
            first_step = steps[0]
            new_state = extend_state(State(state), first_step.text)
            results.append(Continuation(
                text=new_state,
                is_terminal=first_step.is_terminal,
                answer=first_step.answer,
            ))

            # Cache remaining steps for future generate() calls on this path
            if len(steps) > 1:
                new_state_key = hash(str(new_state))
                self._pending_steps[new_state_key] = steps[1:]

        return results
