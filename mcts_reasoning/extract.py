"""
Structure Extraction: discover the tree latent in natural CoT responses.

The model reasons naturally (no constraints, no action tags). Then we
parse the response to find implicit structure:
- Assumptions tested → branches
- Sequential deductions → chains
- Contradictions found → dead-end markers
- Intermediate conclusions → node values

The extracted tree can then be analyzed, scored, and expanded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import Message, State
from .node import Node
from .terminal import TerminalDetector, MarkerTerminalDetector


@dataclass
class ExtractedStep:
    """One reasoning step extracted from a CoT response."""
    text: str
    step_type: str = "deduce"  # deduce, assume, verify, calculate, conclude, contradict
    parent_idx: int = -1       # -1 = follows previous step linearly
    is_branch: bool = False    # True if this starts a new branch (e.g., "alternatively...")
    is_terminal: bool = False
    answer: str | None = None


class LLMExtractor:
    """Uses the LLM to extract tree structure from a CoT response."""

    def __init__(self, provider, terminal_detector: Optional[TerminalDetector] = None):
        self.provider = provider
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    def extract(self, question: str, response: str) -> list[ExtractedStep]:
        """Extract structured steps from a natural CoT response."""
        messages = [
            Message(role="system", content=(
                "You analyze reasoning text and extract its logical structure.\n\n"
                "For each reasoning step, output one line in this exact format:\n"
                "STEP|type|parent|text\n\n"
                "Where:\n"
                "- type is one of: assume, deduce, calculate, verify, contradict, conclude\n"
                "- parent is the step number this builds on (0 for first step, or the step number of the assumption being tested)\n"
                "- text is a brief summary of what this step does\n\n"
                "Mark branches: when the reasoning tests alternative assumptions, those start new branches from the same parent.\n\n"
                "Example:\n"
                "STEP|assume|0|Assume A is a knight\n"
                "STEP|deduce|1|Then B must be a knave since A tells truth\n"
                "STEP|verify|2|Check B's statement: consistent\n"
                "STEP|assume|0|Alternatively assume A is a knave\n"
                "STEP|deduce|4|Then B must be a knight\n"
                "STEP|contradict|5|B's statement contradicts the assumption\n"
                "STEP|conclude|3|Therefore A is a knight\n"
            )),
            Message(role="user", content=(
                f"Question: {question}\n\n"
                f"Reasoning:\n{response}\n\n"
                f"Extract the logical structure:"
            )),
        ]

        result = self.provider.generate(messages, max_tokens=1000, temperature=0.0)
        return self._parse_steps(result, response)

    def _parse_steps(self, extraction: str, original: str) -> list[ExtractedStep]:
        """Parse the STEP|type|parent|text format."""
        steps = []
        for line in extraction.strip().split("\n"):
            line = line.strip()
            if not line.startswith("STEP|"):
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue

            _, step_type, parent_str, text = parts
            step_type = step_type.strip().lower()
            try:
                parent_idx = int(parent_str.strip())
            except ValueError:
                parent_idx = max(0, len(steps))  # default: follows previous

            check = self.terminal_detector.is_terminal(text)

            steps.append(ExtractedStep(
                text=text.strip(),
                step_type=step_type,
                parent_idx=parent_idx,
                is_branch=(step_type == "assume" and parent_idx < len(steps) - 1),
                is_terminal=check.is_terminal or step_type == "conclude",
                answer=check.answer,
            ))

        return steps

    def build_tree(self, question: str, steps: list[ExtractedStep]) -> Node:
        """Convert extracted steps into a Node tree."""
        root = Node(state=State(f"Question: {question}"))
        nodes = [root]  # index 0 = root

        for i, step in enumerate(steps):
            # Find parent node
            if step.parent_idx < len(nodes) and step.parent_idx >= 0:
                parent = nodes[step.parent_idx]
            elif nodes:
                parent = nodes[-1]  # default: append to last node
            else:
                parent = root

            state_text = f"[{step.step_type}] {step.text}"
            child = parent.add_child(
                state=State(f"{parent.state}\n\n{state_text}"),
                is_terminal=step.is_terminal,
                answer=step.answer,
            )
            nodes.append(child)

        return root


class RegexExtractor:
    """Lightweight extraction using regex patterns. No LLM call needed.

    Looks for structural markers in natural CoT text:
    - "Assume..." / "Let's say..." / "If..." → assumption branch
    - "Therefore..." / "So..." / "Thus..." → deduction
    - "But..." / "However..." / "Contradiction" → contradiction
    - "The answer is..." / "ANSWER:" → conclusion
    - Numbers after "=" → calculation
    """

    def __init__(self, terminal_detector: Optional[TerminalDetector] = None):
        self.terminal_detector = terminal_detector or MarkerTerminalDetector()

    def extract(self, question: str, response: str) -> list[ExtractedStep]:
        """Extract steps using regex pattern matching on the response text."""
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        steps = []
        current_branch_root = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            lower = sent.lower()
            check = self.terminal_detector.is_terminal(sent)

            # Classify the sentence
            if re.match(r'(?i)(let\'?s?\s+)?(assume|suppose|say|consider\s+the\s+case)', lower):
                step_type = "assume"
                # Assumptions branch from the current root
                parent = current_branch_root
                is_branch = len(steps) > 0
            elif re.match(r'(?i)(but|however|contradiction|this\s+contradicts|impossible)', lower):
                step_type = "contradict"
                parent = len(steps)  # follows previous
                is_branch = False
            elif re.match(r'(?i)(therefore|thus|so|hence|we\s+can\s+conclude|this\s+means)', lower):
                step_type = "deduce"
                parent = len(steps)
                is_branch = False
            elif re.search(r'=\s*\d', sent):
                step_type = "calculate"
                parent = len(steps)
                is_branch = False
            elif check.is_terminal:
                step_type = "conclude"
                parent = len(steps)
                is_branch = False
            else:
                step_type = "deduce"
                parent = len(steps)
                is_branch = False

            steps.append(ExtractedStep(
                text=sent,
                step_type=step_type,
                parent_idx=parent,
                is_branch=is_branch,
                is_terminal=check.is_terminal or step_type == "conclude",
                answer=check.answer,
            ))

            # After a contradiction, the next assumption branches from earlier
            if step_type == "contradict":
                current_branch_root = 0

        return steps

    def build_tree(self, question: str, steps: list[ExtractedStep]) -> Node:
        """Convert extracted steps into a Node tree."""
        root = Node(state=State(f"Question: {question}"))
        nodes = [root]

        for step in steps:
            parent_idx = min(step.parent_idx, len(nodes) - 1)
            parent = nodes[max(0, parent_idx)]

            state_text = f"[{step.step_type}] {step.text}"
            child = parent.add_child(
                state=State(f"{parent.state}\n\n{state_text}"),
                is_terminal=step.is_terminal,
                answer=step.answer,
            )
            nodes.append(child)

        return root
