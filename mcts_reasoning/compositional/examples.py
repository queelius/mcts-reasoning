"""
Examples for one-shot and few-shot learning in compositional prompting.

Provides structured examples that can be used to guide LLM reasoning
through demonstration.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import random


@dataclass
class Example:
    """
    A single example for few-shot learning.

    Can represent either:
    - A complete solution example (problem + steps + solution)
    - A compositional guidance example (problem + which dimensions to use)
    """

    problem: str
    solution: str
    reasoning_steps: Optional[List[str]] = None
    compositional_vector: Optional[Dict[str, str]] = None  # For RAG type (a)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_string(self, include_steps: bool = True,
                        include_solution: bool = True) -> str:
        """
        Convert example to prompt string for few-shot learning.

        Args:
            include_steps: Whether to include reasoning steps
            include_solution: Whether to include final solution

        Returns:
            Formatted example string
        """
        parts = [f"Problem: {self.problem}"]

        if include_steps and self.reasoning_steps:
            parts.append("\nReasoning:")
            for i, step in enumerate(self.reasoning_steps, 1):
                parts.append(f"{i}. {step}")

        if include_solution:
            parts.append(f"\nSolution: {self.solution}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'problem': self.problem,
            'solution': self.solution,
            'reasoning_steps': self.reasoning_steps,
            'compositional_vector': self.compositional_vector,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """Create from dictionary."""
        return cls(
            problem=data['problem'],
            solution=data['solution'],
            reasoning_steps=data.get('reasoning_steps'),
            compositional_vector=data.get('compositional_vector'),
            metadata=data.get('metadata', {})
        )


class ExampleSet:
    """
    A collection of examples with retrieval capabilities.

    Supports:
    - Storage of multiple examples
    - Similarity-based retrieval
    - Random sampling for few-shot prompting
    """

    def __init__(self, examples: Optional[List[Example]] = None):
        """
        Initialize example set.

        Args:
            examples: Initial list of examples
        """
        self.examples: List[Example] = examples or []
        self._embeddings: Optional[Dict[int, Any]] = None  # For future embedding support

    def add(self, example: Example) -> 'ExampleSet':
        """Add an example to the set."""
        self.examples.append(example)
        return self

    def add_from_dict(self, problem: str, solution: str,
                     reasoning_steps: Optional[List[str]] = None,
                     **metadata) -> 'ExampleSet':
        """
        Add an example from components.

        Args:
            problem: Problem statement
            solution: Solution
            reasoning_steps: Optional reasoning steps
            **metadata: Additional metadata

        Returns:
            self (for chaining)
        """
        example = Example(
            problem=problem,
            solution=solution,
            reasoning_steps=reasoning_steps,
            metadata=metadata
        )
        self.examples.append(example)
        return self

    def retrieve_similar(self, query: str, k: int = 3,
                        method: str = 'keyword') -> List[Example]:
        """
        Retrieve k most similar examples to query.

        Args:
            query: Query problem statement
            k: Number of examples to retrieve
            method: Similarity method ('keyword', 'random', or 'embedding')

        Returns:
            List of k most similar examples
        """
        if not self.examples:
            return []

        if method == 'random':
            return random.sample(self.examples, min(k, len(self.examples)))

        elif method == 'keyword':
            # Simple keyword-based similarity
            query_words = set(query.lower().split())

            scored_examples = []
            for example in self.examples:
                problem_words = set(example.problem.lower().split())
                # Jaccard similarity
                intersection = len(query_words & problem_words)
                union = len(query_words | problem_words)
                similarity = intersection / union if union > 0 else 0
                scored_examples.append((similarity, example))

            # Sort by similarity and return top k
            scored_examples.sort(reverse=True, key=lambda x: x[0])
            return [ex for _, ex in scored_examples[:k]]

        elif method == 'embedding':
            # Future: use embeddings for semantic similarity
            # For now, fall back to keyword
            return self.retrieve_similar(query, k, method='keyword')

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def retrieve_by_metadata(self, **filters) -> List[Example]:
        """
        Retrieve examples by metadata filters.

        Args:
            **filters: Metadata key-value pairs to filter by

        Returns:
            Filtered examples
        """
        results = []
        for example in self.examples:
            match = all(
                example.metadata.get(key) == value
                for key, value in filters.items()
            )
            if match:
                results.append(example)
        return results

    def sample_random(self, k: int = 3) -> List[Example]:
        """Sample k random examples."""
        return random.sample(self.examples, min(k, len(self.examples)))

    def to_few_shot_prompt(self, n_examples: int = 3,
                          query: Optional[str] = None,
                          retrieval_method: str = 'keyword',
                          include_steps: bool = True) -> str:
        """
        Generate a few-shot prompt from examples.

        Args:
            n_examples: Number of examples to include
            query: Query for similarity-based retrieval (if None, uses random)
            retrieval_method: How to retrieve examples
            include_steps: Whether to include reasoning steps

        Returns:
            Formatted few-shot prompt
        """
        if query:
            examples = self.retrieve_similar(query, n_examples, retrieval_method)
        else:
            examples = self.sample_random(n_examples)

        if not examples:
            return ""

        parts = ["Here are some examples:\n"]

        for i, example in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(example.to_prompt_string(include_steps=include_steps))
            parts.append("")  # Blank line between examples

        return "\n".join(parts)

    def __len__(self) -> int:
        """Number of examples in the set."""
        return len(self.examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def __repr__(self):
        return f"ExampleSet({len(self.examples)} examples)"


# ========== Predefined Example Sets ==========

def get_math_examples() -> ExampleSet:
    """Get a predefined set of math problem examples."""
    examples = ExampleSet()

    examples.add_from_dict(
        problem="What is 15 × 23?",
        reasoning_steps=[
            "Break down: 15 × 23 = 15 × (20 + 3)",
            "Distribute: 15 × 20 + 15 × 3",
            "Calculate: 300 + 45",
            "Sum: 345"
        ],
        solution="345",
        domain="arithmetic"
    )

    examples.add_from_dict(
        problem="Solve x² + 5x + 6 = 0",
        reasoning_steps=[
            "Factor the quadratic: (x + 2)(x + 3) = 0",
            "Set each factor to zero: x + 2 = 0 or x + 3 = 0",
            "Solve: x = -2 or x = -3"
        ],
        solution="x = -2 or x = -3",
        domain="algebra"
    )

    examples.add_from_dict(
        problem="Find all prime numbers less than 20",
        reasoning_steps=[
            "List candidates: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19",
            "Remove even numbers (except 2): 2, 3, 5, 7, 9, 11, 13, 15, 17, 19",
            "Remove multiples of 3 (except 3): 2, 3, 5, 7, 11, 13, 17, 19",
            "All remaining numbers are prime"
        ],
        solution="2, 3, 5, 7, 11, 13, 17, 19",
        domain="number_theory"
    )

    return examples


def get_logic_examples() -> ExampleSet:
    """Get a predefined set of logic problem examples."""
    examples = ExampleSet()

    examples.add_from_dict(
        problem="If all birds can fly, and penguins are birds, can penguins fly?",
        reasoning_steps=[
            "Premise 1: All birds can fly",
            "Premise 2: Penguins are birds",
            "Logical conclusion: Penguins can fly",
            "Reality check: This conclusion is false - the first premise is incorrect"
        ],
        solution="The argument is logically valid but unsound because the premise 'all birds can fly' is false.",
        domain="logic"
    )

    return examples


def get_coding_examples() -> ExampleSet:
    """Get a predefined set of coding problem examples."""
    examples = ExampleSet()

    examples.add_from_dict(
        problem="Write a function to check if a number is prime",
        reasoning_steps=[
            "Handle edge cases: numbers ≤ 1 are not prime",
            "2 is the only even prime",
            "For odd numbers, check divisibility up to √n",
            "If no divisors found, number is prime"
        ],
        solution="""def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True""",
        domain="programming"
    )

    return examples


__all__ = [
    'Example',
    'ExampleSet',
    'get_math_examples',
    'get_logic_examples',
    'get_coding_examples',
]
