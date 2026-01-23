"""
Retrieval-Augmented Generation (RAG) for compositional prompting.

Provides two types of RAG:
1. CompositionalRAGStore: Maps problem patterns to compositional dimensions
2. SolutionRAGStore: Maps problems to complete solution examples
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from . import (
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
)
from .examples import Example, ExampleSet


# ========== Type (a): Compositional RAG ==========


@dataclass
class CompositionalGuidance:
    """
    Sparse guidance mapping problem patterns to compositional dimensions.

    This is RAG type (a): Instead of providing full examples, it suggests
    which compositional dimensions to emphasize for specific problem types.
    """

    problem_pattern: str  # Description or keywords identifying problem type
    problem_keywords: List[str] = field(default_factory=list)

    # Recommended dimensions (can be partial)
    recommended_operations: Optional[List[CognitiveOperation]] = None
    recommended_focuses: Optional[List[FocusAspect]] = None
    recommended_styles: Optional[List[ReasoningStyle]] = None
    recommended_connections: Optional[List[ConnectionType]] = None
    recommended_formats: Optional[List[OutputFormat]] = None

    # Weights for biased sampling (higher = more preferred)
    weights: Optional[Dict[str, Dict[Any, float]]] = None

    # Metadata
    domain: Optional[str] = None
    difficulty: Optional[str] = None
    success_rate: float = 0.0  # Track how well this guidance works

    def to_weights_dict(self) -> Dict[str, Dict[Any, float]]:
        """
        Convert recommendations to weight dictionary for ActionSelector.

        Returns:
            Weight dictionary suitable for ComposingPrompt.sample_weighted()
        """
        if self.weights:
            return self.weights

        # Build weights from recommendations
        weights = {}

        if self.recommended_operations:
            weights["cognitive_op"] = {op: 3.0 for op in self.recommended_operations}

        if self.recommended_focuses:
            weights["focus"] = {focus: 2.5 for focus in self.recommended_focuses}

        if self.recommended_styles:
            weights["style"] = {style: 3.0 for style in self.recommended_styles}

        if self.recommended_connections:
            weights["connection"] = {conn: 2.0 for conn in self.recommended_connections}

        if self.recommended_formats:
            weights["output_format"] = {fmt: 2.0 for fmt in self.recommended_formats}

        return weights

    def matches_problem(self, problem: str) -> float:
        """
        Calculate match score for a given problem.

        Args:
            problem: Problem statement

        Returns:
            Match score (0-1)
        """
        problem_lower = problem.lower()

        # Keyword matching
        if self.problem_keywords:
            matches = sum(
                1 for kw in self.problem_keywords if kw.lower() in problem_lower
            )
            return min(matches / len(self.problem_keywords), 1.0)

        # Pattern matching (simple substring)
        if self.problem_pattern.lower() in problem_lower:
            return 0.8

        return 0.0


class CompositionalRAGStore:
    """
    Storage and retrieval for compositional guidance.

    Maps problem types to recommended compositional dimensions.
    This is a sparse representation - just the dimensions, not full examples.
    """

    def __init__(self, guidance_list: Optional[List[CompositionalGuidance]] = None):
        """
        Initialize RAG store.

        Args:
            guidance_list: Initial list of guidance entries
        """
        self.guidance: List[CompositionalGuidance] = guidance_list or []

    def add_guidance(self, guidance: CompositionalGuidance) -> "CompositionalRAGStore":
        """Add guidance entry."""
        self.guidance.append(guidance)
        return self

    def add(
        self,
        problem_pattern: str,
        keywords: Optional[List[str]] = None,
        operations: Optional[List[CognitiveOperation]] = None,
        focuses: Optional[List[FocusAspect]] = None,
        styles: Optional[List[ReasoningStyle]] = None,
        domain: Optional[str] = None,
        **kwargs,
    ) -> "CompositionalRAGStore":
        """
        Add guidance entry from components.

        Args:
            problem_pattern: Problem pattern description
            keywords: Keywords to match
            operations: Recommended operations
            focuses: Recommended focuses
            styles: Recommended reasoning styles
            domain: Problem domain
            **kwargs: Additional metadata

        Returns:
            self (for chaining)
        """
        guidance = CompositionalGuidance(
            problem_pattern=problem_pattern,
            problem_keywords=keywords or [],
            recommended_operations=operations,
            recommended_focuses=focuses,
            recommended_styles=styles,
            domain=domain,
            **kwargs,
        )
        self.guidance.append(guidance)
        return self

    def retrieve(self, problem: str, k: int = 3) -> List[CompositionalGuidance]:
        """
        Retrieve k most relevant guidance entries for a problem.

        Args:
            problem: Problem statement
            k: Number of entries to retrieve

        Returns:
            List of relevant guidance entries
        """
        if not self.guidance:
            return []

        # Score all guidance entries
        scored = [(g.matches_problem(problem), g) for g in self.guidance]

        # Sort by score and return top k
        scored.sort(reverse=True, key=lambda x: x[0])

        # Filter out zero scores
        return [g for score, g in scored[:k] if score > 0]

    def get_recommended_weights(
        self, problem: str, merge_strategy: str = "average"
    ) -> Dict[str, Dict[Any, float]]:
        """
        Get recommended weights for a problem by merging guidance.

        Args:
            problem: Problem statement
            merge_strategy: How to merge multiple guidance entries ('average', 'max', 'first')

        Returns:
            Weight dictionary for ActionSelector
        """
        relevant_guidance = self.retrieve(problem, k=3)

        if not relevant_guidance:
            return {}

        if merge_strategy == "first":
            return relevant_guidance[0].to_weights_dict()

        # Merge multiple guidance entries
        all_weights: Dict[str, Dict[Any, float]] = {}

        for guidance in relevant_guidance:
            weights = guidance.to_weights_dict()

            for dim, dim_weights in weights.items():
                if dim not in all_weights:
                    all_weights[dim] = {}

                for component, weight in dim_weights.items():
                    if component in all_weights[dim]:
                        if merge_strategy == "average":
                            all_weights[dim][component] = (
                                all_weights[dim][component] + weight
                            ) / 2
                        elif merge_strategy == "max":
                            all_weights[dim][component] = max(
                                all_weights[dim][component], weight
                            )
                    else:
                        all_weights[dim][component] = weight

        return all_weights

    def update_success_rate(self, problem: str, success: bool):
        """
        Update success rate for matching guidance.

        Args:
            problem: Problem that was solved
            success: Whether it was successful
        """
        for guidance in self.retrieve(problem, k=1):
            # Simple running average update
            n = getattr(guidance, "_update_count", 0)
            old_rate = guidance.success_rate
            guidance.success_rate = (old_rate * n + (1.0 if success else 0.0)) / (n + 1)
            guidance._update_count = n + 1  # Track update count

    def save(self, filepath: Path):
        """Save RAG store to JSON file."""
        # TODO: Implement serialization
        pass

    @classmethod
    def load(cls, filepath: Path) -> "CompositionalRAGStore":
        """Load RAG store from JSON file."""
        # TODO: Implement deserialization
        return cls()

    def __len__(self) -> int:
        return len(self.guidance)

    def __repr__(self):
        return f"CompositionalRAGStore({len(self.guidance)} guidance entries)"


# ========== Type (b): Solution RAG ==========


class SolutionRAGStore:
    """
    Storage and retrieval for complete solution examples.

    This is RAG type (b): Full examples with problem, reasoning steps,
    and solution. Traditional few-shot learning.
    """

    def __init__(self, examples: Optional[ExampleSet] = None):
        """
        Initialize solution RAG store.

        Args:
            examples: ExampleSet to use for storage
        """
        self.examples = examples or ExampleSet()

    def add_example(self, example: Example) -> "SolutionRAGStore":
        """Add a solution example."""
        self.examples.add(example)
        return self

    def add(
        self,
        problem: str,
        solution: str,
        reasoning_steps: Optional[List[str]] = None,
        **metadata,
    ) -> "SolutionRAGStore":
        """
        Add example from components.

        Args:
            problem: Problem statement
            solution: Solution
            reasoning_steps: Optional reasoning steps
            **metadata: Additional metadata

        Returns:
            self (for chaining)
        """
        self.examples.add_from_dict(
            problem=problem,
            solution=solution,
            reasoning_steps=reasoning_steps,
            **metadata,
        )
        return self

    def retrieve(
        self, problem: str, k: int = 3, method: str = "keyword"
    ) -> List[Example]:
        """
        Retrieve k most similar solution examples.

        Args:
            problem: Problem statement
            k: Number of examples to retrieve
            method: Similarity method

        Returns:
            List of similar examples
        """
        return self.examples.retrieve_similar(problem, k, method)

    def to_few_shot_prompt(
        self, problem: str, n_examples: int = 3, include_steps: bool = True
    ) -> str:
        """
        Generate few-shot prompt for a problem.

        Args:
            problem: Current problem
            n_examples: Number of examples to include
            include_steps: Whether to include reasoning steps

        Returns:
            Formatted few-shot prompt
        """
        return self.examples.to_few_shot_prompt(
            n_examples=n_examples,
            query=problem,
            retrieval_method="keyword",
            include_steps=include_steps,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __repr__(self):
        return f"SolutionRAGStore({len(self.examples)} examples)"


# ========== Predefined RAG Stores ==========


def get_math_compositional_rag() -> CompositionalRAGStore:
    """Get predefined compositional RAG for math problems."""
    store = CompositionalRAGStore()

    # Quadratic equations
    store.add(
        problem_pattern="quadratic equation",
        keywords=["x^2", "x²", "quadratic", "solve equation"],
        operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.ANALYZE],
        focuses=[FocusAspect.STRUCTURE, FocusAspect.PATTERNS],
        styles=[ReasoningStyle.SYSTEMATIC, ReasoningStyle.FORMAL],
        domain="algebra",
    )

    # Prime numbers
    store.add(
        problem_pattern="prime numbers",
        keywords=["prime", "primes", "divisible", "factors"],
        operations=[CognitiveOperation.GENERATE, CognitiveOperation.VERIFY],
        focuses=[FocusAspect.PATTERNS, FocusAspect.CORRECTNESS],
        styles=[ReasoningStyle.SYSTEMATIC],
        domain="number_theory",
    )

    # Arithmetic
    store.add(
        problem_pattern="arithmetic calculation",
        keywords=["multiply", "divide", "add", "subtract", "calculate"],
        operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.ANALYZE],
        focuses=[FocusAspect.STRUCTURE, FocusAspect.EFFICIENCY],
        styles=[ReasoningStyle.SYSTEMATIC],
        domain="arithmetic",
    )

    # Proofs
    store.add(
        problem_pattern="prove",
        keywords=["prove", "proof", "show that", "demonstrate"],
        operations=[CognitiveOperation.VERIFY, CognitiveOperation.ANALYZE],
        focuses=[FocusAspect.CORRECTNESS, FocusAspect.ASSUMPTIONS],
        styles=[ReasoningStyle.FORMAL, ReasoningStyle.CRITICAL],
        domain="proofs",
    )

    return store


def get_coding_compositional_rag() -> CompositionalRAGStore:
    """Get predefined compositional RAG for coding problems."""
    store = CompositionalRAGStore()

    # Algorithm design
    store.add(
        problem_pattern="algorithm",
        keywords=["algorithm", "function", "implement", "design"],
        operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.GENERATE],
        focuses=[FocusAspect.STRUCTURE, FocusAspect.EFFICIENCY],
        styles=[ReasoningStyle.SYSTEMATIC],
        domain="algorithms",
    )

    # Debugging
    store.add(
        problem_pattern="debug",
        keywords=["debug", "fix", "error", "bug", "issue"],
        operations=[CognitiveOperation.ANALYZE, CognitiveOperation.VERIFY],
        focuses=[FocusAspect.ERRORS, FocusAspect.ASSUMPTIONS],
        styles=[ReasoningStyle.CRITICAL, ReasoningStyle.SYSTEMATIC],
        domain="debugging",
    )

    # Optimization
    store.add(
        problem_pattern="optimize",
        keywords=["optimize", "improve", "faster", "efficiency"],
        operations=[CognitiveOperation.EVALUATE, CognitiveOperation.REFINE],
        focuses=[FocusAspect.EFFICIENCY, FocusAspect.ALTERNATIVES],
        styles=[ReasoningStyle.CRITICAL, ReasoningStyle.CREATIVE],
        domain="optimization",
    )

    return store


def get_logic_compositional_rag() -> CompositionalRAGStore:
    """Get predefined compositional RAG for logic problems."""
    store = CompositionalRAGStore()

    # Logical reasoning
    store.add(
        problem_pattern="logical reasoning",
        keywords=["if", "then", "therefore", "premise", "conclusion"],
        operations=[CognitiveOperation.ANALYZE, CognitiveOperation.VERIFY],
        focuses=[FocusAspect.ASSUMPTIONS, FocusAspect.CORRECTNESS],
        styles=[ReasoningStyle.FORMAL, ReasoningStyle.SYSTEMATIC],
        domain="logic",
    )

    # Puzzles
    store.add(
        problem_pattern="puzzle",
        keywords=["puzzle", "riddle", "constraint", "must"],
        operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.GENERATE],
        focuses=[FocusAspect.CONSTRAINTS, FocusAspect.ALTERNATIVES],
        styles=[ReasoningStyle.CREATIVE, ReasoningStyle.SYSTEMATIC],
        domain="puzzles",
    )

    return store


__all__ = [
    "CompositionalGuidance",
    "CompositionalRAGStore",
    "SolutionRAGStore",
    "get_math_compositional_rag",
    "get_coding_compositional_rag",
    "get_logic_compositional_rag",
]
