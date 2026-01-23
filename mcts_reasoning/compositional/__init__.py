"""
Compositional module: LLM providers, examples, and RAG stores.

The compositional action dimensions (enums) are defined here for use by RAG stores
and other components that need to guide reasoning strategies.
"""

from enum import Enum


# ========== Compositional Action Dimensions ==========
# These enums define the compositional action space for guided reasoning.
# Used by CompositionalRAGStore to recommend reasoning approaches.

class CognitiveOperation(Enum):
    """ω: High-level reasoning operations"""
    DECOMPOSE = "decompose"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    ABSTRACT = "abstract"
    CONCRETIZE = "concretize"
    COMPARE = "compare"
    EVALUATE = "evaluate"
    GENERATE = "generate"
    REFINE = "refine"
    FINALIZE = "finalize"


class FocusAspect(Enum):
    """φ: What aspect to focus on"""
    STRUCTURE = "structure"
    DETAILS = "details"
    ASSUMPTIONS = "assumptions"
    CONSTRAINTS = "constraints"
    GOAL = "goal"
    PROGRESS = "progress"
    ERRORS = "errors"
    ALTERNATIVES = "alternatives"
    PATTERNS = "patterns"
    SOLUTION = "solution"
    CORRECTNESS = "correctness"
    EFFICIENCY = "efficiency"
    EXAMPLES = "examples"
    RELATIONSHIPS = "relationships"


class ReasoningStyle(Enum):
    """σ: How to approach the reasoning"""
    SYSTEMATIC = "systematic"
    INTUITIVE = "intuitive"
    FORMAL = "formal"
    EXPLORATORY = "exploratory"
    CRITICAL = "critical"
    CREATIVE = "creative"


class ConnectionType(Enum):
    """κ: How to connect to previous reasoning"""
    CONTINUE = "continue"
    CONTRAST = "contrast"
    ELABORATE = "elaborate"
    SUMMARIZE = "summarize"
    PIVOT = "pivot"
    VERIFY = "verify"
    CONCLUDE = "conclude"
    QUESTION = "question"
    THEREFORE = "therefore"
    HOWEVER = "however"
    BUILDING_ON = "building_on"
    ALTERNATIVELY = "alternatively"


class OutputFormat(Enum):
    """τ: How to structure the output"""
    LIST = "list"
    STEPS = "steps"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    SOLUTION = "solution"
    CODE = "code"
    MATHEMATICAL = "mathematical"
    FREEFORM = "free-form"
    NARRATIVE = "narrative"
    TABLE = "table"


# ========== Re-exports ==========

# LLM Providers
from .providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockLLMProvider,
    get_llm,
)

# Few-shot examples
from .examples import Example, ExampleSet

# RAG stores
from .rag import (
    CompositionalGuidance,
    CompositionalRAGStore,
    SolutionRAGStore,
)

__all__ = [
    # Compositional action dimensions
    "CognitiveOperation",
    "FocusAspect",
    "ReasoningStyle",
    "ConnectionType",
    "OutputFormat",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MockLLMProvider",
    "get_llm",
    # Examples
    "Example",
    "ExampleSet",
    # RAG
    "CompositionalGuidance",
    "CompositionalRAGStore",
    "SolutionRAGStore",
]
