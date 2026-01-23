"""
MCTS-Reasoning: Monte Carlo Tree Search for LLM-based reasoning.

A clean implementation of MCTS for systematic reasoning with LLMs.
Separates Search (MCTS), Generator (LLM), and Evaluator (Judge) concerns.
"""

__version__ = "0.5.2"

# Core MCTS components
from .node import Node
from .mcts import MCTS, SearchResult

# Actions (state-dependent operations)
from .actions import (
    Action,
    ActionResult,
    ActionSpace,
    ContinueAction,
    DefaultActionSpace,
    # Extensions
    CompressAction,
    ExtendedActionSpace,
)

# Terminal detection
from .terminal import (
    TerminalDetector,
    TerminalCheck,
    MarkerTerminalDetector,
    BoxedTerminalDetector,
    MultiMarkerTerminalDetector,
)

# Generator interface and implementations
from .generator import (
    Generator,
    LLMGenerator,
    MockGenerator,
    Continuation,
    ANSWER_MARKER,
)

# Evaluator interface and implementations
from .evaluator import (
    Evaluator,
    LLMEvaluator,
    MockEvaluator,
    GroundTruthEvaluator,
    NumericEvaluator,
    ProcessEvaluator,
    CompositeEvaluator,
    Evaluation,
)

# Sampling strategies
from .sampling import (
    PathSampler,
    SampledPath,
    SamplingStrategy,
)

# LLM provider adapter for v2
from .llm_provider import create_generator, create_evaluator

# LLM Providers (unified interface)
from .compositional.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockLLMProvider,
    get_llm,
)

# Configuration
from .config import Config, get_config

__all__ = [
    # Core MCTS
    "Node",
    "MCTS",
    "SearchResult",
    # Actions
    "Action",
    "ActionResult",
    "ActionSpace",
    "ContinueAction",
    "DefaultActionSpace",
    "CompressAction",
    "ExtendedActionSpace",
    # Terminal Detection
    "TerminalDetector",
    "TerminalCheck",
    "MarkerTerminalDetector",
    "BoxedTerminalDetector",
    "MultiMarkerTerminalDetector",
    # Generator
    "Generator",
    "LLMGenerator",
    "MockGenerator",
    "Continuation",
    "ANSWER_MARKER",
    # Evaluator
    "Evaluator",
    "LLMEvaluator",
    "MockEvaluator",
    "GroundTruthEvaluator",
    "NumericEvaluator",
    "ProcessEvaluator",
    "CompositeEvaluator",
    "Evaluation",
    # Sampling
    "PathSampler",
    "SampledPath",
    "SamplingStrategy",
    # LLM Provider Adapter
    "create_generator",
    "create_evaluator",
    # LLM Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MockLLMProvider",
    "get_llm",
    # Configuration
    "Config",
    "get_config",
]
