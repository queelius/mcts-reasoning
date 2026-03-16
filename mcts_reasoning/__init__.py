"""
MCTS-Reasoning: Monte Carlo Tree Search for LLM-based reasoning.

A clean implementation of MCTS for systematic reasoning with LLMs.
Separates Search (MCTS), Generator (LLM), and Evaluator (Judge) concerns.
"""

__version__ = "0.6.0-dev"

# Core MCTS components
from .node import Node
from .mcts import MCTS, SearchResult

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

# Core types
from .types import (
    State,
    Message,
    SearchState,
    ConsensusResult,
    extend_state,
)

__all__ = [
    # Core MCTS
    "Node",
    "MCTS",
    "SearchResult",
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
    # Types
    "State",
    "Message",
    "SearchState",
    "ConsensusResult",
    "extend_state",
]
