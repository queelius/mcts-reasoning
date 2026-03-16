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
    MarkerTerminalDetector,
    BoxedTerminalDetector,
    MultiMarkerTerminalDetector,
)

# Generator interface and implementations
from .generator import (
    Generator,
    LLMGenerator,
    MockGenerator,
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
)

# Prompt strategies
from .prompt import (
    PromptStrategy,
    StepByStepPrompt,
    FewShotPrompt,
    Example,
    ExampleSource,
    StaticExampleSource,
)

# Sampling strategies
from .sampling import (
    PathSampler,
    SamplingStrategy,
    ValueSampling,
    VisitSampling,
    DiverseSampling,
    TopKSampling,
)

# Consensus strategies
from .consensus import (
    ConsensusStrategy,
    MajorityVote,
    WeightedVote,
)

# Providers
from .providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    detect_provider,
    get_provider,
)

# Core types
from .types import (
    State,
    Message,
    Continuation,
    Evaluation,
    TerminalCheck,
    SampledPath,
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
    # Prompt
    "PromptStrategy",
    "StepByStepPrompt",
    "FewShotPrompt",
    "Example",
    "ExampleSource",
    "StaticExampleSource",
    # Sampling
    "PathSampler",
    "SamplingStrategy",
    "ValueSampling",
    "VisitSampling",
    "DiverseSampling",
    "TopKSampling",
    # Consensus
    "ConsensusStrategy",
    "MajorityVote",
    "WeightedVote",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "detect_provider",
    "get_provider",
    # Types
    "State",
    "Message",
    "Continuation",
    "SampledPath",
    "SearchState",
    "ConsensusResult",
    "extend_state",
]
