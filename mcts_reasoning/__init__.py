"""Composable Monte Carlo Tree Search for LLM reasoning."""

__version__ = "0.6.0"

from .types import (
    Message,
    State,
    extend_state,
    SearchState,
    Continuation,
    Evaluation,
    TerminalCheck,
    SampledPath,
    ConsensusResult,
)
from .node import Node
from .mcts import MCTS
from .generator import Generator, LLMGenerator
from .evaluator import (
    Evaluator,
    GroundTruthEvaluator,
    NumericEvaluator,
    LLMEvaluator,
    ProcessEvaluator,
    CompositeEvaluator,
)
from .terminal import (
    MarkerTerminalDetector,
    BoxedTerminalDetector,
    MultiMarkerTerminalDetector,
)
from .prompt import (
    PromptStrategy,
    StepByStepPrompt,
    FewShotPrompt,
    ExampleSource,
    Example,
    StaticExampleSource,
)
from .sampling import (
    SamplingStrategy,
    ValueSampling,
    VisitSampling,
    DiverseSampling,
    TopKSampling,
    PathSampler,
)
from .consensus import ConsensusStrategy, MajorityVote, WeightedVote
from .providers import get_provider, detect_provider

__all__ = [
    # Version
    "__version__",
    # Types
    "Message",
    "State",
    "extend_state",
    "SearchState",
    "Continuation",
    "Evaluation",
    "TerminalCheck",
    "SampledPath",
    "ConsensusResult",
    # Core MCTS
    "Node",
    "MCTS",
    # Generator
    "Generator",
    "LLMGenerator",
    # Evaluator
    "Evaluator",
    "GroundTruthEvaluator",
    "NumericEvaluator",
    "LLMEvaluator",
    "ProcessEvaluator",
    "CompositeEvaluator",
    # Terminal Detection
    "MarkerTerminalDetector",
    "BoxedTerminalDetector",
    "MultiMarkerTerminalDetector",
    # Prompt
    "PromptStrategy",
    "StepByStepPrompt",
    "FewShotPrompt",
    "ExampleSource",
    "Example",
    "StaticExampleSource",
    # Sampling
    "SamplingStrategy",
    "ValueSampling",
    "VisitSampling",
    "DiverseSampling",
    "TopKSampling",
    "PathSampler",
    # Consensus
    "ConsensusStrategy",
    "MajorityVote",
    "WeightedVote",
    # Providers
    "get_provider",
    "detect_provider",
]
