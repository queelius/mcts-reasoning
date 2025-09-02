"""
MCTS-Reasoning: Monte Carlo Tree Search for LLM-based reasoning

A clean implementation of MCTS with compositional actions for systematic reasoning.
"""

__version__ = "0.1.0"

# Core MCTS
from .core import MCTS, MCTSNode

# Reasoning-specific MCTS
from .reasoning import ReasoningMCTS

# Sampling strategies
from .sampling import MCTSSampler, SampledPath, SamplingMCTS

# LLM adapters
from .llm_adapters import (
    LLMAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    MockLLMAdapter,
    get_llm,
)

# IPC support (if available)
try:
    from .mcts_with_ipc import MCTSWithIPC, ReasoningMCTSWithIPC
    _has_ipc = True
except ImportError:
    _has_ipc = False
    MCTSWithIPC = None
    ReasoningMCTSWithIPC = None

__all__ = [
    # Core
    "MCTS",
    "MCTSNode",
    "ReasoningMCTS",
    
    # Sampling
    "MCTSSampler",
    "SampledPath",
    "SamplingMCTS",
    
    # LLM Adapters
    "LLMAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "MockLLMAdapter",
    "get_llm",
]

# Add IPC classes if available
if _has_ipc:
    __all__.extend(["MCTSWithIPC", "ReasoningMCTSWithIPC"])