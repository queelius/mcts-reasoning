"""
MCTS-Reasoning: Monte Carlo Tree Search for LLM-based reasoning

A clean implementation of MCTS with compositional actions for systematic reasoning.
Integrates advanced compositional prompting with tree search exploration.
"""

__version__ = "0.2.0"

# Core MCTS
from .core import MCTS, MCTSNode

# Reasoning-specific MCTS
from .reasoning import ReasoningMCTS

# Sampling strategies
from .sampling import MCTSSampler, SampledPath, SamplingMCTS

# Unified LLM providers
from .compositional.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockLLMProvider,
    get_llm,
)

# Compositional prompting system
from .compositional import (
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
    ComposingPrompt,
    smart_termination,
)

# Compositional actions for MCTS
from .compositional.actions import (
    CompositionalAction,
    ActionSelector,
)

# Configuration
from .config import Config, get_config

__all__ = [
    # Core
    "MCTS",
    "MCTSNode",
    "ReasoningMCTS",

    # Sampling
    "MCTSSampler",
    "SampledPath",
    "SamplingMCTS",

    # LLM Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MockLLMProvider",
    "get_llm",

    # Compositional prompting
    "CognitiveOperation",
    "FocusAspect",
    "ReasoningStyle",
    "ConnectionType",
    "OutputFormat",
    "ComposingPrompt",
    "smart_termination",

    # Compositional actions
    "CompositionalAction",
    "ActionSelector",

    # Configuration
    "Config",
    "get_config",
]

# Try to import MCP features (optional)
try:
    from .compositional import (
        MCPToolType,
        MCPTool,
        MCPToolCall,
        MCPToolResult,
        MCPClient,
        MCPLLMProvider,
        create_mcp_client,
        create_mcp_provider,
        MCPActionIntent,
        MCPCompositionalAction,
        MCPActionSelector,
        create_mcp_action,
        create_code_execution_action,
        create_research_action,
    )

    __all__.extend([
        # MCP Core
        "MCPToolType",
        "MCPTool",
        "MCPToolCall",
        "MCPToolResult",
        "MCPClient",
        "MCPLLMProvider",
        "create_mcp_client",
        "create_mcp_provider",

        # MCP Actions
        "MCPActionIntent",
        "MCPCompositionalAction",
        "MCPActionSelector",
        "create_mcp_action",
        "create_code_execution_action",
        "create_research_action",
    ])

    _has_mcp = True
except ImportError:
    _has_mcp = False