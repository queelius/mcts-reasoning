"""
MCP Tool Integration for MCTS Reasoning.

Provides tool-augmented reasoning through MCP (Model Context Protocol) servers.

Quick Start:
    # Without tools (unchanged API)
    mcts = MCTS(generator=gen, evaluator=eval)
    result = mcts.search("What is 2+2?", simulations=20)

    # With tools (explicit opt-in)
    tool_context = ToolContext.from_servers({
        "rag": {"command": ["python", "-m", "mcts_reasoning.tools.rag_server"]},
    })
    tool_gen = ToolAwareGenerator(base_generator=gen, tool_context=tool_context)
    mcts = MCTS(generator=tool_gen, evaluator=eval)
    result = mcts.search("Solve x^2 + 5x + 6 = 0", simulations=20)

Testing:
    # Mock context for testing
    context = ToolContext.mock({
        "calculator": {"description": "Calculate", "response": "42"},
    })
"""

# Formats
from .formats import (
    ToolFormat,
    ToolDefinition,
    ToolCall,
    ToolResult,
)

# Execution
from .execution import (
    ToolExecutionResult,
    ToolCallParser,
    ToolCallHandler,
)

# Registry
from .registry import (
    MCPToolRegistry,
    create_tool_from_mcp,
)

# Client
from .client import (
    ServerConfig,
    MCPClientManager,
    MockMCPClientManager,
)

# Context (main entry point)
from .context import ToolContext

# Generator wrapper
from .generator import (
    ToolAwareGenerator,
    wrap_generator_with_tools,
    create_native_tool_generator,
)

# Native function calling
from .native import (
    NativeFunctionCallProvider,
    OpenAINativeWrapper,
    AnthropicNativeWrapper,
    wrap_provider_for_native_tools,
    supports_native_function_calling,
)


# RAG Server (lazy import to avoid requiring mcp)
def create_rag_server(*args, **kwargs):
    """Create an MCP server exposing RAG functionality."""
    from .rag_server import create_rag_server as _create

    return _create(*args, **kwargs)


__all__ = [
    # Formats
    "ToolFormat",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    # Execution
    "ToolExecutionResult",
    "ToolCallParser",
    "ToolCallHandler",
    # Registry
    "MCPToolRegistry",
    "create_tool_from_mcp",
    # Client
    "ServerConfig",
    "MCPClientManager",
    "MockMCPClientManager",
    # Context
    "ToolContext",
    # Generator
    "ToolAwareGenerator",
    "wrap_generator_with_tools",
    "create_native_tool_generator",
    # Native function calling
    "NativeFunctionCallProvider",
    "OpenAINativeWrapper",
    "AnthropicNativeWrapper",
    "wrap_provider_for_native_tools",
    "supports_native_function_calling",
    # RAG Server
    "create_rag_server",
]
