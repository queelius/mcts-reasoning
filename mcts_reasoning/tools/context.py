"""
ToolContext: High-level interface for tool integration with MCTS.

Provides prompt injection, tool call parsing, and execution in a unified API.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .formats import ToolFormat, ToolDefinition
from .execution import ToolCallHandler, ToolExecutionResult
from .client import MCPClientManager, MockMCPClientManager

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """
    Context for tool-augmented reasoning.

    Manages MCP server connections and provides methods for:
    - Injecting tool definitions into prompts
    - Parsing tool calls from LLM output
    - Executing tool calls and formatting results

    This is the main entry point for tool integration with MCTS.
    """

    mcp_manager: MCPClientManager = field(default_factory=MCPClientManager)
    fmt: ToolFormat = ToolFormat.XML
    max_tool_iterations: int = 3

    # Internal state
    _handler: Optional[ToolCallHandler] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the tool call handler."""
        self._handler = ToolCallHandler(
            executor=self._execute_tool,
            fmt=self.fmt,
            max_iterations=self.max_tool_iterations,
        )

    @classmethod
    def from_servers(
        cls,
        servers: Dict[str, Dict[str, Any]],
        fmt: ToolFormat = ToolFormat.XML,
        max_tool_iterations: int = 3,
    ) -> "ToolContext":
        """
        Create a ToolContext from server configurations.

        Args:
            servers: Dict of server_name -> config
                     Config should have 'command' key (list of strings)
            fmt: Tool format for prompts/responses
            max_tool_iterations: Max iterations for tool calls

        Returns:
            Configured ToolContext

        Example:
            context = ToolContext.from_servers({
                "rag": {"command": ["python", "-m", "mcts_reasoning.tools.rag_server"]},
                "calc": {"command": ["npx", "@anthropic/calculator-server"]},
            })
        """
        manager = MCPClientManager()
        manager.add_servers(servers)

        return cls(
            mcp_manager=manager,
            fmt=fmt,
            max_tool_iterations=max_tool_iterations,
        )

    @classmethod
    def mock(
        cls,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        fmt: ToolFormat = ToolFormat.XML,
    ) -> "ToolContext":
        """
        Create a mock ToolContext for testing.

        Args:
            tools: Dict of tool_name -> {"description": ..., "response": ...}
            fmt: Tool format

        Returns:
            Mock ToolContext
        """
        manager = MockMCPClientManager()

        if tools:
            for name, config in tools.items():
                manager.register_mock_tool(
                    name=name,
                    description=config.get("description", ""),
                    parameters=config.get("parameters", {}),
                    response=config.get("response"),
                )

        context = cls(mcp_manager=manager, fmt=fmt)
        return context

    async def start(self) -> None:
        """Start all MCP servers."""
        await self.mcp_manager.start()

    async def stop(self) -> None:
        """Stop all MCP servers."""
        await self.mcp_manager.stop()

    def start_sync(self) -> None:
        """Synchronous start."""
        asyncio.run(self.start())

    def stop_sync(self) -> None:
        """Synchronous stop."""
        asyncio.run(self.stop())

    def get_available_tools(self) -> List[ToolDefinition]:
        """Get all available tools."""
        return self.mcp_manager.get_available_tools()

    def inject_tools_into_prompt(
        self,
        prompt: str,
        tool_names: Optional[List[str]] = None,
    ) -> str:
        """
        Inject tool definitions into a prompt.

        Args:
            prompt: Base prompt
            tool_names: Specific tools to include (None = all)

        Returns:
            Prompt with tool definitions injected
        """
        tools_prompt = self.mcp_manager.registry.format_tools_prompt(
            fmt=self.fmt,
            tool_names=tool_names,
        )

        if not tools_prompt:
            return prompt

        instruction = self.mcp_manager.registry.get_tool_use_instruction(self.fmt)

        return f"""{prompt}

{tools_prompt}

{instruction}"""

    def process_response(self, response: str) -> ToolExecutionResult:
        """
        Parse and execute any tool calls in the response.

        Args:
            response: LLM response text

        Returns:
            ToolExecutionResult with calls, results, and remaining text
        """
        return self._handler.process(response)

    def execute_tool_calls(self, response: str) -> ToolExecutionResult:
        """Alias for process_response."""
        return self.process_response(response)

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call synchronously."""
        result = self.mcp_manager.call_tool_sync(name, arguments)
        if result.error:
            raise RuntimeError(result.error)
        return result.result

    def format_tool_results(self, result: ToolExecutionResult) -> str:
        """Format tool execution results for context injection."""
        return result.format_results(self.fmt)

    def augment_state_with_results(
        self,
        state: str,
        result: ToolExecutionResult,
    ) -> str:
        """
        Augment a reasoning state with tool results.

        Args:
            state: Current reasoning state
            result: Tool execution result

        Returns:
            State with tool results appended
        """
        if not result.has_calls:
            return state

        formatted = self.format_tool_results(result)
        remaining = result.remaining_text

        # Build augmented state
        parts = [state]

        if remaining:
            parts.append(remaining)

        if formatted:
            parts.append(f"\n[Tool Results]\n{formatted}")

        return "\n\n".join(parts)

    @property
    def has_tools(self) -> bool:
        """Whether any tools are available."""
        return len(self.mcp_manager) > 0

    def __len__(self) -> int:
        """Number of available tools."""
        return len(self.mcp_manager)

    def __enter__(self) -> "ToolContext":
        """Context manager entry."""
        self.start_sync()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop_sync()

    async def __aenter__(self) -> "ToolContext":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
