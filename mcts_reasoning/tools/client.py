"""
MCP client manager for connecting to MCP servers.

Uses the official MCP SDK for communication with MCP servers via stdio.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

from .formats import ToolDefinition, ToolResult
from .registry import MCPToolRegistry, create_tool_from_mcp

logger = logging.getLogger(__name__)

# Try to import MCP SDK
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: List[str]  # Command to start the server
    env: Dict[str, str] = field(default_factory=dict)
    args: List[str] = field(default_factory=list)


class MCPClientManager:
    """
    Manages connections to multiple MCP servers.

    Provides:
    - Server lifecycle management (start/stop)
    - Tool discovery and registration
    - Tool call execution routing

    Note: Uses AsyncExitStack to properly manage context manager lifecycles
    for MCP server connections.
    """

    def __init__(self):
        """Initialize the client manager."""
        self.servers: Dict[str, ServerConfig] = {}
        self.sessions: Dict[str, Any] = {}  # server_name -> ClientSession
        self.registry = MCPToolRegistry()
        self._running = False
        self._exit_stack: Optional[AsyncExitStack] = None

    def add_server(self, config: ServerConfig) -> None:
        """
        Add a server configuration.

        Args:
            config: Server configuration
        """
        self.servers[config.name] = config
        logger.info(f"Added MCP server config: {config.name}")

    def add_servers(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Add multiple server configurations.

        Args:
            configs: Dict of server_name -> config dict
                     Config dict should have 'command' key (list of strings)
        """
        for name, cfg in configs.items():
            self.add_server(
                ServerConfig(
                    name=name,
                    command=cfg.get("command", []),
                    env=cfg.get("env", {}),
                    args=cfg.get("args", []),
                )
            )

    async def start(self) -> None:
        """Start all configured servers and discover tools."""
        if not HAS_MCP:
            raise ImportError("MCP SDK not installed. Install with: pip install mcp")

        # Create exit stack to manage context manager lifecycles
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for name, config in self.servers.items():
            await self._start_server(name, config)

        self._running = True
        logger.info(f"Started {len(self.sessions)} MCP servers")

    async def _start_server(self, name: str, config: ServerConfig) -> None:
        """Start a single MCP server."""
        try:
            server_params = StdioServerParameters(
                command=config.command[0],
                args=config.command[1:] + config.args,
                env=config.env or None,
            )

            # Enter the stdio_client context and keep it alive via exit_stack
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            # Enter the ClientSession context and keep it alive
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # Initialize the session
            await session.initialize()

            # Store the session
            self.sessions[name] = session

            # Discover tools
            tools_response = await session.list_tools()
            tools = [
                create_tool_from_mcp(tool.model_dump()) for tool in tools_response.tools
            ]

            # Register tools
            self.registry.register_tools(
                tools,
                server_name=name,
                executor=self._create_executor(name),
            )

            logger.info(f"Server {name}: discovered {len(tools)} tools")

        except Exception as e:
            logger.error(f"Failed to start server {name}: {e}")
            raise

    def _create_executor(self, server_name: str) -> Callable:
        """Create an executor function for a server."""

        async def execute(tool_name: str, arguments: Dict[str, Any]) -> Any:
            session = self.sessions.get(server_name)
            if not session:
                raise RuntimeError(f"Server {server_name} not connected")

            result = await session.call_tool(tool_name, arguments)
            return result.content

        # Return sync wrapper for compatibility
        def sync_execute(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return asyncio.run(execute(tool_name, arguments))

        return sync_execute

    async def stop(self) -> None:
        """Stop all servers and clean up resources."""
        # Close all context managers via exit stack
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None

        self.sessions.clear()
        self._running = False
        logger.info("Stopped all MCP servers")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """
        Call a tool by name.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ToolResult with the result or error
        """
        server_name = self.registry.get_server(tool_name)
        if not server_name:
            return ToolResult(
                call_id=None,
                name=tool_name,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        session = self.sessions.get(server_name)
        if not session:
            return ToolResult(
                call_id=None,
                name=tool_name,
                result=None,
                error=f"Server '{server_name}' not connected",
            )

        try:
            result = await session.call_tool(tool_name, arguments)
            return ToolResult(
                call_id=None,
                name=tool_name,
                result=result.content,
            )
        except Exception as e:
            logger.error(f"Tool call {tool_name} failed: {e}")
            return ToolResult(
                call_id=None,
                name=tool_name,
                result=None,
                error=str(e),
            )

    def call_tool_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Synchronous wrapper for call_tool."""
        return asyncio.run(self.call_tool(tool_name, arguments))

    def get_available_tools(self) -> List[ToolDefinition]:
        """Get all available tools."""
        return list(self.registry.tools.values())

    @property
    def is_running(self) -> bool:
        """Whether the manager is running."""
        return self._running

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self.registry)


class MockMCPClientManager(MCPClientManager):
    """
    Mock MCP client manager for testing.

    Allows registering mock tools and responses without real MCP servers.
    """

    def __init__(self):
        """Initialize mock manager."""
        super().__init__()
        self._mock_responses: Dict[str, Any] = {}
        self._call_history: List[Dict[str, Any]] = []

    def register_mock_tool(
        self,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        response: Any = None,
    ) -> None:
        """
        Register a mock tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
            response: Mock response to return
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
        )
        self.registry.register_tool(tool, server_name="mock")
        self._mock_responses[name] = response

    def set_mock_response(self, tool_name: str, response: Any) -> None:
        """Set the mock response for a tool."""
        self._mock_responses[tool_name] = response

    async def start(self) -> None:
        """Mock start - no actual server connections."""
        self._running = True
        logger.info("Mock MCP manager started")

    async def stop(self) -> None:
        """Mock stop."""
        self._running = False

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Execute mock tool call."""
        self._call_history.append(
            {
                "tool": tool_name,
                "arguments": arguments,
            }
        )

        if tool_name not in self._mock_responses:
            return ToolResult(
                call_id=None,
                name=tool_name,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        response = self._mock_responses[tool_name]

        # Support callable responses for dynamic mocking
        if callable(response):
            try:
                result = response(arguments)
            except Exception as e:
                return ToolResult(
                    call_id=None,
                    name=tool_name,
                    result=None,
                    error=str(e),
                )
        else:
            result = response

        return ToolResult(
            call_id=None,
            name=tool_name,
            result=result,
        )

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of tool calls for assertions."""
        return self._call_history.copy()

    def clear_call_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()
