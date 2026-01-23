"""
Tool registry for MCP integration.

Manages tool discovery and registration from MCP servers.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .formats import ToolDefinition, ToolFormat

logger = logging.getLogger(__name__)


@dataclass
class MCPToolRegistry:
    """
    Registry for tools from MCP servers.

    Aggregates tools from multiple MCP servers and provides:
    - Tool lookup by name
    - Tool schema formatting for prompts
    - Tool routing to appropriate servers
    """

    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    tool_servers: Dict[str, str] = field(default_factory=dict)  # tool_name -> server_name
    _executors: Dict[str, Callable] = field(default_factory=dict)  # server_name -> executor

    def register_tool(
        self,
        tool: ToolDefinition,
        server_name: str,
        executor: Optional[Callable] = None,
    ) -> None:
        """
        Register a tool from an MCP server.

        Args:
            tool: Tool definition
            server_name: Name of the MCP server providing this tool
            executor: Optional callable to execute this tool
        """
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")

        self.tools[tool.name] = tool
        self.tool_servers[tool.name] = server_name

        if executor:
            self._executors[server_name] = executor

    def register_tools(
        self,
        tools: List[ToolDefinition],
        server_name: str,
        executor: Optional[Callable] = None,
    ) -> None:
        """Register multiple tools from a server."""
        for tool in tools:
            self.register_tool(tool, server_name, executor)

    def unregister_server(self, server_name: str) -> None:
        """Remove all tools from a server."""
        tools_to_remove = [
            name for name, server in self.tool_servers.items()
            if server == server_name
        ]
        for name in tools_to_remove:
            del self.tools[name]
            del self.tool_servers[name]

        if server_name in self._executors:
            del self._executors[server_name]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self.tools.get(name)

    def get_server(self, tool_name: str) -> Optional[str]:
        """Get the server name for a tool."""
        return self.tool_servers.get(tool_name)

    def get_executor(self, tool_name: str) -> Optional[Callable]:
        """Get the executor for a tool."""
        server_name = self.tool_servers.get(tool_name)
        if server_name:
            return self._executors.get(server_name)
        return None

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def list_servers(self) -> List[str]:
        """List all server names."""
        return list(set(self.tool_servers.values()))

    def format_tools_prompt(
        self,
        fmt: ToolFormat = ToolFormat.XML,
        tool_names: Optional[List[str]] = None,
    ) -> str:
        """
        Format tools for inclusion in LLM prompt.

        Args:
            fmt: Output format
            tool_names: Specific tools to include (None = all)

        Returns:
            Formatted string for prompt injection
        """
        tools = (
            [self.tools[name] for name in tool_names if name in self.tools]
            if tool_names
            else list(self.tools.values())
        )

        if not tools:
            return ""

        if fmt == ToolFormat.XML:
            return self._format_xml(tools)
        elif fmt == ToolFormat.JSON:
            return self._format_json(tools)
        else:
            # Function format - return JSON for native handling
            return self._format_json(tools)

    def _format_xml(self, tools: List[ToolDefinition]) -> str:
        """Format tools as XML."""
        parts = ["<available_tools>"]
        for tool in tools:
            parts.append(tool.to_xml_schema())
        parts.append("</available_tools>")
        return "\n".join(parts)

    def _format_json(self, tools: List[ToolDefinition]) -> str:
        """Format tools as JSON."""
        import json
        return json.dumps(
            {"tools": [tool.to_json_schema() for tool in tools]},
            indent=2
        )

    def get_tool_use_instruction(self, fmt: ToolFormat = ToolFormat.XML) -> str:
        """Get instruction text for how to use tools."""
        if fmt == ToolFormat.XML:
            return """To use a tool, include a tool call in your response like this:
<tool_call name="tool_name">
  <param_name>value</param_name>
</tool_call>

After calling a tool, you will receive the result and can continue reasoning."""

        elif fmt == ToolFormat.JSON:
            return """To use a tool, include a tool call in your response like this:
{"tool": "tool_name", "arguments": {"param_name": "value"}}

After calling a tool, you will receive the result and can continue reasoning."""

        else:
            return "Use the available functions when needed to help solve the problem."

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self.tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self.tools


def create_tool_from_mcp(mcp_tool: Dict[str, Any]) -> ToolDefinition:
    """
    Create a ToolDefinition from an MCP tool schema.

    Args:
        mcp_tool: Tool schema from MCP server (tools/list response)

    Returns:
        ToolDefinition object
    """
    input_schema = mcp_tool.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    return ToolDefinition(
        name=mcp_tool["name"],
        description=mcp_tool.get("description", ""),
        parameters=properties,
        required=required,
    )
