"""
MCP (Model Context Protocol) Integration for MCTS Reasoning

Provides MCP client functionality and tool-aware LLM providers that can
automatically invoke MCP tools during reasoning.

This enables the LLM to:
- Execute Python code and see results
- Search the web
- Read/write files
- Fetch URLs
- Use any MCP-compatible tool

Tool results are automatically incorporated into the reasoning context.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from .providers import LLMProvider

logger = logging.getLogger(__name__)


# ========== MCP Tool Types ==========

class MCPToolType(Enum):
    """Common MCP tool types."""
    PYTHON_EVAL = "python_eval"
    WEB_SEARCH = "web_search"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    FETCH_URL = "fetch_url"
    BASH_COMMAND = "bash_command"
    DATABASE_QUERY = "database_query"
    CUSTOM = "custom"


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    tool_type: MCPToolType
    parameters_schema: Dict[str, Any]
    server_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "parameters": self.parameters_schema
        }


@dataclass
class MCPToolCall:
    """Represents a tool call made by the LLM."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class MCPToolResult:
    """Result from executing an MCP tool."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None

    def to_text(self) -> str:
        """Format result as text for context."""
        if self.success:
            return f"Tool '{self.tool_name}' result:\n{self.result}"
        else:
            return f"Tool '{self.tool_name}' error: {self.error}"


# ========== MCP Client ==========

class MCPClient:
    """
    MCP client for connecting to and using MCP servers.

    This is a simplified implementation. In production, you'd use
    the official MCP SDK or a more robust client.
    """

    def __init__(self):
        """Initialize MCP client."""
        self.servers: Dict[str, Any] = {}
        self.tools: Dict[str, MCPTool] = {}
        self._tool_handlers: Dict[str, Callable] = {}

    def connect_server(self, server_name: str, server_config: Dict[str, Any]):
        """
        Connect to an MCP server.

        Args:
            server_name: Name of the server
            server_config: Configuration for the server
        """
        logger.info(f"Connecting to MCP server: {server_name}")
        self.servers[server_name] = server_config

        # In a real implementation, this would establish the MCP connection
        # and discover available tools

        # For now, we'll manually register tools based on server type
        self._discover_tools(server_name, server_config)

    def _discover_tools(self, server_name: str, config: Dict[str, Any]):
        """Discover tools from a server."""
        server_type = config.get("type", "custom")

        if server_type == "python":
            self.register_tool(MCPTool(
                name="execute_python",
                description="Execute Python code and return the result. Useful for calculations, data processing, testing algorithms.",
                tool_type=MCPToolType.PYTHON_EVAL,
                parameters_schema={
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30}
                },
                server_name=server_name
            ))
        elif server_type == "web":
            self.register_tool(MCPTool(
                name="search_web",
                description="Search the web for information. Returns relevant results.",
                tool_type=MCPToolType.WEB_SEARCH,
                parameters_schema={
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "number", "description": "Number of results", "default": 5}
                },
                server_name=server_name
            ))
        elif server_type == "filesystem":
            self.register_tool(MCPTool(
                name="read_file",
                description="Read contents of a file.",
                tool_type=MCPToolType.READ_FILE,
                parameters_schema={
                    "path": {"type": "string", "description": "File path to read"}
                },
                server_name=server_name
            ))
            self.register_tool(MCPTool(
                name="write_file",
                description="Write content to a file.",
                tool_type=MCPToolType.WRITE_FILE,
                parameters_schema={
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                server_name=server_name
            ))

    def register_tool(self, tool: MCPTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")

    def register_tool_handler(self, tool_name: str, handler: Callable):
        """
        Register a custom handler for a tool.

        Args:
            tool_name: Name of the tool
            handler: Function that takes (arguments: Dict) -> Any
        """
        self._tool_handlers[tool_name] = handler

    def execute_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            Tool result
        """
        tool_name = tool_call.tool_name

        if tool_name not in self.tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )

        tool = self.tools[tool_name]

        try:
            # Check if we have a custom handler
            if tool_name in self._tool_handlers:
                result = self._tool_handlers[tool_name](tool_call.arguments)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=result
                )

            # Otherwise, use default execution based on tool type
            result = self._execute_default(tool, tool_call.arguments)
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result
            )

        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )

    def _execute_default(self, tool: MCPTool, arguments: Dict[str, Any]) -> Any:
        """Default execution for tools without custom handlers."""
        # This is a placeholder - in production, this would use the MCP protocol
        # to send the tool call to the appropriate server

        if tool.tool_type == MCPToolType.PYTHON_EVAL:
            # For demo purposes, return a mock result
            code = arguments.get("code", "")
            return f"[Would execute Python code: {code[:100]}...]"

        elif tool.tool_type == MCPToolType.WEB_SEARCH:
            query = arguments.get("query", "")
            return f"[Would search web for: {query}]"

        elif tool.tool_type == MCPToolType.READ_FILE:
            path = arguments.get("path", "")
            return f"[Would read file: {path}]"

        else:
            return f"[Would execute {tool.name} with {arguments}]"

    def get_tools_description(self) -> str:
        """Get description of all available tools for LLM context."""
        if not self.tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool_name, tool in self.tools.items():
            lines.append(f"\n{tool_name}:")
            lines.append(f"  Description: {tool.description}")
            lines.append(f"  Parameters: {json.dumps(tool.parameters_schema, indent=2)}")

        return "\n".join(lines)

    def parse_tool_calls(self, text: str) -> List[MCPToolCall]:
        """
        Parse tool calls from LLM response.

        Looks for patterns like:
        <tool_call>
        {
          "tool": "execute_python",
          "arguments": {"code": "print(2 + 2)"}
        }
        </tool_call>

        Or function call format:
        execute_python(code="print(2 + 2)")

        Args:
            text: LLM response text

        Returns:
            List of parsed tool calls
        """
        tool_calls = []

        # Try XML-style tool calls
        import re
        xml_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        matches = re.finditer(xml_pattern, text, re.DOTALL)

        for match in matches:
            try:
                call_json = json.loads(match.group(1))
                tool_calls.append(MCPToolCall(
                    tool_name=call_json.get("tool"),
                    arguments=call_json.get("arguments", {}),
                    call_id=call_json.get("id")
                ))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call JSON: {match.group(1)}")

        # Try function call style (simplified)
        for tool_name in self.tools.keys():
            pattern = rf'{tool_name}\((.*?)\)'
            matches = re.finditer(pattern, text)

            for match in matches:
                # Simple parsing of keyword arguments
                args_str = match.group(1)
                try:
                    # This is very simplified - in production use proper parsing
                    arguments = {}
                    if "=" in args_str:
                        for pair in args_str.split(","):
                            if "=" in pair:
                                key, value = pair.split("=", 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')
                                arguments[key] = value

                    tool_calls.append(MCPToolCall(
                        tool_name=tool_name,
                        arguments=arguments
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse function call: {e}")

        return tool_calls


# ========== MCP-Aware LLM Provider ==========

class MCPLLMProvider(LLMProvider):
    """
    LLM provider with automatic MCP tool access.

    Wraps a base LLM provider and adds MCP tool capabilities:
    - Injects tool descriptions into prompts
    - Parses tool calls from LLM responses
    - Executes tools via MCP client
    - Appends results to context
    - Continues generation if tools were called
    """

    def __init__(self, base_provider: LLMProvider, mcp_client: MCPClient,
                 max_tool_iterations: int = 3):
        """
        Initialize MCP-aware provider.

        Args:
            base_provider: Underlying LLM provider
            mcp_client: MCP client for tool execution
            max_tool_iterations: Maximum number of tool call iterations
        """
        self.base_provider = base_provider
        self.mcp_client = mcp_client
        self.max_tool_iterations = max_tool_iterations

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate with automatic tool execution.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Generated text (including tool results)
        """
        # Build prompt with tool descriptions
        tools_desc = self.mcp_client.get_tools_description()

        if tools_desc and "No tools available" not in tools_desc:
            enhanced_prompt = f"""{prompt}

{tools_desc}

To use a tool, output:
<tool_call>
{{
  "tool": "tool_name",
  "arguments": {{"arg": "value"}}
}}
</tool_call>

You can use tools to gather information, execute code, etc.
"""
        else:
            enhanced_prompt = prompt

        # Generate initial response
        response = self.base_provider.generate(
            enhanced_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Parse and execute tool calls (with iteration limit)
        full_response = response
        iteration = 0

        while iteration < self.max_tool_iterations:
            tool_calls = self.mcp_client.parse_tool_calls(response)

            if not tool_calls:
                break  # No more tool calls

            # Execute tools and collect results
            tool_results = []
            for tool_call in tool_calls:
                result = self.mcp_client.execute_tool(tool_call)
                tool_results.append(result)

            # Append results to context
            results_text = "\n\n".join(r.to_text() for r in tool_results)
            full_response += f"\n\n{results_text}"

            # Continue generation with results
            continuation_prompt = f"""{full_response}

Continue your reasoning with the tool results above.
"""

            response = self.base_provider.generate(
                continuation_prompt,
                max_tokens=max_tokens // 2,  # Fewer tokens for continuation
                temperature=temperature
            )

            full_response += f"\n\n{response}"
            iteration += 1

        return full_response

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"MCP({self.base_provider.get_provider_name()})"

    def is_available(self) -> bool:
        """Check if provider is available."""
        return self.base_provider.is_available()


# ========== Utility Functions ==========

def create_mcp_client(servers_config: Dict[str, Dict[str, Any]]) -> MCPClient:
    """
    Create and configure an MCP client.

    Args:
        servers_config: Dictionary mapping server names to configurations

    Returns:
        Configured MCPClient

    Example:
        client = create_mcp_client({
            "python": {"type": "python"},
            "web": {"type": "web"},
        })
    """
    client = MCPClient()

    for server_name, config in servers_config.items():
        client.connect_server(server_name, config)

    return client


def create_mcp_provider(base_provider: LLMProvider,
                       servers_config: Optional[Dict[str, Dict[str, Any]]] = None,
                       mcp_client: Optional[MCPClient] = None) -> MCPLLMProvider:
    """
    Create an MCP-aware LLM provider.

    Args:
        base_provider: Base LLM provider
        servers_config: Server configurations (if mcp_client not provided)
        mcp_client: Existing MCP client (optional)

    Returns:
        MCPLLMProvider instance

    Example:
        base_llm = get_llm("openai", model="gpt-4")
        mcp_llm = create_mcp_provider(base_llm, {
            "python": {"type": "python"},
            "web": {"type": "web"}
        })
    """
    if mcp_client is None:
        if servers_config is None:
            servers_config = {"python": {"type": "python"}}  # Default to Python
        mcp_client = create_mcp_client(servers_config)

    return MCPLLMProvider(base_provider, mcp_client)


# ========== Export All ==========

__all__ = [
    # Enums
    'MCPToolType',

    # Core classes
    'MCPTool',
    'MCPToolCall',
    'MCPToolResult',
    'MCPClient',
    'MCPLLMProvider',

    # Utilities
    'create_mcp_client',
    'create_mcp_provider',
]
