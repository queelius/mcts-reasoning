"""
Tool call parsing and execution.

Handles parsing tool calls from LLM output and executing them via MCP.
"""

import re
import json
import logging
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from .formats import ToolFormat, ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of executing one or more tool calls."""

    calls: List[ToolCall] = field(default_factory=list)
    results: List[ToolResult] = field(default_factory=list)
    remaining_text: str = ""  # Text after tool calls are removed

    @property
    def has_calls(self) -> bool:
        """Whether any tool calls were found."""
        return len(self.calls) > 0

    @property
    def all_succeeded(self) -> bool:
        """Whether all tool calls succeeded."""
        return all(r.success for r in self.results)

    def format_results(self, fmt: ToolFormat = ToolFormat.XML) -> str:
        """Format all results for injection into LLM context."""
        if fmt == ToolFormat.XML:
            return "\n".join(r.to_xml() for r in self.results)
        elif fmt == ToolFormat.JSON:
            return json.dumps([r.to_json() for r in self.results], indent=2)
        else:
            # Function format - typically handled by the provider
            return json.dumps([r.to_json() for r in self.results])


class ToolCallParser:
    """Parses tool calls from LLM output in various formats."""

    # XML pattern: <tool_call name="...">...</tool_call> or <tool name="...">...</tool>
    XML_TOOL_PATTERN = re.compile(
        r'<tool(?:_call|_use)?\s+name=["\']([^"\']+)["\']>(.*?)</tool(?:_call|_use)?>',
        re.DOTALL | re.IGNORECASE,
    )

    # Alternative XML: <function_call name="..."> or <use_tool name="...">
    XML_ALT_PATTERN = re.compile(
        r'<(?:function_call|use_tool|invoke)\s+name=["\']([^"\']+)["\']>(.*?)</(?:function_call|use_tool|invoke)>',
        re.DOTALL | re.IGNORECASE,
    )

    # XML parameter pattern: <param_name>value</param_name>
    XML_PARAM_PATTERN = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>", re.DOTALL)

    # JSON pattern: {"tool": "...", "arguments": {...}} or {"name": "...", "input": {...}}
    JSON_TOOL_PATTERN = re.compile(
        r'\{[^{}]*"(?:tool|name)"\s*:\s*"([^"]+)"[^{}]*"(?:arguments|input|parameters)"\s*:\s*(\{[^{}]*\})[^{}]*\}',
        re.DOTALL,
    )

    # Code block pattern for extracting JSON from markdown
    CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    def parse(self, text: str, fmt: ToolFormat = ToolFormat.XML) -> List[ToolCall]:
        """
        Parse tool calls from text.

        Args:
            text: LLM output text
            fmt: Expected format of tool calls

        Returns:
            List of parsed ToolCall objects
        """
        if fmt == ToolFormat.XML:
            return self._parse_xml(text)
        elif fmt == ToolFormat.JSON:
            return self._parse_json(text)
        else:
            # For function format, calls come from the API response, not text
            return []

    def _parse_xml(self, text: str) -> List[ToolCall]:
        """Parse XML-formatted tool calls."""
        calls = []

        # Try both XML patterns
        for pattern in [self.XML_TOOL_PATTERN, self.XML_ALT_PATTERN]:
            for match in pattern.finditer(text):
                tool_name = match.group(1)
                content = match.group(2)

                # Parse parameters from content
                arguments = self._parse_xml_params(content)
                calls.append(ToolCall(name=tool_name, arguments=arguments))

        return calls

    def _parse_xml_params(self, content: str) -> Dict[str, Any]:
        """Parse parameters from XML content."""
        arguments = {}

        for param_match in self.XML_PARAM_PATTERN.finditer(content):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            # Try to parse as JSON for complex types
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                arguments[param_name] = param_value

        # If no parameters found, try parsing content as JSON directly
        if not arguments and content.strip():
            try:
                parsed = json.loads(content.strip())
                if isinstance(parsed, dict):
                    arguments = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        return arguments

    def _parse_json(self, text: str) -> List[ToolCall]:
        """Parse JSON-formatted tool calls."""
        calls = []

        # Try to find JSON objects in the text
        for match in self.JSON_TOOL_PATTERN.finditer(text):
            tool_name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
            except json.JSONDecodeError:
                arguments = {}

            calls.append(ToolCall(name=tool_name, arguments=arguments))

        # Try to extract JSON from code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                call = self._extract_tool_call_from_json(data)
                if call:
                    calls.append(call)
            except json.JSONDecodeError:
                pass

        # Also try to parse the entire text as JSON (for cleaner responses)
        try:
            data = json.loads(text)
            call = self._extract_tool_call_from_json(data)
            if call:
                calls.append(call)
            elif isinstance(data, list):
                for item in data:
                    call = self._extract_tool_call_from_json(item)
                    if call:
                        calls.append(call)
        except json.JSONDecodeError:
            pass

        return calls

    def _extract_tool_call_from_json(self, data: Any) -> Optional[ToolCall]:
        """Extract a ToolCall from a JSON object if it matches expected format."""
        if not isinstance(data, dict):
            return None

        # Support multiple JSON formats
        # Format 1: {"tool": "name", "arguments": {...}}
        # Format 2: {"name": "name", "input": {...}}
        # Format 3: {"function": "name", "parameters": {...}}

        name = data.get("tool") or data.get("name") or data.get("function")
        if not name:
            return None

        arguments = (
            data.get("arguments") or data.get("input") or data.get("parameters") or {}
        )

        return ToolCall(name=name, arguments=arguments)

    def remove_tool_calls(self, text: str, fmt: ToolFormat = ToolFormat.XML) -> str:
        """Remove tool call markup from text, returning the remaining content."""
        if fmt == ToolFormat.XML:
            text = self.XML_TOOL_PATTERN.sub("", text)
            text = self.XML_ALT_PATTERN.sub("", text)
        elif fmt == ToolFormat.JSON:
            text = self.JSON_TOOL_PATTERN.sub("", text)
            text = self.CODE_BLOCK_PATTERN.sub("", text)
        return text.strip()


# Type for tool executor functions
ToolExecutor = Callable[[str, Dict[str, Any]], Any]


class ToolCallHandler:
    """Handles parsing and executing tool calls."""

    def __init__(
        self,
        executor: Optional[ToolExecutor] = None,
        fmt: ToolFormat = ToolFormat.XML,
        max_iterations: int = 3,
    ):
        """
        Initialize tool call handler.

        Args:
            executor: Function to execute tool calls: (name, args) -> result
            fmt: Format for tool calls
            max_iterations: Max tool call iterations per generation
        """
        self.executor = executor
        self.fmt = fmt
        self.max_iterations = max_iterations
        self.parser = ToolCallParser()

    def process(self, text: str) -> ToolExecutionResult:
        """
        Parse and execute tool calls from LLM output.

        Args:
            text: LLM output text

        Returns:
            ToolExecutionResult with calls, results, and remaining text
        """
        calls = self.parser.parse(text, self.fmt)
        remaining = self.parser.remove_tool_calls(text, self.fmt)

        results = []
        for call in calls:
            result = self._execute_call(call)
            results.append(result)

        return ToolExecutionResult(
            calls=calls,
            results=results,
            remaining_text=remaining,
        )

    def _execute_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        if self.executor is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result=None,
                error="No executor configured",
            )

        try:
            result = self.executor(call.name, call.arguments)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result=result,
            )
        except Exception as e:
            logger.warning(f"Tool call {call.name} failed: {e}")
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result=None,
                error=str(e),
            )
