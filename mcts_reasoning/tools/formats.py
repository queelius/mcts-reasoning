"""
Tool format definitions for MCP integration.

Defines how tool calls are formatted in LLM prompts and how tool responses
are parsed from LLM outputs.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


class ToolFormat(Enum):
    """Format for tool calls in LLM prompts/responses."""

    XML = "xml"  # <tool_call name="..."><param>...</param></tool_call>
    JSON = "json"  # {"tool": "...", "arguments": {...}}
    FUNCTION = "function"  # Native function calling (OpenAI, Anthropic)


@dataclass
class ToolDefinition:
    """Definition of a tool available to the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    required: List[str] = field(default_factory=list)

    def to_xml_schema(self) -> str:
        """Format tool as XML schema for prompts."""
        params_desc = []
        for param_name, param_info in self.parameters.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            required_marker = " (required)" if param_name in self.required else ""
            params_desc.append(f"    <{param_name}>{param_type}: {param_desc}{required_marker}</{param_name}>")

        params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"

        return f"""<tool name="{self.name}">
  <description>{self.description}</description>
  <parameters>
{params_str}
  </parameters>
</tool>"""

    def to_json_schema(self) -> Dict[str, Any]:
        """Format tool as JSON schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            }
        }

    def to_function_schema(self) -> Dict[str, Any]:
        """Format tool for native function calling (OpenAI format)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                }
            }
        }


@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None  # For tracking in multi-turn conversations


@dataclass
class ToolResult:
    """Result from executing a tool call."""

    call_id: Optional[str]
    name: str
    result: Any
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the tool call succeeded."""
        return self.error is None

    def to_xml(self) -> str:
        """Format result as XML for LLM context."""
        if self.error:
            return f'<tool_result name="{self.name}" error="true">{self.error}</tool_result>'
        return f'<tool_result name="{self.name}">{self.result}</tool_result>'

    def to_json(self) -> Dict[str, Any]:
        """Format result as JSON."""
        result = {
            "name": self.name,
            "result": self.result if self.success else None,
        }
        if self.call_id:
            result["call_id"] = self.call_id
        if self.error:
            result["error"] = self.error
        return result
