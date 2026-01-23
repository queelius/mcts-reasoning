"""
Native function calling support for OpenAI and Anthropic providers.

Provides wrappers that enable native tool_use APIs instead of XML/JSON parsing.
This offers better reliability and lower latency compared to parsing.
"""

import logging
from typing import List, Optional, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass

from .formats import ToolDefinition, ToolCall

logger = logging.getLogger(__name__)


@runtime_checkable
class NativeFunctionCallProvider(Protocol):
    """
    Protocol for providers that support native function calling.

    Providers implementing this protocol can use the API's built-in
    tool/function calling mechanism instead of parsing XML/JSON from text.
    """

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Tuple[str, List[ToolCall]]:
        """
        Generate with native function calling.

        Args:
            prompt: Input prompt
            tools: Available tools
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Tuple of (text_response, tool_calls)
        """
        ...

    def supports_native_tools(self) -> bool:
        """Whether this provider supports native function calling."""
        ...


@dataclass
class NativeGenerationResult:
    """Result from native function calling generation."""

    text: str
    tool_calls: List[ToolCall]
    stop_reason: str = "end_turn"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class OpenAINativeWrapper:
    """
    Wrapper that adds native function calling to OpenAI provider.

    Usage:
        from mcts_reasoning.compositional import OpenAIProvider
        from mcts_reasoning.tools.native import OpenAINativeWrapper

        provider = OpenAIProvider(model="gpt-4")
        native = OpenAINativeWrapper(provider)

        text, calls = native.generate_with_tools(prompt, tools)
    """

    def __init__(self, provider):
        """
        Initialize wrapper.

        Args:
            provider: OpenAIProvider instance
        """
        self.provider = provider
        self._client = None

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = self.provider._get_client()
        return self._client

    def supports_native_tools(self) -> bool:
        """OpenAI supports native function calling."""
        return True

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Tuple[str, List[ToolCall]]:
        """
        Generate with OpenAI's native function calling.

        Args:
            prompt: Input prompt
            tools: Available tools
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Tuple of (text_response, tool_calls)
        """
        client = self._get_client()

        # Convert tools to OpenAI format
        openai_tools = [tool.to_function_schema() for tool in tools]

        try:
            response = client.chat.completions.create(
                model=self.provider.model,
                messages=[{"role": "user", "content": prompt}],
                tools=openai_tools if openai_tools else None,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            message = response.choices[0].message
            text = message.content or ""

            # Parse tool calls from response
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    import json

                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    tool_calls.append(
                        ToolCall(
                            name=tc.function.name,
                            arguments=arguments,
                            call_id=tc.id,
                        )
                    )

            return text, tool_calls

        except Exception as e:
            logger.error(f"OpenAI native tool call failed: {e}")
            raise RuntimeError(f"OpenAI generation with tools failed: {e}")

    def generate(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7
    ) -> str:
        """Delegate to base provider for non-tool generation."""
        return self.provider.generate(prompt, max_tokens, temperature)

    def get_provider_name(self) -> str:
        return f"{self.provider.get_provider_name()}-native"


class AnthropicNativeWrapper:
    """
    Wrapper that adds native tool use to Anthropic provider.

    Usage:
        from mcts_reasoning.compositional import AnthropicProvider
        from mcts_reasoning.tools.native import AnthropicNativeWrapper

        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        native = AnthropicNativeWrapper(provider)

        text, calls = native.generate_with_tools(prompt, tools)
    """

    def __init__(self, provider):
        """
        Initialize wrapper.

        Args:
            provider: AnthropicProvider instance
        """
        self.provider = provider
        self._client = None

    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = self.provider._get_client()
        return self._client

    def supports_native_tools(self) -> bool:
        """Anthropic supports native tool use."""
        return True

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Tuple[str, List[ToolCall]]:
        """
        Generate with Anthropic's native tool use.

        Args:
            prompt: Input prompt
            tools: Available tools
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Tuple of (text_response, tool_calls)
        """
        client = self._get_client()

        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required,
                    },
                }
            )

        try:
            kwargs = {
                "model": self.provider.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = client.messages.create(**kwargs)

            # Parse response content
            text_parts = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            name=block.name,
                            arguments=block.input,
                            call_id=block.id,
                        )
                    )

            text = "\n".join(text_parts)
            return text, tool_calls

        except Exception as e:
            logger.error(f"Anthropic native tool use failed: {e}")
            raise RuntimeError(f"Anthropic generation with tools failed: {e}")

    def generate(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7
    ) -> str:
        """Delegate to base provider for non-tool generation."""
        return self.provider.generate(prompt, max_tokens, temperature)

    def get_provider_name(self) -> str:
        return f"{self.provider.get_provider_name()}-native"


def wrap_provider_for_native_tools(provider) -> Optional[NativeFunctionCallProvider]:
    """
    Wrap a provider with native function calling support if available.

    Args:
        provider: LLMProvider instance

    Returns:
        NativeFunctionCallProvider wrapper, or None if not supported
    """
    provider_name = provider.get_provider_name().lower()

    if "openai" in provider_name or "gpt" in provider_name:
        return OpenAINativeWrapper(provider)

    if "anthropic" in provider_name or "claude" in provider_name:
        return AnthropicNativeWrapper(provider)

    # Ollama and other providers don't support native function calling
    return None


def supports_native_function_calling(provider) -> bool:
    """
    Check if a provider supports native function calling.

    Args:
        provider: LLMProvider instance

    Returns:
        True if native function calling is supported
    """
    if isinstance(provider, NativeFunctionCallProvider):
        return provider.supports_native_tools()

    provider_name = provider.get_provider_name().lower()
    return (
        "openai" in provider_name
        or "anthropic" in provider_name
        or "gpt" in provider_name
        or "claude" in provider_name
    )


__all__ = [
    "NativeFunctionCallProvider",
    "NativeGenerationResult",
    "OpenAINativeWrapper",
    "AnthropicNativeWrapper",
    "wrap_provider_for_native_tools",
    "supports_native_function_calling",
]
