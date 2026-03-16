"""
Anthropic LLM provider.

Uses the anthropic SDK (lazy-imported). Detection checks the ANTHROPIC_API_KEY
environment variable.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import Message
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        **kwargs: Any,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(
        self,
        messages: list[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")

    def is_available(self) -> bool:
        return self.api_key is not None

    @classmethod
    def detect(cls, base_url: str | None = None) -> bool:
        """Detect by checking for ANTHROPIC_API_KEY in the environment."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def get_name(self) -> str:
        return f"Anthropic-{self.model}"
