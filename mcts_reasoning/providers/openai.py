"""
OpenAI LLM provider.

Uses the openai SDK (lazy-imported). Detection checks the OPENAI_API_KEY
environment variable.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import Message
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        **kwargs: Any,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def generate(
        self,
        messages: list[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    def is_available(self) -> bool:
        return self.api_key is not None

    @classmethod
    def detect(cls, base_url: str | None = None) -> bool:
        """Detect by checking for OPENAI_API_KEY in the environment."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    def get_name(self) -> str:
        return f"OpenAI-{self.model}"
