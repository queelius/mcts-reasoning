"""
LLM Provider: Abstract base class for language model providers.

All providers must implement generate(), is_available(), and get_name().
The detect() classmethod enables auto-detection of available providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Message


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a completion from a list of messages.

        Args:
            messages: Conversation messages (role + content dicts).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is ready to generate."""

    @classmethod
    def detect(cls, base_url: str | None = None) -> bool:
        """
        Return True if this provider can be auto-detected.

        Override in subclasses to check environment variables or network
        endpoints. The default returns False (no detection).
        """
        return False

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this provider."""
