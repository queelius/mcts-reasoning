"""
Providers: LLM provider implementations.

Supports OpenAI, Anthropic, and Ollama. Use detect_provider() for auto-detection
or get_provider() to create by explicit name.
"""

from __future__ import annotations

from typing import Any

from .base import LLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider

# Order: env-var checks first (fast), network probes last (slow)
_PROVIDER_CLASSES: list[type[LLMProvider]] = [
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
]


def detect_provider(base_url: str | None = None, **kwargs: Any) -> LLMProvider:
    """
    Auto-detect an available LLM provider.

    Tries env-var-based providers first (OpenAI, Anthropic), then
    network-based providers (Ollama). Raises RuntimeError if none found.
    """
    if base_url:
        for cls in _PROVIDER_CLASSES:
            if cls.detect(base_url):
                return cls(base_url=base_url, **kwargs)
    for cls in _PROVIDER_CLASSES:
        if cls.detect():
            return cls(**kwargs)
    raise RuntimeError(
        "No LLM provider available. "
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama."
    )


def get_provider(name: str, **kwargs: Any) -> LLMProvider:
    """
    Create a provider by explicit name.

    Args:
        name: One of "openai", "anthropic", "ollama".
        **kwargs: Provider-specific configuration.

    Raises:
        ValueError: If name is not a known provider.
    """
    registry: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown provider: {name}. Choose from: {list(registry.keys())}"
        )
    return registry[name](**kwargs)


__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "detect_provider",
    "get_provider",
]
