"""
Ollama LLM provider.

Uses the /api/chat endpoint (NOT /api/generate) for message-based interaction.
Detection GETs the base_url with a 2-second timeout and checks for
"Ollama is running" in the response text.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import Message
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama local/remote LLM provider."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self._session: Any = None

    def _get_session(self) -> Any:
        """Lazy-load a requests session."""
        if self._session is None:
            try:
                import requests
            except ImportError:
                raise ImportError(
                    "requests package not installed. "
                    "Install with: pip install requests"
                )
            self._session = requests.Session()
        return self._session

    def generate(
        self,
        messages: list[Message],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        session = self._get_session()
        try:
            response = session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def is_available(self) -> bool:
        try:
            session = self._get_session()
            response = session.get(self.base_url, timeout=2)
            return "Ollama is running" in response.text
        except Exception:
            return False

    @classmethod
    def detect(cls, base_url: str | None = None) -> bool:
        """
        Detect an Ollama server by GETting the base_url.

        Returns True if the response body contains "Ollama is running".
        Falls back to OLLAMA_BASE_URL env var or localhost:11434.
        """
        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = base_url.rstrip("/")
        try:
            import requests

            response = requests.get(base_url, timeout=2)
            return "Ollama is running" in response.text
        except Exception:
            return False

    def get_name(self) -> str:
        return f"Ollama-{self.model}"
