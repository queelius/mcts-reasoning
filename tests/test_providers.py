"""Tests for the providers package (no network calls)."""

import pytest

from mcts_reasoning.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_provider,
    detect_provider,
)


class TestGetProvider:
    """Tests for get_provider factory."""

    def test_get_provider_by_name_openai(self):
        provider = get_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test-key"

    def test_get_provider_by_name_anthropic(self):
        provider = get_provider("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "test-key"

    def test_get_provider_by_name_ollama(self):
        provider = get_provider("ollama", model="llama3.2")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.2"

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider: banana"):
            get_provider("banana")


class TestDetectProvider:
    """Tests for detect_provider auto-detection."""

    def test_detect_provider_raises_when_none(self, monkeypatch):
        """With no env vars and no reachable Ollama, raises RuntimeError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        # Mock requests.get to always fail
        import types as _types

        def fake_get(*args, **kwargs):
            raise ConnectionError("no server")

        import mcts_reasoning.providers.ollama as ollama_mod

        monkeypatch.setattr(ollama_mod, "requests", type("FakeRequests", (), {"get": staticmethod(fake_get)})(), raising=False)

        # Need to also mock the import inside detect() -- easier to mock at module level
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        import requests as real_requests
        monkeypatch.setattr(real_requests, "get", fake_get)

        with pytest.raises(RuntimeError, match="No LLM provider available"):
            detect_provider()

    def test_detect_with_openai_key(self, monkeypatch):
        """Detects OpenAI when OPENAI_API_KEY is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = detect_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_detect_with_anthropic_key(self, monkeypatch):
        """Detects Anthropic when only ANTHROPIC_API_KEY is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider = detect_provider()
        assert isinstance(provider, AnthropicProvider)


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_detect_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert OpenAIProvider.detect() is True

    def test_detect_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert OpenAIProvider.detect() is False

    def test_is_available_with_key(self):
        p = OpenAIProvider(api_key="sk-test")
        assert p.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = OpenAIProvider()
        assert p.is_available() is False

    def test_get_name(self):
        p = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        assert p.get_name() == "OpenAI-gpt-4o"


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_detect_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert AnthropicProvider.detect() is True

    def test_detect_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert AnthropicProvider.detect() is False

    def test_is_available_with_key(self):
        p = AnthropicProvider(api_key="sk-ant-test")
        assert p.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        p = AnthropicProvider()
        assert p.is_available() is False

    def test_get_name(self):
        p = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        assert p.get_name() == "Anthropic-claude-sonnet-4-20250514"


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_detect_base_url(self, monkeypatch):
        """detect() succeeds when server responds with 'Ollama is running'."""
        import types as _types

        class FakeResponse:
            text = "Ollama is running"
            status_code = 200

        def fake_get(url, timeout=None):
            return FakeResponse()

        # Patch requests at module level for the detect() classmethod
        fake_requests = _types.ModuleType("requests")
        fake_requests.get = fake_get  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

        assert OllamaProvider.detect(base_url="http://fakehost:11434") is True

    def test_detect_no_server(self, monkeypatch):
        """detect() returns False when server is not reachable."""
        import types as _types

        def fake_get(url, timeout=None):
            raise ConnectionError("no server")

        fake_requests = _types.ModuleType("requests")
        fake_requests.get = fake_get  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

        assert OllamaProvider.detect(base_url="http://nohost:11434") is False

    def test_get_name(self):
        p = OllamaProvider(model="llama3.2", base_url="http://localhost:11434")
        assert p.get_name() == "Ollama-llama3.2"

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:11434")
        p = OllamaProvider()
        assert p.base_url == "http://myhost:11434"

    def test_base_url_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        p = OllamaProvider()
        assert p.base_url == "http://localhost:11434"


class TestProviderABCContract:
    """Verify LLMProvider cannot be instantiated."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]
