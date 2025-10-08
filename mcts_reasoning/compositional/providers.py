"""
Unified LLM Provider System

Provides a clean, unified interface for all LLM providers with support for:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Ollama (local or remote models)
- Mock (for testing)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os
import logging

logger = logging.getLogger(__name__)


# ========== Base Provider Interface ==========

class LLMProvider(ABC):
    """
    Unified LLM provider interface.

    This interface combines the best of both LLMAdapter and the original LLMProvider.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0=deterministic, higher=more random)

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name/identifier."""
        pass

    def is_available(self) -> bool:
        """
        Check if provider is available.

        Default implementation attempts a test generation.
        Override for more efficient checks.
        """
        try:
            self.generate("test", max_tokens=1, temperature=0.0)
            return True
        except:
            return False


# ========== Concrete Providers ==========

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API calls."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize mock provider.

        Args:
            responses: Dictionary mapping prompt keywords to responses
        """
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate mock response."""
        self.call_count += 1
        self.last_prompt = prompt

        # Check for specific responses
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Default responses based on prompt patterns
        if "terminal" in prompt.lower() or "complete" in prompt.lower():
            return "YES" if self.call_count > 5 else "NO"

        if "evaluate" in prompt.lower() or "quality" in prompt.lower():
            return "0.75"

        if "analyze" in prompt.lower():
            return "Analysis: The problem requires systematic decomposition."

        if "solve" in prompt.lower():
            return "Solution: Apply the algorithm step by step."

        if "verify" in prompt.lower():
            return "Verification: The solution appears correct."

        # Generic response
        return f"[MOCK LLM Response to: {prompt[:50]}...]"

    def get_provider_name(self) -> str:
        return "MockLLM"

    def is_available(self) -> bool:
        return True


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using OpenAI."""
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    def get_provider_name(self) -> str:
        return f"OpenAI-{self.model}"

    def is_available(self) -> bool:
        return self.api_key is not None


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Claude."""
        client = self._get_client()

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")

    def get_provider_name(self) -> str:
        return f"Anthropic-{self.model}"

    def is_available(self) -> bool:
        return self.api_key is not None


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str = None, host: str = None, port: int = None, base_url: str = None):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama2", "mistral"). Reads from OLLAMA_MODEL env var if not specified.
            host: Ollama server host (default: "localhost")
            port: Ollama server port (default: 11434)
            base_url: Full base URL (e.g., "http://192.168.0.225:11434").
                     Reads from OLLAMA_BASE_URL env var if not specified.
                     If provided, overrides host/port.
        """
        # Read from environment if not provided
        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL")

        if model is None:
            model = os.environ.get("OLLAMA_MODEL", "llama2")

        # Set base_url from either direct param, env var, or construct from host/port
        if base_url:
            self.base_url = base_url.rstrip('/')
            # Extract host and port from base_url for reference
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            self.host = parsed.hostname or "unknown"
            self.port = parsed.port or 11434
        else:
            # Construct from host/port
            self.host = host or os.environ.get("OLLAMA_HOST", "localhost")
            self.port = port or int(os.environ.get("OLLAMA_PORT", "11434"))
            self.base_url = f"http://{self.host}:{self.port}"

        self.model = model
        self._session = None

    def _get_session(self):
        """Lazy load the requests session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise ImportError("requests package not installed. Install with: pip install requests")
        return self._session

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate using Ollama."""
        session = self._get_session()

        try:
            response = session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def get_provider_name(self) -> str:
        return f"Ollama-{self.model}"

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            session = self._get_session()
            response = session.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models on the Ollama server.

        Returns:
            List of model dictionaries with name, size, modified date, etc.
        """
        try:
            session = self._get_session()
            response = session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def get_model_info(self, model_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Model to get info for. Uses self.model if not specified.

        Returns:
            Dictionary with model info (modelfile, parameters, template, etc.)
        """
        model = model_name or self.model
        try:
            session = self._get_session()
            response = session.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get info for model {model}: {e}")
            return None


# ========== Factory and Utility Functions ==========

class ProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(provider: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            provider: One of 'openai', 'anthropic', 'ollama', 'mock'
            **kwargs: Provider-specific configuration

        Returns:
            LLMProvider instance
        """
        providers = {
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
            'ollama': OllamaProvider,
            'mock': MockLLMProvider
        }

        provider_lower = provider.lower()
        if provider_lower not in providers:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")

        return providers[provider_lower](**kwargs)

    @staticmethod
    def from_env() -> LLMProvider:
        """
        Create provider from environment variables.

        Checks in order:
        1. LLM_PROVIDER env var
        2. OPENAI_API_KEY -> OpenAI
        3. ANTHROPIC_API_KEY -> Anthropic
        4. Default to Mock (for testing without setup)
        """
        provider = os.environ.get("LLM_PROVIDER")

        if provider:
            logger.info(f"Using LLM_PROVIDER={provider} from environment")
            return ProviderFactory.create(provider)

        if os.environ.get("OPENAI_API_KEY"):
            logger.info("Detected OPENAI_API_KEY, using OpenAI provider")
            return ProviderFactory.create("openai")

        if os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("Detected ANTHROPIC_API_KEY, using Anthropic provider")
            return ProviderFactory.create("anthropic")

        # Default to Ollama if it's available, otherwise Mock
        try:
            ollama = ProviderFactory.create("ollama")
            if ollama.is_available():
                logger.info("Detected Ollama server, using Ollama provider")
                return ollama
        except:
            pass

        logger.warning("No LLM provider configured, using MockLLMProvider")
        return ProviderFactory.create("mock")


def get_llm(provider: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Get an LLM provider.

    Convenience function that either:
    - Creates a provider by name if specified
    - Auto-detects from environment if provider is None

    Args:
        provider: Provider name ('openai', 'anthropic', 'ollama', 'mock') or None
        **kwargs: Provider-specific configuration

    Returns:
        LLMProvider instance

    Example:
        # Auto-detect from environment
        llm = get_llm()

        # Specify provider
        llm = get_llm("openai", model="gpt-4")
        llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
        llm = get_llm("ollama", model="llama2")
        llm = get_llm("mock")
    """
    if provider:
        return ProviderFactory.create(provider, **kwargs)
    return ProviderFactory.from_env()


def test_provider(provider: LLMProvider) -> bool:
    """
    Test a provider.

    Args:
        provider: Provider to test

    Returns:
        True if test succeeds, False otherwise
    """
    print(f"Testing {provider.get_provider_name()}...")

    if not provider.is_available():
        print("  ❌ Not available")
        return False

    try:
        response = provider.generate("What is 2+2?", max_tokens=10, temperature=0.0)
        print(f"  ✅ Response: {response[:50]}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


# ========== Export All ==========

__all__ = [
    # Base class
    'LLMProvider',

    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'OllamaProvider',
    'MockLLMProvider',

    # Factory
    'ProviderFactory',
    'get_llm',

    # Utilities
    'test_provider',
]


if __name__ == "__main__":
    # Test providers
    print("LLM Provider Tests")
    print("=" * 50)

    # Test Mock
    mock = MockLLMProvider()
    test_provider(mock)

    # Test Ollama if available
    print()
    ollama = OllamaProvider()
    test_provider(ollama)

    # Test auto-detection
    print("\nAuto-detection Test:")
    llm = get_llm()
    print(f"Auto-detected: {llm.get_provider_name()}")
    test_provider(llm)
