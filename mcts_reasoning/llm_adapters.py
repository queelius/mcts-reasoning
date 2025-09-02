"""
LLM Adapters for MCTS

Provides a unified interface for different LLM providers:
- OpenAI
- Anthropic  
- Ollama
- Mock (for testing)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import json


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama LLM."""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "llama2"):
        self.host = host
        self.port = port
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            try:
                from tree_of_thought_mcts.llm.ollama_client import OllamaClient
                self._client = OllamaClient(host=self.host, port=self.port, model=self.model)
            except ImportError:
                raise ImportError("Ollama client not available. Check tree_of_thought_mcts.llm.ollama_client")
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
        """Generate using Ollama."""
        client = self._get_client()
        response = client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.text if hasattr(response, 'text') else str(response)
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            client = self._get_client()
            return client.is_available()
        except:
            return False


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                openai.api_key = self.api_key
                self._client = openai
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
        """Generate using OpenAI."""
        client = self._get_client()
        
        try:
            if self.model.startswith("gpt"):
                # Chat models
                response = client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                # Completion models
                response = client.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].text
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self.api_key is not None


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
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
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
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
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return self.api_key is not None


class MockLLMAdapter(LLMAdapter):
    """Mock LLM for testing without API calls."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
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
        return f"Mock response #{self.call_count}: Processed prompt of {len(prompt)} chars."
    
    def is_available(self) -> bool:
        """Mock is always available."""
        return True


class LLMFactory:
    """Factory for creating LLM adapters."""
    
    @staticmethod
    def create(provider: str, **kwargs) -> LLMAdapter:
        """
        Create an LLM adapter.
        
        Args:
            provider: One of 'ollama', 'openai', 'anthropic', 'mock'
            **kwargs: Provider-specific configuration
        
        Returns:
            LLMAdapter instance
        """
        providers = {
            'ollama': OllamaAdapter,
            'openai': OpenAIAdapter,
            'anthropic': AnthropicAdapter,
            'mock': MockLLMAdapter
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
        
        return providers[provider](**kwargs)
    
    @staticmethod
    def create_from_env() -> LLMAdapter:
        """
        Create an LLM adapter based on environment variables.
        
        Checks in order:
        1. LLM_PROVIDER env var
        2. OPENAI_API_KEY -> OpenAI
        3. ANTHROPIC_API_KEY -> Anthropic  
        4. Default to Ollama
        """
        provider = os.environ.get("LLM_PROVIDER")
        
        if provider:
            return LLMFactory.create(provider)
        
        if os.environ.get("OPENAI_API_KEY"):
            return LLMFactory.create("openai")
        
        if os.environ.get("ANTHROPIC_API_KEY"):
            return LLMFactory.create("anthropic")
        
        # Default to Ollama
        return LLMFactory.create("ollama")


# Convenience functions
def get_llm(provider: Optional[str] = None, **kwargs) -> LLMAdapter:
    """
    Get an LLM adapter.
    
    If provider is None, tries to detect from environment.
    """
    if provider:
        return LLMFactory.create(provider, **kwargs)
    return LLMFactory.create_from_env()


def test_llm_adapter(adapter: LLMAdapter):
    """Test an LLM adapter."""
    print(f"Testing {adapter.__class__.__name__}...")
    
    if not adapter.is_available():
        print("  ❌ Not available")
        return False
    
    try:
        response = adapter.generate("What is 2+2?", temperature=0.0, max_tokens=10)
        print(f"  ✅ Response: {response[:50]}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Test different adapters
    print("LLM Adapter Tests")
    print("=" * 50)
    
    # Test Mock
    mock = MockLLMAdapter()
    test_llm_adapter(mock)
    
    # Test Ollama if available
    ollama = OllamaAdapter()
    test_llm_adapter(ollama)
    
    # Test OpenAI if key is set
    if os.environ.get("OPENAI_API_KEY"):
        openai_adapter = OpenAIAdapter()
        test_llm_adapter(openai_adapter)
    
    # Test factory
    print("\nFactory Test:")
    llm = get_llm()
    print(f"Auto-detected: {llm.__class__.__name__}")
    test_llm_adapter(llm)