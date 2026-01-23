"""
LLM Provider adapter for v2 MCTS.

Bridges the existing LLM providers to the v2 Generator/Evaluator interfaces.
"""

from .generator import LLMGenerator
from .evaluator import LLMEvaluator


class LLMAdapter:
    """
    Adapter that wraps existing LLM providers for use with v2.

    The existing providers have a .generate(prompt, **kwargs) method.
    """

    def __init__(self, provider):
        """
        Initialize adapter with an LLM provider.

        Args:
            provider: Any object with a .generate(prompt, **kwargs) method
        """
        self.provider = provider

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 500
    ) -> str:
        """Generate text from the LLM."""
        return self.provider.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def create_generator(
    llm_provider, temperature: float = 0.7, max_tokens: int = 500
) -> LLMGenerator:
    """
    Create an LLMGenerator from an existing provider.

    Args:
        llm_provider: LLM provider with .generate() method
        temperature: Sampling temperature
        max_tokens: Max tokens per continuation

    Returns:
        LLMGenerator instance
    """
    adapter = LLMAdapter(llm_provider)
    return LLMGenerator(
        llm=adapter,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_evaluator(llm_provider, temperature: float = 0.1) -> LLMEvaluator:
    """
    Create an LLMEvaluator from an existing provider.

    Args:
        llm_provider: LLM provider with .generate() method
        temperature: Sampling temperature (low for consistent scoring)

    Returns:
        LLMEvaluator instance
    """
    adapter = LLMAdapter(llm_provider)
    return LLMEvaluator(
        llm=adapter,
        temperature=temperature,
    )
