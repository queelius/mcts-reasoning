"""
Pytest configuration and shared fixtures.
"""

import pytest
from mcts_reasoning.compositional.providers import MockLLMProvider


@pytest.fixture
def mock_llm():
    """Fixture providing a basic MockLLMProvider."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_responses():
    """Fixture providing MockLLMProvider with predefined responses."""
    responses = {
        "quadratic": "The solution is x = -2 or x = -3",
        "prime": "The prime numbers are 2, 3, 5, 7, 11, 13, 17, 19",
        "multiply": "The result is 345",
        "solve": "Solution: Apply systematic decomposition",
        "analyze": "Analysis: This requires breaking down into steps",
    }
    return MockLLMProvider(responses)


@pytest.fixture
def sample_example_dict():
    """Sample example data as dictionary."""
    return {
        "problem": "What is 2 + 2?",
        "solution": "4",
        "reasoning_steps": ["Add the two numbers", "2 + 2 = 4"],
        "metadata": {"domain": "arithmetic", "difficulty": "easy"}
    }
