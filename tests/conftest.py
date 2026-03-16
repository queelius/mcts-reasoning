"""
Pytest configuration and shared fixtures.

Note: compositional.providers was removed in v0.6 Task 1.
These fixtures are stubs until the provider system is rebuilt (Task 8-9).
"""

import pytest


@pytest.fixture
def sample_example_dict():
    """Sample example data as dictionary."""
    return {
        "problem": "What is 2 + 2?",
        "solution": "4",
        "reasoning_steps": ["Add the two numbers", "2 + 2 = 4"],
        "metadata": {"domain": "arithmetic", "difficulty": "easy"},
    }
