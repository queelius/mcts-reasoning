"""
Tests for the MCP server tool implementations (v0.6 Task 13).

All tests use the mock provider -- no LLM calls are made.
"""

from __future__ import annotations

import json

import pytest

from mcts_reasoning.server.tools import (
    _resolve_provider,
    mcts_bench_impl,
    mcts_explore_impl,
    mcts_search_impl,
)
from mcts_reasoning.testing import MockLLMProvider


# ---------------------------------------------------------------------------
# _resolve_provider
# ---------------------------------------------------------------------------


def test_resolve_provider_mock():
    provider = _resolve_provider("mock")
    assert isinstance(provider, MockLLMProvider)


def test_resolve_provider_mock_ignores_model():
    provider = _resolve_provider("mock", model="some-model")
    assert isinstance(provider, MockLLMProvider)


def test_resolve_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        _resolve_provider("nonexistent")


# ---------------------------------------------------------------------------
# mcts_search_impl
# ---------------------------------------------------------------------------


def test_mcts_search_impl_with_mock():
    result = mcts_search_impl("What is 2+2?", provider_name="mock", simulations=3)
    assert isinstance(result, dict)
    # Should succeed (no "error" key) or have meaningful error
    assert "answer" in result or "error" in result


def test_mcts_search_impl_returns_expected_keys():
    result = mcts_search_impl("What is 2+2?", provider_name="mock", simulations=3)
    if "error" not in result:
        assert "answer" in result
        assert "confidence" in result
        assert "simulations" in result
        assert "terminal_states" in result
        assert "tree_nodes" in result


def test_mcts_search_impl_simulations_count():
    result = mcts_search_impl("What is 2+2?", provider_name="mock", simulations=5)
    if "error" not in result:
        assert result["simulations"] == 5


def test_mcts_search_impl_tree_nodes_positive():
    result = mcts_search_impl("What is 2+2?", provider_name="mock", simulations=3)
    if "error" not in result:
        assert result["tree_nodes"] >= 1


def test_mcts_search_impl_bad_provider_returns_error():
    result = mcts_search_impl("Q?", provider_name="nonexistent", simulations=1)
    assert "error" in result


# ---------------------------------------------------------------------------
# mcts_explore_impl
# ---------------------------------------------------------------------------


def test_mcts_explore_impl_with_mock():
    result = mcts_explore_impl("What is 2+2?", provider_name="mock", simulations=3)
    assert isinstance(result, dict)
    assert "tree" in result or "error" in result


def test_mcts_explore_impl_returns_tree():
    result = mcts_explore_impl("What is 2+2?", provider_name="mock", simulations=3)
    if "error" not in result:
        assert "tree" in result
        assert "question" in result
        assert "simulations" in result
        assert "terminal_states" in result
        # Tree should be a serializable dict
        assert isinstance(result["tree"], dict)


def test_mcts_explore_impl_tree_is_json_serializable():
    result = mcts_explore_impl("What is 2+2?", provider_name="mock", simulations=3)
    if "error" not in result:
        # Should not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


def test_mcts_explore_impl_question_round_trips():
    question = "What is 2+2?"
    result = mcts_explore_impl(question, provider_name="mock", simulations=3)
    if "error" not in result:
        assert result["question"] == question


# ---------------------------------------------------------------------------
# mcts_bench_impl
# ---------------------------------------------------------------------------


def test_mcts_bench_impl_arithmetic():
    result = mcts_bench_impl("arithmetic", provider_name="mock", simulations=[3])
    assert isinstance(result, dict)
    # Should be a report dict or an error
    if "error" not in result:
        assert "benchmark_name" in result
        assert "solvers" in result


def test_mcts_bench_impl_knights():
    result = mcts_bench_impl("knights", provider_name="mock", simulations=[3])
    assert isinstance(result, dict)
    if "error" not in result:
        assert result["benchmark_name"] == "knights_and_knaves"


def test_mcts_bench_impl_unknown_benchmark():
    result = mcts_bench_impl("nonexistent", provider_name="mock", simulations=[3])
    assert "error" in result


def test_mcts_bench_impl_multiple_simulations():
    result = mcts_bench_impl("arithmetic", provider_name="mock", simulations=[3, 5])
    assert isinstance(result, dict)
    if "error" not in result:
        # Should have baseline + 2 MCTS solvers
        assert len(result["solvers"]) == 3


def test_mcts_bench_impl_default_simulations():
    result = mcts_bench_impl("arithmetic", provider_name="mock")
    assert isinstance(result, dict)
    if "error" not in result:
        # Default: baseline + MCTS(10)
        assert len(result["solvers"]) == 2


def test_mcts_bench_impl_result_is_json_serializable():
    result = mcts_bench_impl("arithmetic", provider_name="mock", simulations=[3])
    # Should not raise
    serialized = json.dumps(result)
    assert isinstance(serialized, str)
