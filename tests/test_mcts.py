"""Tests for the stateless MCTS implementation (v0.6 Task 10)."""

import pytest

from mcts_reasoning.mcts import MCTS
from mcts_reasoning.types import SearchState, State
from mcts_reasoning.testing import MockGenerator, MockEvaluator


def test_search_returns_search_state():
    mcts = MCTS(generator=MockGenerator(terminal_at=2), evaluator=MockEvaluator(score=0.8))
    state = mcts.search("What is 2+2?", simulations=5)
    assert isinstance(state, SearchState)
    assert state.question == "What is 2+2?"
    assert state.simulations_run == 5


def test_search_does_not_mutate_mcts():
    mcts = MCTS(generator=MockGenerator(terminal_at=2), evaluator=MockEvaluator())
    state1 = mcts.search("Q1", simulations=3)
    state2 = mcts.search("Q2", simulations=3)
    assert state1.question == "Q1"
    assert state2.question == "Q2"
    assert state1.root is not state2.root


def test_continue_search():
    mcts = MCTS(generator=MockGenerator(terminal_at=2), evaluator=MockEvaluator())
    state1 = mcts.search("Q", simulations=3)
    state2 = mcts.continue_search(state1, simulations=3)
    assert state2.simulations_run == 6


def test_on_simulation_callback():
    calls = []

    def callback(sim_num, phase, node, state):
        calls.append((sim_num, phase))

    mcts = MCTS(
        generator=MockGenerator(terminal_at=2),
        evaluator=MockEvaluator(),
        on_simulation=callback,
    )
    mcts.search("Q", simulations=2)
    phases = [c[1] for c in calls]
    assert "select" in phases
    assert "expand" in phases
    assert "backprop" in phases


def test_rollout_respects_max_children():
    mcts = MCTS(
        generator=MockGenerator(terminal_at=10),
        evaluator=MockEvaluator(),
        max_children_per_node=2,
    )
    state = mcts.search("Q", simulations=10)

    def check(node):
        assert len(node.children) <= 2, f"Node has {len(node.children)} children"
        for c in node.children:
            check(c)

    check(state.root)


def test_search_finds_terminal_states():
    mcts = MCTS(
        generator=MockGenerator(terminal_at=2),
        evaluator=MockEvaluator(score=1.0),
    )
    state = mcts.search("Q", simulations=5)
    assert len(state.terminal_states) > 0


def test_search_state_save_load_roundtrip(tmp_path):
    mcts = MCTS(generator=MockGenerator(terminal_at=2), evaluator=MockEvaluator())
    state = mcts.search("Q", simulations=3)
    path = str(tmp_path / "state.json")
    state.save(path)
    loaded = SearchState.load(path)
    assert loaded.question == "Q"
    assert loaded.simulations_run == 3
    assert loaded.root.state == state.root.state


def test_callback_receives_all_four_phases_per_simulation():
    """Each simulation must fire exactly four phase callbacks."""
    calls = []

    def callback(sim_num, phase, node, state):
        calls.append((sim_num, phase))

    mcts = MCTS(
        generator=MockGenerator(terminal_at=2),
        evaluator=MockEvaluator(),
        on_simulation=callback,
    )
    mcts.search("Q", simulations=1)
    phases = [c[1] for c in calls]
    assert phases == ["select", "expand", "rollout", "backprop"]


def test_continue_search_preserves_root_identity():
    """continue_search operates on the same root node."""
    mcts = MCTS(generator=MockGenerator(terminal_at=2), evaluator=MockEvaluator())
    state = mcts.search("Q", simulations=3)
    root_before = state.root
    mcts.continue_search(state, simulations=3)
    assert state.root is root_before


def test_backpropagation_uses_running_average():
    """Value stored on nodes should be a running average, not a sum."""
    mcts = MCTS(
        generator=MockGenerator(terminal_at=1),
        evaluator=MockEvaluator(score=0.6),
        max_children_per_node=10,
    )
    state = mcts.search("Q", simulations=5)
    # Every simulation hits terminal at depth 1 with score 0.6,
    # so root's average value should converge to 0.6.
    avg = state.root.value
    assert 0.0 <= avg <= 1.0, f"Average value {avg} out of expected range"


def test_to_search_result():
    """MCTS.to_search_result converts SearchState to legacy SearchResult."""
    from mcts_reasoning.mcts import SearchResult

    mcts = MCTS(
        generator=MockGenerator(terminal_at=2),
        evaluator=MockEvaluator(score=0.9),
    )
    state = mcts.search("Q", simulations=5)
    result = MCTS.to_search_result(state)
    assert isinstance(result, SearchResult)
    assert result.simulations == state.simulations_run
    assert result.root is state.root
