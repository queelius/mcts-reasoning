"""Tests for mcts_reasoning.types module."""

import json
import os
import tempfile

import pytest

from mcts_reasoning.types import (
    ConsensusResult,
    Continuation,
    Evaluation,
    Message,
    SampledPath,
    SearchState,
    State,
    TerminalCheck,
    extend_state,
)


class TestState:
    """Tests for the State NewType."""

    def test_state_is_str(self):
        """State is a NewType of str, so it behaves as str at runtime."""
        s = State("hello")
        assert isinstance(s, str)
        assert s == "hello"

    def test_extend_state_concatenation(self):
        """extend_state joins with double newline."""
        s = State("Question: What is 2+2?")
        result = extend_state(s, "Step 1: Add the numbers.")
        assert result == "Question: What is 2+2?\n\nStep 1: Add the numbers."

    def test_extend_state_returns_state(self):
        """extend_state returns a State (which is str at runtime)."""
        s = State("base")
        result = extend_state(s, "extension")
        assert isinstance(result, str)

    def test_extend_state_multiple(self):
        """Chaining extend_state produces correct multi-step text."""
        s = State("Q")
        s = extend_state(s, "Step 1")
        s = extend_state(s, "Step 2")
        assert s == "Q\n\nStep 1\n\nStep 2"


class TestMessage:
    """Tests for Message TypedDict."""

    def test_message_creation(self):
        msg: Message = {"role": "user", "content": "hello"}
        assert msg["role"] == "user"
        assert msg["content"] == "hello"

    def test_message_is_dict(self):
        msg: Message = {"role": "assistant", "content": "hi"}
        assert isinstance(msg, dict)


class TestContinuation:
    """Tests for the Continuation dataclass."""

    def test_continuation_fields(self):
        c = Continuation(text=State("some text"), is_terminal=True, answer="42")
        assert c.text == "some text"
        assert c.is_terminal is True
        assert c.answer == "42"

    def test_continuation_defaults(self):
        c = Continuation(text=State("text"))
        assert c.is_terminal is False
        assert c.answer is None

    def test_continuation_non_terminal(self):
        c = Continuation(text=State("reasoning step"), is_terminal=False)
        assert not c.is_terminal
        assert c.answer is None


class TestEvaluation:
    """Tests for the Evaluation dataclass."""

    def test_evaluation_fields(self):
        e = Evaluation(score=0.85, explanation="good reasoning")
        assert e.score == 0.85
        assert e.explanation == "good reasoning"

    def test_evaluation_defaults(self):
        e = Evaluation(score=0.5)
        assert e.explanation == ""

    def test_evaluation_score_range(self):
        """Evaluation doesn't enforce range, but stores values correctly."""
        e = Evaluation(score=0.0)
        assert e.score == 0.0
        e = Evaluation(score=1.0)
        assert e.score == 1.0


class TestTerminalCheck:
    """Tests for the TerminalCheck dataclass."""

    def test_terminal_check_is_terminal(self):
        tc = TerminalCheck(is_terminal=True, answer="42")
        assert tc.is_terminal is True
        assert tc.answer == "42"

    def test_terminal_check_not_terminal(self):
        tc = TerminalCheck(is_terminal=False)
        assert tc.is_terminal is False
        assert tc.answer is None

    def test_terminal_check_defaults(self):
        tc = TerminalCheck(is_terminal=False)
        assert tc.answer is None


class TestSampledPath:
    """Tests for the SampledPath dataclass."""

    def test_sampled_path_fields(self):
        sp = SampledPath(
            nodes=["a", "b", "c"],
            answer="42",
            value=0.9,
            visits=10,
        )
        assert sp.nodes == ["a", "b", "c"]
        assert sp.answer == "42"
        assert sp.value == 0.9
        assert sp.visits == 10

    def test_sampled_path_defaults(self):
        sp = SampledPath(nodes=[])
        assert sp.answer is None
        assert sp.value == 0.0
        assert sp.visits == 0


class TestConsensusResult:
    """Tests for the ConsensusResult dataclass."""

    def test_consensus_result_fields(self):
        cr = ConsensusResult(
            answer="42",
            confidence=0.85,
            distribution={"42": 5, "43": 1},
            paths_used=6,
        )
        assert cr.answer == "42"
        assert cr.confidence == 0.85
        assert cr.distribution == {"42": 5, "43": 1}
        assert cr.paths_used == 6

    def test_consensus_result_defaults(self):
        cr = ConsensusResult(answer="yes", confidence=1.0)
        assert cr.distribution == {}
        assert cr.paths_used == 0


class TestSearchState:
    """Tests for the SearchState dataclass."""

    def test_search_state_config_fields(self):
        from mcts_reasoning.node import Node

        root = Node(state=State("root"))
        ss = SearchState(
            root=root,
            question="What is 2+2?",
            exploration_constant=2.0,
            max_children_per_node=5,
            max_rollout_depth=10,
        )
        assert ss.question == "What is 2+2?"
        assert ss.exploration_constant == 2.0
        assert ss.max_children_per_node == 5
        assert ss.max_rollout_depth == 10
        assert ss.simulations_run == 0
        assert ss.terminal_states == []

    def test_search_state_defaults(self):
        from mcts_reasoning.node import Node

        root = Node(state=State("root"))
        ss = SearchState(root=root, question="Q")
        assert ss.exploration_constant == 1.414
        assert ss.max_children_per_node == 3
        assert ss.max_rollout_depth == 5
        assert ss.simulations_run == 0

    def test_search_state_save_load_roundtrip(self):
        """SearchState save/load preserves all fields."""
        from mcts_reasoning.node import Node

        root = Node(state=State("Root question"))
        child = root.add_child(state=State("Step 1"))
        child.visits = 5
        child.value = 3.5
        terminal = child.add_child(
            state=State("Step 1\n\nANSWER: 42"),
            is_terminal=True,
            answer="42",
        )
        terminal.visits = 3
        terminal.value = 2.7

        ss = SearchState(
            root=root,
            question="What is the answer?",
            simulations_run=10,
            terminal_states=[{"answer": "42", "score": 0.9}],
            exploration_constant=2.0,
            max_children_per_node=4,
            max_rollout_depth=8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            ss.save(path)

            # Verify file exists and is valid JSON
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["question"] == "What is the answer?"

            # Load and verify roundtrip
            loaded = SearchState.load(path)
            assert loaded.question == ss.question
            assert loaded.simulations_run == ss.simulations_run
            assert loaded.exploration_constant == ss.exploration_constant
            assert loaded.max_children_per_node == ss.max_children_per_node
            assert loaded.max_rollout_depth == ss.max_rollout_depth
            assert loaded.terminal_states == ss.terminal_states

            # Verify tree structure survived
            assert loaded.root.state == "Root question"
            assert len(loaded.root.children) == 1
            assert loaded.root.children[0].visits == 5
            assert loaded.root.children[0].children[0].is_terminal is True
            assert loaded.root.children[0].children[0].answer == "42"

    def test_search_state_save_load_empty_tree(self):
        """SearchState roundtrip with a single root node."""
        from mcts_reasoning.node import Node

        root = Node(state=State("Just a root"))
        ss = SearchState(root=root, question="simple")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.json")
            ss.save(path)
            loaded = SearchState.load(path)
            assert loaded.root.state == "Just a root"
            assert loaded.root.children == []
