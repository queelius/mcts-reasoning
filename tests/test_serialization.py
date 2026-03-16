"""Tests for MCTS tree serialization and deserialization."""

import json
import os
import tempfile
import pytest

from mcts_reasoning.node import Node
from mcts_reasoning.mcts import MCTS
from mcts_reasoning.generator import MockGenerator
from mcts_reasoning.evaluator import MockEvaluator


class TestNodeSerialization:
    """Tests for Node serialization."""

    def test_node_to_dict_basic(self):
        """Test basic node serialization."""
        node = Node(
            state="Test state",
            visits=10,
            value=5.0,
            is_terminal=False,
            answer=None,
        )

        data = node.to_dict()

        assert data["state"] == "Test state"
        assert data["visits"] == 10
        assert data["value"] == 5.0
        assert data["is_terminal"] is False
        assert data["answer"] is None
        assert data["children"] == []

    def test_node_to_dict_terminal(self):
        """Test terminal node serialization."""
        node = Node(
            state="Final state\nANSWER: 42",
            visits=5,
            value=4.5,
            is_terminal=True,
            answer="42",
        )

        data = node.to_dict()

        assert data["is_terminal"] is True
        assert data["answer"] == "42"

    def test_node_to_dict_with_children(self):
        """Test node with children serialization."""
        root = Node(state="Root")
        child1 = root.add_child(state="Child 1")
        child2 = root.add_child(state="Child 2")
        grandchild = child1.add_child(state="Grandchild")

        data = root.to_dict()

        assert len(data["children"]) == 2
        assert data["children"][0]["state"] == "Child 1"
        assert data["children"][1]["state"] == "Child 2"
        assert len(data["children"][0]["children"]) == 1
        assert data["children"][0]["children"][0]["state"] == "Grandchild"

    def test_node_to_dict_with_continuations(self):
        """Test that continuation cache is serialized."""
        node = Node(state="State")
        node.set_continuations(["cont1", "cont2", "cont3"])
        node._continuation_index = 1

        data = node.to_dict()

        assert data["_continuations"] == ["cont1", "cont2", "cont3"]
        assert data["_continuation_index"] == 1

    def test_node_from_dict_basic(self):
        """Test basic node deserialization."""
        data = {
            "state": "Test state",
            "visits": 10,
            "value": 5.0,
            "is_terminal": False,
            "answer": None,
            "children": [],
        }

        node = Node.from_dict(data)

        assert node.state == "Test state"
        assert node.visits == 10
        assert node.value == 5.0
        assert node.is_terminal is False
        assert node.answer is None
        assert node.parent is None

    def test_node_from_dict_with_parent(self):
        """Test deserialization with parent reference."""
        parent = Node(state="Parent")
        data = {"state": "Child", "children": []}

        child = Node.from_dict(data, parent=parent)

        assert child.parent is parent
        assert child.state == "Child"

    def test_node_from_dict_with_children(self):
        """Test deserialization reconstructs children with parent refs."""
        data = {
            "state": "Root",
            "children": [
                {"state": "Child 1", "children": []},
                {"state": "Child 2", "children": []},
            ],
        }

        root = Node.from_dict(data)

        assert len(root.children) == 2
        assert root.children[0].state == "Child 1"
        assert root.children[1].state == "Child 2"
        # Parent references should be reconstructed
        assert root.children[0].parent is root
        assert root.children[1].parent is root

    def test_node_from_dict_restores_continuations(self):
        """Test that continuation cache is restored."""
        data = {
            "state": "State",
            "_continuations": ["c1", "c2"],
            "_continuation_index": 1,
            "children": [],
        }

        node = Node.from_dict(data)

        assert node._continuations == ["c1", "c2"]
        assert node._continuation_index == 1

    def test_node_round_trip(self):
        """Test serialization round trip preserves structure."""
        # Build a tree
        root = Node(state="Root", visits=100, value=50.0)
        child1 = root.add_child(state="Path 1")
        child1.visits = 60
        child1.value = 30.0
        terminal = child1.add_child(
            state="Path 1\nANSWER: 42",
            is_terminal=True,
            answer="42",
        )
        terminal.visits = 40
        terminal.value = 35.0

        child2 = root.add_child(state="Path 2")
        child2.visits = 40
        child2.value = 20.0

        # Round trip
        data = root.to_dict()
        restored = Node.from_dict(data)

        # Verify structure
        assert restored.state == "Root"
        assert restored.visits == 100
        assert restored.value == 50.0
        assert len(restored.children) == 2
        assert restored.children[0].state == "Path 1"
        assert restored.children[0].children[0].is_terminal is True
        assert restored.children[0].children[0].answer == "42"

    def test_node_to_json(self):
        """Test JSON string serialization."""
        node = Node(state="Test", visits=5, value=2.5)

        json_str = node.to_json()
        data = json.loads(json_str)

        assert data["state"] == "Test"
        assert data["visits"] == 5

    def test_node_from_json(self):
        """Test JSON string deserialization."""
        json_str = '{"state": "Test", "visits": 5, "value": 2.5, "children": []}'

        node = Node.from_json(json_str)

        assert node.state == "Test"
        assert node.visits == 5


class TestSearchStateSerialization:
    """Tests for SearchState serialization (replaces old MCTS serialization)."""

    def _create_search_state(self):
        """Create a SearchState via MCTS search."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)
        state = mcts.search("What is 2+2?", simulations=5)
        return state, generator, evaluator

    def test_search_state_save_load(self):
        """Test SearchState file-based save and load."""
        state, _, _ = self._create_search_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)
            assert os.path.exists(path)

            from mcts_reasoning.types import SearchState
            loaded = SearchState.load(path)
            assert loaded.question == "What is 2+2?"
            assert loaded.root.state == state.root.state

    def test_search_state_preserves_simulations(self):
        """Test that simulations_run is preserved."""
        state, _, _ = self._create_search_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)

            from mcts_reasoning.types import SearchState
            loaded = SearchState.load(path)
            assert loaded.simulations_run == state.simulations_run

    def test_search_state_round_trip_preserves_tree(self):
        """Test that round trip preserves tree structure."""
        state, _, _ = self._create_search_state()

        original_count = state.root.count_nodes()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)

            from mcts_reasoning.types import SearchState
            loaded = SearchState.load(path)
            assert loaded.root.count_nodes() == original_count

    def test_search_state_preserves_config(self):
        """Test that exploration_constant etc. are preserved."""
        state, _, _ = self._create_search_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)

            from mcts_reasoning.types import SearchState
            loaded = SearchState.load(path)
            assert loaded.exploration_constant == state.exploration_constant
            assert loaded.max_children_per_node == state.max_children_per_node
            assert loaded.max_rollout_depth == state.max_rollout_depth

    def test_search_state_json_structure(self):
        """Test that saved JSON contains expected keys."""
        state, _, _ = self._create_search_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)

            with open(path) as f:
                data = json.load(f)

            assert data["question"] == "What is 2+2?"
            assert "root" in data
            assert "terminal_states" in data
            assert "simulations_run" in data


class TestContinueSearch:
    """Tests for continued search functionality."""

    def test_continue_search_adds_simulations(self):
        """Test that continue_search adds more simulations."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        state = mcts.search("What is 2+2?", simulations=5)
        initial_visits = state.root.visits

        # Continue search (stateless -- pass state explicitly)
        state = mcts.continue_search(state, simulations=5)

        assert state.root.visits > initial_visits
        assert state.simulations_run == 10

    def test_continue_search_after_load(self):
        """Test continuing search after loading a saved tree."""
        from mcts_reasoning.types import SearchState

        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        state = mcts.search("What is 2+2?", simulations=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)

            loaded = SearchState.load(path)
            initial_visits = loaded.root.visits

            result = mcts.continue_search(loaded, simulations=5)

            assert result.root.visits > initial_visits

    def test_continue_search_finds_more_terminals(self):
        """Test that continued search can find more terminal states."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        state = mcts.search("What is 2+2?", simulations=3)
        initial_terminals = len(state.terminal_states)

        state = mcts.continue_search(state, simulations=10)

        assert len(state.terminal_states) >= initial_terminals

    def test_continue_search_returns_search_state(self):
        """Test that continue_search returns SearchState."""
        from mcts_reasoning.types import SearchState

        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        state = mcts.search("What is 2+2?", simulations=5)
        state2 = mcts.continue_search(state, simulations=5)

        assert isinstance(state2, SearchState)
        assert state2.root is state.root
        assert state2.simulations_run == 10


class TestSerializationEdgeCases:
    """Edge case tests for serialization."""

    def test_empty_state_serialization(self):
        """Test node with empty state."""
        node = Node(state="")
        data = node.to_dict()
        restored = Node.from_dict(data)

        assert restored.state == ""

    def test_unicode_state_serialization(self):
        """Test node with unicode content."""
        node = Node(state="Question: What is \u03c0? Answer: \u2248 3.14159")
        data = node.to_dict()
        restored = Node.from_dict(data)

        assert restored.state == node.state

    def test_large_tree_serialization(self):
        """Test serialization of larger tree."""
        root = Node(state="Root")

        # Create tree with many nodes
        for i in range(10):
            child = root.add_child(state=f"Child {i}")
            for j in range(5):
                grandchild = child.add_child(state=f"Grandchild {i}-{j}")
                grandchild.visits = i + j
                grandchild.value = float(i * j)

        data = root.to_dict()
        restored = Node.from_dict(data)

        # Verify structure
        assert len(restored.children) == 10
        assert all(len(c.children) == 5 for c in restored.children)

    def test_search_state_preserves_terminal_states(self):
        """Test that terminal_states list is preserved through save/load."""
        from mcts_reasoning.types import SearchState

        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        state = mcts.search("What is 2+2?", simulations=10)
        original_terminals = len(state.terminal_states)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)
            loaded = SearchState.load(path)
            assert len(loaded.terminal_states) == original_terminals
