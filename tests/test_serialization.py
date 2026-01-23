"""Tests for MCTS tree serialization and deserialization."""

import json
import os
import tempfile
import pytest

from mcts_reasoning.node import Node
from mcts_reasoning.mcts import MCTS, SearchResult
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


class TestMCTSSerialization:
    """Tests for MCTS serialization."""

    def _create_mcts_with_tree(self):
        """Create MCTS instance with a searched tree."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        # Run a small search
        mcts.search("What is 2+2?", simulations=5)

        return mcts, generator, evaluator

    def test_mcts_to_dict(self):
        """Test MCTS serialization to dict."""
        mcts, _, _ = self._create_mcts_with_tree()

        data = mcts.to_dict()

        assert data["version"] == "0.4.0"
        assert data["question"] == "What is 2+2?"
        assert "root" in data
        assert "terminal_states" in data
        assert data["exploration_constant"] == 1.414

    def test_mcts_to_dict_no_root_fails(self):
        """Test that serialization fails without a root."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator)

        with pytest.raises(ValueError, match="Cannot serialize"):
            mcts.to_dict()

    def test_mcts_from_dict(self):
        """Test MCTS deserialization from dict."""
        mcts, generator, evaluator = self._create_mcts_with_tree()
        data = mcts.to_dict()

        # Deserialize with new generator/evaluator
        new_generator = MockGenerator()
        new_evaluator = MockEvaluator()
        restored = MCTS.from_dict(data, new_generator, new_evaluator)

        assert restored.question == "What is 2+2?"
        assert restored.root is not None
        assert restored.exploration_constant == data["exploration_constant"]

    def test_mcts_round_trip_preserves_tree(self):
        """Test that round trip preserves tree structure."""
        mcts, generator, evaluator = self._create_mcts_with_tree()

        # Count nodes before
        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.children)

        original_count = count_nodes(mcts.root)

        # Round trip
        data = mcts.to_dict()
        restored = MCTS.from_dict(data, generator, evaluator)

        restored_count = count_nodes(restored.root)
        assert restored_count == original_count

    def test_mcts_save_and_load(self):
        """Test file-based save and load."""
        mcts, generator, evaluator = self._create_mcts_with_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tree.json")

            # Save
            mcts.save(path)
            assert os.path.exists(path)

            # Load
            loaded = MCTS.load(path, generator, evaluator)

            assert loaded.question == mcts.question
            assert loaded.root.state == mcts.root.state

    def test_mcts_save_creates_directory(self):
        """Test that save creates parent directories."""
        mcts, _, _ = self._create_mcts_with_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dirs", "tree.json")

            mcts.save(path)

            assert os.path.exists(path)

    def test_mcts_to_json(self):
        """Test JSON string serialization."""
        mcts, _, _ = self._create_mcts_with_tree()

        json_str = mcts.to_json()
        data = json.loads(json_str)

        assert data["question"] == "What is 2+2?"

    def test_mcts_from_json(self):
        """Test JSON string deserialization."""
        mcts, generator, evaluator = self._create_mcts_with_tree()
        json_str = mcts.to_json()

        restored = MCTS.from_json(json_str, generator, evaluator)

        assert restored.question == mcts.question


class TestContinueSearch:
    """Tests for continued search functionality."""

    def test_continue_search_adds_simulations(self):
        """Test that continue_search adds more simulations."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        # Initial search
        result1 = mcts.search("What is 2+2?", simulations=5)
        initial_visits = mcts.root.visits

        # Continue search
        result2 = mcts.continue_search(simulations=5)

        # Visits should have increased
        assert mcts.root.visits > initial_visits
        assert result2.simulations == mcts.root.visits

    def test_continue_search_without_tree_fails(self):
        """Test that continue_search fails without existing tree."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator)

        with pytest.raises(ValueError, match="Cannot continue search"):
            mcts.continue_search(simulations=5)

    def test_continue_search_after_load(self):
        """Test continuing search after loading a saved tree."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        # Initial search and save
        mcts.search("What is 2+2?", simulations=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tree.json")
            mcts.save(path)

            # Load and continue
            loaded = MCTS.load(path, generator, evaluator)
            initial_visits = loaded.root.visits

            result = loaded.continue_search(simulations=5)

            assert loaded.root.visits > initial_visits
            assert result.best_answer is not None

    def test_continue_search_finds_more_terminals(self):
        """Test that continued search can find more terminal states."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        # Initial search
        result1 = mcts.search("What is 2+2?", simulations=3)
        initial_terminals = len(mcts.terminal_states)

        # Continue with more simulations
        result2 = mcts.continue_search(simulations=10)

        # Should potentially find more terminal states
        # (not guaranteed but likely with more simulations)
        assert len(mcts.terminal_states) >= initial_terminals

    def test_continue_search_returns_search_result(self):
        """Test that continue_search returns proper SearchResult."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        mcts.search("What is 2+2?", simulations=5)
        result = mcts.continue_search(simulations=5)

        assert isinstance(result, SearchResult)
        assert result.root is mcts.root
        assert result.simulations == mcts.root.visits
        assert result.confidence >= 0


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

    def test_continuation_info_serialization(self):
        """Test that _continuation_info is serialized."""
        node = Node(state="Test")
        node._continuation_info = {
            "cont1": (True, "42"),
            "cont2": (False, None),
        }

        data = node.to_dict()
        restored = Node.from_dict(data)

        assert hasattr(restored, "_continuation_info")
        assert restored._continuation_info == node._continuation_info

    def test_mcts_preserves_terminal_states(self):
        """Test that terminal_states list is preserved."""
        generator = MockGenerator()
        evaluator = MockEvaluator()
        mcts = MCTS(generator, evaluator, max_rollout_depth=3)

        mcts.search("What is 2+2?", simulations=10)
        original_terminals = len(mcts.terminal_states)

        data = mcts.to_dict()
        restored = MCTS.from_dict(data, generator, evaluator)

        assert len(restored.terminal_states) == original_terminals
