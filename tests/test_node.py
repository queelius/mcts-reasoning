"""Tests for the refactored mcts_reasoning.node module."""

import math

import pytest

from mcts_reasoning.node import Node
from mcts_reasoning.types import State


class TestNodeCreation:
    """Tests for basic Node construction."""

    def test_node_creation(self):
        """Test basic node creation with defaults."""
        node = Node(state=State("initial state"))
        assert node.state == "initial state"
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.depth == 0
        assert not node.is_terminal
        assert node.answer is None

    def test_node_creation_with_all_fields(self):
        """Test node creation specifying all fields."""
        node = Node(
            state=State("test"),
            visits=10,
            value=5.0,
            is_terminal=True,
            answer="42",
            depth=3,
        )
        assert node.visits == 10
        assert node.value == 5.0
        assert node.is_terminal is True
        assert node.answer == "42"
        assert node.depth == 3


class TestAddChild:
    """Tests for add_child and parent/depth relationships."""

    def test_add_child_sets_parent_and_depth(self):
        """add_child sets parent reference and depth = parent.depth + 1."""
        root = Node(state=State("root"))
        child = root.add_child(state=State("child"))

        assert child.parent is root
        assert child in root.children
        assert child.depth == 1

    def test_depth_zero_for_root(self):
        """Root node has depth 0."""
        root = Node(state=State("root"))
        assert root.depth == 0

    def test_depth_correct_for_deep_node(self):
        """Great-grandchild (3 levels down) has depth=3."""
        root = Node(state=State("root"))
        child = root.add_child(state=State("child"))
        grandchild = child.add_child(state=State("grandchild"))
        great_grandchild = grandchild.add_child(state=State("great-grandchild"))

        assert root.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2
        assert great_grandchild.depth == 3

    def test_add_child_with_optional_params(self):
        """add_child accepts optional value, visits, is_terminal, answer."""
        root = Node(state=State("root"))
        child = root.add_child(
            state=State("terminal"),
            value=3.5,
            visits=5,
            is_terminal=True,
            answer="42",
        )
        assert child.value == 3.5
        assert child.visits == 5
        assert child.is_terminal is True
        assert child.answer == "42"
        assert child.depth == 1

    def test_add_child_default_optional_params(self):
        """add_child defaults: value=0, visits=0, is_terminal=False, answer=None."""
        root = Node(state=State("root"))
        child = root.add_child(state=State("child"))
        assert child.value == 0.0
        assert child.visits == 0
        assert child.is_terminal is False
        assert child.answer is None


class TestUCB1:
    """Tests for UCB1 calculation."""

    def test_ucb1_unvisited(self):
        """Unvisited node returns infinity."""
        node = Node(state=State("unvisited"))
        assert node.ucb1() == float("inf")

    def test_ucb1_calculation(self):
        """Test UCB1 = avg_value + c * sqrt(ln(parent_visits) / visits)."""
        root = Node(state=State("root"))
        root.visits = 10

        child = root.add_child(state=State("child"))
        child.visits = 3
        child.value = 2.1  # avg = 0.7

        expected = 0.7 + 1.414 * math.sqrt(math.log(10) / 3)
        assert abs(child.ucb1() - expected) < 0.001

    def test_ucb1_no_parent(self):
        """Root node returns just average_value."""
        root = Node(state=State("root"))
        root.visits = 5
        root.value = 3.0
        assert root.ucb1() == root.average_value

    def test_best_child_selection(self):
        """best_child returns child with highest UCB1."""
        root = Node(state=State("root"))
        root.visits = 10

        child1 = root.add_child(state=State("child1"))
        child1.visits = 3
        child1.value = 1.5  # avg = 0.5

        child2 = root.add_child(state=State("child2"))
        child2.visits = 2
        child2.value = 1.6  # avg = 0.8

        # child2 has higher avg and fewer visits -> higher UCB1
        best = root.best_child()
        assert best is child2

    def test_best_child_empty(self):
        """best_child returns None for leaf node."""
        node = Node(state=State("leaf"))
        assert node.best_child() is None


class TestProperties:
    """Tests for is_leaf, is_root, average_value."""

    def test_is_leaf_and_is_root(self):
        """Root is root but not leaf after adding child; child is leaf but not root."""
        root = Node(state=State("root"))
        assert root.is_root
        assert root.is_leaf

        child = root.add_child(state=State("child"))
        assert not root.is_leaf
        assert root.is_root
        assert child.is_leaf
        assert not child.is_root

    def test_average_value_zero_visits(self):
        """average_value returns 0.0 when visits=0."""
        node = Node(state=State("test"))
        assert node.average_value == 0.0

    def test_average_value(self):
        node = Node(state=State("test"))
        node.visits = 4
        node.value = 3.2
        assert abs(node.average_value - 0.8) < 1e-9


class TestPathFromRoot:
    """Tests for path_from_root."""

    def test_path_from_root(self):
        """path_from_root returns [root, ..., self]."""
        root = Node(state=State("root"))
        child = root.add_child(state=State("child"))
        grandchild = child.add_child(state=State("grandchild"))

        path = grandchild.path_from_root()
        assert len(path) == 3
        assert path[0] is root
        assert path[1] is child
        assert path[2] is grandchild

    def test_path_from_root_single(self):
        """Root's path_from_root is just [root]."""
        root = Node(state=State("root"))
        path = root.path_from_root()
        assert path == [root]


class TestSerialization:
    """Tests for to_dict / from_dict roundtrip."""

    def test_to_dict_and_from_dict_roundtrip(self):
        """Full tree survives to_dict -> from_dict."""
        root = Node(state=State("Root"), visits=100, value=50.0)
        child1 = root.add_child(state=State("Path 1"))
        child1.visits = 60
        child1.value = 30.0
        terminal = child1.add_child(
            state=State("Path 1\nANSWER: 42"),
            is_terminal=True,
            answer="42",
        )
        terminal.visits = 40
        terminal.value = 35.0

        child2 = root.add_child(state=State("Path 2"))
        child2.visits = 40
        child2.value = 20.0

        data = root.to_dict()
        restored = Node.from_dict(data)

        # Structure
        assert restored.state == "Root"
        assert restored.visits == 100
        assert restored.value == 50.0
        assert len(restored.children) == 2

        # Child 1 subtree
        r_child1 = restored.children[0]
        assert r_child1.state == "Path 1"
        assert r_child1.parent is restored
        assert r_child1.depth == 1
        assert r_child1.visits == 60

        # Terminal grandchild
        r_terminal = r_child1.children[0]
        assert r_terminal.is_terminal is True
        assert r_terminal.answer == "42"
        assert r_terminal.depth == 2
        assert r_terminal.parent is r_child1

        # Child 2
        r_child2 = restored.children[1]
        assert r_child2.depth == 1

    def test_to_dict_preserves_continuations(self):
        """Continuation cache survives roundtrip."""
        node = Node(state=State("State"))
        node.set_continuations(["c1", "c2", "c3"])
        node._continuation_index = 1

        data = node.to_dict()
        restored = Node.from_dict(data)

        assert restored._continuations == ["c1", "c2", "c3"]
        assert restored._continuation_index == 1

    def test_from_dict_with_parent(self):
        """from_dict accepts a parent parameter."""
        parent = Node(state=State("parent"))
        data = {"state": "child", "children": []}
        child = Node.from_dict(data, parent=parent)
        assert child.parent is parent
        assert child.depth == parent.depth + 1

    def test_to_dict_iterative_deep_tree(self):
        """to_dict works on a deep tree without stack overflow (iterative)."""
        root = Node(state=State("root"))
        current = root
        for i in range(200):
            current = current.add_child(state=State(f"level-{i+1}"))

        data = root.to_dict()

        # Walk the dict to verify depth
        d = data
        depth = 0
        while d.get("children"):
            d = d["children"][0]
            depth += 1
        assert depth == 200

    def test_from_dict_iterative_deep_tree(self):
        """from_dict works on a deeply nested dict without stack overflow."""
        # Build deeply nested dict manually
        data = {"state": "root", "children": []}
        current = data
        for i in range(200):
            child = {"state": f"level-{i+1}", "children": []}
            current["children"] = [child]
            current = child

        restored = Node.from_dict(data)

        # Walk the tree to verify
        node = restored
        depth = 0
        while node.children:
            node = node.children[0]
            depth += 1
        assert depth == 200
        assert node.depth == 200


class TestCountNodes:
    """Tests for count_nodes iterative method."""

    def test_count_nodes_single(self):
        """Single root node has count 1."""
        root = Node(state=State("root"))
        assert root.count_nodes() == 1

    def test_count_nodes_tree(self):
        """Count nodes in a small tree."""
        root = Node(state=State("root"))
        c1 = root.add_child(state=State("c1"))
        c2 = root.add_child(state=State("c2"))
        c1.add_child(state=State("c1.1"))
        c1.add_child(state=State("c1.2"))
        c2.add_child(state=State("c2.1"))
        # root + c1 + c2 + c1.1 + c1.2 + c2.1 = 6
        assert root.count_nodes() == 6

    def test_count_nodes_iterative(self):
        """count_nodes works on a 10-level deep tree without stack overflow."""
        root = Node(state=State("root"))
        current = root
        for i in range(10):
            current = current.add_child(state=State(f"level-{i+1}"))
        # 11 nodes: root + 10 levels
        assert root.count_nodes() == 11

    def test_count_nodes_very_deep(self):
        """count_nodes on a very deep tree (500 levels) -- proves iterative."""
        root = Node(state=State("root"))
        current = root
        for i in range(500):
            current = current.add_child(state=State(f"d{i}"))
        assert root.count_nodes() == 501


class TestMaxDepth:
    """Tests for max_depth iterative method."""

    def test_max_depth_single(self):
        """Single root node has max_depth 0."""
        root = Node(state=State("root"))
        assert root.max_depth() == 0

    def test_max_depth_tree(self):
        """max_depth returns the deepest level."""
        root = Node(state=State("root"))
        c1 = root.add_child(state=State("c1"))
        c2 = root.add_child(state=State("c2"))
        c1.add_child(state=State("c1.1"))
        # Deepest: root -> c1 -> c1.1 = depth 2
        assert root.max_depth() == 2

    def test_max_depth_iterative(self):
        """max_depth works on a deep tree without stack overflow."""
        root = Node(state=State("root"))
        current = root
        for i in range(100):
            current = current.add_child(state=State(f"level-{i+1}"))
        assert root.max_depth() == 100
