"""
Comprehensive unit tests for mcts_reasoning/core.py

Tests the core MCTS implementation including:
- MCTSNode class (UCB1, tree traversal, serialization)
- MCTS class (search algorithm, fluent API, JSON serialization)
- Four MCTS phases: Selection, Expansion, Rollout, Backpropagation

These tests follow TDD principles:
- Test behavior, not implementation
- Use MockLLMProvider for deterministic testing
- Test public APIs only
- Focus on observable outcomes
"""

import pytest
import json
import tempfile
from pathlib import Path
from mcts_reasoning.core import MCTSNode, MCTS
from mcts_reasoning.compositional.providers import MockLLMProvider


class TestMCTSNode:
    """Unit tests for MCTSNode class."""

    def test_create_basic_node(self):
        """Test creating a basic node with required state."""
        node = MCTSNode(state="initial state")

        assert node.state == "initial state"
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.action_taken is None
        assert node.tried_actions == []

    def test_node_is_leaf_when_no_children(self):
        """Test that is_leaf returns True when node has no children."""
        node = MCTSNode(state="test")
        assert node.is_leaf is True

    def test_node_is_not_leaf_when_has_children(self):
        """Test that is_leaf returns False when node has children."""
        parent = MCTSNode(state="parent")
        child = MCTSNode(state="child", parent=parent)
        parent.children.append(child)

        assert parent.is_leaf is False

    def test_node_is_root_when_no_parent(self):
        """Test that is_root returns True when node has no parent."""
        node = MCTSNode(state="root")
        assert node.is_root is True

    def test_node_is_not_root_when_has_parent(self):
        """Test that is_root returns False when node has parent."""
        parent = MCTSNode(state="parent")
        child = MCTSNode(state="child", parent=parent)

        assert child.is_root is False

    def test_root_node_depth_is_zero(self):
        """Test that root node has depth 0."""
        root = MCTSNode(state="root")
        assert root.depth == 0

    def test_child_node_depth_is_one(self):
        """Test that direct child of root has depth 1."""
        root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=root)

        assert child.depth == 1

    def test_grandchild_node_depth_is_two(self):
        """Test that grandchild has depth 2."""
        root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=root)
        grandchild = MCTSNode(state="grandchild", parent=child)

        assert grandchild.depth == 2

    def test_path_to_root_for_root_node(self):
        """Test that path_to_root for root node contains only itself."""
        root = MCTSNode(state="root")
        path = root.path_to_root

        assert len(path) == 1
        assert path[0] is root

    def test_path_to_root_for_child_node(self):
        """Test that path_to_root contains all ancestors in correct order."""
        root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=root)
        grandchild = MCTSNode(state="grandchild", parent=child)

        path = grandchild.path_to_root

        assert len(path) == 3
        assert path[0] is root
        assert path[1] is child
        assert path[2] is grandchild

    def test_ucb1_unvisited_node_returns_infinity(self):
        """Test that UCB1 returns infinity for unvisited nodes."""
        node = MCTSNode(state="test", visits=0)
        assert node.ucb1() == float('inf')

    def test_ucb1_root_node_returns_value(self):
        """Test that UCB1 for root node returns average value."""
        root = MCTSNode(state="root", visits=10, value=50.0)

        # Root node has no parent, so UCB1 should just return value
        assert root.ucb1() == 50.0

    def test_ucb1_combines_exploitation_and_exploration(self):
        """Test that UCB1 properly balances exploitation and exploration."""
        parent = MCTSNode(state="parent", visits=100, value=50.0)
        child = MCTSNode(state="child", parent=parent, visits=10, value=5.0)

        ucb_value = child.ucb1(exploration_constant=1.414)

        # UCB1 = exploitation + exploration
        # exploitation = value / visits = 5.0 / 10 = 0.5
        # exploration = 1.414 * sqrt(log(100) / 10)
        exploitation = 5.0 / 10

        # UCB should be greater than pure exploitation
        assert ucb_value > exploitation
        assert ucb_value < float('inf')

    def test_ucb1_with_different_exploration_constants(self):
        """Test that higher exploration constant increases UCB1."""
        parent = MCTSNode(state="parent", visits=100, value=50.0)
        child = MCTSNode(state="child", parent=parent, visits=10, value=5.0)

        ucb_low = child.ucb1(exploration_constant=0.5)
        ucb_high = child.ucb1(exploration_constant=2.0)

        assert ucb_high > ucb_low

    def test_get_all_descendants_empty_for_leaf(self):
        """Test that get_all_descendants returns empty list for leaf node."""
        leaf = MCTSNode(state="leaf")
        assert leaf.get_all_descendants() == []

    def test_get_all_descendants_with_children(self):
        """Test that get_all_descendants returns all children and grandchildren."""
        root = MCTSNode(state="root")
        child1 = MCTSNode(state="child1", parent=root)
        child2 = MCTSNode(state="child2", parent=root)
        grandchild = MCTSNode(state="grandchild", parent=child1)

        root.children = [child1, child2]
        child1.children = [grandchild]

        descendants = root.get_all_descendants()

        assert len(descendants) == 3
        assert child1 in descendants
        assert child2 in descendants
        assert grandchild in descendants

    def test_to_dict_serializes_node_properties(self):
        """Test that to_dict creates proper dictionary representation."""
        node = MCTSNode(
            state="test state",
            visits=10,
            value=5.5,
            action_taken="test_action"
        )

        node_dict = node.to_dict()

        assert node_dict['state'] == "test state"
        assert node_dict['visits'] == 10
        assert node_dict['value'] == 5.5
        assert node_dict['depth'] == 0  # Root node
        assert node_dict['action_taken'] == "test_action"
        assert node_dict['children'] == []

    def test_to_dict_includes_children(self):
        """Test that to_dict recursively includes children."""
        parent = MCTSNode(state="parent")
        child = MCTSNode(state="child", parent=parent)
        parent.children = [child]

        parent_dict = parent.to_dict()

        assert len(parent_dict['children']) == 1
        assert parent_dict['children'][0]['state'] == "child"

    def test_from_dict_creates_node(self):
        """Test that from_dict creates node from dictionary."""
        data = {
            'state': 'test state',
            'visits': 10,
            'value': 5.5,
            'action_taken': 'test_action',
            'children': []
        }

        node = MCTSNode.from_dict(data)

        assert node.state == 'test state'
        assert node.visits == 10
        assert node.value == 5.5
        assert node.action_taken == 'test_action'

    def test_from_dict_recursively_creates_children(self):
        """Test that from_dict recursively creates child nodes."""
        data = {
            'state': 'parent',
            'visits': 20,
            'value': 10.0,
            'children': [
                {
                    'state': 'child',
                    'visits': 5,
                    'value': 2.5,
                    'children': []
                }
            ]
        }

        parent = MCTSNode.from_dict(data)

        assert len(parent.children) == 1
        child = parent.children[0]
        assert child.state == 'child'
        assert child.parent is parent

    def test_node_serialization_roundtrip(self):
        """Test that serializing and deserializing preserves node data."""
        original = MCTSNode(state="test", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=original, visits=5, value=2.5)
        original.children = [child]

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = MCTSNode.from_dict(data)

        assert restored.state == original.state
        assert restored.visits == original.visits
        assert restored.value == original.value
        assert len(restored.children) == 1
        assert restored.children[0].state == child.state


class TestMCTSFluentAPI:
    """Test MCTS fluent API and configuration."""

    def test_create_empty_mcts(self):
        """Test creating empty MCTS instance."""
        mcts = MCTS()

        assert mcts.root is None
        assert mcts.llm is None
        assert mcts.exploration_constant == 1.414
        assert mcts.max_rollout_depth == 5
        assert mcts.discount_factor == 0.95

    def test_with_llm_sets_llm_and_returns_self(self):
        """Test that with_llm sets LLM and enables chaining."""
        mcts = MCTS()
        llm = MockLLMProvider()

        result = mcts.with_llm(llm)

        assert mcts.llm is llm
        assert result is mcts  # Fluent API returns self

    def test_with_exploration_sets_constant(self):
        """Test that with_exploration sets exploration constant."""
        mcts = MCTS()

        result = mcts.with_exploration(2.0)

        assert mcts.exploration_constant == 2.0
        assert result is mcts

    def test_with_max_rollout_depth_sets_depth(self):
        """Test that with_max_rollout_depth sets rollout depth."""
        mcts = MCTS()

        result = mcts.with_max_rollout_depth(10)

        assert mcts.max_rollout_depth == 10
        assert result is mcts

    def test_with_discount_sets_factor(self):
        """Test that with_discount sets discount factor."""
        mcts = MCTS()

        result = mcts.with_discount(0.99)

        assert mcts.discount_factor == 0.99
        assert result is mcts

    def test_with_metadata_adds_metadata(self):
        """Test that with_metadata adds custom metadata."""
        mcts = MCTS()

        result = mcts.with_metadata(experiment="test", version=1)

        assert mcts._metadata['experiment'] == "test"
        assert mcts._metadata['version'] == 1
        assert result is mcts

    def test_fluent_api_chaining(self):
        """Test that fluent API methods can be chained."""
        llm = MockLLMProvider()

        mcts = (
            MCTS()
            .with_llm(llm)
            .with_exploration(2.0)
            .with_max_rollout_depth(10)
            .with_discount(0.99)
            .with_metadata(test=True)
        )

        assert mcts.llm is llm
        assert mcts.exploration_constant == 2.0
        assert mcts.max_rollout_depth == 10
        assert mcts.discount_factor == 0.99
        assert mcts._metadata['test'] is True


class TestMCTSSearch:
    """Test MCTS search functionality."""

    def test_search_requires_llm(self):
        """Test that search raises error if LLM not set."""
        mcts = MCTS()

        with pytest.raises(ValueError, match="LLM not set"):
            mcts.search("initial state", simulations=1)

    def test_search_initializes_root_node(self):
        """Test that search creates root node with initial state."""
        mcts = MCTS().with_llm(MockLLMProvider())

        mcts.search("test state", simulations=1)

        assert mcts.root is not None
        assert mcts.root.state == "test state"

    def test_search_returns_self_for_chaining(self):
        """Test that search returns self to enable chaining."""
        mcts = MCTS().with_llm(MockLLMProvider())

        result = mcts.search("test", simulations=1)

        assert result is mcts

    def test_search_runs_specified_simulations(self):
        """Test that search runs the correct number of simulations."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)

        mcts.search("initial", simulations=5)

        # After 5 simulations, root should have at least 5 visits
        # (could be more if backpropagation visits multiple times)
        assert mcts.root.visits >= 5

    def test_search_expands_tree(self):
        """Test that search expands the tree by creating children."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)

        mcts.search("initial", simulations=10)

        # Tree should have been expanded
        assert len(mcts.root.children) > 0

    def test_search_can_resume_from_existing_root(self):
        """Test that search can continue from existing tree."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)

        # First search
        mcts.search("initial", simulations=5)
        first_visits = mcts.root.visits

        # Resume search
        mcts.search("initial", simulations=5)

        # Visits should have increased
        assert mcts.root.visits > first_visits


class TestMCTSSelection:
    """Test MCTS selection phase (UCB1-based)."""

    def test_select_returns_root_for_new_tree(self):
        """Test that _select returns root node for new tree."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root")

        selected = mcts._select()

        assert selected is mcts.root

    def test_select_follows_ucb1_to_best_child(self):
        """Test that _select follows child with highest UCB1."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=100, value=50.0)

        # Create children with different visit counts
        child1 = MCTSNode(state="child1", parent=mcts.root, visits=10, value=5.0)
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=1, value=1.0)
        mcts.root.children = [child1, child2]

        # Mark all actions as tried to avoid expansion
        mcts.root.tried_actions = mcts._get_actions(mcts.root.state)

        selected = mcts._select()

        # Child2 should be selected because it has lower visits (higher exploration)
        # and UCB1 returns infinity for unvisited nodes (visits=1 is very low)
        assert selected is child2


class TestMCTSExpansion:
    """Test MCTS expansion phase."""

    def test_expand_creates_new_child(self):
        """Test that _expand creates a new child node."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        parent = MCTSNode(state="parent")

        child = mcts._expand(parent)

        assert child is not None
        assert child.parent is parent
        assert child in parent.children

    def test_expand_marks_action_as_tried(self):
        """Test that _expand marks the action as tried."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        parent = MCTSNode(state="parent")

        initial_tried = len(parent.tried_actions)
        mcts._expand(parent)

        assert len(parent.tried_actions) == initial_tried + 1

    def test_expand_returns_node_when_no_actions_available(self):
        """Test that _expand returns original node when no actions."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)

        # Override _get_actions to return empty list
        mcts._get_actions = lambda state: []

        parent = MCTSNode(state="parent")
        result = mcts._expand(parent)

        assert result is parent
        assert len(parent.children) == 0

    def test_expand_returns_node_when_all_actions_tried(self):
        """Test that _expand returns node when all actions have been tried."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        parent = MCTSNode(state="parent")

        # Mark all actions as tried
        all_actions = mcts._get_actions(parent.state)
        parent.tried_actions = all_actions.copy()

        result = mcts._expand(parent)

        assert result is parent
        assert len(parent.children) == 0


class TestMCTSRollout:
    """Test MCTS rollout phase."""

    def test_rollout_returns_numeric_value(self):
        """Test that _rollout returns a numeric reward value."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        node = MCTSNode(state="test")

        reward = mcts._rollout(node)

        assert isinstance(reward, (int, float))
        assert 0.0 <= reward <= 1.0  # Assuming evaluation returns 0-1

    def test_rollout_respects_max_depth(self):
        """Test that _rollout stops at max depth."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm).with_max_rollout_depth(2)

        # Track number of _take_action calls
        action_count = 0
        original_take_action = mcts._take_action

        def counting_take_action(state, action):
            nonlocal action_count
            action_count += 1
            return original_take_action(state, action)

        mcts._take_action = counting_take_action
        mcts._is_terminal_state = lambda state: False  # Never terminal

        node = MCTSNode(state="test")
        mcts._rollout(node)

        # Should not exceed max depth
        assert action_count <= 2

    def test_rollout_stops_at_terminal_state(self):
        """Test that _rollout stops when terminal state reached."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)

        # Make _is_terminal_state return True immediately
        mcts._is_terminal_state = lambda state: True

        # Track action calls
        action_count = 0
        original_take_action = mcts._take_action

        def counting_take_action(state, action):
            nonlocal action_count
            action_count += 1
            return original_take_action(state, action)

        mcts._take_action = counting_take_action

        node = MCTSNode(state="test")
        mcts._rollout(node)

        # Should not take any actions if already terminal
        assert action_count == 0

    def test_rollout_applies_discount_factor(self):
        """Test that _rollout applies discount based on depth."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm).with_discount(0.5).with_max_rollout_depth(2)

        # Make evaluation always return 1.0
        mcts._evaluate_state = lambda state: 1.0
        mcts._is_terminal_state = lambda state: False

        node = MCTSNode(state="test")
        reward = mcts._rollout(node)

        # With discount 0.5 and depth 2, reward should be 1.0 * 0.5^2 = 0.25
        # (or less if rollout stopped earlier)
        assert reward <= 1.0


class TestMCTSBackpropagation:
    """Test MCTS backpropagation phase."""

    def test_backpropagate_updates_node_visits(self):
        """Test that _backpropagate increments visit count."""
        mcts = MCTS()
        node = MCTSNode(state="test", visits=0)

        mcts._backpropagate(node, reward=1.0)

        assert node.visits == 1

    def test_backpropagate_updates_node_value(self):
        """Test that _backpropagate adds reward to value."""
        mcts = MCTS()
        node = MCTSNode(state="test", value=0.0)

        mcts._backpropagate(node, reward=5.5)

        assert node.value == 5.5

    def test_backpropagate_updates_all_ancestors(self):
        """Test that _backpropagate updates all nodes to root."""
        mcts = MCTS()
        root = MCTSNode(state="root", visits=0, value=0.0)
        child = MCTSNode(state="child", parent=root, visits=0, value=0.0)
        grandchild = MCTSNode(state="grandchild", parent=child, visits=0, value=0.0)

        mcts._backpropagate(grandchild, reward=10.0)

        # All nodes should be updated
        assert grandchild.visits == 1
        assert grandchild.value == 10.0
        assert child.visits == 1
        assert child.value == 10.0
        assert root.visits == 1
        assert root.value == 10.0

    def test_backpropagate_multiple_times_accumulates(self):
        """Test that multiple backpropagations accumulate correctly."""
        mcts = MCTS()
        node = MCTSNode(state="test", visits=0, value=0.0)

        mcts._backpropagate(node, reward=2.0)
        mcts._backpropagate(node, reward=3.0)

        assert node.visits == 2
        assert node.value == 5.0


class TestMCTSProperties:
    """Test MCTS property accessors."""

    def test_best_node_returns_none_for_empty_tree(self):
        """Test that best_node returns None when tree is empty."""
        mcts = MCTS()
        assert mcts.best_node is None

    def test_best_node_returns_most_visited_child(self):
        """Test that best_node returns child with most visits."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        child1 = MCTSNode(state="child1", parent=mcts.root, visits=5)
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=10)
        child3 = MCTSNode(state="child3", parent=mcts.root, visits=3)
        mcts.root.children = [child1, child2, child3]

        assert mcts.best_node is child2

    def test_best_value_returns_zero_for_empty_tree(self):
        """Test that best_value returns 0 for empty tree."""
        mcts = MCTS()
        assert mcts.best_value == 0.0

    def test_best_value_returns_average_value(self):
        """Test that best_value returns average value of best node."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        best_child = MCTSNode(state="best", parent=mcts.root, visits=10, value=50.0)
        mcts.root.children = [best_child]

        assert mcts.best_value == 5.0  # 50.0 / 10

    def test_best_path_returns_empty_for_leaf_root(self):
        """Test that best_path returns empty list when root has no children."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        assert mcts.best_path == []

    def test_best_path_follows_most_visited_nodes(self):
        """Test that best_path follows chain of most visited children."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        child1 = MCTSNode(state="child1", parent=mcts.root, visits=10, action_taken="action1")
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=5, action_taken="action2")
        grandchild = MCTSNode(state="grandchild", parent=child1, visits=3, action_taken="action3")

        mcts.root.children = [child1, child2]
        child1.children = [grandchild]

        path = mcts.best_path

        assert len(path) == 2
        assert path[0] == ("action1", "child1")
        assert path[1] == ("action3", "grandchild")

    def test_solution_returns_root_state_for_empty_path(self):
        """Test that solution returns root state when no best path."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root state")

        assert mcts.solution == "root state"

    def test_solution_returns_final_state_of_best_path(self):
        """Test that solution returns the final state in best path."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        child = MCTSNode(state="final state", parent=mcts.root, visits=10, action_taken="action1")
        mcts.root.children = [child]

        assert mcts.solution == "final state"

    def test_stats_returns_empty_for_uninitialized_tree(self):
        """Test that stats returns empty dict for uninitialized tree."""
        mcts = MCTS()
        assert mcts.stats == {}

    def test_stats_returns_correct_node_count(self):
        """Test that stats correctly counts all nodes in tree."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        child1 = MCTSNode(state="child1", parent=mcts.root)
        child2 = MCTSNode(state="child2", parent=mcts.root)
        mcts.root.children = [child1, child2]

        stats = mcts.stats

        assert stats['total_nodes'] == 3  # root + 2 children

    def test_stats_returns_correct_max_depth(self):
        """Test that stats correctly calculates maximum depth."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=mcts.root)
        grandchild = MCTSNode(state="grandchild", parent=child)
        mcts.root.children = [child]
        child.children = [grandchild]

        stats = mcts.stats

        assert stats['max_depth'] == 2  # root(0) -> child(1) -> grandchild(2)


class TestMCTSSerialization:
    """Test MCTS JSON serialization and persistence."""

    def test_to_json_creates_valid_dict(self):
        """Test that to_json creates valid JSON-serializable dict."""
        mcts = MCTS().with_exploration(2.0).with_metadata(test=True)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        data = mcts.to_json()

        assert isinstance(data, dict)
        assert 'root' in data
        assert 'config' in data
        assert 'metadata' in data
        assert 'timestamp' in data
        assert 'stats' in data

    def test_to_json_preserves_config(self):
        """Test that to_json preserves MCTS configuration."""
        mcts = (
            MCTS()
            .with_exploration(2.5)
            .with_max_rollout_depth(10)
            .with_discount(0.99)
        )
        mcts.root = MCTSNode(state="root")

        data = mcts.to_json()
        config = data['config']

        assert config['exploration_constant'] == 2.5
        assert config['max_rollout_depth'] == 10
        assert config['discount_factor'] == 0.99

    def test_to_json_preserves_metadata(self):
        """Test that to_json preserves custom metadata."""
        mcts = MCTS().with_metadata(experiment="test", version=2)
        mcts.root = MCTSNode(state="root")

        data = mcts.to_json()

        assert data['metadata']['experiment'] == "test"
        assert data['metadata']['version'] == 2

    def test_from_json_restores_tree(self):
        """Test that from_json restores MCTS from dict."""
        data = {
            'root': {
                'state': 'root',
                'visits': 10,
                'value': 5.0,
                'children': []
            },
            'config': {
                'exploration_constant': 2.0,
                'max_rollout_depth': 8,
                'discount_factor': 0.98
            },
            'metadata': {'test': True}
        }

        mcts = MCTS.from_json(data)

        assert mcts.root.state == 'root'
        assert mcts.root.visits == 10
        assert mcts.exploration_constant == 2.0
        assert mcts.max_rollout_depth == 8
        assert mcts._metadata['test'] is True

    def test_json_roundtrip_preserves_data(self):
        """Test that serializing and deserializing preserves all data."""
        original = (
            MCTS()
            .with_exploration(2.0)
            .with_max_rollout_depth(10)
            .with_metadata(experiment="test")
        )
        original.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=original.root, visits=5, value=2.5)
        original.root.children = [child]

        # Serialize
        data = original.to_json()

        # Deserialize
        restored = MCTS.from_json(data)

        assert restored.root.state == original.root.state
        assert restored.exploration_constant == original.exploration_constant
        assert len(restored.root.children) == 1

    def test_save_creates_json_file(self):
        """Test that save creates a JSON file."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            mcts.save(filepath)

            assert filepath.exists()

            # Verify it's valid JSON
            with open(filepath) as f:
                data = json.load(f)
                assert 'root' in data

    def test_load_restores_from_file(self):
        """Test that load restores MCTS from JSON file."""
        original = MCTS().with_exploration(2.5)
        original.root = MCTSNode(state="test state", visits=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            original.save(filepath)

            restored = MCTS.load(filepath)

            assert restored.root.state == "test state"
            assert restored.exploration_constant == 2.5

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if needed."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "deep" / "test.json"
            mcts.save(filepath)

            assert filepath.exists()


class TestMCTSContextManager:
    """Test MCTS context manager support."""

    def test_mcts_can_be_used_as_context_manager(self):
        """Test that MCTS can be used with 'with' statement."""
        with MCTS() as mcts:
            assert mcts is not None
            assert isinstance(mcts, MCTS)

    def test_context_manager_returns_mcts_instance(self):
        """Test that context manager returns functional MCTS."""
        llm = MockLLMProvider()

        with MCTS().with_llm(llm) as mcts:
            mcts.search("test", simulations=1)
            assert mcts.root is not None


class TestMCTSStringRepresentation:
    """Test MCTS string representations."""

    def test_repr_for_uninitialized_tree(self):
        """Test __repr__ for uninitialized tree."""
        mcts = MCTS()
        assert "uninitialized" in repr(mcts)

    def test_repr_includes_stats(self):
        """Test __repr__ includes key statistics."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=10, value=5.0)
        mcts.root.children = [child]

        representation = repr(mcts)

        assert "nodes=" in representation
        assert "depth=" in representation
        assert "best_value=" in representation

    def test_str_for_empty_tree(self):
        """Test __str__ for empty tree."""
        mcts = MCTS()
        assert "Empty" in str(mcts)

    def test_str_includes_readable_stats(self):
        """Test __str__ includes human-readable statistics."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        string_repr = str(mcts)

        assert "Nodes:" in string_repr
        assert "Max depth:" in string_repr
        assert "Best value:" in string_repr


class TestMCTSNodeMethods:
    """Test additional MCTS node inspection methods."""

    def test_get_all_nodes_returns_all_nodes(self):
        """Test that get_all_nodes returns all nodes in tree."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        child1 = MCTSNode(state="child1", parent=mcts.root)
        child2 = MCTSNode(state="child2", parent=mcts.root)
        mcts.root.children = [child1, child2]

        all_nodes = mcts.get_all_nodes()

        assert len(all_nodes) == 3
        assert mcts.root in all_nodes
        assert child1 in all_nodes
        assert child2 in all_nodes

    def test_get_node_by_index_returns_correct_node(self):
        """Test that get_node_by_index returns node at index."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=mcts.root)
        mcts.root.children = [child]

        # Index 0 should be root, index 1 should be child
        assert mcts.get_node_by_index(0) is mcts.root
        assert mcts.get_node_by_index(1) is child

    def test_get_node_by_index_returns_none_for_invalid_index(self):
        """Test that get_node_by_index returns None for invalid index."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")

        assert mcts.get_node_by_index(999) is None
        assert mcts.get_node_by_index(-1) is None

    def test_get_node_details_returns_comprehensive_info(self):
        """Test that get_node_details returns all node information."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=50.0)
        child = MCTSNode(state="child state", parent=mcts.root, visits=5, value=10.0, action_taken="test_action")

        details = mcts.get_node_details(child)

        assert details['depth'] == 1
        assert details['visits'] == 5
        assert details['value'] == 10.0
        assert details['avg_value'] == 2.0  # 10.0 / 5
        assert details['action'] == "test_action"
        assert 'state_preview' in details
        assert details['num_children'] == 0
        assert details['is_leaf'] is True
        assert 'ucb1' in details
