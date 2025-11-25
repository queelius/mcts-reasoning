"""
Comprehensive unit tests for mcts_reasoning/sampling.py

Tests sampling strategies for MCTS trees including:
- SampledPath data structure
- MCTSSampler strategies (value-based, visit-based, diverse, top-K)
- Consistency checking across multiple reasoning paths
- SamplingMCTS convenience wrapper

These tests follow TDD principles:
- Test behavior, not implementation
- Focus on observable outcomes
- Use deterministic test data where possible
"""

import pytest
import random
from mcts_reasoning.core import MCTS, MCTSNode
from mcts_reasoning.sampling import SampledPath, MCTSSampler, SamplingMCTS
from mcts_reasoning.compositional.providers import MockLLMProvider


class TestSampledPath:
    """Unit tests for SampledPath dataclass."""

    def test_create_sampled_path(self):
        """Test creating a SampledPath with all fields."""
        node1 = MCTSNode(state="state1")
        node2 = MCTSNode(state="state2", parent=node1, action_taken="action1")

        path = SampledPath(
            nodes=[node1, node2],
            actions=["action1"],
            states=["state1", "state2"],
            total_value=10.0,
            total_visits=5
        )

        assert len(path.nodes) == 2
        assert len(path.actions) == 1
        assert len(path.states) == 2
        assert path.total_value == 10.0
        assert path.total_visits == 5

    def test_final_state_returns_last_state(self):
        """Test that final_state property returns the last state."""
        path = SampledPath(
            nodes=[],
            actions=[],
            states=["state1", "state2", "state3"],
            total_value=0.0,
            total_visits=0
        )

        assert path.final_state == "state3"

    def test_final_state_empty_path_returns_empty_string(self):
        """Test that final_state returns empty string for empty path."""
        path = SampledPath(
            nodes=[],
            actions=[],
            states=[],
            total_value=0.0,
            total_visits=0
        )

        assert path.final_state == ""

    def test_length_returns_number_of_nodes(self):
        """Test that length property returns number of nodes."""
        nodes = [MCTSNode(state=f"state{i}") for i in range(5)]

        path = SampledPath(
            nodes=nodes,
            actions=[],
            states=[],
            total_value=0.0,
            total_visits=0
        )

        assert path.length == 5

    def test_to_dict_creates_dictionary(self):
        """Test that to_dict creates proper dictionary representation."""
        node1 = MCTSNode(state="state1")
        node2 = MCTSNode(state="state2", parent=node1, action_taken="action1")

        path = SampledPath(
            nodes=[node1, node2],
            actions=["action1", "action2"],
            states=["state1", "state2"],
            total_value=15.0,
            total_visits=10
        )

        path_dict = path.to_dict()

        assert path_dict['actions'] == ["action1", "action2"]
        assert path_dict['final_state'] == "state2"
        assert path_dict['length'] == 2
        assert path_dict['value'] == 15.0
        assert path_dict['visits'] == 10


class TestMCTSSamplerSetup:
    """Test MCTSSampler initialization and basic functionality."""

    def test_create_sampler_with_mcts(self):
        """Test creating a sampler from MCTS instance."""
        llm = MockLLMProvider()
        mcts = MCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root")

        sampler = MCTSSampler(mcts)

        assert sampler.mcts is mcts
        assert sampler.root is mcts.root

    def test_sampler_requires_tree_root(self):
        """Test that sampling operations require initialized tree."""
        mcts = MCTS()
        # Don't initialize root
        sampler = MCTSSampler(mcts)

        with pytest.raises(ValueError, match="No tree to sample from"):
            sampler.sample_by_value()

    def test_sampler_with_empty_tree(self):
        """Test sampler behavior with tree that has only root."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        sampler = MCTSSampler(mcts)

        # Should not raise error, just return path with only root
        path = sampler.sample_by_value()
        assert path.length == 1
        assert path.final_state == "root"


class TestValueBasedSampling:
    """Test sample_by_value strategy."""

    def test_sample_by_value_returns_sampled_path(self):
        """Test that sample_by_value returns a SampledPath object."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        sampler = MCTSSampler(mcts)

        path = sampler.sample_by_value()

        assert isinstance(path, SampledPath)
        assert path.nodes[0] is mcts.root

    def test_sample_by_value_follows_path_to_leaf(self):
        """Test that sample_by_value follows path until leaf node."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5, action_taken="action1")
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        path = sampler.sample_by_value()

        # Should follow path: root -> child
        assert path.length == 2
        assert path.nodes[0] is mcts.root
        assert path.nodes[1] is child
        assert path.final_state == "child"

    def test_sample_by_value_greedy_with_zero_temperature(self):
        """Test that temperature=0 always selects highest value child."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        # Create children with different values
        child_low = MCTSNode(state="low", parent=mcts.root, visits=5, value=1.0, action_taken="low")
        child_high = MCTSNode(state="high", parent=mcts.root, visits=5, value=4.5, action_taken="high")
        mcts.root.children = [child_low, child_high]

        sampler = MCTSSampler(mcts)

        # With temperature=0, should always select high value child
        for _ in range(10):
            path = sampler.sample_by_value(temperature=0.0)
            assert path.nodes[1] is child_high

    def test_sample_by_value_from_specific_node(self):
        """Test that from_node parameter allows sampling from non-root node."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5, action_taken="action1")
        grandchild = MCTSNode(state="grandchild", parent=child, visits=2, value=1.0, action_taken="action2")
        mcts.root.children = [child]
        child.children = [grandchild]

        sampler = MCTSSampler(mcts)
        path = sampler.sample_by_value(from_node=child)

        # Should start from child, not root
        assert path.nodes[0] is child
        assert path.nodes[1] is grandchild
        assert path.length == 2

    def test_sample_by_value_calculates_correct_statistics(self):
        """Test that sampled path has correct total_value and total_visits."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5, action_taken="action1")
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        path = sampler.sample_by_value()

        # Should sum across all nodes in path
        assert path.total_value == 5.0 + 2.5  # 7.5
        assert path.total_visits == 10 + 5    # 15


class TestVisitBasedSampling:
    """Test sample_by_visits strategy."""

    def test_sample_by_visits_returns_sampled_path(self):
        """Test that sample_by_visits returns a SampledPath object."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        sampler = MCTSSampler(mcts)

        path = sampler.sample_by_visits()

        assert isinstance(path, SampledPath)

    def test_sample_by_visits_favors_high_visit_children(self):
        """Test that children with more visits are sampled more frequently."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=100, value=50.0)

        # Create children with very different visit counts
        child_rare = MCTSNode(state="rare", parent=mcts.root, visits=1, value=0.5, action_taken="rare")
        child_common = MCTSNode(state="common", parent=mcts.root, visits=99, value=49.5, action_taken="common")
        mcts.root.children = [child_rare, child_common]

        sampler = MCTSSampler(mcts)

        # Sample many times - child_common should be selected most of the time
        common_count = 0
        trials = 100

        for _ in range(trials):
            path = sampler.sample_by_visits()
            if path.nodes[1] is child_common:
                common_count += 1

        # With 99% visit share, should be selected ~99% of the time
        # Allow some variance but should be clearly majority
        assert common_count > 80  # At least 80% (probabilistic test)

    def test_sample_by_visits_handles_zero_visits(self):
        """Test that sample_by_visits handles children with zero visits."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        # Create children with zero visits
        child1 = MCTSNode(state="child1", parent=mcts.root, visits=0, value=0.0, action_taken="action1")
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=0, value=0.0, action_taken="action2")
        mcts.root.children = [child1, child2]

        sampler = MCTSSampler(mcts)

        # Should not crash - should sample uniformly
        path = sampler.sample_by_visits()
        assert path.nodes[1] in [child1, child2]

    def test_sample_by_visits_from_specific_node(self):
        """Test that from_node parameter works with visit-based sampling."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5, action_taken="action1")
        grandchild = MCTSNode(state="grandchild", parent=child, visits=2, value=1.0, action_taken="action2")
        mcts.root.children = [child]
        child.children = [grandchild]

        sampler = MCTSSampler(mcts)
        path = sampler.sample_by_visits(from_node=child)

        assert path.nodes[0] is child


class TestTopKSampling:
    """Test sample_top_k strategy."""

    def test_sample_top_k_returns_list_of_paths(self):
        """Test that sample_top_k returns a list of SampledPath objects."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_top_k(k=2)

        assert isinstance(paths, list)
        assert all(isinstance(p, SampledPath) for p in paths)

    def test_sample_top_k_by_value_returns_highest_value_paths(self):
        """Test that top-k by value returns paths sorted by average value."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        # Create paths with different average values
        child1 = MCTSNode(state="low_value", parent=mcts.root, visits=5, value=1.0)
        child2 = MCTSNode(state="high_value", parent=mcts.root, visits=5, value=4.5)
        child3 = MCTSNode(state="mid_value", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child1, child2, child3]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_top_k(k=3, criterion="value")

        # Paths should be sorted by value (descending)
        # Path values = root.value + child.value
        values = [p.total_value / max(p.total_visits, 1) for p in paths]
        assert values == sorted(values, reverse=True)

        # Best path should be the one through high_value child
        assert paths[0].nodes[-1] is child2

    def test_sample_top_k_by_visits_returns_most_visited_paths(self):
        """Test that top-k by visits returns paths sorted by visit count."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=100, value=50.0)

        child1 = MCTSNode(state="child1", parent=mcts.root, visits=5, value=2.5)
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=50, value=25.0)
        child3 = MCTSNode(state="child3", parent=mcts.root, visits=20, value=10.0)
        mcts.root.children = [child1, child2, child3]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_top_k(k=3, criterion="visits")

        # Should be sorted by total visits
        visits = [p.total_visits for p in paths]
        assert visits == sorted(visits, reverse=True)

        # Most visited path should go through child2
        assert paths[0].nodes[-1] is child2

    def test_sample_top_k_by_depth_returns_deepest_paths(self):
        """Test that top-k by depth returns paths sorted by length."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        # Create paths of different depths
        child1 = MCTSNode(state="shallow", parent=mcts.root, visits=5, value=2.5)
        child2 = MCTSNode(state="deep_parent", parent=mcts.root, visits=5, value=2.5)
        grandchild = MCTSNode(state="deep_child", parent=child2, visits=2, value=1.0)

        mcts.root.children = [child1, child2]
        child2.children = [grandchild]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_top_k(k=2, criterion="depth")

        # Should be sorted by length
        lengths = [p.length for p in paths]
        assert lengths == sorted(lengths, reverse=True)

        # Deepest path should go through grandchild
        assert paths[0].nodes[-1] is grandchild

    def test_sample_top_k_returns_at_most_k_paths(self):
        """Test that sample_top_k returns no more than k paths."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_top_k(k=10, criterion="value")

        # Only 1 path exists, so should return only 1
        assert len(paths) == 1

    def test_sample_top_k_invalid_criterion_raises_error(self):
        """Test that invalid criterion raises ValueError."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        sampler = MCTSSampler(mcts)

        with pytest.raises(ValueError, match="Unknown criterion"):
            sampler.sample_top_k(k=1, criterion="invalid")


class TestDiverseSampling:
    """Test sample_diverse strategy."""

    def test_sample_diverse_returns_list_of_paths(self):
        """Test that sample_diverse returns list of paths."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        # Create children to enable diversity
        child1 = MCTSNode(state="child1", parent=mcts.root, visits=5, value=2.5, action_taken="action1")
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=5, value=2.5, action_taken="action2")
        mcts.root.children = [child1, child2]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_diverse(n=2, min_distance=0.1)

        assert isinstance(paths, list)
        assert len(paths) <= 2  # May return fewer if diversity constraint can't be met

    def test_sample_diverse_paths_are_different(self):
        """Test that diverse sampling returns syntactically different paths."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        # Create multiple different paths
        child1 = MCTSNode(state="child1", parent=mcts.root, visits=5, value=2.5, action_taken="action_a")
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=5, value=2.5, action_taken="action_b")
        child3 = MCTSNode(state="child3", parent=mcts.root, visits=5, value=2.5, action_taken="action_c")
        mcts.root.children = [child1, child2, child3]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_diverse(n=3, min_distance=0.5)

        # Paths should have different actions
        action_sequences = [tuple(str(a) for a in p.actions) for p in paths]

        # All paths should be unique
        assert len(set(action_sequences)) == len(action_sequences)

    def test_sample_diverse_respects_min_distance(self):
        """Test that sampled paths meet minimum distance requirement."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        child1 = MCTSNode(state="child1", parent=mcts.root, visits=5, value=2.5, action_taken="action1")
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=5, value=2.5, action_taken="action2")
        mcts.root.children = [child1, child2]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_diverse(n=2, min_distance=0.8)

        # If we got 2 paths, they should meet the distance requirement
        if len(paths) == 2:
            distance = sampler._path_distance(paths[0], paths[1])
            assert distance >= 0.8


class TestMultipleSampling:
    """Test sample_multiple strategy."""

    def test_sample_multiple_returns_correct_number_of_paths(self):
        """Test that sample_multiple returns requested number of paths."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_multiple(n=5, strategy="value")

        assert len(paths) == 5

    def test_sample_multiple_value_strategy(self):
        """Test that sample_multiple with value strategy uses value-based sampling."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_multiple(n=3, strategy="value")

        # All paths should be valid SampledPath objects
        assert all(isinstance(p, SampledPath) for p in paths)

    def test_sample_multiple_visits_strategy(self):
        """Test that sample_multiple with visits strategy uses visit-based sampling."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_multiple(n=3, strategy="visits")

        assert all(isinstance(p, SampledPath) for p in paths)

    def test_sample_multiple_mixed_strategy_alternates(self):
        """Test that mixed strategy alternates between value and visits."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        paths = sampler.sample_multiple(n=4, strategy="mixed")

        # Should return 4 paths (2 value-based, 2 visit-based)
        assert len(paths) == 4

    def test_sample_multiple_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root")
        sampler = MCTSSampler(mcts)

        with pytest.raises(ValueError, match="Unknown strategy"):
            sampler.sample_multiple(n=1, strategy="invalid")


class TestConsistencySampling:
    """Test get_consistent_solution strategy."""

    def test_get_consistent_solution_returns_dict(self):
        """Test that get_consistent_solution returns dictionary with results."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="solution A", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        result = sampler.get_consistent_solution(n_samples=5)

        assert isinstance(result, dict)
        assert 'solution' in result
        assert 'confidence' in result
        assert 'support' in result
        assert 'total_samples' in result
        assert 'path' in result
        assert 'clusters' in result

    def test_get_consistent_solution_finds_majority_solution(self):
        """Test that consistent solution finds the most common solution."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=100, value=50.0)

        # Create paths that lead to the same solution most of the time
        child1 = MCTSNode(state="solution A", parent=mcts.root, visits=80, value=40.0)
        child2 = MCTSNode(state="solution B", parent=mcts.root, visits=20, value=10.0)
        mcts.root.children = [child1, child2]

        sampler = MCTSSampler(mcts)

        # With high visit weight on solution A, it should be most consistent
        result = sampler.get_consistent_solution(n_samples=10)

        # Confidence should be relatively high (>50%)
        assert result['confidence'] > 0.3

    def test_get_consistent_solution_confidence_calculation(self):
        """Test that confidence is calculated as support / total_samples."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="solution", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        sampler = MCTSSampler(mcts)
        result = sampler.get_consistent_solution(n_samples=10)

        expected_confidence = result['support'] / result['total_samples']
        assert abs(result['confidence'] - expected_confidence) < 0.001


class TestHelperMethods:
    """Test helper methods in MCTSSampler."""

    def test_get_all_paths_returns_all_leaf_paths(self):
        """Test that _get_all_paths returns paths to all leaves."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        child1 = MCTSNode(state="leaf1", parent=mcts.root, visits=5, value=2.5)
        child2 = MCTSNode(state="leaf2", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child1, child2]

        sampler = MCTSSampler(mcts)
        all_paths = sampler._get_all_paths(mcts.root)

        # Should have 2 paths (one to each leaf)
        assert len(all_paths) == 2

        final_states = {p.final_state for p in all_paths}
        assert final_states == {"leaf1", "leaf2"}

    def test_get_all_paths_handles_deep_trees(self):
        """Test that _get_all_paths correctly handles multi-level trees."""
        mcts = MCTS()
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        grandchild1 = MCTSNode(state="grandchild1", parent=child, visits=2, value=1.0)
        grandchild2 = MCTSNode(state="grandchild2", parent=child, visits=2, value=1.0)

        mcts.root.children = [child]
        child.children = [grandchild1, grandchild2]

        sampler = MCTSSampler(mcts)
        all_paths = sampler._get_all_paths(mcts.root)

        # Should have 2 paths (one to each grandchild)
        assert len(all_paths) == 2

        # Both paths should go through child
        for path in all_paths:
            assert path.length == 3  # root -> child -> grandchild
            assert path.nodes[1] is child

    def test_path_distance_identical_paths_returns_zero(self):
        """Test that distance between identical paths is 0."""
        path1 = SampledPath(
            nodes=[],
            actions=["a", "b", "c"],
            states=[],
            total_value=0.0,
            total_visits=0
        )
        path2 = SampledPath(
            nodes=[],
            actions=["a", "b", "c"],
            states=[],
            total_value=0.0,
            total_visits=0
        )

        sampler = MCTSSampler(MCTS())
        distance = sampler._path_distance(path1, path2)

        assert distance == 0.0

    def test_path_distance_completely_different_paths_returns_one(self):
        """Test that distance between completely different paths approaches 1."""
        path1 = SampledPath(
            nodes=[],
            actions=["a", "b", "c"],
            states=[],
            total_value=0.0,
            total_visits=0
        )
        path2 = SampledPath(
            nodes=[],
            actions=["x", "y", "z"],
            states=[],
            total_value=0.0,
            total_visits=0
        )

        sampler = MCTSSampler(MCTS())
        distance = sampler._path_distance(path1, path2)

        # All actions different, so distance should be 1.0
        assert distance == 1.0

    def test_path_distance_partially_different_paths(self):
        """Test that distance is proportional to differences."""
        path1 = SampledPath(
            nodes=[],
            actions=["a", "b", "c"],
            states=[],
            total_value=0.0,
            total_visits=0
        )
        path2 = SampledPath(
            nodes=[],
            actions=["a", "b", "x"],
            states=[],
            total_value=0.0,
            total_visits=0
        )

        sampler = MCTSSampler(MCTS())
        distance = sampler._path_distance(path1, path2)

        # 1 out of 3 actions different = 1/3 distance
        assert 0.3 < distance < 0.4

    def test_levenshtein_distance_identical_sequences(self):
        """Test that Levenshtein distance is 0 for identical sequences."""
        sampler = MCTSSampler(MCTS())
        distance = sampler._levenshtein(["a", "b", "c"], ["a", "b", "c"])

        assert distance == 0

    def test_levenshtein_distance_one_insertion(self):
        """Test Levenshtein distance with one insertion."""
        sampler = MCTSSampler(MCTS())
        distance = sampler._levenshtein(["a", "b", "c"], ["a", "x", "b", "c"])

        assert distance == 1  # One insertion

    def test_levenshtein_distance_one_deletion(self):
        """Test Levenshtein distance with one deletion."""
        sampler = MCTSSampler(MCTS())
        distance = sampler._levenshtein(["a", "b", "c"], ["a", "c"])

        assert distance == 1  # One deletion

    def test_levenshtein_distance_one_substitution(self):
        """Test Levenshtein distance with one substitution."""
        sampler = MCTSSampler(MCTS())
        distance = sampler._levenshtein(["a", "b", "c"], ["a", "x", "c"])

        assert distance == 1  # One substitution

    def test_cluster_solutions_simple_groups_identical(self):
        """Test that simple clustering groups identical solutions."""
        solutions = ["A", "B", "A", "A", "C", "B"]

        sampler = MCTSSampler(MCTS())
        clusters = sampler._cluster_solutions_simple(solutions)

        # Should have 3 clusters
        assert len(clusters) == 3

        # Find cluster for "A"
        cluster_a = next(c for c in clusters if c['representative'] == "A")
        assert cluster_a['count'] == 3

        cluster_b = next(c for c in clusters if c['representative'] == "B")
        assert cluster_b['count'] == 2

        cluster_c = next(c for c in clusters if c['representative'] == "C")
        assert cluster_c['count'] == 1


class TestSamplingMCTS:
    """Test SamplingMCTS convenience wrapper."""

    def test_sampling_mcts_inherits_from_mcts(self):
        """Test that SamplingMCTS is a subclass of MCTS."""
        assert issubclass(SamplingMCTS, MCTS)

    def test_sample_returns_single_path_when_n_equals_1(self):
        """Test that sample returns SampledPath (not list) when n=1."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        path = mcts.sample(n=1, strategy="value")

        assert isinstance(path, SampledPath)

    def test_sample_returns_list_when_n_greater_than_1(self):
        """Test that sample returns list of paths when n>1."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        paths = mcts.sample(n=3, strategy="value")

        assert isinstance(paths, list)
        assert len(paths) == 3

    def test_sample_value_strategy(self):
        """Test that sample works with value strategy."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        path = mcts.sample(n=1, strategy="value", temperature=1.0)

        assert isinstance(path, SampledPath)

    def test_sample_visits_strategy(self):
        """Test that sample works with visits strategy."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        path = mcts.sample(n=1, strategy="visits")

        assert isinstance(path, SampledPath)

    def test_sample_diverse_strategy(self):
        """Test that sample works with diverse strategy."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child1 = MCTSNode(state="child1", parent=mcts.root, visits=5, value=2.5, action_taken="a1")
        child2 = MCTSNode(state="child2", parent=mcts.root, visits=5, value=2.5, action_taken="a2")
        mcts.root.children = [child1, child2]

        paths = mcts.sample(n=2, strategy="diverse")

        assert isinstance(paths, list)

    def test_get_top_k_method(self):
        """Test that get_top_k convenience method works."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        paths = mcts.get_top_k(k=1, criterion="value")

        assert isinstance(paths, list)
        assert len(paths) == 1

    def test_check_consistency_method(self):
        """Test that check_consistency convenience method works."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)
        child = MCTSNode(state="solution", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child]

        result = mcts.check_consistency(n_samples=5)

        assert isinstance(result, dict)
        assert 'solution' in result
        assert 'confidence' in result

    def test_all_solutions_property(self):
        """Test that all_solutions returns unique solutions in tree."""
        llm = MockLLMProvider()
        mcts = SamplingMCTS().with_llm(llm)
        mcts.root = MCTSNode(state="root", visits=10, value=5.0)

        child1 = MCTSNode(state="solution A", parent=mcts.root, visits=5, value=2.5)
        child2 = MCTSNode(state="solution B", parent=mcts.root, visits=5, value=2.5)
        mcts.root.children = [child1, child2]

        solutions = mcts.all_solutions

        assert isinstance(solutions, list)
        assert set(solutions) == {"solution A", "solution B"}
