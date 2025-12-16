"""Tests for sampling strategies."""

import pytest
from mcts_reasoning.node import Node
from mcts_reasoning.sampling import PathSampler, SampledPath, SamplingStrategy


class TestSampledPath:
    """Tests for SampledPath dataclass."""

    def test_path_depth(self):
        """Test depth calculation."""
        root = Node(state="Question")
        child1 = root.add_child(state="Question\nStep 1")
        child2 = child1.add_child(state="Question\nStep 1\nStep 2")

        path = SampledPath(
            nodes=[root, child1, child2],
            terminal=None,
            value=0.5,
            visits=1,
            answer=None,
        )
        assert path.depth == 2  # Two steps from root

    def test_path_steps(self):
        """Test step extraction."""
        root = Node(state="What is 2+2?")
        child1 = root.add_child(state="What is 2+2?\nStep 1: Add the numbers.")
        child2 = child1.add_child(
            state="What is 2+2?\nStep 1: Add the numbers.\nStep 2: 2+2=4"
        )

        path = SampledPath(
            nodes=[root, child1, child2],
            terminal=None,
            value=0.5,
            visits=1,
            answer=None,
        )

        steps = path.steps
        assert len(steps) == 2
        assert "Step 1" in steps[0]
        assert "Step 2" in steps[1]

    def test_path_reasoning(self):
        """Test full reasoning trace."""
        root = Node(state="Question")
        child = root.add_child(
            state="Question\nReasoning trace...\nANSWER: 42",
            is_terminal=True,
            answer="42",
        )

        path = SampledPath(
            nodes=[root, child],
            terminal=child,
            value=1.0,
            visits=5,
            answer="42",
        )

        assert "ANSWER: 42" in path.reasoning
        assert path.answer == "42"


class TestPathSampler:
    """Tests for PathSampler."""

    def _create_test_tree(self):
        """Create a test tree with terminal nodes."""
        root = Node(state="What is 2+2?")

        # Path 1: High value, correct answer
        p1_step1 = root.add_child(state="What is 2+2?\nLet me calculate...")
        p1_step1.visits = 10
        p1_step1._total_value = 8.0

        p1_terminal = p1_step1.add_child(
            state="What is 2+2?\nLet me calculate...\n2+2=4\nANSWER: 4",
            is_terminal=True,
            answer="4",
        )
        p1_terminal.visits = 8
        p1_terminal._total_value = 7.2  # avg 0.9

        # Path 2: Medium value, correct answer
        p2_step1 = root.add_child(state="What is 2+2?\nThinking step by step...")
        p2_step1.visits = 5
        p2_step1._total_value = 3.0

        p2_terminal = p2_step1.add_child(
            state="What is 2+2?\nThinking step by step...\nThe sum is 4.\nANSWER: 4",
            is_terminal=True,
            answer="4",
        )
        p2_terminal.visits = 4
        p2_terminal._total_value = 2.8  # avg 0.7

        # Path 3: Low value, wrong answer
        p3_step1 = root.add_child(state="What is 2+2?\nI'll guess...")
        p3_step1.visits = 2
        p3_step1._total_value = 0.6

        p3_terminal = p3_step1.add_child(
            state="What is 2+2?\nI'll guess...\nMaybe 5?\nANSWER: 5",
            is_terminal=True,
            answer="5",
        )
        p3_terminal.visits = 1
        p3_terminal._total_value = 0.2  # avg 0.2

        return root

    def test_get_terminals(self):
        """Test finding terminal nodes."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        terminals = sampler.get_terminals()
        assert len(terminals) == 3
        assert all(t.is_terminal for t in terminals)

    def test_sample_by_value(self):
        """Test value-based sampling."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        paths = sampler.sample(n=3, strategy="value")

        assert len(paths) == 3
        # Should be ordered by value (descending)
        assert paths[0].value >= paths[1].value >= paths[2].value
        # Highest value should be the correct answer path
        assert paths[0].answer == "4"

    def test_sample_by_visits(self):
        """Test visit-based sampling."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        paths = sampler.sample(n=3, strategy="visits")

        assert len(paths) == 3
        # Should be ordered by visits (descending)
        assert paths[0].visits >= paths[1].visits >= paths[2].visits

    def test_sample_topk(self):
        """Test top-k sampling."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        paths = sampler.sample(n=2, strategy="topk")

        assert len(paths) == 2
        # Top 2 should both be correct answer
        assert paths[0].answer == "4"
        assert paths[1].answer == "4"

    def test_sample_diverse(self):
        """Test diverse sampling."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        paths = sampler.sample(n=3, strategy="diverse")

        assert len(paths) == 3
        # All three paths should be included (they're all different)
        answers = [p.answer for p in paths]
        assert "4" in answers  # Correct answer included
        assert "5" in answers  # Wrong answer included for diversity

    def test_sample_fewer_than_requested(self):
        """Test when requesting more paths than available."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        paths = sampler.sample(n=10, strategy="value")

        # Should return all available (3), not 10
        assert len(paths) == 3

    def test_get_answer_distribution(self):
        """Test answer distribution calculation."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        dist = sampler.get_answer_distribution()

        assert "4" in dist
        assert "5" in dist
        assert dist["4"]["count"] == 2  # Two paths with answer 4
        assert dist["5"]["count"] == 1  # One path with answer 5

    def test_consistency_score(self):
        """Test consistency score calculation."""
        root = self._create_test_tree()
        sampler = PathSampler(root)

        score = sampler.consistency_score()

        # 2 out of 3 have same answer (4)
        assert score == pytest.approx(2 / 3)

    def test_consistency_score_all_same(self):
        """Test consistency when all answers match."""
        root = Node(state="Q")
        for i in range(3):
            child = root.add_child(
                state=f"Q\nPath {i}\nANSWER: 42",
                is_terminal=True,
                answer="42",
            )
            child.visits = 1
            child._total_value = 0.8

        sampler = PathSampler(root)
        assert sampler.consistency_score() == 1.0

    def test_empty_tree(self):
        """Test with tree containing no terminals."""
        root = Node(state="Q")
        root.add_child(state="Q\nStep 1")  # Non-terminal

        sampler = PathSampler(root)

        assert len(sampler.get_terminals()) == 0
        assert len(sampler.sample(n=5, strategy="value")) == 0

    def test_include_non_terminal(self):
        """Test including non-terminal leaves in sampling."""
        root = Node(state="Q")
        leaf = root.add_child(state="Q\nPartial reasoning")
        leaf.visits = 5
        leaf._total_value = 3.0

        sampler = PathSampler(root)

        # Without non-terminal: empty
        paths = sampler.sample(n=5, strategy="value", include_non_terminal=False)
        assert len(paths) == 0

        # With non-terminal: includes leaf
        paths = sampler.sample(n=5, strategy="value", include_non_terminal=True)
        assert len(paths) == 1
        assert paths[0].terminal is None


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert SamplingStrategy.VALUE.value == "value"
        assert SamplingStrategy.VISITS.value == "visits"
        assert SamplingStrategy.DIVERSE.value == "diverse"
        assert SamplingStrategy.TOPK.value == "topk"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        assert SamplingStrategy("value") == SamplingStrategy.VALUE
        assert SamplingStrategy("visits") == SamplingStrategy.VISITS
