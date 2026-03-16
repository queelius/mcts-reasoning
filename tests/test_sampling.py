"""Tests for sampling strategies and PathSampler facade."""

import pytest
from mcts_reasoning.node import Node
from mcts_reasoning.sampling import (
    PathSampler,
    SamplingStrategy,
    ValueSampling,
    VisitSampling,
    DiverseSampling,
    TopKSampling,
)
from mcts_reasoning.types import SampledPath


# ---------------------------------------------------------------------------
# Helper: build a test tree using add_child with proper value/visits
# ---------------------------------------------------------------------------


def _create_test_tree() -> Node:
    """Create a test tree with three terminal nodes at varying quality.

    Tree structure:
        root
        ├── p1_step1 (visits=10, value=8.0)
        │   └── p1_terminal "4" (visits=8, value=7.2 → avg 0.9)
        ├── p2_step1 (visits=5, value=3.0)
        │   └── p2_terminal "4" (visits=4, value=2.8 → avg 0.7)
        └── p3_step1 (visits=2, value=0.6)
            └── p3_terminal "5" (visits=1, value=0.2 → avg 0.2)
    """
    root = Node(state="What is 2+2?")

    # Path 1: high value, correct
    p1_step1 = root.add_child(
        state="What is 2+2?\nLet me calculate...",
        value=8.0,
        visits=10,
    )
    p1_step1.add_child(
        state="What is 2+2?\nLet me calculate...\n2+2=4\nANSWER: 4",
        is_terminal=True,
        answer="4",
        value=7.2,
        visits=8,
    )

    # Path 2: medium value, correct
    p2_step1 = root.add_child(
        state="What is 2+2?\nThinking step by step...",
        value=3.0,
        visits=5,
    )
    p2_step1.add_child(
        state="What is 2+2?\nThinking step by step...\nThe sum is 4.\nANSWER: 4",
        is_terminal=True,
        answer="4",
        value=2.8,
        visits=4,
    )

    # Path 3: low value, wrong
    p3_step1 = root.add_child(
        state="What is 2+2?\nI'll guess...",
        value=0.6,
        visits=2,
    )
    p3_step1.add_child(
        state="What is 2+2?\nI'll guess...\nMaybe 5?\nANSWER: 5",
        is_terminal=True,
        answer="5",
        value=0.2,
        visits=1,
    )

    return root


class TestValueSampling:
    """Tests for ValueSampling strategy."""

    def test_returns_highest_value(self):
        root = _create_test_tree()
        strategy = ValueSampling()
        paths = strategy.sample(root, n=3)

        assert len(paths) == 3
        # Sorted by average_value descending
        assert paths[0].value >= paths[1].value >= paths[2].value
        # Highest value terminal has answer "4" (avg 0.9)
        assert paths[0].answer == "4"

    def test_returns_fewer_when_not_enough(self):
        root = _create_test_tree()
        paths = ValueSampling().sample(root, n=10)
        assert len(paths) == 3


class TestVisitSampling:
    """Tests for VisitSampling strategy."""

    def test_returns_most_visited(self):
        root = _create_test_tree()
        paths = VisitSampling().sample(root, n=3)

        assert len(paths) == 3
        assert paths[0].visits >= paths[1].visits >= paths[2].visits
        # Most visited terminal has 8 visits
        assert paths[0].visits == 8


class TestDiverseSampling:
    """Tests for DiverseSampling strategy."""

    def test_returns_different_answers(self):
        root = _create_test_tree()
        paths = DiverseSampling().sample(root, n=3)

        assert len(paths) == 3
        answers = [p.answer for p in paths]
        assert "4" in answers
        assert "5" in answers

    def test_empty_tree(self):
        root = Node(state="Q")
        root.add_child(state="Q\nStep 1")  # Non-terminal
        paths = DiverseSampling().sample(root, n=5)
        assert len(paths) == 0


class TestTopKSampling:
    """Tests for TopKSampling strategy."""

    def test_topk(self):
        root = _create_test_tree()
        paths = TopKSampling().sample(root, n=2)

        assert len(paths) == 2
        # Top 2 by value should both be answer "4"
        assert paths[0].answer == "4"
        assert paths[1].answer == "4"


class TestPathSampler:
    """Tests for PathSampler facade."""

    def test_facade_default_strategy(self):
        root = _create_test_tree()
        sampler = PathSampler(root)
        paths = sampler.sample(n=3)
        assert len(paths) == 3
        # Default is ValueSampling
        assert paths[0].value >= paths[1].value

    def test_facade_string_strategy(self):
        root = _create_test_tree()
        sampler = PathSampler(root)
        paths = sampler.sample(n=3, strategy="visits")
        assert paths[0].visits >= paths[1].visits

    def test_facade_abc_strategy_override(self):
        root = _create_test_tree()
        sampler = PathSampler(root, strategy=VisitSampling())
        paths = sampler.sample(n=3)
        assert paths[0].visits >= paths[1].visits

    def test_answer_distribution(self):
        root = _create_test_tree()
        sampler = PathSampler(root)
        dist = sampler.get_answer_distribution()

        assert "4" in dist
        assert "5" in dist
        assert dist["4"]["count"] == 2
        assert dist["5"]["count"] == 1

    def test_consistency_score(self):
        root = _create_test_tree()
        sampler = PathSampler(root)
        score = sampler.consistency_score()
        # 2 out of 3 have answer "4"
        assert score == pytest.approx(2 / 3)

    def test_consistency_score_all_same(self):
        root = Node(state="Q")
        for i in range(3):
            root.add_child(
                state=f"Q\nPath {i}\nANSWER: 42",
                is_terminal=True,
                answer="42",
                value=0.8,
                visits=1,
            )
        sampler = PathSampler(root)
        assert sampler.consistency_score() == 1.0

    def test_empty_tree(self):
        root = Node(state="Q")
        root.add_child(state="Q\nStep 1")  # Non-terminal
        sampler = PathSampler(root)
        assert len(sampler.get_terminals()) == 0
        assert len(sampler.sample(n=5, strategy="value")) == 0

    def test_include_non_terminal(self):
        root = Node(state="Q")
        root.add_child(
            state="Q\nPartial reasoning",
            value=3.0,
            visits=5,
        )
        sampler = PathSampler(root)

        # Without non-terminal: empty
        paths = sampler.sample(n=5, strategy="value", include_non_terminal=False)
        assert len(paths) == 0

        # With non-terminal: includes leaf
        paths = sampler.sample(n=5, strategy="value", include_non_terminal=True)
        assert len(paths) == 1
        assert paths[0].answer is None

    def test_get_terminals(self):
        root = _create_test_tree()
        sampler = PathSampler(root)
        terminals = sampler.get_terminals()
        assert len(terminals) == 3
        assert all(t.is_terminal for t in terminals)


class TestSamplingStrategyABC:
    """Tests for SamplingStrategy ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SamplingStrategy()  # type: ignore[abstract]

    def test_unknown_string_strategy_raises(self):
        root = _create_test_tree()
        sampler = PathSampler(root)
        with pytest.raises(ValueError, match="Unknown strategy"):
            sampler.sample(n=3, strategy="nonexistent")


class TestSampledPath:
    """Tests for SampledPath dataclass."""

    def test_basic_fields(self):
        path = SampledPath(
            nodes=[],
            answer="42",
            value=0.9,
            visits=5,
        )
        assert path.answer == "42"
        assert path.value == 0.9
        assert path.visits == 5


class TestSelfConsistency:
    """Tests for self-consistency voting (legacy API)."""

    def _create_voting_tree(self):
        root = Node(state="What is 2+2?")

        for i, val in enumerate([0.9, 0.8, 0.7]):
            step = root.add_child(state=f"What is 2+2?\nPath {i}")
            step.add_child(
                state=f"What is 2+2?\nPath {i}\nANSWER: 4",
                is_terminal=True,
                answer="4",
                value=val,
                visits=1,
            )

        for i, val in enumerate([0.3, 0.2]):
            step = root.add_child(state=f"What is 2+2?\nWrong path {i}")
            step.add_child(
                state=f"What is 2+2?\nWrong path {i}\nANSWER: 5",
                is_terminal=True,
                answer="5",
                value=val,
                visits=1,
            )

        return root

    def test_majority_vote(self):
        root = self._create_voting_tree()
        sampler = PathSampler(root)
        answer, confidence = sampler.majority_vote()
        assert answer == "4"
        assert confidence == pytest.approx(3 / 5)

    def test_weighted_vote(self):
        root = self._create_voting_tree()
        sampler = PathSampler(root)
        answer, confidence = sampler.weighted_vote()
        assert answer == "4"
        # "4" total value: 0.9+0.8+0.7=2.4, "5": 0.3+0.2=0.5, total=2.9
        assert confidence > 0.8

    def test_self_consistency_vote_full_result(self):
        root = self._create_voting_tree()
        sampler = PathSampler(root)
        result = sampler.self_consistency_vote()

        assert result["answer"] == "4"
        assert result["total_votes"] == 5
        assert result["votes"]["4"] == 3
        assert result["votes"]["5"] == 2
        assert "weighted_votes" in result
        assert result["confidence"] > 0

    def test_self_consistency_normalize_answers(self):
        root = Node(state="Q")
        for i, ans in enumerate(["$100", "100", " 100 ", "$100.00"]):
            root.add_child(
                state=f"Q\nPath {i}\nANSWER: {ans}",
                is_terminal=True,
                answer=ans,
                value=0.5,
                visits=1,
            )
        sampler = PathSampler(root)
        result = sampler.self_consistency_vote(normalize_answers=True)
        assert result["total_votes"] == 4
        assert len(result["votes"]) <= 2

    def test_self_consistency_no_normalize(self):
        root = Node(state="Q")
        for ans in ["$100", "100"]:
            root.add_child(
                state=f"Q\nANSWER: {ans}",
                is_terminal=True,
                answer=ans,
                value=0.5,
                visits=1,
            )
        sampler = PathSampler(root)
        result = sampler.self_consistency_vote(normalize_answers=False)
        assert len(result["votes"]) == 2

    def test_self_consistency_empty_tree(self):
        root = Node(state="Q")
        root.add_child(state="Q\nPartial")
        sampler = PathSampler(root)
        result = sampler.self_consistency_vote()
        assert result["answer"] is None
        assert result["confidence"] == 0.0
        assert result["total_votes"] == 0

    def test_self_consistency_single_answer(self):
        root = Node(state="Q")
        root.add_child(
            state="Q\nANSWER: 42",
            is_terminal=True,
            answer="42",
            value=1.0,
            visits=1,
        )
        sampler = PathSampler(root)
        result = sampler.self_consistency_vote()
        assert result["answer"] == "42"
        assert result["confidence"] == 1.0
        assert result["total_votes"] == 1
