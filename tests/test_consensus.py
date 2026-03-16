"""Tests for ConsensusStrategy ABC and implementations."""

import pytest
from mcts_reasoning.consensus import (
    ConsensusStrategy,
    MajorityVote,
    WeightedVote,
)
from mcts_reasoning.types import SampledPath


def _make_path(answer: str | None, value: float = 0.5) -> SampledPath:
    """Helper to create a SampledPath with just answer and value."""
    return SampledPath(nodes=[], answer=answer, value=value, visits=1)


class TestMajorityVote:
    """Tests for MajorityVote consensus strategy."""

    def test_majority_vote(self):
        paths = [
            _make_path("4", 0.9),
            _make_path("4", 0.8),
            _make_path("5", 0.3),
        ]
        result = MajorityVote().vote(paths)
        assert result.answer == "4"
        assert result.confidence == pytest.approx(2 / 3)
        assert result.distribution == {"4": 2, "5": 1}
        assert result.paths_used == 3

    def test_majority_vote_tie_picks_one(self):
        paths = [
            _make_path("4", 0.9),
            _make_path("5", 0.8),
        ]
        result = MajorityVote().vote(paths)
        assert result.answer in ("4", "5")
        assert result.confidence == pytest.approx(0.5)

    def test_vote_with_no_answers(self):
        paths = [
            _make_path(None, 0.5),
            _make_path(None, 0.3),
        ]
        result = MajorityVote().vote(paths)
        assert result.answer == ""
        assert result.confidence == 0.0
        assert result.distribution == {}

    def test_vote_empty_paths(self):
        result = MajorityVote().vote([])
        assert result.answer == ""
        assert result.confidence == 0.0
        assert result.paths_used == 0


class TestWeightedVote:
    """Tests for WeightedVote consensus strategy."""

    def test_weighted_vote(self):
        paths = [
            _make_path("4", 0.9),
            _make_path("4", 0.8),
            _make_path("5", 0.3),
        ]
        result = WeightedVote().vote(paths)
        assert result.answer == "4"
        # "4" total weight = 1.7, "5" weight = 0.3, total = 2.0
        assert result.confidence == pytest.approx(1.7 / 2.0)
        assert result.distribution == {"4": 2, "5": 1}

    def test_weighted_vote_high_value_minority_wins(self):
        """A single high-value path can outweigh many low-value ones."""
        paths = [
            _make_path("rare", 10.0),
            _make_path("common", 0.1),
            _make_path("common", 0.1),
            _make_path("common", 0.1),
        ]
        result = WeightedVote().vote(paths)
        assert result.answer == "rare"

    def test_vote_with_no_answers(self):
        paths = [
            _make_path(None, 0.5),
            _make_path(None, 0.3),
        ]
        result = WeightedVote().vote(paths)
        assert result.answer == ""
        assert result.confidence == 0.0

    def test_vote_empty_paths(self):
        result = WeightedVote().vote([])
        assert result.answer == ""
        assert result.confidence == 0.0
        assert result.paths_used == 0


class TestConsensusStrategyABC:
    """Tests for ConsensusStrategy ABC contract."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ConsensusStrategy()  # type: ignore[abstract]
