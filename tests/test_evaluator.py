"""Tests for the evaluator module."""

import pytest
from mcts_reasoning.evaluator import (
    Evaluator,
    LLMEvaluator,
    GroundTruthEvaluator,
    NumericEvaluator,
    ProcessEvaluator,
    CompositeEvaluator,
    MockEvaluator,
)
from mcts_reasoning.types import Evaluation


class TestNumericEvaluatorExact:
    """Tests for NumericEvaluator exact matching."""

    def test_numeric_evaluator_exact(self):
        evaluator = NumericEvaluator(ground_truth=42.0)
        result = evaluator.evaluate("q", "state", "42")

        assert result.score == 1.0
        assert result.is_correct is True

    def test_numeric_evaluator_exact_decimal(self):
        evaluator = NumericEvaluator(ground_truth=3.14)
        result = evaluator.evaluate("q", "state", "3.14")

        assert result.score == 1.0
        assert result.is_correct is True

    def test_numeric_evaluator_no_number(self):
        evaluator = NumericEvaluator(ground_truth=42.0)
        result = evaluator.evaluate("q", "state", "I don't know")

        assert result.score == 0.0
        assert result.is_correct is False

    def test_numeric_evaluator_fraction(self):
        evaluator = NumericEvaluator(ground_truth=0.5, rel_tol=0.01)
        result = evaluator.evaluate("q", "state", "1/2")

        assert result.score == 1.0

    def test_numeric_evaluator_scientific(self):
        evaluator = NumericEvaluator(ground_truth=1e6, rel_tol=0.01)
        result = evaluator.evaluate("q", "state", "1e6")

        assert result.score == 1.0

    def test_numeric_evaluator_partial_credit_close(self):
        evaluator = NumericEvaluator(ground_truth=100.0, rel_tol=1e-5)
        result = evaluator.evaluate("q", "state", "95")

        assert result.score == 0.8
        assert result.is_correct is False

    def test_numeric_evaluator_partial_credit_far(self):
        evaluator = NumericEvaluator(ground_truth=100.0, partial_credit_factor=0.5)
        result = evaluator.evaluate("q", "state", "70")

        assert result.score == 0.5

    def test_numeric_evaluator_no_credit(self):
        evaluator = NumericEvaluator(ground_truth=100.0)
        result = evaluator.evaluate("q", "state", "10")

        assert result.score == 0.0


class TestGroundTruthEvaluator:
    """Tests for GroundTruthEvaluator."""

    def test_ground_truth_evaluator_exact_match(self):
        evaluator = GroundTruthEvaluator(ground_truth="4")
        result = evaluator.evaluate("What is 2+2?", "reasoning", "4")

        assert result.score == 1.0
        assert result.is_correct is True

    def test_ground_truth_evaluator_no_match(self):
        evaluator = GroundTruthEvaluator(ground_truth="4", partial_credit=False)
        result = evaluator.evaluate("q", "state", "5")

        assert result.score == 0.0
        assert result.is_correct is False

    def test_ground_truth_evaluator_partial_credit(self):
        evaluator = GroundTruthEvaluator(ground_truth="42")
        result = evaluator.evaluate("q", "state", "The answer is 42")

        assert result.score == 0.7

    def test_ground_truth_evaluator_normalization(self):
        evaluator = GroundTruthEvaluator(ground_truth="HELLO WORLD!")
        result = evaluator.evaluate("q", "state", "  hello   world  ")

        assert result.score == 1.0
        assert result.is_correct is True


class TestProcessEvaluator:
    """Tests for ProcessEvaluator."""

    def test_process_evaluator_good_reasoning(self):
        evaluator = ProcessEvaluator()
        state = """
        Step 1: First, let's understand the problem.
        Step 2: We need to calculate 2 + 2.
        Since 2 + 2 = 4, therefore the answer is 4.
        Let's verify: 4 - 2 = 2, which confirms our answer.
        """

        result = evaluator.evaluate("What is 2+2?", state, "4")
        assert result.score > 0.5

    def test_process_evaluator_minimal(self):
        evaluator = ProcessEvaluator()
        result = evaluator.evaluate("q", "The answer is 4.", "4")
        assert result.score < 0.5

    def test_process_evaluator_with_answer_evaluator(self):
        answer_eval = GroundTruthEvaluator(ground_truth="4")
        evaluator = ProcessEvaluator(
            answer_evaluator=answer_eval,
            answer_weight=0.7,
            process_weight=0.3,
        )

        state = "Step 1: 2 + 2 = 4. Therefore, the answer is 4."
        result = evaluator.evaluate("What is 2+2?", state, "4")
        assert result.score > 0.8

    def test_process_evaluator_step_detection(self):
        evaluator = ProcessEvaluator()
        state = "Step 1: Start. Step 2: Continue. Step 3: Done."
        result = evaluator.evaluate("q", state, "a")
        assert "step structure" in result.reasoning.lower()

    def test_process_evaluator_verification_detection(self):
        evaluator = ProcessEvaluator()
        state = "The answer is 4. Let's verify: 2 + 2 = 4. Confirmed."
        result = evaluator.evaluate("q", state, "a")
        assert "verification" in result.reasoning.lower()


class TestParseScore:
    """Tests for LLMEvaluator._parse_score_from_text."""

    def test_parse_score_prefers_0_to_1(self):
        """'The answer includes 2 steps, score: 0.8' should return 0.8, not 2.0."""
        result = LLMEvaluator._parse_score_from_text(
            "The answer includes 2 steps, score: 0.8"
        )
        assert result == 0.8

    def test_parse_score_simple_decimal(self):
        result = LLMEvaluator._parse_score_from_text("0.85")
        assert result == 0.85

    def test_parse_score_with_text(self):
        result = LLMEvaluator._parse_score_from_text("Score: 0.9 because...")
        assert result == 0.9

    def test_parse_score_whole_number_1(self):
        result = LLMEvaluator._parse_score_from_text("1")
        assert result == 1.0

    def test_parse_score_whole_number_0(self):
        result = LLMEvaluator._parse_score_from_text("0")
        assert result == 0.0

    def test_parse_score_clamps_large_number(self):
        result = LLMEvaluator._parse_score_from_text("5.0")
        assert result == 1.0

    def test_parse_score_no_number_defaults_to_05(self):
        result = LLMEvaluator._parse_score_from_text("Unable to evaluate")
        assert result == 0.5

    def test_parse_score_multiple_in_range(self):
        """With multiple 0-1 numbers, return the last one."""
        result = LLMEvaluator._parse_score_from_text(
            "score 0.5 but revised to 0.9"
        )
        assert result == 0.9

    def test_parse_score_out_of_range_only(self):
        """When only out-of-range numbers exist, clamp the last one."""
        result = LLMEvaluator._parse_score_from_text("rated 3 out of 5")
        assert result == 1.0  # last number (5) clamped to 1.0

    def test_parse_score_mixed_range(self):
        """When both in-range and out-of-range exist, prefer in-range."""
        result = LLMEvaluator._parse_score_from_text(
            "After 3 iterations, final score: 0.75"
        )
        assert result == 0.75


class TestLLMEvaluator:
    """Tests for LLMEvaluator end-to-end."""

    class MockLLM:
        def __init__(self, response="0.8"):
            self.response = response
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            return self.response

    def test_basic_evaluation(self):
        llm = self.MockLLM(response="0.85")
        evaluator = LLMEvaluator(llm=llm)

        result = evaluator.evaluate("What is 2+2?", "I calculated 2+2=4", "4")

        assert result.score == 0.85
        assert len(llm.calls) == 1

    def test_default_on_parse_failure(self):
        llm = self.MockLLM(response="Unable to evaluate")
        evaluator = LLMEvaluator(llm=llm)

        result = evaluator.evaluate("q", "s", "a")
        assert result.score == 0.5


class TestCompositeEvaluator:
    """Tests for CompositeEvaluator."""

    def test_equal_weights(self):
        eval1 = MockEvaluator(default_score=0.6)
        eval2 = MockEvaluator(default_score=0.8)
        composite = CompositeEvaluator([eval1, eval2], weights=[1.0, 1.0])

        result = composite.evaluate("q", "state", "42")
        assert 0.68 < result.score < 0.72

    def test_custom_weights(self):
        eval1 = MockEvaluator(default_score=1.0)
        eval2 = MockEvaluator(default_score=0.0)
        composite = CompositeEvaluator([eval1, eval2], weights=[3.0, 1.0])

        result = composite.evaluate("q", "state", "42")
        # 1.0 * 0.75 + 0.0 * 0.25 = 0.75
        # But MockEvaluator gives default_score for numeric answers
        # eval2 default_score=0.0, so 0.0 for "42" (0.0 * 0.8 = 0.0 for non-digit)
        # Actually MockEvaluator gives default_score when answer.isdigit()
        assert result.score == pytest.approx(0.75, abs=0.01)


class TestMockEvaluator:
    """Tests for MockEvaluator."""

    def test_returns_default_score(self):
        evaluator = MockEvaluator(default_score=0.9)
        result = evaluator.evaluate("q", "state", "42")
        assert result.score == 0.9

    def test_counts_calls(self):
        evaluator = MockEvaluator()
        evaluator.evaluate("q", "s", "a")
        evaluator.evaluate("q", "s", "a")
        assert evaluator.call_count == 2

    def test_numeric_answers_get_full_score(self):
        evaluator = MockEvaluator(default_score=0.8)
        numeric = evaluator.evaluate("q", "s", "42")
        text = evaluator.evaluate("q", "s", "forty-two")
        assert numeric.score > text.score
