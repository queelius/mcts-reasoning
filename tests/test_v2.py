"""
Tests for MCTS v2 implementation.

Tests the clean architecture with:
- Node: UCB1 calculation, continuation caching
- Generator: continuation generation, terminal detection
- Evaluator: scoring terminal states
- MCTS: full search algorithm
"""

import pytest
import math
from mcts_reasoning import (
    Node,
    Generator, LLMGenerator,
    Evaluator, LLMEvaluator,
    MCTS, SearchResult,
    MockGenerator, Continuation, ANSWER_MARKER,
    MockEvaluator,
    GroundTruthEvaluator,
    CompositeEvaluator,
    Evaluation,
)


# ============================================================
# Node Tests
# ============================================================

class TestNode:
    """Tests for the Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(state="initial state")
        assert node.state == "initial state"
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert not node.is_terminal
        assert node.answer is None

    def test_node_is_root(self):
        """Test is_root property."""
        root = Node(state="root")
        child = root.add_child("child")
        assert root.is_root
        assert not child.is_root

    def test_node_is_leaf(self):
        """Test is_leaf property."""
        node = Node(state="leaf")
        assert node.is_leaf
        node.add_child("child")
        assert not node.is_leaf

    def test_node_depth(self):
        """Test depth calculation."""
        root = Node(state="root")
        child1 = root.add_child("child1")
        child2 = child1.add_child("child2")
        grandchild = child2.add_child("grandchild")

        assert root.depth == 0
        assert child1.depth == 1
        assert child2.depth == 2
        assert grandchild.depth == 3

    def test_average_value(self):
        """Test average value calculation."""
        node = Node(state="test")
        assert node.average_value == 0.0

        node.visits = 4
        node.value = 3.2
        assert node.average_value == 0.8

    def test_ucb1_unvisited(self):
        """Test UCB1 returns infinity for unvisited nodes."""
        node = Node(state="unvisited")
        assert node.ucb1() == float('inf')

    def test_ucb1_calculation(self):
        """Test UCB1 calculation."""
        parent = Node(state="parent")
        parent.visits = 10

        child = parent.add_child("child")
        child.visits = 3
        child.value = 2.1  # avg = 0.7

        # UCB1 = 0.7 + 1.414 * sqrt(ln(10) / 3)
        expected = 0.7 + 1.414 * math.sqrt(math.log(10) / 3)
        assert abs(child.ucb1() - expected) < 0.001

    def test_continuation_caching(self):
        """Test continuation caching mechanism."""
        node = Node(state="test")

        # Initially has untried continuations (None = not generated yet)
        assert node.has_untried_continuations()
        assert node.get_next_continuation() is None

        # Set continuations
        node.set_continuations(["cont1", "cont2", "cont3"])

        # Now can retrieve them in order
        assert node.has_untried_continuations()
        assert node.get_next_continuation() == "cont1"
        assert node.get_next_continuation() == "cont2"
        assert node.get_next_continuation() == "cont3"

        # All used up
        assert not node.has_untried_continuations()
        assert node.get_next_continuation() is None

    def test_add_child(self):
        """Test adding children."""
        parent = Node(state="parent")
        child = parent.add_child("child state", is_terminal=True, answer="42")

        assert child in parent.children
        assert child.parent is parent
        assert child.state == "child state"
        assert child.is_terminal
        assert child.answer == "42"

    def test_best_child(self):
        """Test best child selection by UCB1."""
        parent = Node(state="parent")
        parent.visits = 10

        child1 = parent.add_child("child1")
        child1.visits = 3
        child1.value = 1.5

        child2 = parent.add_child("child2")
        child2.visits = 2
        child2.value = 1.6

        # child2 should have higher UCB1 (higher avg value and fewer visits)
        best = parent.best_child()
        assert best is child2

    def test_most_visited_child(self):
        """Test most visited child selection."""
        parent = Node(state="parent")

        child1 = parent.add_child("child1")
        child1.visits = 5

        child2 = parent.add_child("child2")
        child2.visits = 10

        assert parent.most_visited_child() is child2

    def test_highest_value_child(self):
        """Test highest value child selection."""
        parent = Node(state="parent")

        child1 = parent.add_child("child1")
        child1.visits = 5
        child1.value = 2.0  # avg = 0.4

        child2 = parent.add_child("child2")
        child2.visits = 5
        child2.value = 4.0  # avg = 0.8

        assert parent.highest_value_child() is child2

    def test_path_from_root(self):
        """Test path reconstruction."""
        root = Node(state="root")
        child = root.add_child("child")
        grandchild = child.add_child("grandchild")

        path = grandchild.path_from_root()
        assert len(path) == 3
        assert path[0] is root
        assert path[1] is child
        assert path[2] is grandchild


# ============================================================
# Generator Tests
# ============================================================

class TestGenerator:
    """Tests for Generator classes."""

    def test_answer_marker(self):
        """Test answer marker constant."""
        assert ANSWER_MARKER == "ANSWER:"

    def test_continuation_dataclass(self):
        """Test Continuation dataclass."""
        cont = Continuation(text="some text", is_terminal=True, answer="42")
        assert cont.text == "some text"
        assert cont.is_terminal
        assert cont.answer == "42"

    def test_mock_generator_basic(self):
        """Test basic mock generator behavior."""
        gen = MockGenerator()

        # Generate a single continuation
        conts = gen.generate("What is 2+2?", "Initial state", n=1)
        assert len(conts) == 1
        assert "Step 1" in conts[0].text

    def test_mock_generator_multiple(self):
        """Test generating multiple continuations."""
        gen = MockGenerator()

        conts = gen.generate("What is 2+2?", "Initial state", n=3)
        assert len(conts) == 3
        for cont in conts:
            assert isinstance(cont, Continuation)

    def test_mock_generator_becomes_terminal(self):
        """Test mock generator produces terminal state after several steps."""
        gen = MockGenerator()

        # Build up state to trigger terminal
        state = "Initial\n\nStep 1: ...\n\nStep 2: ..."
        conts = gen.generate("test", state, n=1)

        # Should produce ANSWER
        assert conts[0].is_terminal
        assert conts[0].answer == "4"

    def test_mock_generator_custom_responses(self):
        """Test mock generator with custom responses."""
        responses = [
            "This is step one.",
            "ANSWER: 42",
        ]
        gen = MockGenerator(responses=responses)

        cont1 = gen.generate("q", "state", n=1)[0]
        assert not cont1.is_terminal
        assert "step one" in cont1.text

        cont2 = gen.generate("q", cont1.text, n=1)[0]
        assert cont2.is_terminal
        assert cont2.answer == "42"

    def test_extract_answer(self):
        """Test answer extraction from text."""
        gen = MockGenerator()

        assert gen.extract_answer("ANSWER: 42") == "42"
        assert gen.extract_answer("Some reasoning... ANSWER: hello world") == "hello world"
        assert gen.extract_answer("No answer here") is None

    def test_is_terminal(self):
        """Test terminal detection."""
        gen = MockGenerator()

        assert gen.is_terminal("ANSWER: done")
        assert not gen.is_terminal("Still working...")


# ============================================================
# Evaluator Tests
# ============================================================

class TestEvaluator:
    """Tests for Evaluator classes."""

    def test_evaluation_dataclass(self):
        """Test Evaluation dataclass."""
        ev = Evaluation(score=0.8, reasoning="good solution", is_correct=True)
        assert ev.score == 0.8
        assert ev.reasoning == "good solution"
        assert ev.is_correct

    def test_mock_evaluator(self):
        """Test mock evaluator."""
        evaluator = MockEvaluator(default_score=0.9)

        result = evaluator.evaluate("What is 2+2?", "Some reasoning", "4")
        assert result.score == 0.9
        assert evaluator.call_count == 1

    def test_mock_evaluator_prefers_numeric_answers(self):
        """Test mock evaluator gives higher scores to numeric answers."""
        evaluator = MockEvaluator(default_score=0.8)

        numeric_result = evaluator.evaluate("q", "state", "4")
        text_result = evaluator.evaluate("q", "state", "four")

        assert numeric_result.score > text_result.score

    def test_ground_truth_evaluator_exact_match(self):
        """Test ground truth evaluator with exact match."""
        evaluator = GroundTruthEvaluator(ground_truth="4")

        result = evaluator.evaluate("What is 2+2?", "reasoning", "4")
        assert result.score == 1.0
        assert result.is_correct

    def test_ground_truth_evaluator_no_match(self):
        """Test ground truth evaluator with no match."""
        evaluator = GroundTruthEvaluator(ground_truth="4", partial_credit=False)

        result = evaluator.evaluate("What is 2+2?", "reasoning", "5")
        assert result.score == 0.0
        assert not result.is_correct

    def test_ground_truth_evaluator_partial_credit(self):
        """Test ground truth evaluator partial credit."""
        evaluator = GroundTruthEvaluator(ground_truth="42")

        # Answer contains truth
        result = evaluator.evaluate("q", "state", "The answer is 42")
        assert result.score == 0.7

    def test_ground_truth_evaluator_normalization(self):
        """Test ground truth evaluator normalizes answers."""
        evaluator = GroundTruthEvaluator(ground_truth="HELLO WORLD!")

        result = evaluator.evaluate("q", "state", "  hello   world  ")
        assert result.score == 1.0
        assert result.is_correct

    def test_composite_evaluator(self):
        """Test composite evaluator."""
        eval1 = MockEvaluator(default_score=0.6)
        eval2 = MockEvaluator(default_score=0.8)

        composite = CompositeEvaluator([eval1, eval2], weights=[1.0, 1.0])

        # Use numeric answer to get full default_score from MockEvaluator
        result = composite.evaluate("q", "state", "42")
        # Equal weights: (0.6 + 0.8) / 2 = 0.7
        assert 0.68 < result.score < 0.72  # Allow for floating point


class TestNumericEvaluator:
    """Tests for NumericEvaluator."""

    def test_exact_match(self):
        """Test numeric evaluator with exact match."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=42.0)
        result = evaluator.evaluate("q", "state", "42")

        assert result.score == 1.0
        assert result.is_correct

    def test_tolerance_match(self):
        """Test numeric evaluator with tolerance."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=3.14159, rel_tol=0.001)
        result = evaluator.evaluate("q", "state", "3.14")

        assert result.score == 1.0
        assert result.is_correct

    def test_partial_credit_close(self):
        """Test partial credit for close but not exact answers."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=100.0, rel_tol=1e-5)
        result = evaluator.evaluate("q", "state", "95")  # 5% error

        assert result.score == 0.8  # Within 10%
        assert not result.is_correct

    def test_partial_credit_further(self):
        """Test partial credit for answers within 50%."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=100.0, partial_credit_factor=0.5)
        result = evaluator.evaluate("q", "state", "70")  # 30% error

        assert result.score == 0.5
        assert not result.is_correct

    def test_no_credit_far(self):
        """Test no credit for far-off answers."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=100.0)
        result = evaluator.evaluate("q", "state", "10")  # 90% error

        assert result.score == 0.0
        assert not result.is_correct

    def test_extract_number_from_text(self):
        """Test extracting numbers from text answers."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=42.0)

        result = evaluator.evaluate("q", "state", "The answer is 42")
        assert result.score == 1.0

        result = evaluator.evaluate("q", "state", "I think it's approximately 42.0")
        assert result.score == 1.0

    def test_scientific_notation(self):
        """Test parsing scientific notation."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=1e6, rel_tol=0.01)
        result = evaluator.evaluate("q", "state", "1e6")

        assert result.score == 1.0

    def test_fraction_parsing(self):
        """Test parsing fractions."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=0.5, rel_tol=0.01)
        result = evaluator.evaluate("q", "state", "1/2")

        assert result.score == 1.0

    def test_no_number_in_answer(self):
        """Test handling when no number can be extracted."""
        from mcts_reasoning.evaluator import NumericEvaluator

        evaluator = NumericEvaluator(ground_truth=42.0)
        result = evaluator.evaluate("q", "state", "I don't know")

        assert result.score == 0.0
        assert not result.is_correct


class TestProcessEvaluator:
    """Tests for ProcessEvaluator."""

    def test_process_only_evaluation(self):
        """Test evaluating process without answer evaluator."""
        from mcts_reasoning.evaluator import ProcessEvaluator

        evaluator = ProcessEvaluator()

        # Good reasoning with steps and logic
        good_state = """
        Step 1: First, let's understand the problem.
        Step 2: We need to calculate 2 + 2.
        Since 2 + 2 = 4, therefore the answer is 4.
        Let's verify: 4 - 2 = 2, which confirms our answer.
        """

        result = evaluator.evaluate("What is 2+2?", good_state, "4")
        assert result.score > 0.5  # Should have high process score

    def test_minimal_process(self):
        """Test evaluation of minimal reasoning."""
        from mcts_reasoning.evaluator import ProcessEvaluator

        evaluator = ProcessEvaluator()

        minimal_state = "The answer is 4."

        result = evaluator.evaluate("q", minimal_state, "4")
        assert result.score < 0.5  # Low process score

    def test_with_answer_evaluator(self):
        """Test process evaluator combined with answer evaluator."""
        from mcts_reasoning.evaluator import ProcessEvaluator, GroundTruthEvaluator

        answer_eval = GroundTruthEvaluator(ground_truth="4")
        evaluator = ProcessEvaluator(
            answer_evaluator=answer_eval,
            answer_weight=0.7,
            process_weight=0.3,
        )

        # Correct answer with good process
        good_state = "Step 1: 2 + 2 = 4. Therefore, the answer is 4."
        result = evaluator.evaluate("What is 2+2?", good_state, "4")
        assert result.score > 0.8

        # Correct answer with minimal process
        minimal_state = "4"
        result = evaluator.evaluate("q", minimal_state, "4")
        assert result.score >= 0.7  # At least the answer weight

    def test_step_structure_detection(self):
        """Test detection of step-by-step structure."""
        from mcts_reasoning.evaluator import ProcessEvaluator

        evaluator = ProcessEvaluator()

        step_state = "Step 1: Start. Step 2: Continue. Step 3: Done."
        result = evaluator.evaluate("q", step_state, "a")
        assert "step structure" in result.reasoning.lower()

    def test_verification_detection(self):
        """Test detection of verification statements."""
        from mcts_reasoning.evaluator import ProcessEvaluator

        evaluator = ProcessEvaluator()

        verify_state = "The answer is 4. Let's verify: 2 + 2 = 4. Confirmed."
        result = evaluator.evaluate("q", verify_state, "a")
        assert "verification" in result.reasoning.lower()


class TestLLMEvaluator:
    """Tests for LLMEvaluator (LLM-as-judge)."""

    class MockLLM:
        """Mock LLM for testing evaluators."""

        def __init__(self, response: str = "0.8"):
            self.response = response
            self.calls = []

        def generate(self, prompt: str, **kwargs) -> str:
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            return self.response

    def test_llm_evaluator_basic(self):
        """Test basic LLM evaluator scoring."""
        from mcts_reasoning.evaluator import LLMEvaluator

        llm = self.MockLLM(response="0.85")
        evaluator = LLMEvaluator(llm=llm)

        result = evaluator.evaluate("What is 2+2?", "I calculated 2+2=4", "4")

        assert result.score == 0.85
        assert len(llm.calls) == 1
        assert "What is 2+2?" in llm.calls[0]["prompt"]

    def test_llm_evaluator_parses_various_formats(self):
        """Test LLM evaluator parses various response formats."""
        from mcts_reasoning.evaluator import LLMEvaluator

        # Test decimal
        llm = self.MockLLM(response="Score: 0.9")
        evaluator = LLMEvaluator(llm=llm)
        result = evaluator.evaluate("q", "s", "a")
        assert result.score == 0.9

        # Test with text
        llm.response = "The score is 0.75 because..."
        result = evaluator.evaluate("q", "s", "a")
        assert result.score == 0.75

        # Test whole number (should be clamped to 1.0)
        llm.response = "1"
        result = evaluator.evaluate("q", "s", "a")
        assert result.score == 1.0

    def test_llm_evaluator_clamps_scores(self):
        """Test LLM evaluator clamps scores to 0-1."""
        from mcts_reasoning.evaluator import LLMEvaluator

        llm = self.MockLLM(response="5.0")  # Out of range
        evaluator = LLMEvaluator(llm=llm)

        result = evaluator.evaluate("q", "s", "a")
        assert result.score == 1.0  # Clamped to max

    def test_llm_evaluator_default_on_parse_failure(self):
        """Test LLM evaluator defaults to 0.5 on parse failure."""
        from mcts_reasoning.evaluator import LLMEvaluator

        llm = self.MockLLM(response="Unable to evaluate")  # No number
        evaluator = LLMEvaluator(llm=llm)

        result = evaluator.evaluate("q", "s", "a")
        assert result.score == 0.5  # Default

    def test_llm_evaluator_uses_temperature(self):
        """Test LLM evaluator passes temperature to LLM."""
        from mcts_reasoning.evaluator import LLMEvaluator

        llm = self.MockLLM()
        evaluator = LLMEvaluator(llm=llm, temperature=0.2)

        evaluator.evaluate("q", "s", "a")

        assert llm.calls[0]["kwargs"]["temperature"] == 0.2

    def test_llm_evaluator_custom_prompt(self):
        """Test LLM evaluator with custom prompt template."""
        from mcts_reasoning.evaluator import LLMEvaluator

        llm = self.MockLLM(response="0.9")
        custom_prompt = "Q: {question}\nA: {answer}\nScore:"
        evaluator = LLMEvaluator(llm=llm, prompt_template=custom_prompt)

        evaluator.evaluate("test question", "state", "test answer")

        assert "Q: test question" in llm.calls[0]["prompt"]
        assert "A: test answer" in llm.calls[0]["prompt"]

    def test_llm_evaluator_truncates_long_states(self):
        """Test LLM evaluator truncates very long states."""
        from mcts_reasoning.evaluator import LLMEvaluator

        llm = self.MockLLM(response="0.7")
        evaluator = LLMEvaluator(llm=llm)

        # Create a very long state
        long_state = "x" * 5000
        evaluator.evaluate("q", long_state, "a")

        # State should be truncated in the prompt
        assert len(llm.calls[0]["prompt"]) < 5000


# ============================================================
# MCTS Tests
# ============================================================

class TestMCTS:
    """Tests for MCTS class."""

    def test_mcts_creation(self):
        """Test MCTS creation."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        mcts = MCTS(gen, eval_, exploration_constant=2.0, max_children_per_node=5)
        assert mcts.exploration_constant == 2.0
        assert mcts.max_children_per_node == 5

    def test_mcts_search_basic(self):
        """Test basic MCTS search."""
        gen = MockGenerator()
        eval_ = MockEvaluator(default_score=0.8)

        mcts = MCTS(gen, eval_, max_children_per_node=2)
        result = mcts.search("What is 2+2?", simulations=10)

        assert isinstance(result, SearchResult)
        assert result.simulations == 10
        assert result.root is not None

    def test_mcts_finds_answer(self):
        """Test MCTS finds an answer."""
        gen = MockGenerator()
        eval_ = MockEvaluator(default_score=0.8)

        mcts = MCTS(gen, eval_)
        result = mcts.search("What is 2+2?", simulations=20)

        assert result.best_answer is not None
        assert result.confidence > 0

    def test_mcts_builds_tree(self):
        """Test MCTS builds a tree with depth."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        mcts = MCTS(gen, eval_, max_children_per_node=3)
        result = mcts.search("What is 2+2?", simulations=30)

        stats = result.stats
        assert stats["total_nodes"] > 1
        assert stats["max_depth"] > 0

    def test_mcts_finds_terminal_states(self):
        """Test MCTS records terminal states."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        mcts = MCTS(gen, eval_)
        result = mcts.search("What is 2+2?", simulations=20)

        assert len(result.terminal_states) > 0
        for ts in result.terminal_states:
            assert "answer" in ts
            assert "score" in ts

    def test_mcts_respects_max_children(self):
        """Test MCTS respects max_children_per_node."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        mcts = MCTS(gen, eval_, max_children_per_node=2)
        result = mcts.search("What is 2+2?", simulations=30)

        # Check root doesn't exceed max children
        assert len(result.root.children) <= 2

    def test_mcts_tree_visualization(self):
        """Test tree visualization."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        mcts = MCTS(gen, eval_)
        mcts.search("What is 2+2?", simulations=10)

        viz = mcts.get_tree_visualization(max_depth=2)
        assert isinstance(viz, str)
        assert len(viz) > 0
        assert "v=" in viz  # Contains visit count

    def test_mcts_stats(self):
        """Test SearchResult stats property."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        mcts = MCTS(gen, eval_)
        result = mcts.search("What is 2+2?", simulations=15)

        stats = result.stats
        assert "total_nodes" in stats
        assert "max_depth" in stats
        assert "simulations" in stats
        assert "terminal_states_found" in stats
        assert stats["simulations"] == 15

    def test_mcts_exploration_vs_exploitation(self):
        """Test that exploration constant affects behavior."""
        gen = MockGenerator()
        eval_ = MockEvaluator()

        # High exploration - should explore more
        mcts_explore = MCTS(gen, eval_, exploration_constant=3.0, max_children_per_node=3)
        result_explore = mcts_explore.search("test", simulations=30)

        # Low exploration - should exploit more
        gen2 = MockGenerator()
        mcts_exploit = MCTS(gen2, eval_, exploration_constant=0.1, max_children_per_node=3)
        result_exploit = mcts_exploit.search("test", simulations=30)

        # Both should complete, but tree shapes may differ
        assert result_explore.root is not None
        assert result_exploit.root is not None


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for the full system."""

    def test_full_reasoning_flow(self):
        """Test complete reasoning flow from question to answer."""
        responses = [
            "Step 1: I need to calculate 2 + 2.",
            "Step 2: Adding 2 and 2 gives 4. ANSWER: 4",
        ]
        gen = MockGenerator(responses=responses)
        eval_ = GroundTruthEvaluator(ground_truth="4")

        mcts = MCTS(gen, eval_, max_children_per_node=2)
        result = mcts.search("What is 2+2?", simulations=10)

        # Should find the correct answer
        assert "4" in str(result.best_answer)
        assert result.confidence > 0.5

    def test_multiple_solution_paths(self):
        """Test that MCTS explores multiple paths."""
        gen = MockGenerator()  # Default mock produces variations
        eval_ = MockEvaluator(default_score=0.7)

        mcts = MCTS(gen, eval_, max_children_per_node=3)
        result = mcts.search("Solve this problem", simulations=30)

        # Should have explored multiple children at root
        assert len(result.root.children) > 1

    def test_backpropagation_updates_values(self):
        """Test that backpropagation correctly updates node values."""
        gen = MockGenerator()
        eval_ = MockEvaluator(default_score=0.8)

        mcts = MCTS(gen, eval_)
        result = mcts.search("test", simulations=20)

        # Root should have accumulated visits
        assert result.root.visits == 20
        assert result.root.value > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
