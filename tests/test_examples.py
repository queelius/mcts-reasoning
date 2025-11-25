"""
Tests for mcts_reasoning/compositional/examples.py

Tests the Example and ExampleSet classes for few-shot learning.
"""

import pytest
from mcts_reasoning.compositional.examples import (
    Example,
    ExampleSet,
    get_math_examples,
    get_logic_examples,
    get_coding_examples,
)


class TestExample:
    """Tests for the Example class."""

    def test_create_basic_example(self):
        """Test creating a basic example."""
        ex = Example(
            problem="What is 2 + 2?",
            solution="4"
        )
        assert ex.problem == "What is 2 + 2?"
        assert ex.solution == "4"
        assert ex.reasoning_steps is None
        assert ex.compositional_vector is None
        assert ex.metadata == {}

    def test_create_example_with_steps(self):
        """Test creating example with reasoning steps."""
        steps = ["Add the numbers", "2 + 2 = 4"]
        ex = Example(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=steps
        )
        assert ex.reasoning_steps == steps

    def test_create_example_with_metadata(self):
        """Test creating example with metadata."""
        metadata = {"domain": "arithmetic", "difficulty": "easy"}
        ex = Example(
            problem="What is 2 + 2?",
            solution="4",
            metadata=metadata
        )
        assert ex.metadata == metadata

    def test_create_example_with_compositional_vector(self):
        """Test creating example with compositional vector."""
        vector = {"omega": "decompose", "phi": "structure"}
        ex = Example(
            problem="Solve equation",
            solution="x = 2",
            compositional_vector=vector
        )
        assert ex.compositional_vector == vector

    def test_to_prompt_string_basic(self):
        """Test converting example to prompt string."""
        ex = Example(
            problem="What is 2 + 2?",
            solution="4"
        )
        prompt = ex.to_prompt_string()
        assert "Problem: What is 2 + 2?" in prompt
        assert "Solution: 4" in prompt

    def test_to_prompt_string_with_steps(self):
        """Test prompt string includes reasoning steps."""
        ex = Example(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=["Add the numbers", "2 + 2 = 4"]
        )
        prompt = ex.to_prompt_string(include_steps=True)
        assert "Reasoning:" in prompt
        assert "1. Add the numbers" in prompt
        assert "2. 2 + 2 = 4" in prompt

    def test_to_prompt_string_no_steps(self):
        """Test prompt string without steps when disabled."""
        ex = Example(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=["Add the numbers", "2 + 2 = 4"]
        )
        prompt = ex.to_prompt_string(include_steps=False)
        assert "Reasoning:" not in prompt
        assert "Add the numbers" not in prompt

    def test_to_prompt_string_no_solution(self):
        """Test prompt string without solution."""
        ex = Example(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=["Add the numbers"]
        )
        prompt = ex.to_prompt_string(include_solution=False)
        assert "Problem: What is 2 + 2?" in prompt
        assert "Solution: 4" not in prompt

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        ex = Example(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=["Add numbers"],
            compositional_vector={"omega": "analyze"},
            metadata={"domain": "math"}
        )
        data = ex.to_dict()
        assert data["problem"] == "What is 2 + 2?"
        assert data["solution"] == "4"
        assert data["reasoning_steps"] == ["Add numbers"]
        assert data["compositional_vector"] == {"omega": "analyze"}
        assert data["metadata"] == {"domain": "math"}

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "problem": "What is 2 + 2?",
            "solution": "4",
            "reasoning_steps": ["Add numbers"],
            "compositional_vector": {"omega": "analyze"},
            "metadata": {"domain": "math"}
        }
        ex = Example.from_dict(data)
        assert ex.problem == "What is 2 + 2?"
        assert ex.solution == "4"
        assert ex.reasoning_steps == ["Add numbers"]
        assert ex.compositional_vector == {"omega": "analyze"}
        assert ex.metadata == {"domain": "math"}

    def test_round_trip_serialization(self):
        """Test that to_dict/from_dict round-trip works."""
        original = Example(
            problem="Test problem",
            solution="Test solution",
            reasoning_steps=["Step 1", "Step 2"],
            metadata={"key": "value"}
        )
        data = original.to_dict()
        restored = Example.from_dict(data)
        assert restored.problem == original.problem
        assert restored.solution == original.solution
        assert restored.reasoning_steps == original.reasoning_steps
        assert restored.metadata == original.metadata


class TestExampleSet:
    """Tests for the ExampleSet class."""

    def test_create_empty_set(self):
        """Test creating empty example set."""
        es = ExampleSet()
        assert len(es) == 0
        assert es.examples == []

    def test_create_with_examples(self):
        """Test creating set with initial examples."""
        examples = [
            Example("Problem 1", "Solution 1"),
            Example("Problem 2", "Solution 2")
        ]
        es = ExampleSet(examples)
        assert len(es) == 2

    def test_add_example(self):
        """Test adding example to set."""
        es = ExampleSet()
        ex = Example("Problem", "Solution")
        result = es.add(ex)
        assert len(es) == 1
        assert es.examples[0] == ex
        assert result is es  # Check fluent API

    def test_add_from_dict(self):
        """Test adding example from components."""
        es = ExampleSet()
        result = es.add_from_dict(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=["Add numbers"],
            domain="math"
        )
        assert len(es) == 1
        assert es.examples[0].problem == "What is 2 + 2?"
        assert es.examples[0].solution == "4"
        assert es.examples[0].reasoning_steps == ["Add numbers"]
        assert es.examples[0].metadata["domain"] == "math"
        assert result is es  # Check fluent API

    def test_add_from_dict_chaining(self):
        """Test fluent API chaining with add_from_dict."""
        es = (ExampleSet()
              .add_from_dict("Problem 1", "Solution 1")
              .add_from_dict("Problem 2", "Solution 2")
              .add_from_dict("Problem 3", "Solution 3"))
        assert len(es) == 3

    def test_iterate_examples(self):
        """Test iterating over examples."""
        examples = [
            Example("P1", "S1"),
            Example("P2", "S2"),
            Example("P3", "S3")
        ]
        es = ExampleSet(examples)
        collected = list(es)
        assert collected == examples

    def test_retrieve_similar_keyword_basic(self):
        """Test keyword-based similarity retrieval."""
        es = ExampleSet()
        es.add_from_dict("Solve quadratic equation x^2 + 5x + 6", "x = -2 or x = -3")
        es.add_from_dict("Find prime numbers less than 10", "2, 3, 5, 7")
        es.add_from_dict("Solve linear equation 2x + 4 = 8", "x = 2")

        # Query for quadratic problems
        results = es.retrieve_similar("quadratic equation x^2", k=2, method='keyword')
        assert len(results) <= 2
        assert "quadratic" in results[0].problem.lower()

    def test_retrieve_similar_keyword_scoring(self):
        """Test that keyword similarity scores correctly."""
        es = ExampleSet()
        # Add examples with varying similarity
        es.add_from_dict("apple banana cherry", "fruit1")
        es.add_from_dict("apple banana", "fruit2")
        es.add_from_dict("apple", "fruit3")
        es.add_from_dict("orange grape", "fruit4")

        results = es.retrieve_similar("apple banana cherry", k=4, method='keyword')
        # First result should be exact match
        assert results[0].problem == "apple banana cherry"

    def test_retrieve_similar_empty_set(self):
        """Test retrieval from empty set."""
        es = ExampleSet()
        results = es.retrieve_similar("test query", k=3)
        assert results == []

    def test_retrieve_similar_k_greater_than_size(self):
        """Test retrieval when k > number of examples."""
        es = ExampleSet()
        es.add_from_dict("Problem 1", "Solution 1")
        es.add_from_dict("Problem 2", "Solution 2")

        results = es.retrieve_similar("Problem", k=10)
        assert len(results) == 2  # Returns all available

    def test_retrieve_similar_random_method(self):
        """Test random retrieval method."""
        es = ExampleSet()
        for i in range(5):
            es.add_from_dict(f"Problem {i}", f"Solution {i}")

        results = es.retrieve_similar("query", k=3, method='random')
        assert len(results) == 3

    def test_retrieve_similar_embedding_fallback(self):
        """Test that embedding method falls back to keyword."""
        es = ExampleSet()
        es.add_from_dict("Test problem", "Test solution")

        # Embedding method should fall back to keyword
        results = es.retrieve_similar("Test", k=1, method='embedding')
        assert len(results) == 1

    def test_retrieve_similar_invalid_method(self):
        """Test that invalid method raises ValueError."""
        es = ExampleSet()
        es.add_from_dict("Problem", "Solution")

        with pytest.raises(ValueError, match="Unknown similarity method"):
            es.retrieve_similar("query", k=1, method='invalid')

    def test_retrieve_by_metadata(self):
        """Test retrieving examples by metadata."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1", domain="math", difficulty="easy")
        es.add_from_dict("P2", "S2", domain="math", difficulty="hard")
        es.add_from_dict("P3", "S3", domain="logic", difficulty="easy")

        # Filter by single metadata field
        math_examples = es.retrieve_by_metadata(domain="math")
        assert len(math_examples) == 2

        # Filter by multiple fields
        easy_math = es.retrieve_by_metadata(domain="math", difficulty="easy")
        assert len(easy_math) == 1
        assert easy_math[0].problem == "P1"

    def test_retrieve_by_metadata_no_matches(self):
        """Test metadata retrieval with no matches."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1", domain="math")

        results = es.retrieve_by_metadata(domain="physics")
        assert results == []

    def test_sample_random(self):
        """Test random sampling."""
        es = ExampleSet()
        for i in range(10):
            es.add_from_dict(f"Problem {i}", f"Solution {i}")

        sample = es.sample_random(k=3)
        assert len(sample) == 3
        assert all(isinstance(ex, Example) for ex in sample)

    def test_sample_random_k_greater_than_size(self):
        """Test random sampling when k > size."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1")
        es.add_from_dict("P2", "S2")

        sample = es.sample_random(k=5)
        assert len(sample) == 2  # Returns all available

    def test_to_few_shot_prompt_with_query(self):
        """Test generating few-shot prompt with query."""
        es = ExampleSet()
        es.add_from_dict("Solve x^2 = 4", "x = ±2", reasoning_steps=["Take square root"])
        es.add_from_dict("Solve x + 3 = 7", "x = 4", reasoning_steps=["Subtract 3"])

        prompt = es.to_few_shot_prompt(n_examples=2, query="Solve x^2", retrieval_method='keyword')
        assert "Here are some examples:" in prompt
        assert "Example 1:" in prompt
        assert "Problem:" in prompt
        assert "Solution:" in prompt

    def test_to_few_shot_prompt_without_query(self):
        """Test generating few-shot prompt without query (random)."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1")
        es.add_from_dict("P2", "S2")

        prompt = es.to_few_shot_prompt(n_examples=1, query=None)
        assert "Here are some examples:" in prompt
        assert "Example 1:" in prompt

    def test_to_few_shot_prompt_with_steps(self):
        """Test few-shot prompt includes reasoning steps."""
        es = ExampleSet()
        es.add_from_dict(
            "What is 2 + 2?",
            "4",
            reasoning_steps=["Add the numbers", "2 + 2 = 4"]
        )

        prompt = es.to_few_shot_prompt(n_examples=1, include_steps=True)
        assert "Reasoning:" in prompt
        assert "1. Add the numbers" in prompt

    def test_to_few_shot_prompt_without_steps(self):
        """Test few-shot prompt without reasoning steps."""
        es = ExampleSet()
        es.add_from_dict(
            "What is 2 + 2?",
            "4",
            reasoning_steps=["Add the numbers"]
        )

        prompt = es.to_few_shot_prompt(n_examples=1, include_steps=False)
        assert "Reasoning:" not in prompt
        assert "Add the numbers" not in prompt

    def test_to_few_shot_prompt_empty_set(self):
        """Test few-shot prompt from empty set."""
        es = ExampleSet()
        prompt = es.to_few_shot_prompt(n_examples=3)
        assert prompt == ""

    def test_len(self):
        """Test __len__ method."""
        es = ExampleSet()
        assert len(es) == 0
        es.add_from_dict("P1", "S1")
        assert len(es) == 1
        es.add_from_dict("P2", "S2")
        assert len(es) == 2

    def test_repr(self):
        """Test __repr__ method."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1")
        es.add_from_dict("P2", "S2")
        assert repr(es) == "ExampleSet(2 examples)"


class TestPredefinedExampleSets:
    """Tests for predefined example sets."""

    def test_get_math_examples(self):
        """Test math example set."""
        examples = get_math_examples()
        assert len(examples) > 0
        assert isinstance(examples, ExampleSet)

        # Check that examples have expected structure
        for ex in examples:
            assert isinstance(ex, Example)
            assert ex.problem
            assert ex.solution
            assert ex.reasoning_steps is not None

    def test_get_math_examples_domains(self):
        """Test math examples have correct domains."""
        examples = get_math_examples()

        # Check metadata
        arithmetic = examples.retrieve_by_metadata(domain="arithmetic")
        assert len(arithmetic) > 0

        algebra = examples.retrieve_by_metadata(domain="algebra")
        assert len(algebra) > 0

    def test_get_logic_examples(self):
        """Test logic example set."""
        examples = get_logic_examples()
        assert len(examples) > 0
        assert isinstance(examples, ExampleSet)

    def test_get_coding_examples(self):
        """Test coding example set."""
        examples = get_coding_examples()
        assert len(examples) > 0
        assert isinstance(examples, ExampleSet)

        # Check that at least one has code in solution
        has_code = any("def " in ex.solution for ex in examples)
        assert has_code

    def test_predefined_examples_are_retrievable(self):
        """Test that predefined examples can be retrieved by similarity."""
        math_ex = get_math_examples()

        # Query for prime number problems
        results = math_ex.retrieve_similar("prime numbers", k=1, method='keyword')
        assert len(results) > 0
        assert "prime" in results[0].problem.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions for robust behavior."""

    def test_example_with_empty_strings(self):
        """Test creating examples with empty strings."""
        # Empty problem and solution should be allowed
        ex = Example("", "")
        assert ex.problem == ""
        assert ex.solution == ""

        # Should produce valid prompt string
        prompt = ex.to_prompt_string()
        assert isinstance(prompt, str)

    def test_example_with_whitespace_only(self):
        """Test examples with whitespace-only content."""
        ex = Example("   ", "\t\n")
        prompt = ex.to_prompt_string()
        assert isinstance(prompt, str)
        # Whitespace should be preserved
        assert "   " in prompt or "\t" in prompt or prompt.count("\n") > 0

    def test_example_with_unicode_content(self):
        """Test that examples handle Unicode characters correctly."""
        ex = Example(
            "Solve x² + 2x = 0",
            "x = 0 or x = −2",
            reasoning_steps=["Factor: x(x + 2) = 0", "Solutions: x ∈ {0, −2}"]
        )

        prompt = ex.to_prompt_string()
        assert "x²" in prompt
        assert "−2" in prompt  # Unicode minus sign
        assert "∈" in prompt  # Element-of symbol

    def test_example_with_special_regex_characters(self):
        """Test examples with regex special characters don't cause issues."""
        ex = Example(
            "Match pattern: .*+?[](){}|^$",
            "Use escaping: \\.\\*\\+\\?\\[\\]\\(\\)\\{\\}\\|\\^\\$"
        )

        prompt = ex.to_prompt_string()
        assert ".*+?[](){}|^$" in prompt

    def test_example_with_very_long_content(self):
        """Test examples with very long strings."""
        long_problem = "Solve: " + "x + " * 1000 + "1 = 0"
        long_solution = "x = " + "-1" * 500

        ex = Example(long_problem, long_solution)
        prompt = ex.to_prompt_string()

        # Should handle without crashing
        assert len(prompt) > 1000
        assert long_problem in prompt

    def test_retrieve_similar_with_empty_query(self):
        """Test that empty query string is handled gracefully."""
        es = ExampleSet()
        es.add_from_dict("Solve x = 1", "x = 1")
        es.add_from_dict("Solve y = 2", "y = 2")

        results = es.retrieve_similar("", k=2, method='keyword')
        # Should return a list (may be empty or may return all)
        assert isinstance(results, list)

    def test_retrieve_similar_with_whitespace_query(self):
        """Test retrieval with whitespace-only query."""
        es = ExampleSet()
        es.add_from_dict("Test", "Solution")

        results = es.retrieve_similar("   \t\n", k=1, method='keyword')
        assert isinstance(results, list)

    def test_retrieve_similar_with_unicode_query(self):
        """Test retrieval with Unicode characters in query."""
        es = ExampleSet()
        es.add_from_dict("Solve x² + 2x = 0", "x = 0 or x = −2")
        es.add_from_dict("Calculate π × 2", "≈ 6.28")

        # Should handle Unicode in queries
        results = es.retrieve_similar("x² equation", k=1, method='keyword')
        assert isinstance(results, list)
        if len(results) > 0:
            assert "x²" in results[0].problem

    def test_retrieve_similar_k_zero(self):
        """Test retrieval with k=0."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1")
        es.add_from_dict("P2", "S2")

        results = es.retrieve_similar("P1", k=0)
        assert results == []

    def test_retrieve_similar_k_larger_than_set(self):
        """Test retrieval when k exceeds number of examples."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1")
        es.add_from_dict("P2", "S2")

        results = es.retrieve_similar("query", k=100)
        # Should return all available examples
        assert len(results) <= 2

    def test_retrieve_similar_empty_set(self):
        """Test retrieval from empty ExampleSet."""
        es = ExampleSet()
        results = es.retrieve_similar("query", k=5)
        assert results == []

    def test_example_set_with_duplicate_examples(self):
        """Test that duplicate examples can be added."""
        es = ExampleSet()
        es.add_from_dict("P1", "S1")
        es.add_from_dict("P1", "S1")  # Exact duplicate

        assert len(es) == 2  # Both should be stored

    def test_few_shot_prompt_with_no_examples(self):
        """Test generating few-shot prompt from empty set."""
        es = ExampleSet()
        prompt = es.to_few_shot_prompt(n_examples=3, query="test")

        # Should return empty string or handle gracefully
        assert isinstance(prompt, str)
        assert len(prompt) == 0 or "examples" not in prompt.lower()

    def test_metadata_with_special_types(self):
        """Test metadata can contain various Python types."""
        ex = Example(
            "Test problem",
            "Test solution",
            metadata={
                "difficulty": 5,
                "tags": ["math", "algebra"],
                "nested": {"key": "value"},
                "bool_flag": True,
                "float_val": 3.14
            }
        )

        # Metadata should be preserved
        assert ex.metadata["difficulty"] == 5
        assert "math" in ex.metadata["tags"]
        assert ex.metadata["nested"]["key"] == "value"

        # Should serialize/deserialize correctly
        data = ex.to_dict()
        restored = Example.from_dict(data)
        assert restored.metadata == ex.metadata
