"""
Tests for mcts_reasoning/compositional/rag.py

Tests the RAG (Retrieval-Augmented Generation) system including:
- CompositionalGuidance
- CompositionalRAGStore (Type a: problem -> dimensions)
- SolutionRAGStore (Type b: problem -> full solutions)
"""

import pytest
from mcts_reasoning.compositional.rag import (
    CompositionalGuidance,
    CompositionalRAGStore,
    SolutionRAGStore,
    get_math_compositional_rag,
    get_coding_compositional_rag,
    get_logic_compositional_rag,
)
from mcts_reasoning.compositional import (
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
)
from mcts_reasoning.compositional.examples import Example, ExampleSet


class TestCompositionalGuidance:
    """Tests for CompositionalGuidance class."""

    def test_create_basic_guidance(self):
        """Test creating basic guidance."""
        guidance = CompositionalGuidance(
            problem_pattern="quadratic equation",
            problem_keywords=["quadratic", "x^2"]
        )
        assert guidance.problem_pattern == "quadratic equation"
        assert guidance.problem_keywords == ["quadratic", "x^2"]

    def test_create_guidance_with_operations(self):
        """Test creating guidance with recommended operations."""
        guidance = CompositionalGuidance(
            problem_pattern="quadratic",
            recommended_operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.ANALYZE]
        )
        assert len(guidance.recommended_operations) == 2
        assert CognitiveOperation.DECOMPOSE in guidance.recommended_operations

    def test_create_guidance_with_all_dimensions(self):
        """Test creating guidance with all compositional dimensions."""
        guidance = CompositionalGuidance(
            problem_pattern="test",
            recommended_operations=[CognitiveOperation.DECOMPOSE],
            recommended_focuses=[FocusAspect.STRUCTURE],
            recommended_styles=[ReasoningStyle.SYSTEMATIC],
            recommended_connections=[ConnectionType.THEREFORE],
            recommended_formats=[OutputFormat.STEPS]
        )
        assert guidance.recommended_operations is not None
        assert guidance.recommended_focuses is not None
        assert guidance.recommended_styles is not None
        assert guidance.recommended_connections is not None
        assert guidance.recommended_formats is not None

    def test_create_guidance_with_domain(self):
        """Test creating guidance with domain metadata."""
        guidance = CompositionalGuidance(
            problem_pattern="test",
            domain="mathematics",
            difficulty="hard"
        )
        assert guidance.domain == "mathematics"
        assert guidance.difficulty == "hard"

    def test_matches_problem_by_keyword(self):
        """Test problem matching by keywords."""
        guidance = CompositionalGuidance(
            problem_pattern="quadratic",
            problem_keywords=["quadratic", "x^2", "x²"]
        )

        # Should match if keywords present
        score1 = guidance.matches_problem("Solve the quadratic equation x^2 + 5x + 6 = 0")
        assert score1 > 0

        # Should not match unrelated problem
        score2 = guidance.matches_problem("Find all prime numbers less than 20")
        assert score2 == 0

    def test_matches_problem_by_pattern(self):
        """Test problem matching by pattern."""
        guidance = CompositionalGuidance(
            problem_pattern="prime numbers",
            problem_keywords=[]
        )

        score = guidance.matches_problem("Find all prime numbers less than 20")
        assert score == 0.8  # Pattern match returns 0.8

    def test_matches_problem_keyword_scoring(self):
        """Test keyword match scoring."""
        guidance = CompositionalGuidance(
            problem_pattern="test",
            problem_keywords=["apple", "banana", "cherry"]
        )

        # All keywords match
        score_all = guidance.matches_problem("apple banana cherry")
        assert score_all == 1.0

        # Partial match
        score_partial = guidance.matches_problem("apple banana")
        assert 0 < score_partial < 1.0

    def test_matches_problem_case_insensitive(self):
        """Test that matching is case insensitive."""
        guidance = CompositionalGuidance(
            problem_pattern="test",
            problem_keywords=["Quadratic", "Equation"]
        )

        score = guidance.matches_problem("solve quadratic equation")
        assert score > 0

    def test_matches_problem_no_match(self):
        """Test that non-matching problems return 0."""
        guidance = CompositionalGuidance(
            problem_pattern="quadratic equations",  # Won't match substring
            problem_keywords=[]  # Empty keywords, so falls through to pattern check
        )

        score = guidance.matches_problem("completely unrelated problem about primes")
        assert score == 0.0

    def test_to_weights_dict_from_recommendations(self):
        """Test converting recommendations to weight dictionary."""
        guidance = CompositionalGuidance(
            problem_pattern="test",
            recommended_operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.ANALYZE],
            recommended_focuses=[FocusAspect.STRUCTURE],
            recommended_styles=[ReasoningStyle.SYSTEMATIC]
        )

        weights = guidance.to_weights_dict()
        assert 'cognitive_op' in weights
        assert CognitiveOperation.DECOMPOSE in weights['cognitive_op']
        assert weights['cognitive_op'][CognitiveOperation.DECOMPOSE] == 3.0

        assert 'focus' in weights
        assert FocusAspect.STRUCTURE in weights['focus']

        assert 'style' in weights
        assert ReasoningStyle.SYSTEMATIC in weights['style']

    def test_to_weights_dict_with_explicit_weights(self):
        """Test that explicit weights override defaults."""
        explicit_weights = {
            'cognitive_op': {CognitiveOperation.VERIFY: 5.0}
        }
        guidance = CompositionalGuidance(
            problem_pattern="test",
            weights=explicit_weights
        )

        weights = guidance.to_weights_dict()
        assert weights == explicit_weights

    def test_to_weights_dict_empty_guidance(self):
        """Test weights dict for guidance with no recommendations."""
        guidance = CompositionalGuidance(
            problem_pattern="test"
        )
        weights = guidance.to_weights_dict()
        assert weights == {}

    def test_to_weights_dict_with_all_dimensions(self):
        """Test weights dict with all compositional dimensions."""
        guidance = CompositionalGuidance(
            problem_pattern="test",
            recommended_operations=[CognitiveOperation.VERIFY],
            recommended_focuses=[FocusAspect.CORRECTNESS],
            recommended_styles=[ReasoningStyle.FORMAL],
            recommended_connections=[ConnectionType.THEREFORE],
            recommended_formats=[OutputFormat.MATHEMATICAL]
        )
        weights = guidance.to_weights_dict()

        # Check all dimensions are present
        assert 'cognitive_op' in weights
        assert 'focus' in weights
        assert 'style' in weights
        assert 'connection' in weights
        assert 'output_format' in weights

        # Check specific values
        assert ConnectionType.THEREFORE in weights['connection']
        assert weights['connection'][ConnectionType.THEREFORE] == 2.0
        assert OutputFormat.MATHEMATICAL in weights['output_format']
        assert weights['output_format'][OutputFormat.MATHEMATICAL] == 2.0

    def test_success_rate_tracking(self):
        """Test success rate tracking."""
        guidance = CompositionalGuidance(problem_pattern="test")
        assert guidance.success_rate == 0.0


class TestCompositionalRAGStore:
    """Tests for CompositionalRAGStore class."""

    def test_create_empty_store(self):
        """Test creating empty RAG store."""
        store = CompositionalRAGStore()
        assert len(store) == 0
        assert store.guidance == []

    def test_create_with_guidance(self):
        """Test creating store with initial guidance."""
        guidance_list = [
            CompositionalGuidance("pattern1"),
            CompositionalGuidance("pattern2")
        ]
        store = CompositionalRAGStore(guidance_list)
        assert len(store) == 2

    def test_add_guidance(self):
        """Test adding guidance to store."""
        store = CompositionalRAGStore()
        guidance = CompositionalGuidance("test pattern")
        result = store.add_guidance(guidance)

        assert len(store) == 1
        assert store.guidance[0] == guidance
        assert result is store  # Check fluent API

    def test_add_from_components(self):
        """Test adding guidance from components."""
        store = CompositionalRAGStore()
        result = store.add(
            problem_pattern="quadratic equation",
            keywords=["quadratic", "x^2"],
            operations=[CognitiveOperation.DECOMPOSE],
            focuses=[FocusAspect.STRUCTURE],
            styles=[ReasoningStyle.SYSTEMATIC],
            domain="algebra"
        )

        assert len(store) == 1
        guidance = store.guidance[0]
        assert guidance.problem_pattern == "quadratic equation"
        assert guidance.problem_keywords == ["quadratic", "x^2"]
        assert guidance.domain == "algebra"
        assert result is store  # Check fluent API

    def test_add_chaining(self):
        """Test fluent API chaining."""
        store = (CompositionalRAGStore()
                 .add("pattern1", keywords=["key1"])
                 .add("pattern2", keywords=["key2"])
                 .add("pattern3", keywords=["key3"]))
        assert len(store) == 3

    def test_retrieve_by_problem(self):
        """Test retrieving guidance by problem."""
        store = CompositionalRAGStore()
        store.add("quadratic", keywords=["quadratic", "x^2"])
        store.add("prime", keywords=["prime", "divisible"])
        store.add("linear", keywords=["linear", "x"])

        results = store.retrieve("Solve quadratic equation x^2 + 5x + 6", k=2)
        assert len(results) <= 2
        assert any("quadratic" in g.problem_pattern for g in results)

    def test_retrieve_returns_best_matches(self):
        """Test that retrieve returns best matching guidance."""
        store = CompositionalRAGStore()
        store.add("exact match", keywords=["apple", "banana", "cherry"])
        store.add("partial match", keywords=["apple", "banana"])
        store.add("weak match", keywords=["apple"])
        store.add("no match", keywords=["orange"])

        results = store.retrieve("apple banana cherry", k=3)
        # Should get best matches (not zero-score matches)
        assert len(results) == 3
        assert all(g.matches_problem("apple banana cherry") > 0 for g in results)

    def test_retrieve_empty_store(self):
        """Test retrieve from empty store."""
        store = CompositionalRAGStore()
        results = store.retrieve("test problem", k=3)
        assert results == []

    def test_retrieve_filters_zero_scores(self):
        """Test that retrieve filters out zero-score matches."""
        store = CompositionalRAGStore()
        store.add("unrelated", keywords=["xyz", "abc"])

        results = store.retrieve("completely different problem", k=5)
        assert len(results) == 0  # No matches with score > 0

    def test_get_recommended_weights_single_guidance(self):
        """Test getting weights from single guidance."""
        store = CompositionalRAGStore()
        store.add(
            "quadratic",
            keywords=["quadratic"],
            operations=[CognitiveOperation.DECOMPOSE],
            focuses=[FocusAspect.STRUCTURE]
        )

        weights = store.get_recommended_weights("solve quadratic equation", merge_strategy='first')
        assert 'cognitive_op' in weights
        assert CognitiveOperation.DECOMPOSE in weights['cognitive_op']

    def test_get_recommended_weights_merge_average(self):
        """Test averaging weights from multiple guidance."""
        store = CompositionalRAGStore()
        store.add(
            "pattern1",
            keywords=["test"],
            operations=[CognitiveOperation.DECOMPOSE]
        )
        store.add(
            "pattern2",
            keywords=["test"],
            operations=[CognitiveOperation.DECOMPOSE]
        )

        weights = store.get_recommended_weights("test problem", merge_strategy='average')
        # Both recommend DECOMPOSE with weight 3.0, average should be 3.0
        assert weights['cognitive_op'][CognitiveOperation.DECOMPOSE] == 3.0

    def test_get_recommended_weights_merge_max(self):
        """Test max weights from multiple guidance."""
        store = CompositionalRAGStore()

        # First guidance with weight 3.0 (default)
        guidance1 = CompositionalGuidance(
            problem_pattern="p1",
            problem_keywords=["test"],
            recommended_operations=[CognitiveOperation.DECOMPOSE]
        )

        # Second guidance with explicit higher weight
        guidance2 = CompositionalGuidance(
            problem_pattern="p2",
            problem_keywords=["test"],
            weights={
                'cognitive_op': {CognitiveOperation.DECOMPOSE: 5.0}
            }
        )

        store.add_guidance(guidance1)
        store.add_guidance(guidance2)

        weights = store.get_recommended_weights("test problem", merge_strategy='max')
        # Should take max weight
        assert weights['cognitive_op'][CognitiveOperation.DECOMPOSE] == 5.0

    def test_get_recommended_weights_no_matches(self):
        """Test getting weights when no guidance matches."""
        store = CompositionalRAGStore()
        store.add("unrelated", keywords=["xyz"])

        weights = store.get_recommended_weights("completely different problem")
        assert weights == {}

    def test_update_success_rate(self):
        """Test updating success rate for guidance."""
        store = CompositionalRAGStore()
        store.add("test", keywords=["test"])

        # Initial success rate should be 0
        assert store.guidance[0].success_rate == 0.0

        # Update with success
        store.update_success_rate("test problem", success=True)
        assert store.guidance[0].success_rate == 1.0

        # Update with failure
        store.update_success_rate("test problem", success=False)
        # Running average: (1.0 + 0.0) / 2 = 0.5
        assert store.guidance[0].success_rate == 0.5

    def test_update_success_rate_multiple_updates(self):
        """Test multiple success rate updates."""
        store = CompositionalRAGStore()
        store.add("test", keywords=["test"])

        # Multiple successes
        for _ in range(3):
            store.update_success_rate("test problem", success=True)
        assert store.guidance[0].success_rate == 1.0

        # One failure
        store.update_success_rate("test problem", success=False)
        # 3 successes + 1 failure = 0.75
        assert store.guidance[0].success_rate == 0.75

    def test_len(self):
        """Test __len__ method."""
        store = CompositionalRAGStore()
        assert len(store) == 0
        store.add("p1")
        assert len(store) == 1
        store.add("p2")
        assert len(store) == 2

    def test_repr(self):
        """Test __repr__ method."""
        store = CompositionalRAGStore()
        store.add("p1")
        store.add("p2")
        assert repr(store) == "CompositionalRAGStore(2 guidance entries)"

    # NOTE: save() and load() methods are currently placeholders.
    # Tests will be added when these methods are properly implemented.


class TestSolutionRAGStore:
    """Tests for SolutionRAGStore class."""

    def test_create_empty_store(self):
        """Test creating empty solution store."""
        store = SolutionRAGStore()
        assert len(store) == 0
        assert isinstance(store.examples, ExampleSet)

    def test_create_with_example_set(self):
        """Test creating store with existing ExampleSet."""
        examples = ExampleSet()
        examples.add_from_dict("Problem", "Solution")

        store = SolutionRAGStore(examples)
        assert len(store) == 1

    def test_add_example(self):
        """Test adding example to store."""
        store = SolutionRAGStore()
        example = Example("Problem", "Solution")
        result = store.add_example(example)

        assert len(store) == 1
        assert result is store  # Check fluent API

    def test_add_from_components(self):
        """Test adding example from components."""
        store = SolutionRAGStore()
        result = store.add(
            problem="What is 2 + 2?",
            solution="4",
            reasoning_steps=["Add the numbers"],
            domain="math"
        )

        assert len(store) == 1
        assert store.examples.examples[0].problem == "What is 2 + 2?"
        assert result is store  # Check fluent API

    def test_add_chaining(self):
        """Test fluent API chaining."""
        store = (SolutionRAGStore()
                 .add("P1", "S1")
                 .add("P2", "S2")
                 .add("P3", "S3"))
        assert len(store) == 3

    def test_retrieve_similar_examples(self):
        """Test retrieving similar examples."""
        store = SolutionRAGStore()
        store.add("Solve quadratic equation x^2 + 5x + 6", "x = -2 or x = -3")
        store.add("Find prime numbers less than 10", "2, 3, 5, 7")
        store.add("Solve linear equation 2x + 4 = 8", "x = 2")

        results = store.retrieve("quadratic equation x^2", k=2)
        assert len(results) <= 2
        assert isinstance(results[0], Example)

    def test_retrieve_uses_keyword_method(self):
        """Test that retrieve uses keyword similarity by default."""
        store = SolutionRAGStore()
        store.add("apple banana cherry", "fruit")
        store.add("dog cat mouse", "animal")

        results = store.retrieve("apple banana", k=1, method='keyword')
        assert "apple" in results[0].problem

    def test_to_few_shot_prompt(self):
        """Test generating few-shot prompt."""
        store = SolutionRAGStore()
        store.add(
            "What is 2 + 2?",
            "4",
            reasoning_steps=["Add the numbers", "2 + 2 = 4"]
        )
        store.add(
            "What is 3 + 3?",
            "6",
            reasoning_steps=["Add the numbers", "3 + 3 = 6"]
        )

        prompt = store.to_few_shot_prompt("What is 5 + 5?", n_examples=2, include_steps=True)
        assert "Here are some examples:" in prompt
        assert "Problem:" in prompt
        assert "Solution:" in prompt
        assert "Reasoning:" in prompt

    def test_to_few_shot_prompt_without_steps(self):
        """Test few-shot prompt without reasoning steps."""
        store = SolutionRAGStore()
        store.add("Problem", "Solution", reasoning_steps=["Step 1"])

        prompt = store.to_few_shot_prompt("Query", n_examples=1, include_steps=False)
        assert "Reasoning:" not in prompt

    def test_len(self):
        """Test __len__ method."""
        store = SolutionRAGStore()
        assert len(store) == 0
        store.add("P1", "S1")
        assert len(store) == 1

    def test_repr(self):
        """Test __repr__ method."""
        store = SolutionRAGStore()
        store.add("P1", "S1")
        store.add("P2", "S2")
        assert repr(store) == "SolutionRAGStore(2 examples)"


class TestPredefinedRAGStores:
    """Tests for predefined RAG stores."""

    def test_get_math_compositional_rag(self):
        """Test predefined math compositional RAG."""
        store = get_math_compositional_rag()
        assert isinstance(store, CompositionalRAGStore)
        assert len(store) > 0

    def test_math_rag_quadratic_guidance(self):
        """Test that math RAG has quadratic equation guidance."""
        store = get_math_compositional_rag()
        results = store.retrieve("solve quadratic equation x^2 + 5x + 6 = 0", k=1)
        assert len(results) > 0
        guidance = results[0]
        assert guidance.domain == "algebra"

    def test_math_rag_prime_guidance(self):
        """Test that math RAG has prime number guidance."""
        store = get_math_compositional_rag()
        results = store.retrieve("find all prime numbers less than 30", k=1)
        assert len(results) > 0

    def test_math_rag_proof_guidance(self):
        """Test that math RAG has proof guidance."""
        store = get_math_compositional_rag()
        results = store.retrieve("prove that the sum is even", k=1)
        assert len(results) > 0
        guidance = results[0]
        assert CognitiveOperation.VERIFY in guidance.recommended_operations

    def test_get_coding_compositional_rag(self):
        """Test predefined coding compositional RAG."""
        store = get_coding_compositional_rag()
        assert isinstance(store, CompositionalRAGStore)
        assert len(store) > 0

    def test_coding_rag_algorithm_guidance(self):
        """Test that coding RAG has algorithm guidance."""
        store = get_coding_compositional_rag()
        results = store.retrieve("implement algorithm for sorting", k=1)
        assert len(results) > 0

    def test_coding_rag_debug_guidance(self):
        """Test that coding RAG has debug guidance."""
        store = get_coding_compositional_rag()
        results = store.retrieve("debug this code that has an error", k=1)
        assert len(results) > 0
        guidance = results[0]
        assert guidance.domain == "debugging"

    def test_coding_rag_optimize_guidance(self):
        """Test that coding RAG has optimization guidance."""
        store = get_coding_compositional_rag()
        results = store.retrieve("optimize this function for better performance", k=1)
        assert len(results) > 0

    def test_get_logic_compositional_rag(self):
        """Test predefined logic compositional RAG."""
        store = get_logic_compositional_rag()
        assert isinstance(store, CompositionalRAGStore)
        assert len(store) > 0

    def test_logic_rag_reasoning_guidance(self):
        """Test that logic RAG has logical reasoning guidance."""
        store = get_logic_compositional_rag()
        results = store.retrieve("if premise then conclusion", k=1)
        assert len(results) > 0

    def test_logic_rag_puzzle_guidance(self):
        """Test that logic RAG has puzzle guidance."""
        store = get_logic_compositional_rag()
        results = store.retrieve("solve this puzzle with constraints", k=1)
        assert len(results) > 0

    def test_predefined_stores_have_weights(self):
        """Test that predefined stores provide valid weights."""
        stores = [
            get_math_compositional_rag(),
            get_coding_compositional_rag(),
            get_logic_compositional_rag()
        ]

        for store in stores:
            # Each store should have guidance
            assert len(store) > 0

            # Each guidance should provide weights
            for guidance in store.guidance:
                weights = guidance.to_weights_dict()
                assert isinstance(weights, dict)
                # Should have at least cognitive_op or other dimensions
                assert len(weights) > 0


class TestRAGEdgeCases:
    """Test edge cases and boundary conditions for RAG stores."""

    def test_compositional_guidance_with_empty_keywords(self):
        """Test guidance with no keywords falls back to pattern matching."""
        from mcts_reasoning.compositional.rag import CompositionalGuidance
        from mcts_reasoning.compositional import CognitiveOperation

        guidance = CompositionalGuidance(
            problem_pattern="test pattern",
            problem_keywords=[]  # Empty!
        )

        # Should fall back to pattern matching
        score = guidance.matches_problem("This has test pattern in it")
        assert score == 0.8  # Pattern match score

        score_no_match = guidance.matches_problem("Different text")
        assert score_no_match < score

    def test_compositional_guidance_with_unicode_keywords(self):
        """Test guidance with Unicode in keywords."""
        from mcts_reasoning.compositional.rag import CompositionalGuidance

        guidance = CompositionalGuidance(
            problem_pattern="math",
            problem_keywords=["π", "θ", "²", "√"]
        )

        score = guidance.matches_problem("Calculate π × radius²")
        assert score > 0

    def test_compositional_rag_store_empty_retrieve(self):
        """Test retrieving from empty RAG store."""
        store = CompositionalRAGStore()
        results = store.retrieve("any problem", k=5)
        assert results == []

    def test_compositional_rag_store_no_matches(self):
        """Test retrieval when no guidance matches."""
        store = CompositionalRAGStore()
        store.add("specific keyword", keywords=["veryrareword12345"])

        results = store.retrieve("completely different problem", k=5)
        # Should return empty or low-score matches filtered out
        assert isinstance(results, list)

    def test_solution_rag_store_empty_retrieve(self):
        """Test retrieving from empty solution RAG store."""
        store = SolutionRAGStore()
        results = store.retrieve("problem", k=3)
        assert results == []

    def test_solution_rag_few_shot_empty(self):
        """Test few-shot prompt generation from empty store."""
        store = SolutionRAGStore()
        prompt = store.to_few_shot_prompt("problem", n_examples=3)
        assert isinstance(prompt, str)
        assert len(prompt) == 0

    def test_get_recommended_weights_no_matches(self):
        """Test weight generation when no guidance matches."""
        store = CompositionalRAGStore()
        store.add("specific", keywords=["veryspecific"])

        weights = store.get_recommended_weights("different problem")
        # Should return empty dict when no matches
        assert weights == {}

    def test_get_recommended_weights_merge_strategies(self):
        """Test different merge strategies for multiple matching guidance."""
        from mcts_reasoning.compositional import CognitiveOperation

        store = CompositionalRAGStore()
        store.add("test1", keywords=["test"], operations=[CognitiveOperation.ANALYZE])
        store.add("test2", keywords=["test"], operations=[CognitiveOperation.DECOMPOSE])

        # Test different merge strategies
        weights_avg = store.get_recommended_weights("test problem", merge_strategy='average')
        weights_max = store.get_recommended_weights("test problem", merge_strategy='max')
        weights_first = store.get_recommended_weights("test problem", merge_strategy='first')

        # All should return valid weights
        assert 'cognitive_op' in weights_avg
        assert 'cognitive_op' in weights_max
        assert 'cognitive_op' in weights_first

    def test_compositional_guidance_matches_problem_edge_cases(self):
        """Test pattern matching with edge case inputs."""
        from mcts_reasoning.compositional.rag import CompositionalGuidance

        guidance = CompositionalGuidance(
            problem_pattern="test",
            problem_keywords=["key1", "key2"]
        )

        # Empty string
        assert guidance.matches_problem("") == 0.0

        # Whitespace only
        assert guidance.matches_problem("   ") == 0.0

        # Case insensitivity
        score_lower = guidance.matches_problem("key1 problem")
        score_upper = guidance.matches_problem("KEY1 PROBLEM")
        assert score_lower == score_upper

    def test_success_rate_tracking_edge_cases(self):
        """Test success rate tracking with edge cases."""
        from mcts_reasoning.compositional.rag import CompositionalGuidance

        store = CompositionalRAGStore()
        store.add("test", keywords=["test"])

        # Update success rate multiple times
        store.update_success_rate("test problem", success=True)
        store.update_success_rate("test problem", success=True)
        store.update_success_rate("test problem", success=False)

        # Success rate should be updated (implementation dependent)
        results = store.retrieve("test problem", k=1)
        if results:
            # Success rate should be calculated correctly
            assert 0 <= results[0].success_rate <= 1.0

    def test_weights_dict_with_no_recommendations(self):
        """Test weight dict generation with empty recommendations."""
        from mcts_reasoning.compositional.rag import CompositionalGuidance

        guidance = CompositionalGuidance(
            problem_pattern="test",
            # No recommended operations, focuses, etc.
        )

        weights = guidance.to_weights_dict()
        # Should return empty dict or handle gracefully
        assert isinstance(weights, dict)

    def test_solution_rag_with_very_long_examples(self):
        """Test solution RAG with very long content."""
        store = SolutionRAGStore()

        long_problem = "Solve: " + "x + " * 500 + "1 = 0"
        long_solution = "x = " + "-1 " * 250
        long_steps = ["Step " + str(i) for i in range(100)]

        store.add(long_problem, long_solution, reasoning_steps=long_steps)

        # Should handle without crashing
        results = store.retrieve(long_problem[:100], k=1)
        assert len(results) > 0

    def test_compositional_rag_with_all_dimensions(self):
        """Test guidance with all dimensions specified."""
        from mcts_reasoning.compositional import (
            CognitiveOperation, FocusAspect, ReasoningStyle,
            ConnectionType, OutputFormat
        )

        store = CompositionalRAGStore()
        store.add(
            "comprehensive",
            keywords=["test"],
            operations=[CognitiveOperation.ANALYZE],
            focuses=[FocusAspect.STRUCTURE],
            styles=[ReasoningStyle.SYSTEMATIC]
            # Note: connections and formats not supported in add() yet
        )

        weights = store.get_recommended_weights("test problem")
        # Should have weights for specified dimensions
        assert 'cognitive_op' in weights
        assert 'focus' in weights
        assert 'style' in weights
